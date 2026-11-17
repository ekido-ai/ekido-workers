"""llm-lora/v1: LoRA fine-tuning of a small causal language model on a customer's recipe dataset.

The same rules as the other workloads keep a job movable between devices:
- the base model is pinned to a commit, the dataset to its SHA-256, and LoRA's initial weights to the job seed;
- every example is padded to the same length, so memory use stays what the probe measured;
- the loss counts only the answer tokens (the prompt and the padding are masked with -100), so the model learns to
  answer, not to repeat the question.

The dataset is one JSONL file in the run's folder (dataset.jsonl), fetched by the agent from the artifact store and
checked here against the spec's hash. Each line holds a "split" ("train" or "test") and the recipe's columns.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Sequence

import torch
from ekido_core.recipes import recipe

IGNORE = -100


def load_tokenizer(model_id: str, revision: str | None = None):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_id, revision=revision)


def load_model(model_id: str, revision: str | None = None, *, device: str = 'cpu',
               dtype: torch.dtype = torch.float32, load_in_4bit: bool = False):
    from transformers import AutoModelForCausalLM
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        return AutoModelForCausalLM.from_pretrained(model_id, revision=revision,
                                                     quantization_config=config, device_map=device)
    return AutoModelForCausalLM.from_pretrained(model_id, revision=revision, torch_dtype=dtype).to(device)


def load_rows(path: Path, sha256: str, split: str = "train") -> list[dict]:
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError("dataset.jsonl doesn't match the job's dataset.sha256")
    rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
    return [r for r in rows if r.get("split") == split]


def encode(tokenizer, examples: Sequence[dict], max_len: int) -> dict[str, torch.Tensor]:
    """Prompt + completion + end-of-text, padded or cut to max_len. Labels are -100 except on the answer."""
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    ids, masks, labels = [], [], []
    for example in examples:
        prompt = tokenizer(example["prompt"], add_special_tokens=False)["input_ids"]
        answer = tokenizer(example["completion"], add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
        # Keep the whole answer if possible; cut the prompt from its start, where the instruction text is least needed.
        room = max_len - len(answer)
        if room < 1:
            prompt, answer = prompt[-1:], answer[: max_len - 1]
        elif len(prompt) > room:
            prompt = prompt[-room:]
        tokens = prompt + answer
        target = [IGNORE] * len(prompt) + answer
        fill = max_len - len(tokens)
        ids.append(tokens + [pad] * fill)
        masks.append([1] * len(tokens) + [0] * fill)
        labels.append(target + [IGNORE] * fill)
    return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks), "labels": torch.tensor(labels)}


def _pretrained(spec: dict, exec_config: dict, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_spec = spec["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_spec["id"], revision=model_spec["revision"])
    model = AutoModelForCausalLM.from_pretrained(
        model_spec["id"], revision=model_spec["revision"], attn_implementation=exec_config.get("attention_impl", "sdpa"), torch_dtype=torch.float32
    )
    return tokenizer, model


def build_llm(spec: dict, exec_config: dict, device: str, data_dir: Path | None, with_data: bool = True,
              loader: Callable | None = None, canary_examples: int = 16):
    """Returns (model, dataset_size, batch_fn, canary_batch). `loader` lets tests use a tiny model."""
    from peft import LoraConfig, TaskType, get_peft_model

    tokenizer, base = (loader or _pretrained)(spec, exec_config, device)
    lora = spec["lora"]
    torch.manual_seed(spec["seed"])  # LoRA's initial weights depend only on the job seed.
    model = get_peft_model(
        base,
        LoraConfig(task_type=TaskType.CAUSAL_LM, r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"], target_modules=lora["target_modules"]),
    ).to(device)
    # Evaluation generates text, so it needs the tokenizer that belongs to these weights.
    model.ekido_tokenizer = tokenizer
    if exec_config.get("grad_checkpointing"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    if not with_data:
        return model, 0, None, {}
    if data_dir is None:
        raise ValueError("llm-lora/v1 needs the run folder that holds dataset.jsonl")
    rows = load_rows(Path(data_dir) / "dataset.jsonl", spec["dataset"]["sha256"])
    if len(rows) != spec["dataset"]["train_rows"]:
        raise ValueError(f"dataset has {len(rows)} training rows, the spec says {spec['dataset']['train_rows']}")
    r = recipe(spec["recipe_id"])
    encoded = encode(tokenizer, [r.example(row) for row in rows], spec["max_seq_length"])

    def batch_fn(indices: Sequence[int]) -> dict[str, torch.Tensor]:
        index = torch.as_tensor(list(indices), dtype=torch.long) % len(rows)
        return {k: v[index] for k, v in encoded.items()}

    return model, len(rows), batch_fn, batch_fn(range(min(canary_examples, len(rows))))


@torch.no_grad()
def generate_answers(model, tokenizer, prompts: Sequence[str], *, max_new_tokens: int, max_prompt_tokens: int, device: str) -> list[str]:
    """The model's answer to each prompt. Greedy and one prompt at a time, so the same weights always give the
    same text, whatever the device or batch size."""
    model.eval()
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    answers = []
    for prompt in prompts:
        ids = tokenizer(prompt, add_special_tokens=False)["input_ids"][-max_prompt_tokens:]
        inputs = torch.tensor([ids], device=device)
        output = model.generate(
            input_ids=inputs, attention_mask=torch.ones_like(inputs), max_new_tokens=max_new_tokens, do_sample=False,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=pad,
        )
        answers.append(tokenizer.decode(output[0, len(ids):].tolist(), skip_special_tokens=True))
    return answers
