"""The llm-lora/v1 training workload.

One job spec (validated by ``ekido_core.job_spec``) plus one dataset blob in, a trained
LoRA adapter and an evidence file out. The rules that make a run resumable on different
hardware all come from ``ekido_core`` and are executed here, not reinvented:

- which examples a step trains on is a pure function of the spec
  (``ekido_core.logical.logical_batch_indices``), so a resume needs no sampler state;
- the learning rate is a pure function of the step (``logical.learning_rate``);
- how the logical batch is split into micro-batches is execution detail only, chosen by
  the adaptation compiler from a real probe (``compiler.compile_for_target``), never
  changing what a logical step means;
- checkpoints are GPCK/1 (``ekido_core.gpck``): trainable safetensors + optimizer
  safetensors + a manifest carrying the spec's hash, so a checkpoint from the wrong job
  is refused at load.

Dataset blob: one JSONL file. Each line is one recipe row plus a ``split`` field
(``"train"`` or ``"test"``); the spec's ``dataset`` section pins the blob's SHA-256, byte
size and row counts, and a blob that does not match its spec is refused before any
training happens. The test rows are the user's own examples and are never trained on.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch

from ekido_core.canonical import document_hash
from ekido_core.compiler import Target, compile_for_target
from ekido_core.gpck import load_checkpoint, optimizer_state_to_tensors, restore_optimizer_state, write_checkpoint
from ekido_core.job_spec import spec_problems
from ekido_core.logical import (
    check_exec_config,
    learning_rate,
    logical_batch_indices,
    micro_batch_seed,
    micro_loss_weight,
    split_micro_batches,
)
from ekido_core.recipes import eval_set_hash, passes, recipe, score

from .llm import generate_answers, load_model, load_tokenizer
from .model_probe import make_probe

DEFAULT_CPU_CAP_MIB = 4096.0


# ---------------------------------------------------------------- dataset blob


def read_dataset(path: Path) -> tuple[list[dict], list[dict], str, int]:
    """(train rows, test rows, sha256 of the file bytes, byte size)."""
    raw = Path(path).read_bytes()
    train_rows: list[dict] = []
    test_rows: list[dict] = []
    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        split = row.pop("split", None)
        if split == "train":
            train_rows.append(row)
        elif split == "test":
            test_rows.append(row)
        else:
            raise ValueError(f"dataset row has no valid split: {line[:80]!r}")
    return train_rows, test_rows, hashlib.sha256(raw).hexdigest(), len(raw)


def write_dataset(path: Path, train_rows: list[dict], test_rows: list[dict]) -> tuple[str, int]:
    """Writes a blob and returns (sha256, bytes) for the spec's dataset section."""
    lines = [json.dumps({"split": "train", **row}, ensure_ascii=False) for row in train_rows]
    lines += [json.dumps({"split": "test", **row}, ensure_ascii=False) for row in test_rows]
    raw = ("\n".join(lines) + "\n").encode("utf-8")
    Path(path).write_bytes(raw)
    return hashlib.sha256(raw).hexdigest(), len(raw)


def _check_dataset(spec: dict, train_rows: list[dict], test_rows: list[dict], digest: str, nbytes: int) -> None:
    declared = spec["dataset"]
    if digest != declared["sha256"]:
        raise ValueError("dataset sha256 does not match the spec")
    if nbytes != declared["bytes"]:
        raise ValueError("dataset byte size does not match the spec")
    if len(train_rows) != declared["train_rows"] or len(test_rows) != declared["test_rows"]:
        raise ValueError(
            f"dataset row counts do not match the spec: {len(train_rows)} train, {len(test_rows)} test"
        )


# ---------------------------------------------------------------- examples


def encode_example(tokenizer, example: dict, max_seq_length: int) -> tuple[list[int], list[int]]:
    """Prompt tokens are masked out of the loss; the completion ends with EOS so the
    model learns when to stop."""
    prompt_ids = tokenizer(example["prompt"], add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(example["completion"], add_special_tokens=False)["input_ids"]
    completion_ids = completion_ids + [tokenizer.eos_token_id]
    if len(completion_ids) >= max_seq_length:
        completion_ids = completion_ids[:max_seq_length - 1] + [tokenizer.eos_token_id]
        prompt_ids = []
    elif len(prompt_ids) + len(completion_ids) > max_seq_length:
        prompt_ids = prompt_ids[-(max_seq_length - len(completion_ids)):]
    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids
    return input_ids, labels


def _collate(encoded: list[tuple[list[int], list[int]]], pad_id: int, device: str) -> dict:
    longest = max(len(ids) for ids, _ in encoded)
    input_ids, labels, attention = [], [], []
    for ids, lab in encoded:
        pad = longest - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        labels.append(lab + [-100] * pad)
        attention.append([1] * len(ids) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention, dtype=torch.long, device=device),
    }


# ---------------------------------------------------------------- model


def apply_lora(model, spec: dict):
    from peft import LoraConfig, get_peft_model

    lora = spec["lora"]
    config = LoraConfig(
        r=int(lora["r"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        target_modules=list(lora["target_modules"]),
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)


def _trainable(model) -> dict[str, torch.nn.Parameter]:
    return {name: param for name, param in model.named_parameters() if param.requires_grad}


# ---------------------------------------------------------------- checkpoints


def save_checkpoint(directory: Path, *, step: int, model, optimizer, scaler, spec: dict, exec_config: dict) -> Path:
    named = _trainable(model)
    trainable = {name: param.detach().to("cpu", torch.float32).clone() for name, param in named.items()}
    optimizer_tensors = optimizer_state_to_tensors(optimizer, {id(param): name for name, param in named.items()})
    manifest = {
        "immutable_hash": document_hash(spec),
        "logical_step": step,
        "recipe_id": spec["recipe_id"],
        "model": spec["model"],
        "dataset_sha256": spec["dataset"]["sha256"],
        "exec_config": exec_config,
        "scaler": scaler.state_dict() if scaler is not None else None,
    }
    write_checkpoint(Path(directory), trainable, optimizer_tensors, manifest)
    return Path(directory)


def resume_from(directory: Path, *, spec: dict, model, optimizer, scaler) -> int:
    """Loads a checkpoint into model and optimizer. Returns the completed logical step."""
    named = _trainable(model)
    shapes = {name: tuple(param.shape) for name, param in named.items()}
    manifest, trainable, optimizer_tensors = load_checkpoint(
        Path(directory), shapes, dtype=torch.float32, expected_immutable_hash=document_hash(spec)
    )
    with torch.no_grad():
        for name, param in named.items():
            param.copy_(trainable[name].to(param.device, param.dtype))
    restore_optimizer_state(optimizer, named, optimizer_tensors)
    if scaler is not None and manifest.get("scaler"):
        scaler.load_state_dict(manifest["scaler"])
    return int(manifest["logical_step"])


# ---------------------------------------------------------------- eval


def evaluate(model, tokenizer, recipe_id: str, test_rows: list[dict], *, device: str, max_seq_length: int) -> dict:
    r = recipe(recipe_id)
    prompts = [r.prompt(row[r.columns[0]]) for row in test_rows]
    references = [row[r.columns[1]] for row in test_rows]
    answers = generate_answers(
        model,
        tokenizer,
        prompts,
        max_new_tokens=r.max_answer_tokens,
        max_prompt_tokens=max_seq_length - r.max_answer_tokens,
        device=device,
    )
    passed = [passes(recipe_id, answer, ref) for answer, ref in zip(answers, references)]
    return {
        "pass_rate": sum(passed) / len(passed) if passed else 0.0,
        "score": score(recipe_id, answers, references),
        "metric": r.metric,
        "examples": len(test_rows),
        "eval_set_hash": eval_set_hash(recipe_id, test_rows),
        "floor": r.floor,
        "meets_floor": (sum(passed) / len(passed)) >= r.floor if passed else False,
        "samples": [
            {"reference": ref, "answer": answer, "passed": ok}
            for ref, answer, ok in list(zip(references, answers, passed))[:10]
        ],
    }


# ---------------------------------------------------------------- the run


def _target_for(device: str) -> Target:
    if device.startswith("cuda"):
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        free, _total = torch.cuda.mem_get_info(index)
        return Target(
            provider_id=f"cuda:{index}",
            online=True,
            capacity_mib=props.total_memory / (1024 * 1024),
            free_mib=free / (1024 * 1024),
            compute_cap=f"{props.major}.{props.minor}",
            torch_version=torch.__version__,
        )
    return Target(
        provider_id="cpu",
        online=True,
        capacity_mib=DEFAULT_CPU_CAP_MIB,
        free_mib=DEFAULT_CPU_CAP_MIB,
        compute_cap="cpu",
        torch_version=torch.__version__,
    )


def decide_exec_config(spec: dict, model, tokenizer, sample_texts: list[str], *, device: str) -> dict:
    """The compiler decides, from a real probe. Refusal is a normal outcome and raises."""
    target = _target_for(device)
    probe = make_probe(model, tokenizer, sample_texts, max_seq_length=int(spec["max_seq_length"]), device=device)
    decision = compile_for_target(
        target,
        int(spec["logical_batch_size"]),
        probe,
        remaining_steps=int(spec["total_logical_steps"]),
        seconds_to_deadline=24 * 3600.0,
        allow_cpu=device == "cpu",
    )
    if not decision.approved or decision.exec_config is None:
        raise RuntimeError(f"no exec config approved for this target: {decision.code}: {decision.reason}")
    return decision.exec_config


def run_training(
    spec: dict,
    dataset_path: Path,
    out_dir: Path,
    *,
    device: str | None = None,
    model=None,
    tokenizer=None,
    resume: Path | None = None,
    exec_config: dict | None = None,
    eval_before: bool = False,
    stop_after: int | None = None,
    log=print,
) -> dict:
    """Trains (or resumes) one llm-lora/v1 job. Returns the evidence dict, also written
    to ``out_dir/evidence.json``. Injected ``model``/``tokenizer`` are how CI runs the
    tiny CPU config; a provider run leaves them out and the spec's base is downloaded."""
    problems = spec_problems(spec)
    if problems:
        raise ValueError("invalid job spec: " + "; ".join(problems))
    train_rows, test_rows, digest, nbytes = read_dataset(Path(dataset_path))
    _check_dataset(spec, train_rows, test_rows, digest, nbytes)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    r = recipe(spec["recipe_id"])
    max_seq_length = int(spec["max_seq_length"])
    total_steps = int(spec["total_logical_steps"])
    logical_batch = int(spec["logical_batch_size"])
    seed = int(spec["seed"])

    if tokenizer is None:
        tokenizer = load_tokenizer(spec["model"]["id"], spec["model"]["revision"])
    if model is None:
        model = load_model(spec["model"]["id"], spec["model"]["revision"], device=device)
    else:
        model = model.to(device)
    if not hasattr(model, "peft_config"):
        torch.manual_seed(seed)
        model = apply_lora(model, spec)

    train_examples = [r.example(row) for row in train_rows]
    encoded = [encode_example(tokenizer, example, max_seq_length) for example in train_examples]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    if exec_config is None:
        sample_texts = [example["prompt"] + example["completion"] for example in train_examples[:logical_batch]]
        exec_config = decide_exec_config(spec, model, tokenizer, sample_texts, device=device)
    else:
        check_exec_config(exec_config, logical_batch)
    micro_batch = int(exec_config["micro_batch"])
    amp = exec_config.get("amp", "none")
    if exec_config.get("grad_checkpointing") and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    optimizer_spec = spec["optimizer"]
    if optimizer_spec["name"] != "adamw":
        raise ValueError(f"unsupported optimizer: {optimizer_spec['name']}")
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(optimizer_spec["lr"]),
        betas=tuple(float(b) for b in optimizer_spec["betas"]),
        eps=float(optimizer_spec["eps"]),
        weight_decay=float(optimizer_spec["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(amp == "fp16" and device.startswith("cuda")))

    start_step = 0
    if resume is not None:
        start_step = resume_from(Path(resume), spec=spec, model=model, optimizer=optimizer, scaler=scaler)
        log(f"resumed at logical step {start_step}")

    base_eval = None
    if eval_before and hasattr(model, "disable_adapter"):
        with model.disable_adapter():
            base_eval = evaluate(model, tokenizer, spec["recipe_id"], test_rows, device=device, max_seq_length=max_seq_length)
        log(f"base model: {base_eval['pass_rate']:.1%} pass on {base_eval['examples']} test rows")

    checkpoint_every = int(spec.get("checkpoint_every_steps", total_steps))
    # A provider can be stopped (or killed) mid-job; it runs the steps it can and the
    # checkpoints are where the next provider resumes. The spec never changes.
    last_step = total_steps if stop_after is None else min(total_steps, start_step + stop_after)
    checkpoints: list[str] = []
    losses: list[float] = []
    started = time.perf_counter()
    model.train()
    for step in range(start_step, last_step):
        lr = learning_rate(spec["lr_schedule"], float(optimizer_spec["lr"]), step, total_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        indices = logical_batch_indices(seed, step, logical_batch, len(encoded))
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for micro_index, micro in enumerate(split_micro_batches(indices, micro_batch)):
            torch.manual_seed(micro_batch_seed(seed, step, micro_index))
            batch = _collate([encoded[i] for i in micro], pad_id, device)
            with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", dtype=torch.float16 if amp == "fp16" else torch.bfloat16, enabled=amp in ("fp16", "bf16")):
                loss = model(**batch).loss * micro_loss_weight(len(micro), logical_batch)
            scaler.scale(loss).backward()
            step_loss += float(loss.detach())
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable_params, float(optimizer_spec["max_grad_norm"]))
        scaler.step(optimizer)
        scaler.update()
        losses.append(step_loss)
        done = step + 1
        if done % checkpoint_every == 0 or done == last_step:
            path = save_checkpoint(
                out_dir / "checkpoints" / f"step-{done}",
                step=done, model=model, optimizer=optimizer, scaler=scaler, spec=spec, exec_config=exec_config,
            )
            checkpoints.append(str(path))
        log(f"step {done}/{total_steps} loss {step_loss:.4f} lr {lr:.2e}")

    trained_eval = evaluate(model, tokenizer, spec["recipe_id"], test_rows, device=device, max_seq_length=max_seq_length)
    log(f"trained model: {trained_eval['pass_rate']:.1%} pass on {trained_eval['examples']} test rows")
    evidence = {
        "template": spec["template"],
        "recipe_id": spec["recipe_id"],
        "model": spec["model"],
        "device": device,
        "exec_config": exec_config,
        "logical_steps": total_steps,
        "completed_step": last_step,
        "resumed_from_step": start_step,
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "mean_loss_last_10": sum(losses[-10:]) / len(losses[-10:]) if losses else None,
        "base_eval": base_eval,
        "eval": trained_eval,
        "checkpoints": checkpoints,
        "seconds": round(time.perf_counter() - started, 1),
    }
    (out_dir / "evidence.json").write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return evidence


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run one llm-lora/v1 training job.")
    parser.add_argument("--spec", type=Path, required=True, help="job spec JSON (ekido_core.job_spec)")
    parser.add_argument("--dataset", type=Path, required=True, help="dataset blob JSONL (see module docstring)")
    parser.add_argument("--out", type=Path, required=True, help="output directory for checkpoints and evidence")
    parser.add_argument("--device", default=None, help="cuda or cpu; cuda is used when available")
    parser.add_argument("--resume", type=Path, default=None, help="a GPCK/1 checkpoint directory to resume from")
    parser.add_argument("--exec-config", type=Path, default=None, help="exec config JSON; the compiler probes if omitted")
    parser.add_argument("--tiny", action="store_true", help="CI smoke run: a tiny random Qwen2 instead of the spec's base")
    parser.add_argument("--eval-before", action="store_true", help="score the base model (adapter disabled) first")
    args = parser.parse_args(argv)

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    exec_config = json.loads(args.exec_config.read_text(encoding="utf-8")) if args.exec_config else None
    model = tokenizer = None
    if args.tiny:
        from .tiny import build_tiny_model, build_tiny_tokenizer

        train_rows, test_rows, _digest, _nbytes = read_dataset(args.dataset)
        r = recipe(spec["recipe_id"])
        texts = [r.prompt(row[r.columns[0]]) + row[r.columns[1]] for row in train_rows + test_rows]
        tokenizer = build_tiny_tokenizer(texts)
        model = build_tiny_model(tokenizer)
    evidence = run_training(
        spec, args.dataset, args.out,
        device=args.device, model=model, tokenizer=tokenizer,
        resume=args.resume, exec_config=exec_config, eval_before=args.eval_before,
    )
    print(f"done: {evidence['eval']['pass_rate']:.1%} pass, checkpoints in {args.out / 'checkpoints'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
