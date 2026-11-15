"""Allowlisted workloads, plus custom-python/v1, whose client code runs only inside the Docker sandbox (see custom.py)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
from ekido_core.data_identity import check_split, validate_spec_identity

from .synthetic import SyntheticClassifier, SyntheticData
from .trainer import trainable_named_params

CANARY_EXAMPLES = 16


@dataclass
class Workload:
    model: torch.nn.Module
    dataset_size: int
    batch_fn: object  # Callable[[Sequence[int]], Mapping[str, Tensor]]
    canary_batch: Mapping[str, torch.Tensor]

    def param_shapes(self) -> dict[str, tuple[int, ...]]:
        return {name: tuple(p.shape) for name, p in trainable_named_params(self.model).items()}


def _build_synthetic(spec: dict, exec_config: dict, device: str) -> Workload:
    data = SyntheticData(size=spec.get("dataset_size", 2048), seed=spec["seed"])
    # Initial weights depend only on the job seed, so a fresh start is reproducible.
    torch.manual_seed(spec["seed"])
    model = SyntheticClassifier(dropout=spec.get("dropout", 0.0)).to(device)
    return Workload(model, data.size, data.batch, data.batch(range(CANARY_EXAMPLES)))


def _tokenize(tokenizer, texts: Sequence[str], max_len: int) -> dict[str, torch.Tensor]:
    # Fixed-length padding keeps memory use predictable, so a probe's result stays valid for the whole job.
    encoded = tokenizer(list(texts), padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
    return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}


def _build_distilbert(spec: dict, exec_config: dict, device: str, with_data: bool = True) -> Workload:
    from datasets import load_dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_spec = spec["model"]
    has_identity = validate_spec_identity(spec)
    tokenizer = AutoTokenizer.from_pretrained(model_spec["id"], revision=model_spec["revision"])
    torch.manual_seed(spec["seed"])
    base = AutoModelForSequenceClassification.from_pretrained(
        model_spec["id"],
        revision=model_spec["revision"],
        num_labels=4,
        attn_implementation=exec_config.get("attention_impl", "sdpa"),
    )
    lora = spec["lora"]
    model = get_peft_model(
        base,
        LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora["r"],
            lora_alpha=lora["alpha"],
            lora_dropout=lora["dropout"],
            target_modules=lora["target_modules"],
        ),
    ).to(device)
    if exec_config.get("grad_checkpointing"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    if not with_data:
        return Workload(model, 0, None, {})

    dataset_spec = spec["dataset"]
    dataset = load_dataset(dataset_spec["id"], split=dataset_spec["split"], **({"revision": dataset_spec["revision"]} if has_identity else {}))
    if has_identity:
        check_split(dataset, dataset_spec["content_sha256"], "training data")
    limit = spec.get("train_examples")
    if limit:
        # A fixed subset, independent of the training order, so every provider sees the same data.
        order = np.random.default_rng(spec["seed"]).permutation(len(dataset))[:limit]
        dataset = dataset.select(order.tolist())
    encoded = _tokenize(tokenizer, dataset["text"], spec["max_seq_length"])
    labels = torch.as_tensor(list(dataset["label"]))

    def batch_fn(indices: Sequence[int]) -> dict[str, torch.Tensor]:
        index = torch.as_tensor(list(indices), dtype=torch.long)
        return {"input_ids": encoded["input_ids"][index], "attention_mask": encoded["attention_mask"][index], "labels": labels[index]}

    return Workload(model, len(dataset), batch_fn, batch_fn(range(CANARY_EXAMPLES)))


def build_workload(spec: dict, exec_config: dict, device: str, *, with_data: bool = True) -> Workload:
    """with_data=False skips the training data, for evaluation, which never touches it."""
    template = spec["template"]
    if template == "synthetic-linear/v1":
        return _build_synthetic(spec, exec_config, device)
    if template == "distilbert-lora-agnews/v1":
        return _build_distilbert(spec, exec_config, device, with_data)
    if template == "custom-python/v1":
        from .custom import build_custom

        model, size, batch_fn, canary = build_custom(spec, device, with_data=with_data, canary_examples=CANARY_EXAMPLES)
        return Workload(model, size, batch_fn, canary)
    raise ValueError(f"workload not on the allowlist: {template}")
