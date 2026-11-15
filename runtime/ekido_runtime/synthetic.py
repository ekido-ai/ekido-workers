"""Fast, deterministic workload for correctness tests and the CPU demo fallback. Needs no GPU or downloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass
class Output:
    loss: torch.Tensor
    logits: torch.Tensor


class SyntheticClassifier(torch.nn.Module):
    def __init__(self, features: int = 16, hidden: int = 32, classes: int = 4, dropout: float = 0.0):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(features, hidden),
            torch.nn.LayerNorm(hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, classes),
        )

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> Output:
        logits = self.net(features)
        return Output(loss=torch.nn.functional.cross_entropy(logits, labels), logits=logits)


class SyntheticData:
    """Labels come from a fixed random linear rule, so the task can be learned and always repeats.

    The rule has its own generator: it depends only on the seed, never on the sample size or split,
    so training and evaluation data always describe the same task.
    """

    def __init__(self, size: int = 2048, features: int = 16, classes: int = 4, seed: int = 0, dtype: torch.dtype = torch.float32, split: int = 0):
        rule = torch.randn(features, classes, generator=torch.Generator().manual_seed(seed), dtype=dtype)
        samples = torch.Generator().manual_seed(seed + 1 + split)
        self.x = torch.randn(size, features, generator=samples, dtype=dtype)
        self.y = (self.x @ rule).argmax(dim=1)
        self.size = size

    def batch(self, indices: Sequence[int]) -> dict[str, torch.Tensor]:
        index = torch.as_tensor(list(indices), dtype=torch.long)
        return {"features": self.x[index], "labels": self.y[index]}


def synthetic_spec(total_steps: int = 40, logical_batch_size: int = 32, seed: int = 1337, lr: float = 0.01) -> dict:
    return {
        "logical_batch_size": logical_batch_size,
        "total_logical_steps": total_steps,
        "seed": seed,
        "optimizer": {"name": "adamw", "lr": lr, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01, "max_grad_norm": 1.0},
        "lr_schedule": {"name": "linear_warmup_linear_decay", "warmup_steps": 4},
    }


def exec_config(micro_batch: int, logical_batch_size: int = 32, amp: str = "none") -> dict:
    return {
        "micro_batch": micro_batch,
        "accumulation": logical_batch_size // micro_batch,
        "amp": amp,
        "grad_checkpointing": False,
        "attention_impl": "sdpa",
        "loader_workers": 0,
    }
