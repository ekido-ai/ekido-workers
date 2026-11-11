"""The training loop. One logical step means the same thing on every GPU.

Master weights and optimizer state stay at their base dtype, and only autocast varies. Learning
rate, data order and dropout seeds are pure functions of the step, so a resume needs nothing from
the previous machine beyond the model and optimizer state in the checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import torch

from gpuchain_core.logical import (
    check_exec_config,
    learning_rate,
    logical_batch_indices,
    micro_batch_seed,
    micro_loss_weight,
    split_micro_batches,
)

# Given example indices, returns the keyword arguments for the model, including `labels`.
BatchFn = Callable[[Sequence[int]], Mapping[str, torch.Tensor]]


@dataclass(frozen=True)
class StepResult:
    step: int  # The logical step that was just completed, zero-based.
    loss: float
    grad_norm: float
    lr: float


def amp_dtype(amp: str) -> torch.dtype | None:
    try:
        return {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[amp]
    except KeyError:
        raise ValueError(f"unknown amp mode: {amp}") from None


def native_bf16() -> bool:
    """Never use is_bf16_supported() with its default: it returns True through emulation on a T4."""
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False)


def make_scaler(amp: str, device: str) -> torch.amp.GradScaler | None:
    """FP16 only. BF16 and FP32 have the range to do without one."""
    return torch.amp.GradScaler("cuda") if amp == "fp16" and device.startswith("cuda") else None


def trainable_named_params(model: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    return {name: p for name, p in model.named_parameters() if p.requires_grad}


def accumulate_gradients(model: torch.nn.Module, batch_fn: BatchFn, indices: Sequence[int], micro_batch: int, amp: str, device: str, scaler, job_seed: int, step: int) -> float:
    """Runs forward and backward over the micro-batches. Returns the logical batch's mean loss."""
    total = len(indices)
    dtype = amp_dtype(amp)
    use_autocast = dtype is not None and device.startswith("cuda")
    loss_sum = 0.0
    for micro_index, part in enumerate(split_micro_batches(list(indices), micro_batch)):
        torch.manual_seed(micro_batch_seed(job_seed, step, micro_index))
        batch = {k: v.to(device) for k, v in batch_fn(part).items()}
        with torch.autocast("cuda", dtype=dtype, enabled=use_autocast):
            loss = model(**batch).loss
        # Weight by share, so unequal micro-batches still give the true mean over the logical batch.
        weighted = loss * micro_loss_weight(len(part), total)
        loss_sum += weighted.detach().item()
        (scaler.scale(weighted) if scaler is not None else weighted).backward()
    return loss_sum


def run_logical_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch_fn: BatchFn,
    spec: dict,
    exec_config: dict,
    step: int,
    dataset_size: int,
    device: str,
    scaler=None,
) -> StepResult:
    check_exec_config(exec_config, spec["logical_batch_size"])
    optimizer_spec = spec["optimizer"]
    lr = learning_rate(spec["lr_schedule"], optimizer_spec["lr"], step, spec["total_logical_steps"])
    for group in optimizer.param_groups:
        group["lr"] = lr

    indices = logical_batch_indices(spec["seed"], step, spec["logical_batch_size"], dataset_size)
    optimizer.zero_grad(set_to_none=True)
    loss = accumulate_gradients(model, batch_fn, indices, exec_config["micro_batch"], exec_config["amp"], device, scaler, spec["seed"], step)

    params = list(trainable_named_params(model).values())
    if scaler is not None:
        scaler.unscale_(optimizer)  # Clip real gradients, not scaled ones.
    # Clipped only after accumulation, so the norm matches a full-batch step.
    grad_norm = torch.nn.utils.clip_grad_norm_(params, optimizer_spec["max_grad_norm"])
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    return StepResult(step=step, loss=loss, grad_norm=float(grad_norm), lr=lr)


@torch.no_grad()
def canary_loss(model: torch.nn.Module, batch: Mapping[str, torch.Tensor], device: str) -> float:
    """Eval-mode loss without autocast. Compared after a restore, to catch a broken resume early."""
    was_training = model.training
    model.eval()
    try:
        return float(model(**{k: v.to(device) for k, v in batch.items()}).loss.item())
    finally:
        model.train(was_training)


def canary_matches(saved: float, restored: float, *, relative_tolerance: float = 1e-3, absolute_tolerance: float = 1e-5) -> bool:
    """Starting tolerance. Device differences are measured in the cross-GPU run, then this gets tuned."""
    return abs(saved - restored) <= max(absolute_tolerance, relative_tolerance * abs(saved))


def create_optimizer(params: Sequence[torch.nn.Parameter], optimizer_spec: dict) -> torch.optim.AdamW:
    if optimizer_spec["name"] != "adamw":
        raise ValueError(f"unsupported optimizer: {optimizer_spec['name']}")
    return torch.optim.AdamW(
        params,
        lr=optimizer_spec["lr"],
        betas=tuple(optimizer_spec["betas"]),
        eps=optimizer_spec["eps"],
        weight_decay=optimizer_spec["weight_decay"],
    )
