"""Bring-your-own-code jobs (template custom-python/v1). See docs/custom-workloads.md.

The client's job.py is untrusted. It's only ever imported inside the agent's Docker sandbox: no network,
read-only system, non-root, with memory and process limits. The spawner sets GPUCHAIN_SANDBOX=1 there and
nowhere else, and this module refuses to run without it. The code's and every file's SHA-256 must match
the spec, which is fixed on-chain by hash before training, so a provider runs exactly what was agreed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
from typing import Callable, Sequence

import torch

TEMPLATE = "custom-python/v1"
SANDBOX_ENV = "GPUCHAIN_SANDBOX"
CODE_ENV = "GPUCHAIN_JOB_CODE"  # Path to job.py, mounted read-only.
FILES_ENV = "GPUCHAIN_JOB_FILES"  # Folder of the client's files, mounted read-only.
REQUIRED = ("build_model", "dataset_size", "make_batch")


class CustomJobError(ValueError):
    """The custom job can't run here. The message says why, in words a client can act on."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sandbox() -> None:
    if os.environ.get(SANDBOX_ENV) != "1":
        raise CustomJobError("custom code only runs inside the provider's Docker sandbox, and this process isn't in it")


def check_files(spec: dict, files_dir: Path) -> None:
    """Every file the spec lists must be present with exactly that hash. Files not listed are ignored by nothing: refuse them too."""
    listed = spec.get("files") or {}
    present = {p.name for p in files_dir.iterdir() if p.is_file()} if files_dir.is_dir() else set()
    for name, expected in listed.items():
        path = files_dir / name
        if name not in present:
            raise CustomJobError(f"file {name} is missing")
        if sha256_file(path) != expected:
            raise CustomJobError(f"file {name} doesn't match its agreed hash")
    extra = present - set(listed)
    if extra:
        raise CustomJobError(f"files that weren't agreed are present: {', '.join(sorted(extra))}")


def load_user_module(code_path: Path, expected_sha256: str):
    require_sandbox()
    if sha256_file(code_path) != expected_sha256:
        raise CustomJobError("job.py doesn't match its agreed hash")
    spec = importlib.util.spec_from_file_location("gpuchain_user_job", code_path)
    if spec is None or spec.loader is None:
        raise CustomJobError("job.py can't be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    missing = [name for name in REQUIRED if not callable(getattr(module, name, None))]
    if missing:
        raise CustomJobError(f"job.py must define {', '.join(missing)}")
    return module


def build_custom(spec: dict, device: str, *, with_data: bool = True, canary_examples: int = 16):
    """Returns (model, dataset_size, batch_fn, canary_batch) for the runtime's Workload."""
    code_path = Path(os.environ.get(CODE_ENV, ""))
    files_dir = Path(os.environ.get(FILES_ENV, ""))
    if not code_path.is_file():
        raise CustomJobError("the job's code wasn't mounted")
    module = load_user_module(code_path, spec["code"]["sha256"])
    check_files(spec, files_dir)

    torch.manual_seed(spec["seed"])  # Same starting weights on every machine.
    model = module.build_model(spec)
    if not isinstance(model, torch.nn.Module):
        raise CustomJobError("build_model must return a torch.nn.Module")
    for name, param in model.named_parameters():
        if param.requires_grad and param.dtype != torch.float32:
            raise CustomJobError(f"trainable parameter {name} must be float32, so a checkpoint resumes the same on any GPU")
    model = model.to(device)

    size = int(module.dataset_size(spec, files_dir))
    if size < 1:
        raise CustomJobError("dataset_size must be at least 1")
    make_batch: Callable[..., dict] = module.make_batch

    def batch_fn(indices: Sequence[int]) -> dict:
        batch = make_batch(spec, files_dir, list(indices))
        if not isinstance(batch, dict) or "labels" not in batch:
            raise CustomJobError('make_batch must return a dict of tensors that includes "labels"')
        return batch

    canary = batch_fn(range(min(canary_examples, size)))
    return model, size, batch_fn, canary
