"""Runs one training attempt. The agent starts this and reads its events.

    python -m ekido_runtime.run <config.json>

Talks to the agent only through files, so the container can run with --network none:
  events.jsonl       appended here, tailed by the agent
  lease              touched by the agent; if it goes stale, this process exits
  checkpoint_now     created by the agent to request a checkpoint at the next step boundary

Exit codes: 0 finished, 2 lease lost, 3 out of memory, 4 refused checkpoint, 5 other error.
"""

from __future__ import annotations

import functools
import json
import math
import os
import random
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch

from ekido_core.canonical import document_hash
from ekido_core.gpck import (
    CheckpointError,
    load_checkpoint,
    manifest_hash,
    optimizer_state_to_tensors,
    restore_optimizer_state,
    write_checkpoint,
)
from ekido_core.logical import step_position

from .trainer import canary_loss, canary_matches, create_optimizer, make_scaler, run_logical_step, trainable_named_params
from .workloads import build_workload

EXIT_OK, EXIT_LEASE, EXIT_OOM, EXIT_BAD_CHECKPOINT, EXIT_ERROR = 0, 2, 3, 4, 5


class Events:
    def __init__(self, path: Path):
        self._handle = open(path, "a", encoding="utf-8", buffering=1)

    def emit(self, kind: str, **fields) -> None:
        self._handle.write(json.dumps({"kind": kind, "ts": time.time(), **fields}) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


class TrainingDiverged(Exception):
    """The weights went to NaN or Infinity. Reported plainly, since no valid checkpoint can be written."""


def start_lease_watchdog(path: Path, timeout: float, events: Events) -> None:
    """Dead-man switch. Runs in a thread, so a long step can't delay it.

    If the agent dies, a leftover process would hold VRAM and make its replacement run out of
    memory for the wrong reason on a shared GPU. So it exits, hard, without waiting on anything.
    """

    def watch() -> None:
        while True:
            time.sleep(min(1.0, timeout / 4))
            try:
                age = time.time() - path.stat().st_mtime
            except OSError:
                age = float("inf")
            if age > timeout:
                events.emit("run.exited", reason="LEASE_LOST", age=age)
                os._exit(EXIT_LEASE)

    threading.Thread(target=watch, daemon=True, name="lease-watchdog").start()


def apply_memory_cap(capacity_mib: int | None, device: str) -> None:
    """A real allocator limit: going over raises a genuine torch.OutOfMemoryError."""
    if capacity_mib and device.startswith("cuda"):
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, capacity_mib * 2**20 / total), 0)


@functools.lru_cache(maxsize=1)
def driver_version() -> str:
    """Stored in every manifest, so cross-GPU evidence says which driver produced it."""
    try:
        output = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True, timeout=5, check=True).stdout
        return output.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        return "unknown"


def environment(device: str) -> dict:
    is_gpu = device.startswith("cuda")
    return {
        "gpu": torch.cuda.get_device_name(0) if is_gpu else "cpu",
        "driver": driver_version() if is_gpu else "none",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "image_digest": os.environ.get("IMAGE_DIGEST"),
    }


class Faults:
    """Injected on purpose for the demo and tests. Every use is reported as injected."""

    def __init__(self, config: dict | None, events: Events, workdir: Path):
        config = config or {}
        self.kind = config.get("kind")
        self.at_step = config.get("at_step", 0)
        self.events = events
        self.workdir = workdir
        self.ballast = None
        self.poison_reported = False

    def before_step(self, step: int, model: torch.nn.Module, device: str) -> None:
        # Faults can also arrive during a run, as marker files from the agent.
        if (self.workdir / "inject_oom").exists():
            (self.workdir / "inject_oom").unlink(missing_ok=True)
            self.kind, self.at_step = "OOM", step
        if (self.workdir / "inject_crash_checkpoint").exists():
            (self.workdir / "inject_crash_checkpoint").unlink(missing_ok=True)
            self.kind, self.at_step = "CRASH_MID_CHECKPOINT", step
        if self.kind == "OOM" and step >= self.at_step and self.ballast is None:
            self.events.emit("fault.applied", fault="OOM", step=step, injected=True)
            self.ballast = []
            self.trigger_oom(device)

    @staticmethod
    def trigger_oom(device: str) -> None:
        """Raises a real OOM from a limit, and must never actually fill the GPU.

        An earlier version allocated until the card was full. Windows then pages GPU memory into system
        RAM, which appeared to freeze the whole laptop. So this lowers this process's own limit to what
        it already uses, and one small allocation then fails at once, using almost no extra memory.
        """
        if not device.startswith("cuda"):
            raise torch.OutOfMemoryError("Injected out-of-memory fault on CPU")
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, torch.cuda.memory_reserved(0) / total), 0)
        torch.zeros(256 * 2**20, dtype=torch.uint8, device=device)
        raise torch.OutOfMemoryError("Injected out-of-memory fault: the allocator's limit didn't trigger")

    def after_step(self, step: int, model: torch.nn.Module) -> None:
        if self.kind == "POISON" and step >= self.at_step:
            # Replaces weights with small noise. It stays finite on purpose: this provider keeps running
            # and reporting normal-looking progress, so only an independent evaluation can catch it.
            with torch.no_grad():
                for param in trainable_named_params(model).values():
                    param.copy_(torch.randn_like(param) * 0.02)
            if not self.poison_reported:
                self.poison_reported = True
                self.events.emit("fault.applied", fault="POISON", step=step, injected=True)

    def during_checkpoint(self, step: int, directory: Path) -> None:
        if self.kind == "CRASH_MID_CHECKPOINT" and step >= self.at_step:
            self.events.emit("fault.applied", fault="CRASH_MID_CHECKPOINT", step=step, injected=True)
            os._exit(EXIT_ERROR)

    def freeze_weights(self) -> bool:
        """A lazy provider: reports progress without training."""
        return self.kind == "NO_OP"


def scaler_to_manifest(scaler) -> dict:
    """Keep all GradScaler state needed for an exact FP16 resume."""
    state = scaler.state_dict()
    return {"scale": float(state["scale"]), "growth_factor": float(state["growth_factor"]), "backoff_factor": float(state["backoff_factor"]), "growth_interval": int(state["growth_interval"]), "growth_tracker": int(state["_growth_tracker"])}


def scaler_from_manifest(saved: dict) -> dict:
    """Convert full state; older two-field checkpoints used PyTorch's defaults."""
    extra = {"growth_factor", "backoff_factor", "growth_interval"}
    present = extra.intersection(saved)
    if present and present != extra:
        raise ValueError("incomplete GradScaler state")
    state = {"scale": float(saved["scale"]), "growth_factor": float(saved.get("growth_factor", 2.0)), "backoff_factor": float(saved.get("backoff_factor", 0.5)), "growth_interval": int(saved.get("growth_interval", 2000)), "_growth_tracker": int(saved["growth_tracker"])}
    if not all(math.isfinite(state[k]) for k in ("scale", "growth_factor", "backoff_factor")) or state["scale"] <= 0 or state["growth_factor"] <= 1 or not 0 < state["backoff_factor"] < 1 or state["growth_interval"] < 1 or state["_growth_tracker"] < 0:
        raise ValueError("invalid GradScaler state")
    return state


def save_checkpoint(config: dict, workload, optimizer, scaler, completed_step: int, parent: str | None, exec_config: dict, device: str, faults: Faults, adaptation_hash: str | None, directory: Path) -> tuple[Path, dict]:
    spec = config["spec"]
    canary = canary_loss(workload.model, workload.canary_batch, device)
    if not math.isfinite(canary):
        raise TrainingDiverged(f"canary loss is {canary} at step {completed_step}")
    named = trainable_named_params(workload.model)
    names = {id(p): n for n, p in named.items()}
    # Copy to CPU first, at a step boundary, so a slow disk can't disturb the training state.
    trainable = {n: p.detach().cpu().clone() for n, p in named.items()}
    optimizer_tensors = optimizer_state_to_tensors(optimizer, names)
    scaler_state = None
    if scaler is not None:
        scaler_state = scaler_to_manifest(scaler)
    position = step_position(max(0, completed_step - 1), spec["logical_batch_size"], workload.dataset_size)
    manifest = {
        "job_id": config["job_id"],
        "attempt": config["attempt"],
        "fencing_token": config["fencing_token"],
        "logical_step": completed_step,
        "epoch": position.epoch,
        "immutable_hash": document_hash(spec),
        "model": {k: spec["model"][k] for k in ("id", "revision") if k in spec.get("model", {})} or {"id": spec["template"], "revision": "n/a"},
        "dataset_fingerprint": spec.get("dataset", {}).get("fingerprint", "0" * 64),
        "exec": exec_config,
        "scaler": scaler_state,
        "canary": {"loss_fp32": canary},
        "environment": environment(device),
        "parent_manifest_hash": parent,
        "adaptation_manifest_hash": adaptation_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "signer": config.get("signer"),
    }
    target = directory / f"step_{completed_step:08d}"
    target.mkdir(parents=True, exist_ok=True)
    faults.during_checkpoint(completed_step, target)
    return target, write_checkpoint(target, trainable, optimizer_tensors, manifest)


def restore(config: dict, workload, optimizer, events: Events, device: str):
    """Returns (next step, parent hash, scaler state). Refuses anything that fails validation."""
    source = config.get("resume_from")
    if not source:
        return 0, None, None
    t0 = time.perf_counter()
    named = trainable_named_params(workload.model)
    manifest, trainable, optimizer_tensors = load_checkpoint(Path(source), workload.param_shapes(), expected_immutable_hash=document_hash(config["spec"]))
    with torch.no_grad():
        for name, param in named.items():
            param.copy_(trainable[name].to(param.device))
    restore_optimizer_state(optimizer, named, optimizer_tensors)
    measured = canary_loss(workload.model, workload.canary_batch, device)
    saved = manifest["canary"]["loss_fp32"]
    ok = canary_matches(saved, measured, relative_tolerance=config.get("canary_tolerance", 1e-3))
    events.emit("run.restored", step=manifest["logical_step"], canary_saved=saved, canary_measured=measured, canary_ok=ok, seconds=round(time.perf_counter() - t0, 3))
    if not ok:
        # The hashes were fine here: the model just reproduced a different loss.
        raise CheckpointError("CANARY_MISMATCH", f"canary loss {measured} differs from saved {saved}")
    return manifest["logical_step"], manifest_hash(manifest), manifest["scaler"]


def run(config: dict) -> int:
    workdir = Path(config["workdir"])
    workdir.mkdir(parents=True, exist_ok=True)
    events = Events(workdir / "events.jsonl")
    device = config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    spec, exec_config = config["spec"], config["exec"]

    lease = config.get("lease_file")
    if lease:
        start_lease_watchdog(Path(lease), config.get("lease_timeout", 15.0), events)

    apply_memory_cap(config.get("capacity_mib"), device)
    faults = Faults(config.get("fault"), events, workdir)
    checkpoint_now = workdir / "checkpoint_now"

    try:
        events.emit("run.starting", attempt=config["attempt"], fencing_token=config["fencing_token"], device=device, exec=exec_config)
        random.seed(spec["seed"])
        workload = build_workload(spec, exec_config, device, data_dir=workdir)
        params = list(trainable_named_params(workload.model).values())
        optimizer = create_optimizer(params, spec["optimizer"])
        scaler = make_scaler(exec_config["amp"], device)
        start, parent, scaler_state = restore(config, workload, optimizer, events, device)
        if scaler is not None and scaler_state:
            scaler.load_state_dict(scaler_from_manifest(scaler_state))
        workload.model.train()

        total = config.get("stop_at_step") or spec["total_logical_steps"]
        every = spec.get("checkpoint_every_steps", 25)
        events.emit("run.started", start_step=start, total_steps=total)
        window_start, window_steps = time.perf_counter(), 0
        previous_step = None
        for step in range(start, total):
            faults.before_step(step, workload.model, device)
            if faults.freeze_weights():
                time.sleep(0.01)
                result = None
            else:
                result = _step_with_fault(workload, optimizer, spec, exec_config, step, device, scaler, faults)
            faults.after_step(step, workload.model)
            completed = step + 1
            window_steps += 1

            now = time.perf_counter()
            if completed == total or now - window_start >= config.get("progress_seconds", 0.5):
                peak = torch.cuda.max_memory_allocated() / 2**20 if device.startswith("cuda") else None
                events.emit(
                    "run.progress",
                    step=completed,
                    total_steps=total,
                    loss=result.loss if result else None,
                    lr=result.lr if result else None,
                    grad_norm=result.grad_norm if result else None,
                    steps_per_s=round(window_steps / max(now - window_start, 1e-6), 3),
                    peak_mem_mib=round(peak, 1) if peak is not None else None,
                )
                window_start, window_steps = now, 0

            requested = checkpoint_now.exists()
            if completed % every == 0 or completed == total or requested:
                if requested:
                    checkpoint_now.unlink(missing_ok=True)
                t0 = time.perf_counter()
                target, manifest = save_checkpoint(config, workload, optimizer, scaler, completed, parent, exec_config, device, faults, config.get("adaptation_manifest_hash"), workdir / "checkpoints")
                parent = manifest_hash(manifest)
                events.emit("checkpoint.ready", step=completed, path=str(target), manifest=manifest, manifest_hash=parent, seconds=round(time.perf_counter() - t0, 3), emergency=requested)
            previous_step = completed
        events.emit("run.exited", reason="FINISHED", step=previous_step)
        return EXIT_OK
    except torch.OutOfMemoryError as exc:
        events.emit("run.exited", reason="OOM", detail=str(exc)[:300])
        return EXIT_OOM
    except CheckpointError as exc:
        events.emit("run.exited", reason="BAD_CHECKPOINT", code=exc.code, detail=str(exc)[:300])
        return EXIT_BAD_CHECKPOINT
    except TrainingDiverged as exc:
        events.emit("run.exited", reason="DIVERGED", detail=str(exc)[:300])
        return EXIT_ERROR
    except Exception as exc:  # The agent must always learn why a run ended.
        events.emit("run.exited", reason="ERROR", detail=f"{type(exc).__name__}: {exc}"[:500], traceback=traceback.format_exc()[-1500:])
        return EXIT_ERROR
    finally:
        # A leftover lease watchdog only exists in a real runtime process, which exits right after this.
        if not lease:
            events.close()


def _step_with_fault(workload, optimizer, spec, exec_config, step, device, scaler, faults: Faults):
    if faults.kind == "LR_TAMPER" and step >= faults.at_step:
        tampered = json.loads(json.dumps(spec))
        tampered["optimizer"]["lr"] *= 50
        spec = tampered
    return run_logical_step(workload.model, optimizer, workload.batch_fn, spec, exec_config, step, workload.dataset_size, device, scaler)


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return EXIT_ERROR
    return run(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))


if __name__ == "__main__":
    sys.exit(main())
