"""Tries a candidate config for real and measures it. This is what lets the compiler approve one.

    python -m ekido_runtime.probe <config.json> <result.json>

Runs as its own process, so a failed attempt can't leave fragmented or stuck GPU memory behind.
Exit code is 0 whenever a result was written, including out of memory, which is a valid answer.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

from .run import apply_memory_cap
from .oom import is_memory_exhaustion
from .trainer import create_optimizer, make_scaler, run_logical_step, trainable_named_params
from .workloads import build_workload

DEFAULT_TIMED_STEPS = 3


def probe(config: dict, data_dir: Path | None = None) -> dict:
    """data_dir holds the job's files (dataset.jsonl for llm-lora/v1), fetched by the agent beside result.json."""
    spec, exec_config = config["spec"], config["exec"]
    device = config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    gpu = device.startswith("cuda")
    t0 = time.perf_counter()
    try:
        apply_memory_cap(config.get("capacity_mib"), device)
        if gpu:
            torch.cuda.reset_peak_memory_stats()
        workload = build_workload(spec, exec_config, device, data_dir=data_dir)
        optimizer = create_optimizer(list(trainable_named_params(workload.model).values()), spec["optimizer"])
        scaler = make_scaler(exec_config["amp"], device)
        workload.model.train()

        def one_step(step: int) -> None:
            run_logical_step(workload.model, optimizer, workload.batch_fn, spec, exec_config, step, workload.dataset_size, device, scaler)

        # The first step allocates optimizer state, so it counts for memory but not for speed.
        one_step(0)
        timed = config.get("timed_steps", DEFAULT_TIMED_STEPS)
        if gpu:
            torch.cuda.synchronize()
        started = time.perf_counter()
        for step in range(1, timed + 1):
            one_step(step)
        if gpu:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        # Reserved memory, not allocated: reserved is what the allocator's cap is enforced against.
        peak = torch.cuda.max_memory_reserved() / 2**20 if gpu else None
        return {
            "ok": True,
            "oom": False,
            "peak_mem_mib": round(peak, 1) if peak is not None else 0.0,
            "steps_per_s": round(timed / max(elapsed, 1e-9), 3),
            "duration_s": round(time.perf_counter() - t0, 3),
            "device": torch.cuda.get_device_name(0) if gpu else "cpu",
            "exec": exec_config,
        }
    except torch.OutOfMemoryError as exc:
        return {"ok": False, "oom": True, "error": str(exc)[:300], "duration_s": round(time.perf_counter() - t0, 3), "exec": exec_config}
    except Exception as exc:
        oom = isinstance(exc, MemoryError) or (isinstance(exc, RuntimeError) and is_memory_exhaustion(str(exc)))
        return {"ok": False, "oom": oom, "error": f"{type(exc).__name__}: {exc}"[:500], "duration_s": round(time.perf_counter() - t0, 3), "exec": exec_config}


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    # The probe folder is the one result.json goes into, in both process and Docker mode.
    result = probe(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")), data_dir=Path(sys.argv[2]).resolve().parent)
    Path(sys.argv[2]).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
