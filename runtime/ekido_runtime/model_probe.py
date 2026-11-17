"""The memory probe: one real training step, measured, on the actual model.

The adaptation compiler (``ekido_core.compiler.compile_for_target``) only approves an
exec config a probe has proven on the target. This is that probe for llm-lora/v1: it runs
one forward and backward pass at the requested micro-batch — no optimizer step, gradients
zeroed afterwards — and reports the peak memory it really used. A config is never
approved from an estimate.
"""

from __future__ import annotations

import time

import torch

from ekido_core.compiler import ProbeResult, Target


def _rss_mib() -> float:
    # Windows does not provide resource; use its process peak working set.
    import sys
    if sys.platform == 'win32':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
                        ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                        ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                        ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                        ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        get_memory = ctypes.WinDLL('psapi', use_last_error=True).GetProcessMemoryInfo
        get_memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        get_memory.restype = wintypes.BOOL
        if not get_memory(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return counters.PeakWorkingSetSize / 1024**2
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024**2 if sys.platform == 'darwin' else 1024)


def probe_step(model, tokenizer, texts: list[str], *, micro_batch: int, max_seq_length: int, device: str) -> ProbeResult:
    """One forward+backward at ``micro_batch``. Weights are not updated."""
    if not texts:
        return ProbeResult(ok=False, error="probe needs at least one text")
    batch_texts = [texts[i % len(texts)] for i in range(micro_batch)]
    encoded = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_length)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    was_training = model.training
    model.train()
    model.zero_grad(set_to_none=True)
    try:
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)
            before = torch.cuda.memory_allocated(device)
        else:
            before = _rss_mib()
        started = time.perf_counter()
        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
        loss.backward()
        seconds = time.perf_counter() - started
        if device.startswith("cuda"):
            peak = (torch.cuda.max_memory_allocated(device) - 0) / (1024 * 1024)
            peak = max(peak, (torch.cuda.memory_allocated(device) - before) / (1024 * 1024))
        else:
            peak = max(_rss_mib() - 0.0, 0.0)  # process peak so far; a floor, never an underestimate of the step's cost on a fresh provider process it is the step's own peak
        return ProbeResult(ok=True, peak_mem_mib=float(peak), steps_per_s=float(1.0 / seconds) if seconds > 0 else None, duration_s=float(seconds))
    except (torch.cuda.OutOfMemoryError, RuntimeError, MemoryError) as exc:
        message = str(exc)
        oom = isinstance(exc, (torch.cuda.OutOfMemoryError, MemoryError)) or "out of memory" in message.lower()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        return ProbeResult(ok=False, oom=oom, error=message[:300])
    finally:
        model.zero_grad(set_to_none=True)
        if not was_training:
            model.eval()


def make_probe(model, tokenizer, texts: list[str], *, max_seq_length: int, device: str):
    """A ``ProbeFn`` for ``compile_for_target`` bound to this model and device."""

    def probe(target: Target, exec_config: dict) -> ProbeResult:
        return probe_step(
            model,
            tokenizer,
            texts,
            micro_batch=int(exec_config["micro_batch"]),
            max_seq_length=max_seq_length,
            device=device,
        )

    return probe
