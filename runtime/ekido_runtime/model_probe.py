"""The memory probe: one real training step, measured, on the actual model.

The adaptation compiler (``ekido_core.compiler.compile_for_target``) only approves an
exec config a probe has proven on the target. The probe executes an accumulated logical
step including the AdamW update and its lazily allocated state, then restores weights.
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


def probe_step(model, tokenizer, texts: list[str], *, micro_batch: int, max_seq_length: int, device: str,
               accumulation: int = 1, amp: str = 'none', grad_checkpointing: bool = False) -> ProbeResult:
    """Measure a complete logical step and restore trainable weights afterwards."""
    if not texts:
        return ProbeResult(ok=False, error="probe needs at least one text")
    if accumulation < 1:
        return ProbeResult(ok=False, error="accumulation must be positive")
    was_training = model.training
    was_checkpointing = bool(getattr(model, 'is_gradient_checkpointing', False))
    params = [p for p in model.parameters() if p.requires_grad]
    originals = [p.detach().cpu().clone() for p in params]
    optimizer = None
    try:
        if not params:
            return ProbeResult(ok=False, error="probe needs trainable parameters")
        model.train()
        if grad_checkpointing and not was_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
        elif was_checkpointing and not grad_checkpointing:
            model.gradient_checkpointing_disable()
        batch_texts = [texts[i % len(texts)] for i in range(micro_batch)]
        encoded = tokenizer(batch_texts, return_tensors="pt", padding="max_length", truncation=True, max_length=max_seq_length)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        optimizer = torch.optim.AdamW(params)
        scaler = torch.amp.GradScaler('cuda', enabled=(amp == 'fp16' and device.startswith('cuda')))
        optimizer.zero_grad(set_to_none=True)
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(accumulation):
            with torch.autocast(device_type='cuda' if device.startswith('cuda') else 'cpu',
                                dtype=torch.float16 if amp == 'fp16' else torch.bfloat16,
                                enabled=amp in ('fp16', 'bf16')):
                loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss / accumulation
            scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(optimizer)
        scaler.update()
        if device.startswith('cuda'):
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started
        if device.startswith("cuda"):
            peak = torch.cuda.max_memory_reserved(device) / (1024 * 1024)
        else:
            peak = _rss_mib()
        return ProbeResult(ok=True, peak_mem_mib=float(peak), steps_per_s=float(1.0 / seconds) if seconds > 0 else None, duration_s=float(seconds))
    except (torch.cuda.OutOfMemoryError, RuntimeError, MemoryError) as exc:
        message = str(exc)
        oom = isinstance(exc, (torch.cuda.OutOfMemoryError, MemoryError)) or "out of memory" in message.lower()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        return ProbeResult(ok=False, oom=oom, error=message[:300])
    finally:
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            for param, original in zip(params, originals):
                param.copy_(original.to(param.device, param.dtype))
        if grad_checkpointing and not was_checkpointing:
            model.gradient_checkpointing_disable()
            if hasattr(model, 'disable_input_require_grads'):
                model.disable_input_require_grads()
        elif was_checkpointing and not grad_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train(was_training)


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
            accumulation=int(exec_config['accumulation']),
            amp=exec_config['amp'],
            grad_checkpointing=exec_config['grad_checkpointing'],
        )

    return probe
