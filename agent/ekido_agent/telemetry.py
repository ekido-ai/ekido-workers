"""Host GPU telemetry through nvidia-smi. Self-reported by the provider: it never affects payment."""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass

QUERY = "name,uuid,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw,compute_cap,driver_version"


@dataclass(frozen=True)
class GpuSample:
    name: str
    uuid: str
    memory_total_mib: float | None
    memory_used_mib: float | None
    util_pct: float | None
    temp_c: float | None
    power_w: float | None
    compute_cap: str | None
    driver: str | None
    source: str = "HOST_NVML"

    def to_dict(self) -> dict:
        return asdict(self)


def _number(text: str) -> float | None:
    """nvidia-smi reports what it can't read as [N/A] and similar. Unknown stays None, never 0."""
    try:
        return float(text.strip())
    except ValueError:
        return None


def _text(text: str) -> str | None:
    text = text.strip()
    return None if not text or text.startswith("[") else text


def parse_nvidia_smi(output: str) -> list[GpuSample]:
    samples = []
    for line in output.strip().splitlines():
        fields = [f.strip() for f in line.split(",")]
        if len(fields) != 9:
            continue
        samples.append(
            GpuSample(
                name=fields[0],
                uuid=fields[1],
                memory_total_mib=_number(fields[2]),
                memory_used_mib=_number(fields[3]),
                util_pct=_number(fields[4]),
                temp_c=_number(fields[5]),
                power_w=_number(fields[6]),
                compute_cap=_text(fields[7]),
                driver=_text(fields[8]),
            )
        )
    return samples


def read_gpus(timeout: float = 5.0) -> list[GpuSample]:
    """An empty list means no readable GPU. A failed read must never crash the agent's heartbeat."""
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return parse_nvidia_smi(result.stdout) if result.returncode == 0 else []


def native_bf16(compute_cap: str | None) -> bool | None:
    """BF16 is native from Ampere (compute capability 8.0). Turing cards like the T4 only emulate it."""
    if not compute_cap:
        return None
    try:
        return int(compute_cap.split(".")[0]) >= 8
    except ValueError:
        return None
