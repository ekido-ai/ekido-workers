"""Validate coordinator-provided identities before using them in filesystem paths."""
from pathlib import Path
from uuid import UUID


def run_directory(root: Path, job_id: str, attempt: int) -> Path:
    try:
        parsed = UUID(job_id)
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("invalid job identifier") from exc
    if parsed.version != 4 or str(parsed) != job_id:
        raise ValueError("job identifier must be a canonical UUIDv4")
    if type(attempt) is not int or attempt < 1:
        raise ValueError("attempt must be a positive integer")
    base = Path(root).resolve()
    parent = base / job_id[:12]
    target = parent / f"attempt_{attempt}"
    for path in (parent, target):
        if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
            raise ValueError("run directory contains a link")
    if not target.resolve().is_relative_to(base):
        raise ValueError("run directory escapes the agent workdir")
    return target
