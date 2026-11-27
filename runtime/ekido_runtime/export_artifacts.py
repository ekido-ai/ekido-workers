"""Local GGUF export metadata; this is not a signed network manifest."""
from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import tempfile
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def export_lock(export_dir: Path):
    """Prevent concurrent exporters; the OS releases the lock on process exit."""
    root = Path(export_dir)
    root.mkdir(parents=True, exist_ok=True)
    fd = os.open(root / '.export.lock', os.O_RDWR | os.O_CREAT, 0o600)
    locked = False
    try:
        if os.fstat(fd).st_size == 0:
            os.write(fd, b'0')
        os.lseek(fd, 0, os.SEEK_SET)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError('export directory already has a writer') from exc
        locked = True
        yield
    finally:
        if locked:
            if os.name == 'nt':
                import msvcrt
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def record_gguf(export_dir: Path, artifact: Path, quantization: str) -> dict:
    """Hash a completed GGUF and atomically add it to the local export record.

    The quantization is the converter/quantizer setting supplied by the caller;
    checking a header is not a full GGUF or quantization validation. The caller
    owns the work directory and must serialize exports into it.
    """
    root = Path(export_dir).resolve(strict=True)
    path = Path(artifact).resolve(strict=True)
    try:
        relative = path.relative_to(root)
    except ValueError:
        raise ValueError('GGUF must be inside the export directory') from None
    if (not path.is_file() or path.suffix.lower() != '.gguf' or not isinstance(quantization, str)
            or not re.fullmatch(r'[A-Za-z0-9_]{1,32}', quantization)):
        raise ValueError('invalid GGUF artifact or quantization')
    info_path = root / 'export.json'
    if info_path.is_symlink():
        raise ValueError('export metadata must not be a symlink')
    info = json.loads(info_path.read_text(encoding='utf-8'))
    if not isinstance(info, dict) or not isinstance(info.get('artifacts', {}), dict):
        raise ValueError('invalid export metadata')
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        before = os.fstat(handle.fileno())
        head = handle.read(24)
        if len(head) != 24:
            raise ValueError('GGUF header is truncated')
        magic, version, tensors, metadata = struct.unpack('<4sIQQ', head)
        if magic != b'GGUF' or version not in (2, 3) or not 1 <= tensors <= 1_000_000 or not 1 <= metadata <= 1_000_000:
            raise ValueError('invalid GGUF header')
        digest.update(head)
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
        after = os.fstat(handle.fileno())
    current = path.stat()
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    if identity(before) != identity(after) or identity(after) != identity(current):
        raise ValueError('GGUF changed while hashing')
    entry = {'path': relative.as_posix(), 'format': 'gguf', 'quantization': quantization,
             'bytes': after.st_size, 'sha256': digest.hexdigest()}
    info.setdefault('artifacts', {})[relative.as_posix()] = entry
    data = (json.dumps(info, indent=2, ensure_ascii=False) + '\n').encode('utf-8')
    fd, temporary = tempfile.mkstemp(prefix='.export-', dir=root)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, info_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return entry
