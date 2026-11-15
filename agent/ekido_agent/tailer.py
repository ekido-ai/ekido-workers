"""Reads the runtime's events.jsonl as it grows. Treats its content as untrusted."""

from __future__ import annotations

import json
from pathlib import Path


class EventTailer:
    """Returns only complete lines, since a read can land in the middle of a write."""

    MAX_LINE_BYTES = 8 * 1024 * 1024

    def __init__(self, path: Path):
        self.path = Path(path)
        self._offset = 0
        self._partial = b""
        self.malformed = 0

    def read_new(self) -> list[dict]:
        try:
            with open(self.path, "rb") as handle:
                handle.seek(self._offset)
                chunk = handle.read()
                self._offset = handle.tell()
        except OSError:
            return []
        data = self._partial + chunk
        lines = data.split(b"\n")
        self._partial = lines.pop()
        if len(self._partial) > self.MAX_LINE_BYTES:
            self._partial = b""
            self.malformed += 1
        events = []
        for line in lines:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except ValueError:
                self.malformed += 1
                continue
            if isinstance(event, dict) and isinstance(event.get("kind"), str):
                events.append(event)
            else:
                self.malformed += 1
        return events
