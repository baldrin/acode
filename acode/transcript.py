"""Session transcripts: an append-only JSONL audit log per session.

One file per session (default: ~/.acode/logs/<stamp>-<workspace>.jsonl).
Each line is a JSON object with a timestamp and an "event" field: tasks,
assistant text, tool calls, approval outcomes, tool results, usage, and
warnings. Large payloads are clipped — the transcript answers "what did it
do and did I approve it", it is not a data store. Logging failures never
interrupt a session.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import IO, Any

CLIP_CHARS = 2_000


def clip(text: str, limit: int = CLIP_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"… [clipped {len(text) - limit:,} chars]"


def clip_args(args: dict[str, Any]) -> dict[str, Any]:
    return {
        key: clip(value) if isinstance(value, str) else value
        for key, value in args.items()
    }


class Transcript:
    def __init__(self, path: Path, handle: IO[str]) -> None:
        self.path = path
        self._handle = handle
        self._pending_text: list[str] = []
        self._closed = False

    @classmethod
    def open(cls, root: Path, directory: Path) -> Transcript | None:
        """Create this session's log file; None if the filesystem says no."""
        try:
            directory.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            path = directory / f"{stamp}-{root.name}.jsonl"
            handle = path.open("a", encoding="utf-8")
        except OSError:
            return None
        return cls(path, handle)

    def log(self, event: str, **fields: Any) -> None:
        self._flush_text()
        self._write({"event": event, **fields})

    def text(self, chunk: str) -> None:
        """Accumulate streamed assistant text; written as one event when the
        next non-text event arrives (or on close)."""
        self._pending_text.append(chunk)

    def close(self) -> None:
        if self._closed:
            return
        self._flush_text()
        self._closed = True
        self._handle.close()

    def _flush_text(self) -> None:
        if self._pending_text:
            text = "".join(self._pending_text)
            self._pending_text = []
            self._write({"event": "assistant_text", "text": text})

    def _write(self, record: dict[str, Any]) -> None:
        if self._closed:
            return
        stamped = {"ts": datetime.now().astimezone().isoformat(timespec="seconds"), **record}
        try:
            self._handle.write(json.dumps(stamped, default=str) + "\n")
            self._handle.flush()  # crash-safe: every event lands on disk
        except OSError:
            # An audit log that kills the session it audits is worse than a
            # gap in the log; drop the record and keep working.
            pass
