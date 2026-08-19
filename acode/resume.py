"""Resume: rebuild a session's conversation from a --log-full transcript.

Only full transcripts are resumable — the default log clips payloads and
omits complete messages, so the exact conversation cannot be rebuilt from
it. Reconstruction replays the events that shape the conversation:

- ``task``      → the user message the CLI appended
- ``message``   → assistant turns and tool-result user messages, verbatim
- ``command`` /new → the conversation was cleared
- ``compact``   → the conversation was replaced by the summary pair logged
                  immediately before the event

A crash can truncate the final line, and a session killed mid-task can end
with a tool call whose results never arrived; the reader stops at the last
parseable line and trims back to the last clean boundary (an assistant
message with no unresolved tool calls), reporting how much it dropped.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

RESUME_NOTE = (
    "[Session resumed from a saved transcript on {date}. Time has passed "
    "since it was recorded and the repository may have changed — verify "
    "current file contents before relying on remembered details.]"
)


class ResumeError(Exception):
    """The transcript cannot be resumed; the message says why."""


@dataclass(frozen=True)
class ResumedSession:
    messages: list[dict[str, Any]]
    source: Path
    model: str | None  # what the transcript was recorded with
    dropped: int  # incomplete trailing messages trimmed away


def resolve_resume(
    spec: str, root: Path, log_dir: Path, *, exclude: Path | None = None
) -> ResumedSession:
    """Load the session for ``--resume <spec>`` — a path, or "last"."""
    if spec == "last":
        path = find_latest_log(root, log_dir, exclude=exclude)
        if path is None:
            raise ResumeError(f"no previous transcript for this workspace in {log_dir}")
    else:
        path = Path(spec).expanduser()
    return load_session(path, root)


def find_latest_log(root: Path, log_dir: Path, *, exclude: Path | None = None) -> Path | None:
    """The newest transcript recorded in this workspace, by filename stamp.

    Filenames only carry the workspace's basename, so each candidate's
    session_start is checked against the full path — same-named repositories
    elsewhere do not match. ``exclude`` keeps the running session's own
    (empty) transcript out of consideration.
    """
    for path in sorted(log_dir.glob(f"*-{root.name}.jsonl"), reverse=True):
        if exclude is not None and path == exclude:
            continue
        if _recorded_workspace(path) == str(root):
            return path
    return None


def _recorded_workspace(path: Path) -> str | None:
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                event = _parse(line)
                if event is None:
                    return None
                if event.get("event") == "session_start":
                    return event.get("workspace")
    except OSError:
        return None
    return None


def load_session(path: Path, root: Path) -> ResumedSession:
    """Rebuild the conversation recorded at ``path`` for workspace ``root``."""
    started = False
    full = False
    workspace: str | None = None
    model: str | None = None
    messages: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                event = _parse(line)
                if event is None:
                    break  # crash-truncated or corrupt tail; stop at the last good line
                match event.get("event"):
                    case "session_start":
                        started = True
                        full = bool(event.get("log_full"))
                        workspace = event.get("workspace")
                        model = event.get("model")
                    case "task" if isinstance(event.get("text"), str):
                        messages.append({"role": "user", "content": event["text"]})
                    case "message":
                        message = _rebuild_message(event)
                        if message is None:
                            # A block the writer could not serialize faithfully;
                            # this message and everything after it is unreliable.
                            break
                        messages.append(message)
                    case "command" if event.get("command") == "/new":
                        messages.clear()
                    case "compact":
                        messages[:] = messages[-2:]
    except OSError as exc:
        raise ResumeError(f"cannot read {path}: {exc}") from exc
    if not started:
        raise ResumeError(f"{path} is not an acode transcript")
    if not full:
        raise ResumeError(
            f"{path} is not a --log-full transcript; only full logs record "
            "the complete messages resume needs"
        )
    if workspace != str(root):
        raise ResumeError(
            f"{path} was recorded in workspace {workspace} — run acode there to resume it"
        )
    dropped = _trim(messages)
    if not messages:
        raise ResumeError(f"{path} contains no resumable conversation")
    return ResumedSession(messages=messages, source=path, model=model, dropped=dropped)


def resume_note_messages() -> list[dict[str, Any]]:
    """The staleness note appended after the restored conversation.

    A user/assistant pair, like compaction's, so roles keep alternating
    when the next task arrives.
    """
    note = RESUME_NOTE.format(date=datetime.now().strftime("%Y-%m-%d"))
    return [
        {"role": "user", "content": note},
        {"role": "assistant", "content": "Understood — resuming from the restored conversation."},
    ]


def _parse(line: str) -> dict[str, Any] | None:
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return None
    return event if isinstance(event, dict) else None


def _rebuild_message(event: dict[str, Any]) -> dict[str, Any] | None:
    role = event.get("role")
    content = event.get("content")
    if role not in {"user", "assistant"}:
        return None
    if isinstance(content, str):
        return {"role": role, "content": content}
    if isinstance(content, list) and all(isinstance(block, dict) for block in content):
        return {"role": role, "content": content}
    return None


def _trim(messages: list[dict[str, Any]]) -> int:
    """Drop trailing messages until the conversation ends cleanly.

    A trailing user message is a task or tool results the model never
    answered; a trailing assistant message with tool_use blocks is a call
    whose results never arrived. Either would poison the next request.
    """
    dropped = 0
    while messages:
        last = messages[-1]
        if last["role"] == "assistant" and not _has_tool_use(last["content"]):
            break
        messages.pop()
        dropped += 1
    return dropped


def _has_tool_use(content: Any) -> bool:
    if not isinstance(content, list):
        return False
    return any(
        isinstance(block, dict) and block.get("type") == "tool_use" for block in content
    )
