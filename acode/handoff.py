"""Handoff notes: a per-workspace memory of where the last session left off.

On exit, the user can have the model write a handoff summary of the session
(the same summary compaction uses) to a file keyed by the workspace's full
path; the next session in that workspace loads it into the system prompt.
One note per workspace, overwritten by each save — delete the file to
forget. File errors never block a session: saving reports failure, loading
just comes back empty.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

HANDOFF_DIR = Path.home() / ".acode" / "handoff"


def handoff_path(root: Path, directory: Path = HANDOFF_DIR) -> Path:
    # The full path as the filename (slashes flattened), so same-named
    # repositories in different places do not share a note.
    slug = root.resolve().as_posix().strip("/").replace("/", "-")
    return directory / f"{slug}.md"


def save_handoff(root: Path, note: str, directory: Path = HANDOFF_DIR) -> Path | None:
    """Write the workspace's handoff note; None when the filesystem says no."""
    path = handoff_path(root, directory)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        path.write_text(note.rstrip("\n") + "\n", encoding="utf-8")
    except OSError:
        return None
    return path


def load_handoff(root: Path, directory: Path = HANDOFF_DIR) -> tuple[str, str] | None:
    """(date, note) of the workspace's saved handoff, or None.

    The date is the file's modification time — when the note was written.
    """
    path = handoff_path(root, directory)
    try:
        if not path.is_file():
            return None
        note = path.read_text(encoding="utf-8", errors="replace").strip()
        date = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d")
    except OSError:
        return None
    return (date, note) if note else None
