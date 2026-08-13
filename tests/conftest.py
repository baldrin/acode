"""Shared fixtures. All file and shell tools operate only on tmp_path
workspaces — never on this project's own repository."""

from __future__ import annotations

from pathlib import Path

import pytest

MAIN_PY = '''\
def main() -> None:
    print("hello")


if __name__ == "__main__":
    main()
'''


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A resolved temporary directory seeded like a small project."""
    root = tmp_path.resolve()  # resolve: /tmp is a symlink on macOS
    (root / "src").mkdir()
    (root / "src" / "main.py").write_text(MAIN_PY)
    (root / "README.md").write_text("# Sample project\n\nA fixture repository for tests.\n")
    (root / ".git").mkdir()
    (root / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    return root
