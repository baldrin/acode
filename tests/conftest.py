"""Shared fixtures. All file and shell tools operate only on tmp_path
workspaces — never on this project's own repository."""

from __future__ import annotations

import shutil
import subprocess
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


def git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=True
    )
    return completed.stdout.strip()


@pytest.fixture
def git_workspace(workspace: Path) -> Path:
    """The workspace fixture turned into a real git repository with one commit."""
    shutil.rmtree(workspace / ".git")  # replace the fake .git with a real repo
    git(workspace, "init", "-q", "-b", "main")
    git(workspace, "config", "user.name", "Test")
    git(workspace, "config", "user.email", "test@example.com")
    git(workspace, "add", "-A")
    git(workspace, "commit", "-q", "-m", "initial")
    return workspace
