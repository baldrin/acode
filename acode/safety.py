"""Safety rails: path confinement, a command denylist, and command execution.

The launch directory (the "workspace root") is a hard boundary for all file
access. The command denylist is a tripwire for obvious footguns, not a
security boundary — the approval gate in the agent loop is the real control.
"""

from __future__ import annotations

import contextlib
import os
import re
import shlex
import signal
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .sandbox import Sandbox


class ToolError(Exception):
    """A tool call failed; the message is fed back to the model as an error."""


class SafetyError(ToolError):
    """A tool call violated a safety rail before it was executed."""


def resolve_path(path: str, root: Path) -> Path:
    """Resolve *path* against *root*, rejecting anything that escapes it.

    Symlinks and ``..`` segments are resolved before the containment check.
    The ``.git`` directory is off limits to every file tool — git operations
    go through ``run_command`` so they hit the approval gate.
    """
    resolved = (root / os.path.expanduser(path)).resolve()
    if not resolved.is_relative_to(root):
        raise SafetyError(f"path {path!r} is outside the workspace {root}")
    if ".git" in resolved.relative_to(root).parts:
        raise SafetyError(
            f"path {path!r} is inside .git; use run_command for git operations"
        )
    return resolved


_SUDO = re.compile(r"(?:^|[\s;&|(`])sudo\b")
_SHELLS = frozenset({"sh", "bash", "zsh", "dash", "fish"})
_DOWNLOADERS = frozenset({"curl", "wget"})


def check_command(command: str, root: Path) -> None:
    """Reject obvious footguns before the command is even shown for approval.

    This is a tripwire, not a security boundary: it catches accidents, not an
    adversary. The y/n approval gate is the real control.
    """
    if _SUDO.search(command):
        raise SafetyError("command uses sudo, which is blocked")
    if _pipes_download_to_shell(command):
        raise SafetyError("piping a download into a shell is blocked")
    for tokens in _simple_commands(command):
        if _is_dangerous_rm(tokens, root):
            raise SafetyError(
                "rm -rf targeting the workspace root or a directory above it is blocked"
            )


def _tokenize(segment: str) -> list[str]:
    try:
        return shlex.split(segment)
    except ValueError:  # unbalanced quotes etc. — fall back to a crude split
        return segment.split()


def _simple_commands(command: str) -> list[list[str]]:
    """Split a shell command on ;, &, |, &&, || and tokenize each part."""
    return [_tokenize(segment) for segment in re.split(r"[;&|]+", command)]


def _pipes_download_to_shell(command: str) -> bool:
    segments = command.split("|")
    for i, segment in enumerate(segments[:-1]):
        tokens = _tokenize(segment)
        if not tokens or Path(tokens[0]).name not in _DOWNLOADERS:
            continue
        for downstream in segments[i + 1 :]:
            tokens = [
                t
                for t in _tokenize(downstream)
                if t not in {"sudo", "env"} and "=" not in t
            ]
            if tokens and Path(tokens[0]).name in _SHELLS:
                return True
    return False


def _is_dangerous_rm(tokens: list[str], root: Path) -> bool:
    """True for an ``rm`` that is recursive+forced at or above the workspace."""
    if not tokens or Path(tokens[0]).name != "rm":
        return False
    recursive = force = False
    targets: list[str] = []
    for token in tokens[1:]:
        if token.startswith("--"):
            recursive |= token == "--recursive"
            force |= token == "--force"
        elif token.startswith("-") and len(token) > 1:
            recursive |= bool(set("rR") & set(token))
            force |= "f" in token
        else:
            targets.append(token)
    if not (recursive and force):
        return False
    for target in targets:
        candidate = Path(os.path.expandvars(os.path.expanduser(target)))
        if not candidate.is_absolute():
            candidate = root / candidate
        while candidate.name in {"*", "**"}:  # treat "dir/*" as "dir"
            candidate = candidate.parent
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if root.is_relative_to(resolved):
            return True
    return False


_SECRET_ENV_SUFFIXES = (
    "_TOKEN",
    "_KEY",
    "_SECRET",
    "_PASSWORD",
    "_PASSWD",
    "_CREDENTIALS",
)
_SECRET_ENV_PREFIXES = ("ANTHROPIC_", "AWS_")


def scrubbed_environment() -> dict[str, str]:
    """os.environ minus anything that looks like a credential.

    Commands the model runs (and any code they invoke) get this environment,
    so approved-but-hostile code cannot simply read API keys or cloud
    credentials out of the process environment.
    """

    def is_secret(name: str) -> bool:
        upper = name.upper()
        return upper.startswith(_SECRET_ENV_PREFIXES) or upper.endswith(_SECRET_ENV_SUFFIXES)

    return {name: value for name, value in os.environ.items() if not is_secret(name)}


@dataclass(frozen=True)
class CommandResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool


def execute_command(
    command: str,
    root: Path,
    *,
    timeout: float = 120.0,
    sandbox: Sandbox | None = None,
) -> CommandResult:
    """Run *command* through the shell with *root* as the working directory.

    All shell execution funnels through this one function. With *sandbox*
    set, the command runs under the OS sandbox (writes confined to the
    workspace, temp, and cache dirs); without it, plain shell execution.
    On timeout the whole process group is killed and the result says so.
    """
    process = subprocess.Popen(
        sandbox.wrap(command) if sandbox else command,
        shell=sandbox is None,
        cwd=root,
        env=scrubbed_environment(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        errors="replace",
        start_new_session=True,  # own process group, so timeout kills children too
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
    return CommandResult(
        stdout=stdout,
        stderr=stderr,
        exit_code=process.returncode,
        timed_out=timed_out,
    )


SNAPSHOT_REF = "refs/acode/snapshot"


def create_snapshot(root: Path) -> str | None:
    """Snapshot the workspace state to a git commit, best-effort.

    Captures tracked *and* untracked files (gitignore respected) using a
    throwaway index, so the user's working tree, index, and branches are
    never touched. The commit is pinned at SNAPSHOT_REF and its hash is
    returned; restore with ``git restore --source <hash> -- .``.

    Returns None when the workspace is not a git repository or snapshotting
    fails for any reason — the harness proceeds without the safety net.
    """
    if not (root / ".git").exists():
        return None
    identity = {
        "GIT_AUTHOR_NAME": "acode",
        "GIT_AUTHOR_EMAIL": "acode@localhost",
        "GIT_COMMITTER_NAME": "acode",
        "GIT_COMMITTER_EMAIL": "acode@localhost",
    }
    try:
        with tempfile.TemporaryDirectory(prefix="acode-git-") as tmp:
            env = {**os.environ, **identity, "GIT_INDEX_FILE": str(Path(tmp) / "index")}
            _git(root, env, "add", "-A", ".")
            tree = _git(root, env, "write-tree")
            try:
                parent = ["-p", _git(root, env, "rev-parse", "--verify", "HEAD")]
            except subprocess.CalledProcessError:  # repo has no commits yet
                parent = []
            commit = _git(
                root, env, "commit-tree", tree, *parent, "-m", "acode workspace snapshot"
            )
            _git(root, env, "update-ref", SNAPSHOT_REF, commit)
            return commit
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def _git(root: Path, env: dict[str, str], *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        env=env,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=True,
        timeout=30,
    )
    return completed.stdout.strip()
