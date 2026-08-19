"""OS-level sandboxing for shell commands: Seatbelt on macOS, bubblewrap on Linux.

Defense-in-depth behind the approval gate, not a replacement for it. The
sandbox confines filesystem *writes* to the workspace plus temp and cache
directories; reads and network are untouched (commands are individually
approved, and the environment is already scrubbed of credentials).

Both mechanisms wrap the command rather than the harness, so a sandbox
failure can never corrupt acode itself. ``detect_sandbox`` probes the
mechanism at startup — a trivial command must succeed *and* a write outside
the writable roots must fail — so a present-but-broken sandbox is reported
instead of silently trusted.
"""

from __future__ import annotations

import contextlib
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

PROBE_TIMEOUT = 15.0


class SandboxUnavailable(Exception):
    """No working sandbox on this system; the message says why and what to do."""


@dataclass(frozen=True)
class Sandbox:
    kind: str  # "seatbelt" (macOS) or "bubblewrap" (Linux)
    writable: tuple[Path, ...]  # resolved roots commands may write beneath

    def wrap(self, command: str) -> list[str]:
        """The argv that runs *command* under this sandbox via /bin/sh."""
        if self.kind == "seatbelt":
            return [
                "sandbox-exec",
                "-p",
                seatbelt_profile(self.writable),
                "/bin/sh",
                "-c",
                command,
            ]
        return bwrap_argv(command, self.writable)

    def describe(self) -> str:
        return f"{self.kind} — command writes confined to the workspace, temp, and cache dirs"


def writable_roots(root: Path) -> tuple[Path, ...]:
    """Where sandboxed commands may write: workspace, temp, and user caches.

    Temp and cache dirs stay writable so ordinary tooling (uv, pip, npm,
    compilers) keeps working; on macOS the whole per-user /var/folders tree
    is included because compilers and frameworks cache there. Paths are
    resolved so symlinked forms (/tmp, /var/folders) match what the kernel
    sees; nonexistent candidates and roots nested under an earlier one are
    dropped.
    """
    temp = Path(tempfile.gettempdir())
    candidates = [root, temp, Path("/tmp")]
    if sys.platform == "darwin":
        candidates.append(temp.parent)  # per-user /var/folders/<xx>/<id> (T and C)
        candidates.append(Path.home() / "Library" / "Caches")
    else:
        candidates.append(Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"))
    roots: list[Path] = []
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            if not resolved.is_dir():
                continue
        except OSError:
            continue
        if resolved not in roots:
            roots.append(resolved)
    # Keep only the outermost roots: a root nested under another adds nothing.
    return tuple(
        root
        for root in roots
        if not any(root != other and root.is_relative_to(other) for other in roots)
    )


def seatbelt_profile(writable: tuple[Path, ...]) -> str:
    """An SBPL profile: allow everything except writes outside the roots.

    SBPL is last-match-wins, so the trailing allows punch holes in the
    file-write deny. Device writes that every shell pipeline needs (null,
    ttys, fd redirections) are allowed explicitly.
    """
    subpaths = "\n".join(f"    (subpath {_sbpl_string(path)})" for path in writable)
    return f"""\
(version 1)
(allow default)
(deny file-write*)
(allow file-write*
{subpaths})
(allow file-write-data file-write-flags
    (literal "/dev/null")
    (literal "/dev/zero")
    (literal "/dev/stdout")
    (literal "/dev/stderr")
    (regex #"^/dev/tty")
    (regex #"^/dev/fd/"))
"""


def _sbpl_string(path: Path) -> str:
    escaped = str(path).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def bwrap_argv(command: str, writable: tuple[Path, ...]) -> list[str]:
    """bubblewrap argv: the whole filesystem read-only, roots bound writable.

    Only the mount namespace is unshared (bwrap's minimum), so network, pids,
    and process groups behave exactly as unsandboxed commands do — the
    timeout's process-group kill still reaches every child.
    """
    argv = ["bwrap", "--ro-bind", "/", "/", "--proc", "/proc", "--dev", "/dev"]
    for path in writable:
        argv += ["--bind", str(path), str(path)]
    argv += ["--die-with-parent", "--", "/bin/sh", "-c", command]
    return argv


def detect_sandbox(root: Path) -> Sandbox:
    """The working sandbox for this platform, probed end to end.

    Raises SandboxUnavailable — with an actionable message — when the
    platform has no mechanism, the mechanism is not installed, or it fails
    either probe: a trivial command must succeed, and a write outside the
    writable roots must be blocked.
    """
    if sys.platform == "darwin":
        if not shutil.which("sandbox-exec"):
            raise SandboxUnavailable("sandbox-exec not found (expected on every macOS)")
        sandbox = Sandbox(kind="seatbelt", writable=writable_roots(root))
    elif sys.platform.startswith("linux"):
        if not shutil.which("bwrap"):
            raise SandboxUnavailable(
                "bubblewrap is not installed — install the 'bubblewrap' package "
                "(e.g. apt install bubblewrap) to sandbox commands"
            )
        sandbox = Sandbox(kind="bubblewrap", writable=writable_roots(root))
    else:
        raise SandboxUnavailable(f"no sandbox mechanism for platform {sys.platform!r}")
    _probe(sandbox)
    return sandbox


def _probe(sandbox: Sandbox) -> None:
    """Verify the sandbox both runs commands and actually confines writes."""
    if _run_probe(sandbox.wrap("true")) != 0:
        raise SandboxUnavailable(
            f"{sandbox.kind} failed to run a trivial command "
            "(common in containers and nested sandboxes)"
        )
    target = _outside_probe_path(sandbox)
    if target is None:
        return  # nowhere outside the writable roots to probe from; trust the mechanism
    try:
        _run_probe(sandbox.wrap(f"touch {shlex.quote(str(target))}"))
        if target.exists():
            target.unlink()
            raise SandboxUnavailable(
                f"{sandbox.kind} did not block a write outside the workspace — "
                "refusing to pretend commands are confined"
            )
    finally:
        with contextlib.suppress(OSError):
            if target.parent.name == ".acode" and not any(target.parent.iterdir()):
                target.parent.rmdir()


def _outside_probe_path(sandbox: Sandbox) -> Path | None:
    """A creatable path outside every writable root, or None if home is inside one."""
    probe_dir = Path.home() / ".acode"
    try:
        resolved = probe_dir.resolve()
        if any(resolved.is_relative_to(root) for root in sandbox.writable):
            return None
        probe_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return resolved / f"sandbox-probe-{os.getpid()}"


def _run_probe(argv: list[str]) -> int:
    try:
        return subprocess.run(
            argv,
            capture_output=True,
            timeout=PROBE_TIMEOUT,
            stdin=subprocess.DEVNULL,
        ).returncode
    except (OSError, subprocess.TimeoutExpired):
        return -1
