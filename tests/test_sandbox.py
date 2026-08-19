"""Sandbox tests: pure argv/profile construction, detection logic with fakes,
and real end-to-end confinement on platforms whose mechanism is present
(Seatbelt on any macOS, bubblewrap on Linux where installed)."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

from acode import sandbox as sandbox_module
from acode.cli import _parse_args
from acode.safety import execute_command
from acode.sandbox import (
    Sandbox,
    SandboxUnavailable,
    bwrap_argv,
    detect_sandbox,
    seatbelt_profile,
    writable_roots,
)


class TestWritableRoots:
    def test_workspace_and_temp_are_covered(self, workspace: Path) -> None:
        # Coverage, not membership: a workspace under the temp tree (as
        # pytest's is on macOS) may be subsumed by a broader temp root.
        roots = writable_roots(workspace)
        assert any(workspace.is_relative_to(r) for r in roots)
        assert any(Path(tempfile.gettempdir()).resolve().is_relative_to(r) for r in roots)

    def test_roots_are_resolved(self, tmp_path: Path) -> None:
        real = (tmp_path / "real").resolve()
        real.mkdir()
        link = tmp_path / "link"
        link.symlink_to(real)
        roots = writable_roots(link)
        assert all(r == r.resolve() for r in roots)
        assert any(real.is_relative_to(r) for r in roots)

    def test_nested_roots_are_deduplicated(self, workspace: Path) -> None:
        roots = writable_roots(workspace)
        for root in roots:
            others = [r for r in roots if r != root]
            assert not any(root.is_relative_to(other) for other in others)


class TestSeatbeltProfile:
    def test_denies_writes_then_allows_the_roots(self, workspace: Path) -> None:
        profile = seatbelt_profile((workspace, Path("/private/tmp")))
        assert "(deny file-write*)" in profile
        assert f'(subpath "{workspace}")' in profile
        assert '(subpath "/private/tmp")' in profile
        # SBPL is last-match-wins: the allow must come after the deny.
        assert profile.index("(deny file-write*)") < profile.index(f'"{workspace}"')

    def test_quotes_in_paths_are_escaped(self) -> None:
        profile = seatbelt_profile((Path('/tmp/odd"name'),))
        assert '/tmp/odd\\"name' in profile

    def test_device_writes_stay_allowed(self, workspace: Path) -> None:
        profile = seatbelt_profile((workspace,))
        assert '"/dev/null"' in profile


class TestBwrapArgv:
    def test_readonly_root_with_writable_binds(self, workspace: Path) -> None:
        argv = bwrap_argv("make test", (workspace, Path("/tmp")))
        assert argv[:3] == ["bwrap", "--ro-bind", "/"]
        assert ["--bind", str(workspace), str(workspace)] == argv[
            argv.index(str(workspace)) - 1 : argv.index(str(workspace)) + 2
        ]
        assert argv[-3:] == ["/bin/sh", "-c", "make test"]

    def test_command_is_a_single_argv_element(self, workspace: Path) -> None:
        command = "echo 'a b'; ls | wc -l"
        assert bwrap_argv(command, (workspace,))[-1] == command


class FakeWrapSandbox(Sandbox):
    """A 'sandbox' that runs commands unconfined — a broken mechanism."""

    def wrap(self, command: str) -> list[str]:
        return ["/bin/sh", "-c", command]


class TestDetection:
    def test_unsupported_platform(self, monkeypatch: pytest.MonkeyPatch, workspace: Path) -> None:
        monkeypatch.setattr(sys, "platform", "win32")
        with pytest.raises(SandboxUnavailable, match="win32"):
            detect_sandbox(workspace)

    def test_linux_without_bubblewrap(
        self, monkeypatch: pytest.MonkeyPatch, workspace: Path
    ) -> None:
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr(sandbox_module.shutil, "which", lambda name: None)
        with pytest.raises(SandboxUnavailable, match="bubblewrap"):
            detect_sandbox(workspace)

    def test_mechanism_that_cannot_run_commands(
        self, monkeypatch: pytest.MonkeyPatch, workspace: Path
    ) -> None:
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(sandbox_module.shutil, "which", lambda name: f"/usr/bin/{name}")
        monkeypatch.setattr(sandbox_module, "_run_probe", lambda argv: 1)
        with pytest.raises(SandboxUnavailable, match="trivial command"):
            detect_sandbox(workspace)

    def test_broken_confinement_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Home is redirected into tmp so the probe file lands somewhere safe;
        # the fake wrap runs the touch unconfined, so the write "escapes" and
        # the probe must refuse the sandbox. Home must not sit inside the
        # writable root, or the escape probe would be skipped as unprobeable.
        home = tmp_path / "home"
        home.mkdir()
        ws = (tmp_path / "ws").resolve()
        ws.mkdir()
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
        broken = FakeWrapSandbox(kind="seatbelt", writable=(ws,))
        with pytest.raises(SandboxUnavailable, match="did not block"):
            sandbox_module._probe(broken)
        assert not list((home / ".acode").glob("sandbox-probe-*"))  # cleaned up

    def test_probe_skipped_when_home_is_inside_a_writable_root(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
        # Workspace contains home, so there is nowhere outside to probe from —
        # the (broken) negative probe cannot run, and detection still passes.
        sandbox_module._probe(FakeWrapSandbox(kind="seatbelt", writable=(tmp_path.resolve(),)))


class TestExecuteCommandWithSandbox:
    def test_wrap_is_used(self, workspace: Path) -> None:
        class Tagging(Sandbox):
            def wrap(self, command: str) -> list[str]:
                return ["/bin/sh", "-c", f"echo wrapped; {command}"]

        result = execute_command(
            "echo inner", workspace, sandbox=Tagging(kind="fake", writable=(workspace,))
        )
        assert result.stdout == "wrapped\ninner\n"
        assert result.exit_code == 0

    def test_none_means_plain_shell(self, workspace: Path) -> None:
        assert execute_command("echo plain", workspace).stdout == "plain\n"


@pytest.fixture
def real_sandbox(workspace: Path) -> Sandbox:
    """The platform's actual sandbox, or skip where none is available."""
    try:
        return detect_sandbox(workspace)
    except SandboxUnavailable as exc:
        pytest.skip(f"no working sandbox here: {exc}")


class TestRealConfinement:
    """End-to-end against the real OS mechanism (macOS always; Linux w/ bwrap)."""

    def test_write_inside_workspace_is_allowed(
        self, workspace: Path, real_sandbox: Sandbox
    ) -> None:
        result = execute_command("echo data > created.txt", workspace, sandbox=real_sandbox)
        assert result.exit_code == 0
        assert (workspace / "created.txt").read_text() == "data\n"

    def test_write_outside_workspace_is_blocked(
        self, workspace: Path, real_sandbox: Sandbox
    ) -> None:
        target = Path.home() / f"acode-sandbox-escape-{os.getpid()}"
        try:
            result = execute_command(
                f"touch {target}", workspace, sandbox=real_sandbox, timeout=30
            )
            assert result.exit_code != 0
            assert not target.exists()
        finally:
            target.unlink(missing_ok=True)

    def test_reads_outside_workspace_stay_open(
        self, workspace: Path, real_sandbox: Sandbox
    ) -> None:
        result = execute_command("head -1 /etc/hosts", workspace, sandbox=real_sandbox)
        assert result.exit_code == 0

    def test_pipes_exit_codes_and_stderr_pass_through(
        self, workspace: Path, real_sandbox: Sandbox
    ) -> None:
        result = execute_command(
            "echo out | tr a-z A-Z; echo err >&2; exit 3", workspace, sandbox=real_sandbox
        )
        assert result.stdout == "OUT\n"
        assert result.stderr == "err\n"
        assert result.exit_code == 3

    def test_timeout_still_kills_the_process_group(
        self, workspace: Path, real_sandbox: Sandbox
    ) -> None:
        result = execute_command("sleep 10", workspace, sandbox=real_sandbox, timeout=0.5)
        assert result.timed_out


class TestCliFlag:
    def test_no_sandbox_flag(self) -> None:
        assert _parse_args(["--no-sandbox"]).no_sandbox is True
        assert _parse_args([]).no_sandbox is False
