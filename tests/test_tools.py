from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from acode import tools
from acode.safety import SafetyError, ToolError
from acode.tools import (
    GREP_MAX_MATCHES,
    READ_MAX_LINES,
    build_preview,
    run_tool,
)


class TestReadFile:
    def test_reads_file(self, workspace: Path) -> None:
        result = run_tool("read_file", {"path": "src/main.py"}, workspace)
        assert 'print("hello")' in result

    def test_missing_file(self, workspace: Path) -> None:
        with pytest.raises(ToolError, match="not a file"):
            run_tool("read_file", {"path": "nope.txt"}, workspace)

    def test_directory_rejected(self, workspace: Path) -> None:
        with pytest.raises(ToolError, match="not a file"):
            run_tool("read_file", {"path": "src"}, workspace)

    def test_line_truncation(self, workspace: Path) -> None:
        (workspace / "big.txt").write_text("\n".join(f"line {i}" for i in range(3000)))
        result = run_tool("read_file", {"path": "big.txt"}, workspace)
        assert f"[truncated: showing {READ_MAX_LINES} of 3000 lines]" in result
        assert "line 1999" in result
        assert "line 2000" not in result

    def test_byte_truncation(self, workspace: Path) -> None:
        (workspace / "wide.txt").write_text("x" * 60_000)
        result = run_tool("read_file", {"path": "wide.txt"}, workspace)
        assert "[truncated: output exceeded" in result

    def test_missing_argument(self, workspace: Path) -> None:
        with pytest.raises(ToolError, match="'path' must be a string"):
            run_tool("read_file", {}, workspace)


class TestListFiles:
    def test_lists_directory(self, workspace: Path) -> None:
        result = run_tool("list_files", {"path": "."}, workspace)
        assert "src/" in result
        assert "README.md" in result
        assert "bytes" in result

    def test_empty_directory(self, workspace: Path) -> None:
        (workspace / "empty").mkdir()
        assert run_tool("list_files", {"path": "empty"}, workspace) == "(empty directory)"

    def test_file_rejected(self, workspace: Path) -> None:
        with pytest.raises(ToolError, match="not a directory"):
            run_tool("list_files", {"path": "README.md"}, workspace)

    def test_git_rejected(self, workspace: Path) -> None:
        with pytest.raises(SafetyError, match="inside .git"):
            run_tool("list_files", {"path": ".git"}, workspace)


class TestGlob:
    def test_finds_files(self, workspace: Path) -> None:
        result = run_tool("glob", {"pattern": "**/*.py"}, workspace)
        assert "src/main.py" in result

    def test_excludes_git(self, workspace: Path) -> None:
        (workspace / ".git" / "hook.py").write_text("# hook")
        result = run_tool("glob", {"pattern": "**/*.py"}, workspace)
        assert ".git" not in result

    def test_no_matches(self, workspace: Path) -> None:
        assert run_tool("glob", {"pattern": "*.rs"}, workspace) == "(no matches)"

    def test_parent_pattern_rejected(self, workspace: Path) -> None:
        with pytest.raises(ToolError, match="must be relative"):
            run_tool("glob", {"pattern": "../**/*.py"}, workspace)

    def test_absolute_pattern_rejected(self, workspace: Path) -> None:
        with pytest.raises(ToolError, match="must be relative"):
            run_tool("glob", {"pattern": "/etc/*"}, workspace)


class TestGrepPythonFallback:
    @pytest.fixture(autouse=True)
    def no_ripgrep(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(tools, "_find_rg", lambda: None)

    def test_finds_match_with_line_number(self, workspace: Path) -> None:
        result = run_tool("grep", {"pattern": r"def main", "path": "."}, workspace)
        assert "src/main.py:1: def main() -> None:" in result

    def test_searches_single_file(self, workspace: Path) -> None:
        result = run_tool("grep", {"pattern": "hello", "path": "src/main.py"}, workspace)
        assert "src/main.py:2:" in result

    def test_no_matches(self, workspace: Path) -> None:
        assert run_tool("grep", {"pattern": "zzz_absent", "path": "."}, workspace) == "(no matches)"

    def test_invalid_regex(self, workspace: Path) -> None:
        with pytest.raises(ToolError, match="invalid regex"):
            run_tool("grep", {"pattern": "(unclosed", "path": "."}, workspace)

    def test_missing_path(self, workspace: Path) -> None:
        with pytest.raises(ToolError, match="does not exist"):
            run_tool("grep", {"pattern": "x", "path": "nope"}, workspace)

    def test_match_cap(self, workspace: Path) -> None:
        (workspace / "many.txt").write_text("\n".join("needle" for _ in range(300)))
        result = run_tool("grep", {"pattern": "needle", "path": "many.txt"}, workspace)
        assert f"[truncated: showing first {GREP_MAX_MATCHES} matches]" in result
        assert result.count("needle") == GREP_MAX_MATCHES

    def test_skips_binary_files(self, workspace: Path) -> None:
        (workspace / "blob.bin").write_bytes(b"\x00\x01needle\x00")
        assert run_tool("grep", {"pattern": "needle", "path": "."}, workspace) == "(no matches)"

    def test_skips_hidden_files(self, workspace: Path) -> None:
        (workspace / ".secret").write_text("needle")
        assert run_tool("grep", {"pattern": "needle", "path": "."}, workspace) == "(no matches)"


@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep not installed")
class TestGrepRipgrep:
    def test_finds_match_with_line_number(self, workspace: Path) -> None:
        result = run_tool("grep", {"pattern": r"def main", "path": "."}, workspace)
        assert "src/main.py:1:" in result

    def test_no_matches(self, workspace: Path) -> None:
        assert run_tool("grep", {"pattern": "zzz_absent", "path": "."}, workspace) == "(no matches)"


class TestEditFile:
    def test_replaces_unique_text(self, workspace: Path) -> None:
        result = run_tool(
            "edit_file",
            {"path": "src/main.py", "old_text": 'print("hello")', "new_text": 'print("goodbye")'},
            workspace,
        )
        assert result == "Edited src/main.py"
        assert 'print("goodbye")' in (workspace / "src" / "main.py").read_text()

    def test_missing_text(self, workspace: Path) -> None:
        with pytest.raises(ToolError, match="not found"):
            run_tool(
                "edit_file",
                {"path": "src/main.py", "old_text": "absent", "new_text": "x"},
                workspace,
            )

    def test_ambiguous_text(self, workspace: Path) -> None:
        with pytest.raises(ToolError, match="appears 2 times"):
            run_tool(
                "edit_file",
                {"path": "src/main.py", "old_text": "main()", "new_text": "start()"},
                workspace,
            )

    def test_empty_old_text(self, workspace: Path) -> None:
        with pytest.raises(ToolError, match="must not be empty"):
            run_tool(
                "edit_file",
                {"path": "src/main.py", "old_text": "", "new_text": "x"},
                workspace,
            )

    def test_missing_file(self, workspace: Path) -> None:
        with pytest.raises(ToolError, match="does not exist"):
            run_tool(
                "edit_file", {"path": "nope.py", "old_text": "a", "new_text": "b"}, workspace
            )


class TestWriteFile:
    def test_creates_file_with_parents(self, workspace: Path) -> None:
        result = run_tool(
            "write_file", {"path": "deep/nested/new.txt", "content": "hi"}, workspace
        )
        assert result == "Created deep/nested/new.txt (2 bytes)"
        assert (workspace / "deep" / "nested" / "new.txt").read_text() == "hi"

    def test_overwrites_file(self, workspace: Path) -> None:
        result = run_tool("write_file", {"path": "README.md", "content": "new"}, workspace)
        assert result.startswith("Overwrote")
        assert (workspace / "README.md").read_text() == "new"

    def test_escape_rejected(self, workspace: Path) -> None:
        with pytest.raises(SafetyError, match="outside the workspace"):
            run_tool("write_file", {"path": "../evil.txt", "content": "x"}, workspace)


class TestRunCommand:
    def test_captures_output_and_exit_code(self, workspace: Path) -> None:
        result = run_tool("run_command", {"command": "echo out; echo err >&2; exit 2"}, workspace)
        assert "exit code: 2" in result
        assert "stdout:\nout" in result
        assert "stderr:\nerr" in result

    def test_denylist_applies(self, workspace: Path) -> None:
        with pytest.raises(SafetyError, match="sudo"):
            run_tool("run_command", {"command": "sudo ls"}, workspace)

    def test_output_truncated(self, workspace: Path) -> None:
        command = "head -c 20000 /dev/zero | tr '\\0' 'x'"
        result = run_tool("run_command", {"command": command}, workspace)
        assert "[truncated: output exceeded" in result

    def test_timeout_reported(self, workspace: Path) -> None:
        result = run_tool("run_command", {"command": "sleep 10"}, workspace, timeout=0.3)
        assert "timed out after 0.3s" in result


class TestBuildPreview:
    def test_run_command_preview(self, workspace: Path) -> None:
        assert build_preview("run_command", {"command": "ls -la"}, workspace) == "$ ls -la"

    def test_run_command_denylist_raises_before_prompt(self, workspace: Path) -> None:
        with pytest.raises(SafetyError, match="sudo"):
            build_preview("run_command", {"command": "sudo ls"}, workspace)

    def test_edit_preview_is_a_diff(self, workspace: Path) -> None:
        preview = build_preview(
            "edit_file",
            {"path": "src/main.py", "old_text": 'print("hello")', "new_text": 'print("bye")'},
            workspace,
        )
        assert "--- a/src/main.py" in preview
        assert '-    print("hello")' in preview
        assert '+    print("bye")' in preview
        # preview must not modify the file
        assert 'print("hello")' in (workspace / "src" / "main.py").read_text()

    def test_edit_preview_rejects_bad_edit(self, workspace: Path) -> None:
        with pytest.raises(ToolError, match="not found"):
            build_preview(
                "edit_file",
                {"path": "src/main.py", "old_text": "absent", "new_text": "x"},
                workspace,
            )

    def test_write_new_file_shows_full_content(self, workspace: Path) -> None:
        preview = build_preview(
            "write_file", {"path": "new.txt", "content": "full content here"}, workspace
        )
        assert preview.startswith("new file: new.txt")
        assert "full content here" in preview

    def test_write_existing_file_shows_diff(self, workspace: Path) -> None:
        preview = build_preview(
            "write_file", {"path": "README.md", "content": "# Rewritten\n"}, workspace
        )
        assert "--- a/README.md" in preview
        assert "+# Rewritten" in preview


class TestDispatch:
    def test_unknown_tool(self, workspace: Path) -> None:
        with pytest.raises(ToolError, match="unknown tool"):
            run_tool("teleport", {}, workspace)

    def test_definitions_cover_dispatch(self) -> None:
        names = {definition["name"] for definition in tools.TOOL_DEFINITIONS}
        assert names == {
            "read_file",
            "list_files",
            "glob",
            "grep",
            "edit_file",
            "write_file",
            "run_command",
        }
        assert tools.MUTATING_TOOLS < names
