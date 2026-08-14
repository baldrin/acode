"""The seven tools: API schemas, implementations, previews, and dispatch.

Read-only tools (read_file, list_files, glob, grep) run without approval.
Mutating tools (edit_file, write_file, run_command) are previewed via
``build_preview`` and gated behind a y/n prompt in the agent loop.
Every result string is truncated before it enters the conversation.
"""

from __future__ import annotations

import difflib
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .safety import (
    SafetyError,
    ToolError,
    check_command,
    execute_command,
    resolve_path,
)

READ_MAX_BYTES = 50_000
READ_MAX_LINES = 2_000
LIST_MAX_ENTRIES = 500
GLOB_MAX_RESULTS = 500
GREP_MAX_MATCHES = 100
OUTPUT_MAX_BYTES = 10_000

MUTATING_TOOLS = frozenset({"edit_file", "write_file", "run_command"})

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "read_file",
        "description": (
            "Read a file's contents. Call this before editing a file so your "
            "edit_file old_text matches exactly. Output is truncated past "
            f"{READ_MAX_LINES} lines or {READ_MAX_BYTES} bytes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path, relative to the workspace root.",
                }
            },
            "required": ["path"],
        },
    },
    {
        "name": "list_files",
        "description": (
            "List the entries of a single directory (non-recursive). "
            "Use glob to find files across the tree."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path relative to the workspace root; '.' for the root.",
                }
            },
            "required": ["path"],
        },
    },
    {
        "name": "glob",
        "description": (
            "Find files matching a glob pattern like '**/*.py', relative to "
            "the workspace root. Returns matching paths, .git excluded."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern, e.g. 'src/**/*.py'. Must be relative.",
                }
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "grep",
        "description": (
            "Search file contents for a regular expression. Returns matching "
            "lines as path:line: text, capped at "
            f"{GREP_MAX_MATCHES} matches. Keep to common regex syntax."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regular expression to search for.",
                },
                "path": {
                    "type": "string",
                    "description": "File or directory to search, relative to the workspace root; '.' for everything.",
                },
            },
            "required": ["pattern", "path"],
        },
    },
    {
        "name": "edit_file",
        "description": (
            "Replace one exact, unique occurrence of old_text in a file with "
            "new_text. Fails if old_text is missing or appears more than once "
            "— read the file first and include enough surrounding context to "
            "make it unique. Requires user approval."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path, relative to the workspace root.",
                },
                "old_text": {
                    "type": "string",
                    "description": "Exact text to replace. Must occur exactly once in the file.",
                },
                "new_text": {
                    "type": "string",
                    "description": "Replacement text.",
                },
            },
            "required": ["path", "old_text", "new_text"],
        },
    },
    {
        "name": "write_file",
        "description": (
            "Create a new file or overwrite an existing one with the given "
            "content. Parent directories are created as needed. Prefer "
            "edit_file for small changes to existing files. Requires user "
            "approval."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path, relative to the workspace root.",
                },
                "content": {
                    "type": "string",
                    "description": "Full file content to write.",
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "run_command",
        "description": (
            "Run a shell command with the workspace root as the working "
            "directory and return its stdout, stderr, and exit code. Use this "
            "for git, tests, builds, and anything the file tools don't cover. "
            "Commands time out after a configurable limit (default 120s) and "
            "run with an environment scrubbed of credential-like variables "
            "(names ending in _TOKEN/_KEY/_SECRET etc.), so commands that "
            "need such credentials will not find them. Requires user approval."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute.",
                }
            },
            "required": ["command"],
        },
    },
]


def read_file(path: str, *, root: Path) -> str:
    target = resolve_path(path, root)
    if not target.is_file():
        raise ToolError(f"{path!r} is not a file")
    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ToolError(f"could not read {path!r}: {exc}") from exc
    return _truncate(text, max_bytes=READ_MAX_BYTES, max_lines=READ_MAX_LINES)


def list_files(path: str, *, root: Path) -> str:
    target = resolve_path(path, root)
    if not target.is_dir():
        raise ToolError(f"{path!r} is not a directory")
    entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name))
    lines = []
    for entry in entries[:LIST_MAX_ENTRIES]:
        if entry.is_dir():
            lines.append(f"{entry.name}/")
        else:
            lines.append(f"{entry.name}  ({entry.stat().st_size} bytes)")
    if len(entries) > LIST_MAX_ENTRIES:
        lines.append(f"[truncated: showing {LIST_MAX_ENTRIES} of {len(entries)} entries]")
    return "\n".join(lines) if lines else "(empty directory)"


def glob_files(pattern: str, *, root: Path) -> str:
    parts = Path(pattern).parts
    if Path(pattern).is_absolute() or ".." in parts:
        raise ToolError(f"glob pattern must be relative and not contain '..': {pattern!r}")
    try:
        matches = sorted(root.glob(pattern))
    except ValueError as exc:
        raise ToolError(f"invalid glob pattern {pattern!r}: {exc}") from exc
    results: list[str] = []
    truncated = False
    for match in matches:
        relative = match.relative_to(root)
        if ".git" in relative.parts:
            continue
        if len(results) == GLOB_MAX_RESULTS:
            truncated = True
            break
        results.append(str(relative))
    if truncated:
        results.append(f"[truncated: more than {GLOB_MAX_RESULTS} matches]")
    return "\n".join(results) if results else "(no matches)"


def grep(pattern: str, path: str, *, root: Path) -> str:
    try:
        regex = re.compile(pattern)
    except re.error as exc:
        raise ToolError(f"invalid regex {pattern!r}: {exc}") from exc
    target = resolve_path(path, root)
    if not target.exists():
        raise ToolError(f"{path!r} does not exist")
    if _find_rg():
        return _grep_ripgrep(pattern, target, root)
    return _grep_python(regex, target, root)


def _find_rg() -> str | None:
    return shutil.which("rg")


def _grep_ripgrep(pattern: str, target: Path, root: Path) -> str:
    relative = target.relative_to(root)
    completed = subprocess.run(
        [
            "rg",
            "--line-number",
            "--no-heading",
            "--color=never",
            "--max-count",
            str(GREP_MAX_MATCHES),
            "-e",
            pattern,
            "--",
            str(relative),
        ],
        cwd=root,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    if completed.returncode == 1:
        return "(no matches)"
    if completed.returncode != 0:
        raise ToolError(f"ripgrep failed: {completed.stderr.strip()}")
    return _cap_matches(completed.stdout.splitlines())


def _grep_python(regex: re.Pattern[str], target: Path, root: Path) -> str:
    files = [target] if target.is_file() else sorted(
        p
        for p in target.rglob("*")
        if p.is_file() and not any(part.startswith(".") for part in p.relative_to(root).parts)
    )
    matches: list[str] = []
    for file in files:
        try:
            data = file.read_bytes()
        except OSError:
            continue
        if b"\x00" in data[:8192]:  # skip binary files
            continue
        text = data.decode("utf-8", errors="replace")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                matches.append(f"{file.relative_to(root)}:{line_number}: {line}")
                if len(matches) > GREP_MAX_MATCHES:
                    return _cap_matches(matches)
    return _cap_matches(matches)


def _cap_matches(matches: list[str]) -> str:
    if not matches:
        return "(no matches)"
    if len(matches) > GREP_MAX_MATCHES:
        return "\n".join(
            matches[:GREP_MAX_MATCHES] + [f"[truncated: showing first {GREP_MAX_MATCHES} matches]"]
        )
    return "\n".join(matches)


def edit_file(path: str, old_text: str, new_text: str, *, root: Path) -> str:
    target = resolve_path(path, root)
    updated = _apply_edit(target, path, old_text, new_text)
    target.write_text(updated, encoding="utf-8")
    return f"Edited {path}"


def _apply_edit(target: Path, path: str, old_text: str, new_text: str) -> str:
    """Validate an edit and return the resulting file content."""
    if not target.is_file():
        raise ToolError(f"{path!r} does not exist; use write_file to create it")
    if not old_text:
        raise ToolError("old_text must not be empty")
    content = target.read_text(encoding="utf-8", errors="replace")
    count = content.count(old_text)
    if count == 0:
        raise ToolError(
            f"old_text was not found in {path!r}; read the file and retry with the exact text"
        )
    if count > 1:
        raise ToolError(
            f"old_text appears {count} times in {path!r}; include more surrounding "
            "context to make it unique"
        )
    return content.replace(old_text, new_text, 1)


def write_file(path: str, content: str, *, root: Path) -> str:
    target = resolve_path(path, root)
    existed = target.exists()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    verb = "Overwrote" if existed else "Created"
    return f"{verb} {path} ({len(content)} bytes)"


def run_command(command: str, *, root: Path, timeout: float) -> str:
    check_command(command, root)
    result = execute_command(command, root, timeout=timeout)
    parts = []
    if result.timed_out:
        parts.append(f"Command timed out after {timeout:g}s and was killed.")
    parts.append(f"exit code: {result.exit_code}")
    if result.stdout:
        parts.append("stdout:\n" + _truncate(result.stdout, max_bytes=OUTPUT_MAX_BYTES))
    if result.stderr:
        parts.append("stderr:\n" + _truncate(result.stderr, max_bytes=OUTPUT_MAX_BYTES))
    return "\n".join(parts)


def build_preview(name: str, args: dict[str, Any], root: Path) -> str:
    """Build the plain-text preview shown at the approval prompt.

    Raises ToolError/SafetyError for calls that must fail before the user is
    ever prompted (denylisted commands, edits that can't apply).
    """
    if name == "run_command":
        command = _require_str(args, "command")
        check_command(command, root)
        return f"$ {command}"
    if name == "edit_file":
        path = _require_str(args, "path")
        target = resolve_path(path, root)
        current = target.read_text(encoding="utf-8", errors="replace") if target.is_file() else ""
        updated = _apply_edit(
            target, path, _require_str(args, "old_text"), _require_str(args, "new_text")
        )
        return _diff(current, updated, path)
    if name == "write_file":
        path = _require_str(args, "path")
        content = _require_str(args, "content")
        target = resolve_path(path, root)
        if target.is_file():
            current = target.read_text(encoding="utf-8", errors="replace")
            return _diff(current, content, path)
        return f"new file: {path}\n{'-' * 40}\n{content}"
    raise ToolError(f"no preview for tool {name!r}")


def _diff(old: str, new: str, path: str) -> str:
    lines = difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
    )
    return "".join(lines) or "(no changes)"


def run_tool(name: str, args: dict[str, Any], root: Path, *, timeout: float = 120.0) -> str:
    """Execute a tool by name. Raises ToolError (or SafetyError) on failure."""
    match name:
        case "read_file":
            return read_file(_require_str(args, "path"), root=root)
        case "list_files":
            return list_files(_require_str(args, "path"), root=root)
        case "glob":
            return glob_files(_require_str(args, "pattern"), root=root)
        case "grep":
            return grep(_require_str(args, "pattern"), _require_str(args, "path"), root=root)
        case "edit_file":
            return edit_file(
                _require_str(args, "path"),
                _require_str(args, "old_text"),
                _require_str(args, "new_text"),
                root=root,
            )
        case "write_file":
            return write_file(
                _require_str(args, "path"), _require_str(args, "content"), root=root
            )
        case "run_command":
            return run_command(_require_str(args, "command"), root=root, timeout=timeout)
        case _:
            raise ToolError(f"unknown tool {name!r}")


def _require_str(args: dict[str, Any], key: str) -> str:
    value = args.get(key)
    if not isinstance(value, str):
        raise ToolError(f"argument {key!r} must be a string, got {type(value).__name__}")
    return value


def _truncate(text: str, *, max_bytes: int, max_lines: int | None = None) -> str:
    notes: list[str] = []
    if max_lines is not None:
        lines = text.splitlines(keepends=True)
        if len(lines) > max_lines:
            text = "".join(lines[:max_lines])
            notes.append(f"[truncated: showing {max_lines} of {len(lines)} lines]")
    data = text.encode("utf-8", errors="replace")
    if len(data) > max_bytes:
        text = data[:max_bytes].decode("utf-8", errors="ignore")
        notes.append(f"[truncated: output exceeded {max_bytes} bytes]")
    if notes:
        return text.rstrip("\n") + "\n" + "\n".join(notes)
    return text
