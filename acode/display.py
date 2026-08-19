"""Terminal presentation: streamed text, tool calls, approvals, warnings.

Plain ANSI escapes only — no terminal libraries. ``ConsoleUI`` implements the
``agent.UI`` protocol, plus ``ask_approval`` (passed to the loop as its
approver) and ``error`` (used by the CLI).
"""

from __future__ import annotations

import sys
from typing import Any

from .agent import TurnStats

RESET = "\x1b[0m"
BOLD = "\x1b[1m"
DIM = "\x1b[2m"
RED = "\x1b[31m"
GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
CYAN = "\x1b[36m"

CLEAR_LINE = "\r\x1b[2K"

RESULT_PREVIEW_LINES = 6
ARG_PREVIEW_CHARS = 80


class ConsoleUI:
    def __init__(self, *, color: bool | None = None) -> None:
        self._color = sys.stdout.isatty() if color is None else color
        self._midline = False  # streamed text is sitting on an unfinished line
        self._thinking = False  # the "thinking…" indicator is on screen

    # -- agent.UI ----------------------------------------------------------

    def on_turn_start(self) -> None:
        if self._color:
            print(self._paint("thinking…", DIM), end="", flush=True)
            self._thinking = True

    def on_text(self, chunk: str) -> None:
        self._clear_indicator()
        print(chunk, end="", flush=True)
        if chunk:
            self._midline = not chunk.endswith("\n")

    def on_tool_call(self, name: str, args: dict[str, Any]) -> None:
        self._fresh_line()
        print(self._paint(f"● {name}({_format_args(args)})", CYAN))

    def on_tool_result(self, content: str, *, is_error: bool) -> None:
        self._fresh_line()
        lines = content.splitlines() or [""]
        shown = lines[:RESULT_PREVIEW_LINES]
        if len(lines) > RESULT_PREVIEW_LINES:
            shown.append(f"… ({len(lines) - RESULT_PREVIEW_LINES} more lines)")
        style = RED if is_error else DIM
        for line in shown:
            print(self._paint(f"  {line}", style))

    def on_message(self, message: dict[str, Any]) -> None:
        """No-op: messages are already rendered via text/tool events. Exists
        so transcript-writing subclasses can capture complete messages."""

    def on_turn_stats(self, stats: TurnStats) -> None:
        self._fresh_line()
        print(
            self._paint(
                f"[tokens: in {stats.input_tokens:,} | out {stats.output_tokens:,} | "
                f"cache read {stats.cache_read_tokens:,} | "
                f"cache write {stats.cache_creation_tokens:,}]",
                DIM,
            )
        )

    def warn(self, message: str) -> None:
        self._fresh_line()
        print(self._paint(f"! {message}", YELLOW))

    # -- CLI extras --------------------------------------------------------

    def error(self, message: str) -> None:
        self._fresh_line()
        print(self._paint(f"✗ {message}", BOLD, RED))

    def info(self, message: str) -> None:
        self._fresh_line()
        print(self._paint(message, DIM))

    def confirm(self, prompt: str) -> bool:
        """A one-line y/N question outside the approval gate (default: no)."""
        self._fresh_line()
        try:
            reply = input(self._paint(f"{prompt} [y/N] ", BOLD))
        except (EOFError, KeyboardInterrupt):
            print()
            return False
        return reply.strip().lower() in {"y", "yes"}

    def on_auto_approved(self, name: str, preview: str) -> None:
        """--trust-edits: show what is happening without stopping for a yes."""
        self._fresh_line()
        print(self._paint(f"⚡ {name} (auto-approved)", CYAN))
        print(self._colorize_preview(preview))

    def ask_approval(self, name: str, preview: str) -> bool:
        """The gate: show what is about to happen, wait for an explicit yes."""
        self._fresh_line()
        print()
        print(self._paint(f"⏸ approval required — {name}", BOLD, YELLOW))
        print(self._colorize_preview(preview))
        try:
            reply = input(self._paint("  proceed? [y/N] ", BOLD))
        except EOFError:
            print()
            return False
        return reply.strip().lower() in {"y", "yes"}

    # -- internals ---------------------------------------------------------

    def _paint(self, text: str, *codes: str) -> str:
        if not self._color:
            return text
        return "".join(codes) + text + RESET

    def _clear_indicator(self) -> None:
        if self._thinking:
            print(CLEAR_LINE, end="", flush=True)
            self._thinking = False

    def _fresh_line(self) -> None:
        """Make sure the next print starts at column zero on its own line."""
        self._clear_indicator()
        if self._midline:
            print()
            self._midline = False

    def _colorize_preview(self, preview: str) -> str:
        if not self._color:
            return preview
        colored = []
        for line in preview.splitlines():
            if line.startswith("+") and not line.startswith("+++"):
                colored.append(self._paint(line, GREEN))
            elif line.startswith("-") and not line.startswith("---"):
                colored.append(self._paint(line, RED))
            elif line.startswith("@@"):
                colored.append(self._paint(line, CYAN))
            else:
                colored.append(line)
        return "\n".join(colored)


def _format_args(args: dict[str, Any]) -> str:
    parts = []
    for key, value in args.items():
        text = repr(value)
        if len(text) > ARG_PREVIEW_CHARS:
            text = text[: ARG_PREVIEW_CHARS - 1] + "…"
        parts.append(f"{key}={text}")
    return ", ".join(parts)
