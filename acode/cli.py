"""Command-line entry point: argument parsing, the REPL, API error handling."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import anthropic

from . import __version__
from .agent import AnthropicClient, TurnStats, build_system_prompt, run_task
from .display import ConsoleUI

DIRECT_DEFAULT_MODEL = "claude-sonnet-5"
GATEWAY_DEFAULT_MODEL = "databricks-claude-sonnet-4-6"
DEFAULT_COMMAND_TIMEOUT = 120.0
DEFAULT_MAX_TOKENS = 32_000


# USD per million tokens, keyed by model-id prefix:
# (input, output, cache write @5min TTL, cache read).
# Rates match intelligent-chunker's table (verified against Anthropic pricing
# 2026-07; gateway-served Claude bills the same per-token rates per our
# tenant's rate card, 2026-08). Version-less prefixes so any served revision
# matches. Unknown models report tokens without a dollar estimate.
PRICING = {
    "claude-haiku-4-5": (1.00, 5.00, 1.25, 0.10),
    "claude-sonnet-4": (3.00, 15.00, 3.75, 0.30),
    "claude-sonnet-5": (3.00, 15.00, 3.75, 0.30),
    "claude-opus-4": (5.00, 25.00, 6.25, 0.50),
    "claude-opus-5": (5.00, 25.00, 6.25, 0.50),
    "databricks-claude-haiku": (1.00, 5.00, 1.25, 0.10),
    "databricks-claude-sonnet": (3.00, 15.00, 3.75, 0.30),
    "databricks-claude-opus": (5.00, 25.00, 6.25, 0.50),
}


@dataclass
class SessionUsage:
    """Token totals accumulated across every model request in a session."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0

    def add(self, stats: TurnStats) -> None:
        self.input_tokens += stats.input_tokens
        self.output_tokens += stats.output_tokens
        self.cache_read_tokens += stats.cache_read_tokens
        self.cache_creation_tokens += stats.cache_creation_tokens

    def estimated_cost_usd(self, model: str) -> float | None:
        """Dollar estimate, or None when the model's pricing is unknown."""
        for prefix, (p_in, p_out, p_write, p_read) in PRICING.items():
            if model.startswith(prefix):
                return (
                    self.input_tokens * p_in
                    + self.output_tokens * p_out
                    + self.cache_creation_tokens * p_write
                    + self.cache_read_tokens * p_read
                ) / 1_000_000
        return None

    def summary(self, model: str) -> str:
        cost = self.estimated_cost_usd(model)
        tail = (
            f"est. ${cost:.4f}"
            if cost is not None
            else f"no pricing known for {model!r}"
        )
        return (
            f"session usage: in {self.input_tokens:,} | out {self.output_tokens:,} | "
            f"cache read {self.cache_read_tokens:,} | "
            f"cache write {self.cache_creation_tokens:,} — {tail}"
        )


class TrackingUI(ConsoleUI):
    """ConsoleUI that also accumulates per-request usage into a SessionUsage."""

    def __init__(self, usage: SessionUsage) -> None:
        super().__init__()
        self._usage = usage

    def on_turn_stats(self, stats: TurnStats) -> None:
        self._usage.add(stats)
        super().on_turn_stats(stats)


def resolve_model(flag: str | None) -> str:
    """An explicit --model wins; otherwise the default is route-aware.

    Gateway endpoints (ANTHROPIC_BASE_URL set) serve models under their own
    names, so the direct-endpoint default would not resolve there.
    """
    if flag:
        return flag
    if os.environ.get("ANTHROPIC_BASE_URL"):
        return GATEWAY_DEFAULT_MODEL
    return DIRECT_DEFAULT_MODEL


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    root = Path.cwd().resolve()
    model = resolve_model(args.model)
    usage = SessionUsage()
    ui = TrackingUI(usage)

    try:
        client = AnthropicClient(
            model=model,
            max_tokens=args.max_tokens,
            system=build_system_prompt(root),
        )
    except anthropic.AnthropicError as exc:
        ui.error(str(exc))
        ui.info("Set the ANTHROPIC_API_KEY environment variable and try again.")
        return 1

    ui.info(f"acode {__version__} — {model} — workspace {root}")
    if base_url := os.environ.get("ANTHROPIC_BASE_URL"):
        ui.info(f"endpoint: {base_url}")
    ui.info("Ctrl+C interrupts a turn; Ctrl+D or /quit exits; /new clears; /cost shows usage.")

    messages: list[dict[str, object]] = []
    task: str | None = args.task
    while True:
        if task is None:
            line = _read_line()
            if line is None or line in {"/quit", "/exit"}:
                break
            if _handle_command(line, messages=messages, usage=usage, model=model, ui=ui):
                continue
            task = line
        messages.append({"role": "user", "content": task})
        task = None
        _run_one_task(client, messages, root=root, ui=ui, model=model, timeout=args.timeout)

    ui.info(usage.summary(model))
    return 0


def _handle_command(
    line: str,
    *,
    messages: list[dict[str, object]],
    usage: SessionUsage,
    model: str,
    ui: ConsoleUI,
) -> bool:
    """Handle a /command; returns True if the line was consumed."""
    if line == "/new":
        messages.clear()
        ui.info("Conversation cleared — next task starts fresh.")
        return True
    if line == "/cost":
        ui.info(usage.summary(model))
        return True
    if line.startswith("/"):
        ui.warn(f"Unknown command {line!r}. Commands: /new, /cost, /quit")
        return True
    return False


def _run_one_task(
    client: AnthropicClient,
    messages: list[dict[str, object]],
    *,
    root: Path,
    ui: ConsoleUI,
    model: str,
    timeout: float,
) -> None:
    """Run one task; API failures are reported and the REPL survives them."""
    try:
        run_task(
            client,
            messages,
            root=root,
            ui=ui,
            approve=ui.ask_approval,
            command_timeout=timeout,
        )
    except anthropic.AuthenticationError:
        ui.error("Authentication failed — check ANTHROPIC_API_KEY.")
    except anthropic.PermissionDeniedError as exc:
        ui.error(f"The API key lacks permission for this request: {exc.message}")
    except anthropic.NotFoundError:
        ui.error(f"Model {model!r} was not found — check --model.")
    except anthropic.RateLimitError as exc:
        retry_after = exc.response.headers.get("retry-after")
        hint = f"retry after {retry_after}s" if retry_after else "try again shortly"
        ui.error(f"Rate limited — {hint}.")
    except anthropic.BadRequestError as exc:
        ui.error(f"The API rejected the request: {exc.message}")
    except anthropic.APIStatusError as exc:
        ui.error(f"API error {exc.status_code}: {exc.message}")
    except anthropic.APIConnectionError:
        ui.error("Could not reach the model endpoint — check your network connection.")


def _read_line() -> str | None:
    """Read the next non-empty input line; None means exit."""
    while True:
        try:
            line = input("\n› ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if line:
            return line


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="acode",
        description=(
            "A minimal agentic coding harness. Run it inside the repository "
            "you want to work on; the current directory becomes the workspace "
            "boundary."
        ),
    )
    parser.add_argument(
        "task",
        nargs="?",
        default=None,
        help="initial task in plain English; if omitted, you are prompted",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            f"Claude model ID (default: {DIRECT_DEFAULT_MODEL}, or "
            f"{GATEWAY_DEFAULT_MODEL} when ANTHROPIC_BASE_URL is set)"
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_COMMAND_TIMEOUT,
        help=f"per-command timeout for run_command, in seconds (default: {DEFAULT_COMMAND_TIMEOUT:g})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        dest="max_tokens",
        help=f"maximum output tokens per model response (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument("--version", action="version", version=f"acode {__version__}")
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
