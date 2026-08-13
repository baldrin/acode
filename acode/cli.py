"""Command-line entry point: argument parsing, the REPL, API error handling."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import anthropic

from . import __version__
from .agent import AnthropicClient, build_system_prompt, run_task
from .display import ConsoleUI

DIRECT_DEFAULT_MODEL = "claude-sonnet-5"
GATEWAY_DEFAULT_MODEL = "databricks-claude-sonnet-4-6"
DEFAULT_COMMAND_TIMEOUT = 120.0
DEFAULT_MAX_TOKENS = 32_000


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
    ui = ConsoleUI()

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
    ui.info("Ctrl+C interrupts a turn; Ctrl+D or /quit exits.")

    messages: list[dict[str, object]] = []
    task: str | None = args.task
    while True:
        if task is None:
            task = _read_task(ui)
            if task is None:
                return 0
        messages.append({"role": "user", "content": task})
        task = None
        _run_one_task(client, messages, root=root, ui=ui, model=model, timeout=args.timeout)


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


def _read_task(ui: ConsoleUI) -> str | None:
    """Prompt for the next task; None means exit."""
    while True:
        try:
            line = input("\n› ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if not line:
            continue
        if line in {"/quit", "/exit"}:
            return None
        if line.startswith("/"):
            ui.warn(f"Unknown command {line!r}. Commands: /quit")
            continue
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
