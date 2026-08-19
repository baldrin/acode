"""Command-line entry point: argument parsing, the REPL, API error handling."""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import anthropic

from . import __version__
from .agent import (
    COMPACT_THRESHOLD_TOKENS,
    CONTEXT_WINDOW_TOKENS,
    AnthropicClient,
    Approver,
    ModelClient,
    TurnStats,
    build_system_prompt,
    compact_conversation,
    load_project_instructions,
    run_task,
    summarize_conversation,
)
from .display import ConsoleUI
from .handoff import HANDOFF_DIR, handoff_path, load_handoff, save_handoff
from .resume import ResumedSession, ResumeError, resolve_resume, resume_note_messages
from .sandbox import Sandbox, SandboxUnavailable, detect_sandbox
from .tools import TOOL_DEFINITIONS
from .transcript import Transcript, serialize_content

LOG_DIR = Path.home() / ".acode" / "logs"

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
    # Size of the conversation after the most recent response — the last
    # request's full prompt plus what the model added to it. Reset to 0 when
    # the conversation is cleared or compacted (unknown until the next request).
    context_tokens: int = 0

    def add(self, stats: TurnStats) -> None:
        self.input_tokens += stats.input_tokens
        self.output_tokens += stats.output_tokens
        self.cache_read_tokens += stats.cache_read_tokens
        self.cache_creation_tokens += stats.cache_creation_tokens
        self.context_tokens = stats.total_prompt_tokens + stats.output_tokens

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
    """ConsoleUI that also accumulates usage and writes the transcript."""

    def __init__(self, usage: SessionUsage, transcript: Transcript | None = None) -> None:
        super().__init__()
        self._usage = usage
        self._transcript = transcript

    def on_text(self, chunk: str) -> None:
        if self._transcript:
            self._transcript.text(chunk)
        super().on_text(chunk)

    def on_tool_call(self, name: str, args: dict[str, object]) -> None:
        if self._transcript:
            self._transcript.log(
                "tool_call", tool=name, args=self._transcript.clipped_args(dict(args))
            )
        super().on_tool_call(name, args)

    def on_tool_result(self, content: str, *, is_error: bool) -> None:
        if self._transcript:
            self._transcript.log(
                "tool_result", is_error=is_error, content=self._transcript.clipped(content)
            )
        super().on_tool_result(content, is_error=is_error)

    def on_message(self, message: dict[str, object]) -> None:
        # Complete messages (thinking blocks, tool_use ids, full results) are
        # only recorded in --log-full mode; the clipped default already
        # captures the human-readable view via the other events.
        if self._transcript and self._transcript.full:
            self._transcript.log(
                "message",
                role=message["role"],
                content=serialize_content(message["content"]),
            )
        super().on_message(message)

    def on_turn_stats(self, stats: TurnStats) -> None:
        self._usage.add(stats)
        if self._transcript:
            self._transcript.log(
                "usage",
                input=stats.input_tokens,
                output=stats.output_tokens,
                cache_read=stats.cache_read_tokens,
                cache_write=stats.cache_creation_tokens,
            )
        super().on_turn_stats(stats)

    def warn(self, message: str) -> None:
        if self._transcript:
            self._transcript.log("warning", message=message)
        super().warn(message)

    def info(self, message: str) -> None:
        if self._transcript:
            self._transcript.log("info", message=message)
        super().info(message)

    def error(self, message: str) -> None:
        if self._transcript:
            self._transcript.log("error", message=message)
        super().error(message)


TRUSTED_EDIT_TOOLS = frozenset({"edit_file", "write_file"})


def make_approver(
    ui: ConsoleUI, *, trust_edits: bool, transcript: Transcript | None = None
) -> Approver:
    """The approval policy for this session.

    Default: every mutating tool prompts. With --trust-edits, file edits run
    without prompting (their previews are still shown, and the pre-task git
    snapshot remains the safety net); shell commands always stay gated.
    Every outcome — and what was shown to get it — goes to the transcript.
    """

    def approve(name: str, preview: str) -> bool:
        if trust_edits and name in TRUSTED_EDIT_TOOLS:
            ui.on_auto_approved(name, preview)
            approved, outcome = True, "auto_approved"
        else:
            approved = ui.ask_approval(name, preview)
            outcome = "approved" if approved else "denied"
        if transcript:
            transcript.log(
                "approval", tool=name, outcome=outcome, preview=transcript.clipped(preview)
            )
        return approved

    return approve


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
    instructions = load_project_instructions(root)
    # A resumed conversation already carries its own past; loading a handoff
    # note on top would inject a second, competing "previous session".
    handoff = None if args.resume else load_handoff(root)
    system = build_system_prompt(root, instructions=instructions, handoff=handoff)
    usage = SessionUsage()
    transcript = Transcript.open(root, LOG_DIR, full=args.log_full)
    ui = TrackingUI(usage, transcript)

    try:
        try:
            client = AnthropicClient(
                model=model,
                max_tokens=args.max_tokens,
                system=system,
            )
        except anthropic.AnthropicError as exc:
            ui.error(str(exc))
            ui.info("Set the ANTHROPIC_API_KEY environment variable and try again.")
            return 1

        resumed: ResumedSession | None = None
        if args.resume:
            try:
                resumed = resolve_resume(
                    args.resume,
                    root,
                    LOG_DIR,
                    exclude=transcript.path if transcript else None,
                )
            except ResumeError as exc:
                ui.error(f"cannot resume: {exc}")
                return 1

        sandbox: Sandbox | None = None
        if args.no_sandbox:
            sandbox_note = "sandbox disabled by --no-sandbox — command writes are unconfined."
        else:
            try:
                sandbox = detect_sandbox(root)
                sandbox_note = f"sandbox: {sandbox.describe()}"
            except SandboxUnavailable as exc:
                sandbox_note = f"sandbox unavailable: {exc} — command writes are unconfined."

        if transcript:
            transcript.log(
                "session_start",
                version=__version__,
                model=model,
                workspace=str(root),
                trust_edits=args.trust_edits,
                log_full=args.log_full,
                sandbox=sandbox.kind if sandbox else None,
                instructions=instructions[0] if instructions else None,
                handoff=handoff[0] if handoff else None,
                endpoint=os.environ.get("ANTHROPIC_BASE_URL"),
            )
            if transcript.full:
                transcript.log("system_prompt", text=system)
                transcript.log("tool_definitions", tools=TOOL_DEFINITIONS)
        ui.info(f"acode {__version__} — {model} — workspace {root}")
        if base_url := os.environ.get("ANTHROPIC_BASE_URL"):
            ui.info(f"endpoint: {base_url}")
        if sandbox:
            ui.info(sandbox_note)
        else:
            ui.warn(sandbox_note)
        if instructions:
            ui.info(f"project instructions: {instructions[0]}")
        if handoff:
            ui.info(f"handoff note from {handoff[0]} loaded — delete to forget: {handoff_path(root)}")
        if transcript:
            ui.info(f"transcript: {transcript.path}")
        else:
            ui.warn(f"transcript logging unavailable (cannot write to {LOG_DIR})")
        if resumed:
            note = f"resumed {len(resumed.messages)} messages from {resumed.source.name}"
            if resumed.dropped:
                note += f" ({resumed.dropped} incomplete trailing messages dropped)"
            ui.info(note)
            if resumed.model and resumed.model != model:
                ui.warn(
                    f"that transcript was recorded with model {resumed.model!r}; "
                    f"resuming with {model!r} may be rejected by the API — "
                    f"consider --model {resumed.model}"
                )
        ui.info("Ctrl+C interrupts a turn; Ctrl+D or /quit exits; /help lists commands.")
        if args.trust_edits:
            if (root / ".git").exists():
                ui.warn(
                    "--trust-edits: file edits run without approval (git snapshots "
                    "remain the safety net); commands still require approval."
                )
            else:
                ui.warn(
                    "--trust-edits in a non-git workspace: edits run without "
                    "approval AND without snapshots — clobbered files are gone."
                )
        approve = make_approver(ui, trust_edits=args.trust_edits, transcript=transcript)

        messages: list[dict[str, object]] = []
        if resumed:
            messages.extend(resumed.messages)
            messages.extend(resume_note_messages())
            if transcript:
                transcript.log(
                    "resumed_from",
                    path=str(resumed.source),
                    messages=len(resumed.messages),
                    dropped=resumed.dropped,
                )
                if transcript.full:
                    # Replay the restored conversation into this session's own
                    # log so a resumed session is itself resumable.
                    for message in messages:
                        transcript.log(
                            "message",
                            role=message["role"],
                            content=serialize_content(message["content"]),
                        )
        task: str | None = args.task
        while True:
            if task is None:
                line = _read_line()
                if line is None or line in {"/quit", "/exit"}:
                    break
                if _handle_command(
                    line,
                    messages=messages,
                    usage=usage,
                    model=model,
                    ui=ui,
                    transcript=transcript,
                    client=client,
                ):
                    continue
                task = line
            if transcript:
                transcript.log("task", text=task)
            messages.append({"role": "user", "content": task})
            task = None
            _run_one_task(
                client,
                messages,
                root=root,
                ui=ui,
                approve=approve,
                model=model,
                timeout=args.timeout,
                sandbox=sandbox,
            )
            if usage.context_tokens >= COMPACT_THRESHOLD_TOKENS:
                ui.warn(
                    f"Context reached {usage.context_tokens:,} of "
                    f"~{CONTEXT_WINDOW_TOKENS:,} tokens — compacting automatically."
                )
                _compact(
                    client, messages, ui=ui, usage=usage, model=model, transcript=transcript
                )

        _offer_handoff(client, messages, root=root, ui=ui, model=model, transcript=transcript)
        ui.info(usage.summary(model))
        return 0
    finally:
        if transcript:
            transcript.log(
                "session_end",
                input=usage.input_tokens,
                output=usage.output_tokens,
                cache_read=usage.cache_read_tokens,
                cache_write=usage.cache_creation_tokens,
                estimated_cost_usd=usage.estimated_cost_usd(model),
            )
            transcript.close()


COMMANDS = (
    ("/new", "clear the conversation and start fresh"),
    ("/compact", "summarize the conversation into a handoff note and continue from it"),
    ("/cost", "show session token totals, estimated cost, and context size"),
    ("/help", "show this list"),
    ("/quit", "exit (also /exit or Ctrl+D)"),
)


def _handle_command(
    line: str,
    *,
    messages: list[dict[str, object]],
    usage: SessionUsage,
    model: str,
    ui: ConsoleUI,
    transcript: Transcript | None = None,
    client: ModelClient | None = None,
) -> bool:
    """Handle a /command; returns True if the line was consumed."""
    if transcript and line.startswith("/"):
        transcript.log("command", command=line)
    if line == "/new":
        messages.clear()
        usage.context_tokens = 0
        ui.info("Conversation cleared — next task starts fresh.")
        return True
    if line == "/compact":
        if client is None:
            ui.warn("No model client available to compact with.")
        else:
            _compact(client, messages, ui=ui, usage=usage, model=model, transcript=transcript)
        return True
    if line == "/cost":
        ui.info(usage.summary(model))
        ui.info(
            f"context: ~{usage.context_tokens:,} of ~{CONTEXT_WINDOW_TOKENS:,} tokens "
            f"(auto-compacts at {COMPACT_THRESHOLD_TOKENS:,})"
        )
        return True
    if line == "/help":
        width = max(len(name) for name, _ in COMMANDS)
        for name, description in COMMANDS:
            ui.info(f"{name:<{width}}  {description}")
        return True
    if line.startswith("/"):
        ui.warn(f"Unknown command {line!r} — /help lists the commands.")
        return True
    return False


def _offer_handoff(
    client: ModelClient,
    messages: list[dict[str, object]],
    *,
    root: Path,
    ui: ConsoleUI,
    model: str,
    transcript: Transcript | None,
) -> None:
    """On exit with a conversation, offer to save a handoff note.

    Ctrl+D exits leave no stdin to answer with, so the question falls
    through to "no" — /quit is the path that gets asked.
    """
    if not messages:
        return
    if not ui.confirm("Save a handoff note for the next session?"):
        return
    summary: str | None = None
    try:
        with _api_errors(ui, model):
            summary = summarize_conversation(client, messages, ui=ui)
    except KeyboardInterrupt:
        ui.warn("Handoff skipped.")
        return
    if not summary:
        ui.warn("No handoff note was produced — nothing saved.")
        return
    path = save_handoff(root, summary)
    if path is None:
        ui.warn(f"Could not write the handoff note under {HANDOFF_DIR} — nothing saved.")
        return
    if transcript:
        transcript.log("handoff_saved", path=str(path), note=transcript.clipped(summary))
    ui.info(f"Handoff note saved — the next session here starts with it: {path}")


def _run_one_task(
    client: AnthropicClient,
    messages: list[dict[str, object]],
    *,
    root: Path,
    ui: ConsoleUI,
    approve: Approver,
    model: str,
    timeout: float,
    sandbox: Sandbox | None = None,
) -> None:
    """Run one task; API failures are reported and the REPL survives them."""
    with _api_errors(ui, model):
        run_task(
            client,
            messages,
            root=root,
            ui=ui,
            approve=approve,
            command_timeout=timeout,
            sandbox=sandbox,
        )


@contextlib.contextmanager
def _api_errors(ui: ConsoleUI, model: str):
    """Report API failures readably instead of crashing the REPL."""
    try:
        yield
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


def _compact(
    client: ModelClient,
    messages: list[dict[str, object]],
    *,
    ui: ConsoleUI,
    usage: SessionUsage,
    model: str,
    transcript: Transcript | None,
) -> None:
    """Compact the conversation, reporting the outcome; failures change nothing."""
    if not messages:
        ui.info("Nothing to compact — the conversation is empty.")
        return
    before = usage.context_tokens
    ui.info(f"compacting conversation (~{before:,} tokens)…")
    compacted: bool | None = None  # None: the request itself failed (already reported)
    try:
        with _api_errors(ui, model):
            compacted = compact_conversation(client, messages, ui=ui)
    except KeyboardInterrupt:
        ui.warn("Compaction interrupted — conversation unchanged.")
        return
    if compacted:
        usage.context_tokens = 0
        summary = messages[0]["content"]
        if transcript:
            transcript.log(
                "compact", tokens_before=before, summary=transcript.clipped(str(summary))
            )
        ui.info("Conversation compacted — the session continues from the handoff summary.")
    elif compacted is False:
        ui.warn("Compaction produced no summary — conversation unchanged.")
    else:
        ui.warn("Conversation unchanged.")


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
        "--log-full",
        action="store_true",
        dest="log_full",
        help=(
            "log everything to the transcript, unclipped: the system prompt, "
            "tool definitions, complete messages to and from the model "
            "(thinking summaries included), and full tool results. The "
            "default transcript clips large payloads. Note: a full log "
            "contains everything the model read, including file contents."
        ),
    )
    parser.add_argument(
        "--resume",
        default=None,
        metavar="LOG|last",
        help=(
            "resume a previous session from its --log-full transcript: a "
            "path under ~/.acode/logs, or 'last' for this workspace's most "
            "recent transcript. The conversation is restored exactly (an "
            "incomplete final exchange is trimmed); only full transcripts "
            "are resumable."
        ),
    )
    parser.add_argument(
        "--trust-edits",
        action="store_true",
        dest="trust_edits",
        help=(
            "run file edits (edit_file/write_file) without asking; their "
            "previews are still shown and the pre-task git snapshot remains "
            "the safety net. Shell commands always require approval."
        ),
    )
    parser.add_argument(
        "--no-sandbox",
        action="store_true",
        dest="no_sandbox",
        help=(
            "run shell commands without the OS sandbox. By default (when "
            "sandbox-exec on macOS or bubblewrap on Linux is available and "
            "passes a startup probe) commands may only write inside the "
            "workspace, temp, and cache directories."
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
