"""The agent loop, readable top to bottom.

One task = one call to ``run_task``: send the conversation, stream the model's
text, execute any tool calls (mutating ones behind the approval gate), append
the results, and repeat until the model responds with no tool calls or the
user interrupts.

The model backend is abstracted behind the small ``ModelClient`` protocol so
tests drive the loop with scripted responses and no network. The real
implementation, ``AnthropicClient``, lives at the bottom of this file.
"""

from __future__ import annotations

import os
import ssl
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol

import anthropic
import httpx

from .safety import ToolError, create_snapshot
from .sandbox import Sandbox
from .tools import MUTATING_TOOLS, TOOL_DEFINITIONS, build_preview, run_tool

CONTEXT_WINDOW_TOKENS = 1_000_000
COMPACT_THRESHOLD_TOKENS = 750_000  # auto-compact between tasks above this
CONTEXT_WARNING_TOKENS = 800_000  # mid-task warning; compaction runs at the task boundary

DECLINED_MESSAGE = "User declined this action."

COMPACT_PROMPT = """\
The conversation is approaching the context limit. Write a handoff summary
that would let you continue this session in a fresh conversation with no
other memory of it. Include: the user's task(s) and their current status;
what was changed (files, commands run) and why; key discoveries, decisions,
and constraints; approaches that failed and should not be retried; and what
remains to be done, with concrete next steps. Reply with only the summary —
do not use tools."""

COMPACT_HEADER = (
    "[The conversation so far was compacted to stay within the context "
    "window. Handoff summary of the session:]"
)


def compact_conversation(client: ModelClient, messages: list[dict[str, Any]], *, ui: UI) -> bool:
    """Replace the conversation with a model-written handoff summary.

    One extra model request; its usage is reported through the normal stats
    channel so session accounting stays truthful. The summary is not streamed
    to the terminal — it exists for the model, not the user. Returns False
    (leaving ``messages`` untouched) when the model produced no summary text.
    """
    ui.on_turn_start()
    turn = client.create_turn(messages + [{"role": "user", "content": COMPACT_PROMPT}])
    try:
        for _ in turn.stream_text():
            pass
        response = turn.final_message()
    finally:
        turn.close()
    ui.on_turn_stats(_stats(response.usage))
    summary = "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    ).strip()
    if not summary:
        return False
    # A user/assistant pair keeps roles alternating for the next task's
    # user message, whatever the endpoint's strictness about ordering.
    messages[:] = [
        {"role": "user", "content": f"{COMPACT_HEADER}\n\n{summary}"},
        {
            "role": "assistant",
            "content": "Understood — I have the context from the summary and will continue from there.",
        },
    ]
    for message in messages:
        ui.on_message(message)
    return True


def build_system_prompt(root: Path) -> str:
    # Static per session — nothing volatile may be interpolated here, or the
    # prompt-cache prefix breaks on every request.
    return f"""\
You are acode, a coding agent working inside the repository at {root}.

Work on the user's task with the provided tools: read files, edit files, and
run shell commands. All paths are relative to the repository root; files
outside it are inaccessible, and .git is off limits to the file tools — use
git through run_command.

Conventions:
- Read a file before editing it so edit_file's old_text matches exactly.
- Prefer edit_file for targeted changes; write_file for new files or rewrites.
- edit_file, write_file, and run_command each require the user's approval.
  A declined action is not an error — adjust your approach or ask the user.
- Verify your work when practical (run the tests, run the code).
- When the task is done, stop and summarize what you did. If you are blocked
  or need a decision only the user can make, ask instead of guessing."""


# --- Interfaces ------------------------------------------------------------

class ModelTurn(Protocol):
    """One in-flight model response."""

    def stream_text(self) -> Iterator[str]: ...

    def final_message(self) -> Any: ...

    def close(self) -> None: ...


class ModelClient(Protocol):
    """The model backend: fixed configuration in, one turn per request out."""

    def create_turn(self, messages: list[dict[str, Any]]) -> ModelTurn: ...


class UI(Protocol):
    """Where the loop reports what is happening (terminal in production)."""

    def on_turn_start(self) -> None: ...

    def on_text(self, chunk: str) -> None: ...

    def on_tool_call(self, name: str, args: dict[str, Any]) -> None: ...

    def on_tool_result(self, content: str, *, is_error: bool) -> None: ...

    def on_message(self, message: dict[str, Any]) -> None: ...

    def on_turn_stats(self, stats: TurnStats) -> None: ...

    def warn(self, message: str) -> None: ...

    def info(self, message: str) -> None: ...


Approver = Callable[[str, str], bool]  # (tool name, preview) -> approved?


@dataclass(frozen=True)
class TurnStats:
    """Token usage for one model request, straight from ``response.usage``."""

    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int

    @property
    def total_prompt_tokens(self) -> int:
        return self.input_tokens + self.cache_read_tokens + self.cache_creation_tokens


# --- The loop --------------------------------------------------------------

def run_task(
    client: ModelClient,
    messages: list[dict[str, Any]],
    *,
    root: Path,
    ui: UI,
    approve: Approver,
    command_timeout: float = 120.0,
    sandbox: Sandbox | None = None,
) -> None:
    """Run one task until the model stops calling tools.

    ``messages`` is mutated in place so the caller keeps the full conversation
    for follow-up tasks in the same session. On Ctrl+C the partially built
    turn is discarded, leaving the conversation consistent.
    """
    snapshot_pending = True

    def ensure_snapshot() -> None:
        # Best-effort git snapshot of the workspace, once per task, taken
        # just before the first approved mutation so clobbered work is
        # recoverable. A non-git workspace silently has no safety net.
        nonlocal snapshot_pending
        if not snapshot_pending:
            return
        snapshot_pending = False
        commit = create_snapshot(root)
        if commit:
            ui.info(
                f"workspace snapshot {commit[:12]} — "
                f"restore with: git restore --source {commit[:12]} -- ."
            )

    while True:
        checkpoint = len(messages)
        try:
            ui.on_turn_start()
            turn = client.create_turn(messages)
            try:
                for chunk in turn.stream_text():
                    ui.on_text(chunk)
                response = turn.final_message()
            finally:
                turn.close()

            # Append the full content — tool_use and thinking blocks included;
            # dropping them breaks the next request.
            messages.append({"role": "assistant", "content": response.content})
            ui.on_message(messages[-1])

            stats = _stats(response.usage)
            ui.on_turn_stats(stats)
            if stats.total_prompt_tokens >= CONTEXT_WARNING_TOKENS:
                ui.warn(
                    f"Context is at {stats.total_prompt_tokens:,} of "
                    f"~{CONTEXT_WINDOW_TOKENS:,} tokens; the conversation "
                    "will be compacted when this task finishes."
                )

            match response.stop_reason:
                case "tool_use":
                    results = _execute_tool_calls(
                        response.content,
                        root=root,
                        ui=ui,
                        approve=approve,
                        ensure_snapshot=ensure_snapshot,
                        command_timeout=command_timeout,
                        sandbox=sandbox,
                    )
                    messages.append({"role": "user", "content": results})
                    ui.on_message(messages[-1])
                case "pause_turn":
                    continue  # server paused mid-turn; re-send to resume
                case "end_turn" | "stop_sequence":
                    return
                case "max_tokens":
                    ui.warn(
                        "The response was cut off by the output-token limit; "
                        "ask the model to continue, or raise --max-tokens."
                    )
                    return
                case "refusal":
                    ui.warn(_refusal_message(response))
                    return
                case other:
                    ui.warn(f"Model stopped for an unexpected reason: {other!r}.")
                    return
        except KeyboardInterrupt:
            del messages[checkpoint:]
            ui.warn("Interrupted — partial turn discarded.")
            return


def _execute_tool_calls(
    content: list[Any],
    *,
    root: Path,
    ui: UI,
    approve: Approver,
    ensure_snapshot: Callable[[], None],
    command_timeout: float,
    sandbox: Sandbox | None,
) -> list[dict[str, Any]]:
    """Run every tool_use block and return their tool_result blocks.

    All results go back in a single user message — splitting them across
    messages trains the model to stop making parallel calls.
    """
    results: list[dict[str, Any]] = []
    for block in content:
        if getattr(block, "type", None) != "tool_use":
            continue
        args = block.input if isinstance(block.input, dict) else None
        ui.on_tool_call(block.name, args or {})
        if args is None:
            output, is_error = f"tool input must be an object, got {block.input!r}", True
        else:
            output, is_error = _run_one_tool(
                block.name,
                args,
                root=root,
                approve=approve,
                ensure_snapshot=ensure_snapshot,
                command_timeout=command_timeout,
                sandbox=sandbox,
            )
        ui.on_tool_result(output, is_error=is_error)
        result: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": output,
        }
        if is_error:
            result["is_error"] = True
        results.append(result)
    return results


def _run_one_tool(
    name: str,
    args: dict[str, Any],
    *,
    root: Path,
    approve: Approver,
    ensure_snapshot: Callable[[], None],
    command_timeout: float,
    sandbox: Sandbox | None,
) -> tuple[str, bool]:
    """Execute one tool call; mutating tools pass the approval gate first."""
    try:
        if name in MUTATING_TOOLS:
            preview = build_preview(name, args, root)
            if not approve(name, preview):
                return DECLINED_MESSAGE, False
            ensure_snapshot()
        return run_tool(name, args, root, timeout=command_timeout, sandbox=sandbox), False
    except ToolError as exc:
        return str(exc), True


def _stats(usage: Any) -> TurnStats:
    def count(field: str) -> int:
        return getattr(usage, field, 0) or 0

    return TurnStats(
        input_tokens=count("input_tokens"),
        output_tokens=count("output_tokens"),
        cache_read_tokens=count("cache_read_input_tokens"),
        cache_creation_tokens=count("cache_creation_input_tokens"),
    )


def _refusal_message(response: Any) -> str:
    details = getattr(response, "stop_details", None)
    category = getattr(details, "category", None) if details else None
    suffix = f" (category: {category})" if category else ""
    return f"The model declined this request for safety reasons{suffix}."


# --- The real backend ------------------------------------------------------

CA_BUNDLE_ENV_VARS = ("ACODE_CA_BUNDLE", "CHUNKER_CA_BUNDLE")


def _tls_http_client() -> httpx.Client | None:
    """TLS trust for gateway endpoints behind a private CA.

    Order: an explicit CA bundle (ACODE_CA_BUNDLE, falling back to
    CHUNKER_CA_BUNDLE so companion tools on the same endpoint can share one
    setting), then the OS trust store via the optional ``truststore``
    package, else None — meaning the SDK's default HTTP client is used.
    """
    for var in CA_BUNDLE_ENV_VARS:
        bundle = os.environ.get(var)
        if bundle:
            return httpx.Client(verify=bundle)
    try:
        import truststore
    except ImportError:
        return None
    return httpx.Client(verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT))


class AnthropicClient:
    """Streams turns from the Claude endpoint with prompt caching.

    Caching layout: one cache_control breakpoint on the system prompt block
    (tools render before system, so this caches both together) and the
    top-level cache_control parameter, which auto-places a breakpoint on the
    last message block so the growing conversation is reused turn over turn.
    Verify against usage: cache_read_input_tokens should be nonzero from the
    second request of a session onward.
    """

    def __init__(self, *, model: str, max_tokens: int, system: str) -> None:
        # Endpoint routing is the SDK's own: ANTHROPIC_BASE_URL (+
        # ANTHROPIC_AUTH_TOKEN as a Bearer token) redirects to a gateway;
        # otherwise ANTHROPIC_API_KEY hits the default Claude endpoint.
        http_client = _tls_http_client()
        self._client = (
            anthropic.Anthropic()
            if http_client is None
            else anthropic.Anthropic(http_client=http_client)
        )
        # The SDK only validates credentials at request time (with a raw
        # TypeError); check up front so a missing key fails at startup.
        if (
            self._client.api_key is None
            and self._client.auth_token is None
            and getattr(self._client, "credentials", None) is None
        ):
            raise anthropic.AnthropicError(
                "No API credentials found in the environment."
            )
        self._model = model
        self._max_tokens = max_tokens
        self._system = [
            {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
        ]

    def create_turn(self, messages: list[dict[str, Any]]) -> ModelTurn:
        manager = self._client.messages.stream(
            model=self._model,
            max_tokens=self._max_tokens,
            system=self._system,
            tools=TOOL_DEFINITIONS,
            messages=messages,
            # Visibility only: the model thinks (and bills) the same either
            # way, but "summarized" returns a server-written summary of the
            # reasoning in thinking blocks, which --log-full can record.
            thinking={"type": "adaptive", "display": "summarized"},
            cache_control={"type": "ephemeral"},
        )
        return _AnthropicTurn(manager)


class _AnthropicTurn:
    def __init__(self, manager: anthropic.lib.streaming.MessageStreamManager) -> None:
        self._manager = manager
        self._stream = manager.__enter__()
        self._closed = False

    def stream_text(self) -> Iterator[str]:
        return self._stream.text_stream

    def final_message(self) -> Any:
        return self._stream.get_final_message()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._manager.__exit__(None, None, None)
