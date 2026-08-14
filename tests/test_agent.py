"""Agent-loop tests driven by a fake model client — no network anywhere."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import pytest

from acode.agent import DECLINED_MESSAGE, TurnStats, run_task
from acode.safety import SNAPSHOT_REF
from tests.conftest import MAIN_PY, git


# --- Fakes -----------------------------------------------------------------

@dataclass
class FakeUsage:
    input_tokens: int = 100
    output_tokens: int = 10
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass
class TextBlock:
    text: str
    type: str = "text"


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: Any
    type: str = "tool_use"


@dataclass
class StopDetails:
    category: str | None = None


@dataclass
class FakeResponse:
    content: list[Any]
    stop_reason: str
    usage: FakeUsage = field(default_factory=FakeUsage)
    stop_details: StopDetails | None = None


class FakeTurn:
    def __init__(self, response: FakeResponse) -> None:
        self._response = response
        self.closed = False

    def stream_text(self) -> Iterator[str]:
        for block in self._response.content:
            if block.type == "text":
                yield block.text

    def final_message(self) -> FakeResponse:
        return self._response

    def close(self) -> None:
        self.closed = True


class InterruptingTurn(FakeTurn):
    """Simulates Ctrl+C arriving while the response is streaming."""

    def stream_text(self) -> Iterator[str]:
        yield "partial "
        raise KeyboardInterrupt


class FakeModelClient:
    def __init__(self, turns: list[FakeTurn]) -> None:
        self._turns = iter(turns)
        self.requests: list[list[dict[str, Any]]] = []
        self.created_turns: list[FakeTurn] = []

    def create_turn(self, messages: list[dict[str, Any]]) -> FakeTurn:
        self.requests.append([dict(m) for m in messages])
        turn = next(self._turns)
        self.created_turns.append(turn)
        return turn


class RecordingUI:
    def __init__(self) -> None:
        self.text: list[str] = []
        self.tool_calls: list[tuple[str, dict[str, Any]]] = []
        self.tool_results: list[tuple[str, bool]] = []
        self.stats: list[TurnStats] = []
        self.warnings: list[str] = []
        self.infos: list[str] = []
        self.turn_starts = 0

    def on_turn_start(self) -> None:
        self.turn_starts += 1

    def on_text(self, chunk: str) -> None:
        self.text.append(chunk)

    def on_tool_call(self, name: str, args: dict[str, Any]) -> None:
        self.tool_calls.append((name, args))

    def on_tool_result(self, content: str, *, is_error: bool) -> None:
        self.tool_results.append((content, is_error))

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def info(self, message: str) -> None:
        self.infos.append(message)

    def on_turn_stats(self, stats: TurnStats) -> None:
        self.stats.append(stats)


# --- Helpers ---------------------------------------------------------------

def text_turn(text: str = "Done.", *, stop_reason: str = "end_turn", **response_kwargs: Any) -> FakeTurn:
    return FakeTurn(FakeResponse([TextBlock(text)], stop_reason, **response_kwargs))


def tool_turn(*blocks: ToolUseBlock) -> FakeTurn:
    return FakeTurn(FakeResponse(list(blocks), "tool_use"))


def run(
    client: FakeModelClient,
    root: Path,
    *,
    approve: Any = None,
    messages: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], RecordingUI]:
    ui = RecordingUI()
    if messages is None:
        messages = [{"role": "user", "content": "do the task"}]
    run_task(
        client,
        messages,
        root=root,
        ui=ui,
        approve=approve if approve is not None else (lambda name, preview: True),
        command_timeout=5.0,
    )
    return messages, ui


def tool_results_of(message: dict[str, Any]) -> list[dict[str, Any]]:
    assert message["role"] == "user"
    return message["content"]


# --- Tests -----------------------------------------------------------------

class TestBasicLoop:
    def test_text_only_response_ends_loop(self, workspace: Path) -> None:
        client = FakeModelClient([text_turn("All done.")])
        messages, ui = run(client, workspace)
        assert len(client.requests) == 1
        assert ui.text == ["All done."]
        assert [m["role"] for m in messages] == ["user", "assistant"]

    def test_tool_call_roundtrip(self, workspace: Path) -> None:
        client = FakeModelClient(
            [
                tool_turn(ToolUseBlock("t1", "read_file", {"path": "src/main.py"})),
                text_turn("The file prints hello."),
            ]
        )
        messages, ui = run(client, workspace)
        assert len(client.requests) == 2
        assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]
        [result] = tool_results_of(messages[2])
        assert result["type"] == "tool_result"
        assert result["tool_use_id"] == "t1"
        assert 'print("hello")' in result["content"]
        assert "is_error" not in result
        assert ui.tool_calls == [("read_file", {"path": "src/main.py"})]

    def test_parallel_tool_calls_return_in_one_message(self, workspace: Path) -> None:
        client = FakeModelClient(
            [
                tool_turn(
                    ToolUseBlock("t1", "read_file", {"path": "README.md"}),
                    ToolUseBlock("t2", "list_files", {"path": "."}),
                ),
                text_turn(),
            ]
        )
        messages, _ = run(client, workspace)
        results = tool_results_of(messages[2])
        assert [r["tool_use_id"] for r in results] == ["t1", "t2"]

    def test_every_turn_is_closed(self, workspace: Path) -> None:
        client = FakeModelClient(
            [tool_turn(ToolUseBlock("t1", "list_files", {"path": "."})), text_turn()]
        )
        run(client, workspace)
        assert all(turn.closed for turn in client.created_turns)

    def test_pause_turn_resumes_automatically(self, workspace: Path) -> None:
        client = FakeModelClient(
            [text_turn("working…", stop_reason="pause_turn"), text_turn("done")]
        )
        messages, _ = run(client, workspace)
        assert len(client.requests) == 2
        assert [m["role"] for m in messages] == ["user", "assistant", "assistant"]


class TestApprovalGate:
    def test_mutating_tool_asks_for_approval(self, workspace: Path) -> None:
        seen: list[tuple[str, str]] = []

        def approve(name: str, preview: str) -> bool:
            seen.append((name, preview))
            return True

        client = FakeModelClient(
            [
                tool_turn(ToolUseBlock("t1", "write_file", {"path": "new.txt", "content": "hi"})),
                text_turn(),
            ]
        )
        run(client, workspace, approve=approve)
        assert len(seen) == 1
        assert seen[0][0] == "write_file"
        assert "hi" in seen[0][1]
        assert (workspace / "new.txt").read_text() == "hi"

    def test_denied_tool_is_not_executed(self, workspace: Path) -> None:
        client = FakeModelClient(
            [
                tool_turn(ToolUseBlock("t1", "write_file", {"path": "new.txt", "content": "hi"})),
                text_turn(),
            ]
        )
        messages, _ = run(client, workspace, approve=lambda name, preview: False)
        assert not (workspace / "new.txt").exists()
        [result] = tool_results_of(messages[2])
        assert result["content"] == DECLINED_MESSAGE
        assert "is_error" not in result  # a denial is guidance, not a failure

    def test_read_only_tool_skips_approval(self, workspace: Path) -> None:
        def approve(name: str, preview: str) -> bool:
            raise AssertionError("read-only tools must not prompt for approval")

        client = FakeModelClient(
            [tool_turn(ToolUseBlock("t1", "read_file", {"path": "README.md"})), text_turn()]
        )
        run(client, workspace, approve=approve)

    def test_denylisted_command_fails_without_prompting(self, workspace: Path) -> None:
        def approve(name: str, preview: str) -> bool:
            raise AssertionError("denylisted commands must fail before the prompt")

        client = FakeModelClient(
            [tool_turn(ToolUseBlock("t1", "run_command", {"command": "sudo ls"})), text_turn()]
        )
        messages, _ = run(client, workspace, approve=approve)
        [result] = tool_results_of(messages[2])
        assert result["is_error"] is True
        assert "sudo" in result["content"]


class TestErrorPaths:
    def test_tool_error_becomes_is_error_result(self, workspace: Path) -> None:
        client = FakeModelClient(
            [tool_turn(ToolUseBlock("t1", "read_file", {"path": "missing.txt"})), text_turn()]
        )
        messages, ui = run(client, workspace)
        [result] = tool_results_of(messages[2])
        assert result["is_error"] is True
        assert "not a file" in result["content"]
        assert ui.tool_results == [(result["content"], True)]

    def test_non_object_tool_input_is_an_error(self, workspace: Path) -> None:
        client = FakeModelClient(
            [tool_turn(ToolUseBlock("t1", "read_file", "not-a-dict")), text_turn()]
        )
        messages, _ = run(client, workspace)
        [result] = tool_results_of(messages[2])
        assert result["is_error"] is True
        assert "must be an object" in result["content"]

    def test_unknown_tool_is_an_error(self, workspace: Path) -> None:
        client = FakeModelClient(
            [tool_turn(ToolUseBlock("t1", "teleport", {})), text_turn()]
        )
        messages, _ = run(client, workspace)
        [result] = tool_results_of(messages[2])
        assert result["is_error"] is True
        assert "unknown tool" in result["content"]


class TestStopReasons:
    def test_refusal_is_surfaced(self, workspace: Path) -> None:
        client = FakeModelClient(
            [text_turn("", stop_reason="refusal", stop_details=StopDetails(category="cyber"))]
        )
        _, ui = run(client, workspace)
        assert any("declined" in w and "cyber" in w for w in ui.warnings)

    def test_max_tokens_is_surfaced(self, workspace: Path) -> None:
        client = FakeModelClient([text_turn("truncated…", stop_reason="max_tokens")])
        _, ui = run(client, workspace)
        assert any("output-token limit" in w for w in ui.warnings)

    def test_unexpected_stop_reason_is_surfaced(self, workspace: Path) -> None:
        client = FakeModelClient([text_turn(stop_reason="banana")])
        _, ui = run(client, workspace)
        assert any("banana" in w for w in ui.warnings)


class TestContextAndStats:
    def test_stats_are_reported(self, workspace: Path) -> None:
        usage = FakeUsage(
            input_tokens=50,
            output_tokens=20,
            cache_read_input_tokens=400,
            cache_creation_input_tokens=30,
        )
        client = FakeModelClient([text_turn(usage=usage)])
        _, ui = run(client, workspace)
        [stats] = ui.stats
        assert stats.cache_read_tokens == 400
        assert stats.total_prompt_tokens == 480

    def test_context_warning_near_limit(self, workspace: Path) -> None:
        client = FakeModelClient(
            [text_turn(usage=FakeUsage(input_tokens=900_000))]
        )
        _, ui = run(client, workspace)
        assert any("Context is at" in w for w in ui.warnings)

    def test_no_context_warning_when_small(self, workspace: Path) -> None:
        client = FakeModelClient([text_turn()])
        _, ui = run(client, workspace)
        assert ui.warnings == []


class TestWorkspaceSnapshot:
    def test_snapshot_taken_before_first_approved_mutation(self, git_workspace: Path) -> None:
        client = FakeModelClient(
            [
                tool_turn(
                    ToolUseBlock(
                        "t1",
                        "edit_file",
                        {"path": "src/main.py", "old_text": "hello", "new_text": "goodbye"},
                    )
                ),
                text_turn(),
            ]
        )
        _, ui = run(client, git_workspace)
        snapshot = git(git_workspace, "rev-parse", SNAPSHOT_REF)
        # The snapshot holds the pre-edit content; the working tree is edited.
        assert git(git_workspace, "show", f"{snapshot}:src/main.py") == MAIN_PY.rstrip("\n")
        assert "goodbye" in (git_workspace / "src" / "main.py").read_text()
        assert any("workspace snapshot" in m for m in ui.infos)

    def test_one_snapshot_covers_the_whole_task(self, git_workspace: Path) -> None:
        client = FakeModelClient(
            [
                tool_turn(ToolUseBlock("t1", "write_file", {"path": "a.txt", "content": "a"})),
                tool_turn(ToolUseBlock("t2", "write_file", {"path": "b.txt", "content": "b"})),
                text_turn(),
            ]
        )
        _, ui = run(client, git_workspace)
        snapshot = git(git_workspace, "rev-parse", SNAPSHOT_REF)
        # Taken before the first write: neither generated file is in it.
        files = git(git_workspace, "ls-tree", "-r", "--name-only", snapshot)
        assert "a.txt" not in files
        assert "b.txt" not in files
        assert sum("workspace snapshot" in m for m in ui.infos) == 1

    def test_no_snapshot_when_denied(self, git_workspace: Path) -> None:
        client = FakeModelClient(
            [
                tool_turn(ToolUseBlock("t1", "write_file", {"path": "a.txt", "content": "a"})),
                text_turn(),
            ]
        )
        _, ui = run(client, git_workspace, approve=lambda name, preview: False)
        with pytest.raises(Exception):
            git(git_workspace, "rev-parse", "--verify", SNAPSHOT_REF)
        assert not any("workspace snapshot" in m for m in ui.infos)

    def test_read_only_task_takes_no_snapshot(self, git_workspace: Path) -> None:
        client = FakeModelClient(
            [tool_turn(ToolUseBlock("t1", "read_file", {"path": "README.md"})), text_turn()]
        )
        run(client, git_workspace)
        with pytest.raises(Exception):
            git(git_workspace, "rev-parse", "--verify", SNAPSHOT_REF)

    def test_non_git_workspace_mutates_without_snapshot(self, workspace: Path) -> None:
        client = FakeModelClient(
            [
                tool_turn(ToolUseBlock("t1", "write_file", {"path": "a.txt", "content": "a"})),
                text_turn(),
            ]
        )
        _, ui = run(client, workspace)
        assert (workspace / "a.txt").read_text() == "a"
        assert not any("workspace snapshot" in m for m in ui.infos)


class TestInterrupt:
    def test_interrupt_discards_partial_turn(self, workspace: Path) -> None:
        client = FakeModelClient(
            [InterruptingTurn(FakeResponse([TextBlock("partial")], "end_turn"))]
        )
        messages, ui = run(client, workspace)
        assert [m["role"] for m in messages] == ["user"]  # nothing half-appended
        assert any("Interrupted" in w for w in ui.warnings)
        assert client.created_turns[0].closed

    def test_interrupt_at_approval_prompt_discards_turn(self, workspace: Path) -> None:
        def approve(name: str, preview: str) -> bool:
            raise KeyboardInterrupt  # Ctrl+C at the y/N prompt

        client = FakeModelClient(
            [tool_turn(ToolUseBlock("t1", "write_file", {"path": "new.txt", "content": "hi"}))]
        )
        messages, ui = run(client, workspace, approve=approve)
        assert [m["role"] for m in messages] == ["user"]
        assert not (workspace / "new.txt").exists()
        assert any("Interrupted" in w for w in ui.warnings)
