"""Resume tests: replay rules, trimming, validation, and a full round trip
through the real transcript writer — no network anywhere."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from acode.cli import SessionUsage, TrackingUI
from acode.agent import run_task
from acode.resume import (
    ResumeError,
    find_latest_log,
    load_session,
    resolve_resume,
    resume_note_messages,
)
from acode.transcript import Transcript, serialize_content
from tests.test_agent import FakeModelClient, ToolUseBlock, text_turn, tool_turn


def start_event(ws: Path, **overrides: Any) -> dict[str, Any]:
    return {
        "event": "session_start",
        "workspace": str(ws),
        "model": "claude-sonnet-5",
        "log_full": True,
        **overrides,
    }


def text_message(role: str, text: str) -> dict[str, Any]:
    return {"event": "message", "role": role, "content": [{"type": "text", "text": text}]}


def tool_exchange(tool_id: str = "t1") -> list[dict[str, Any]]:
    """An assistant tool call and its results, as two message events."""
    return [
        {
            "event": "message",
            "role": "assistant",
            "content": [{"type": "tool_use", "id": tool_id, "name": "read_file", "input": {}}],
        },
        {
            "event": "message",
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": "data"}],
        },
    ]


def write_log(directory: Path, ws: Path, events: list[dict[str, Any]], stamp: str = "20260819-120000") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stamp}-{ws.name}.jsonl"
    path.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    return path


class TestLoadSession:
    def test_roundtrip_in_order(self, tmp_path: Path, workspace: Path) -> None:
        events = [
            start_event(workspace),
            {"event": "task", "text": "fix the bug"},
            *tool_exchange(),
            text_message("assistant", "Fixed."),
        ]
        session = load_session(write_log(tmp_path, workspace, events), workspace)
        assert [m["role"] for m in session.messages] == ["user", "assistant", "user", "assistant"]
        assert session.messages[0]["content"] == "fix the bug"
        assert session.messages[-1]["content"][0]["text"] == "Fixed."
        assert session.dropped == 0
        assert session.model == "claude-sonnet-5"

    def test_dangling_tool_call_is_trimmed(self, tmp_path: Path, workspace: Path) -> None:
        events = [
            start_event(workspace),
            {"event": "task", "text": "task"},
            text_message("assistant", "ok"),
            *tool_exchange()[:1],  # tool_use whose results never arrived
        ]
        session = load_session(write_log(tmp_path, workspace, events), workspace)
        assert session.dropped == 1
        assert session.messages[-1]["content"][0]["text"] == "ok"

    def test_trim_cascades_through_unconsumed_results(
        self, tmp_path: Path, workspace: Path
    ) -> None:
        # tool results arrived but the model never responded to them: the
        # results AND the call before them must both go.
        events = [
            start_event(workspace),
            {"event": "task", "text": "task"},
            text_message("assistant", "clean end"),
            *tool_exchange(),
        ]
        session = load_session(write_log(tmp_path, workspace, events), workspace)
        assert session.dropped == 2
        assert session.messages[-1]["content"][0]["text"] == "clean end"

    def test_crash_truncated_tail_is_tolerated(self, tmp_path: Path, workspace: Path) -> None:
        path = write_log(
            tmp_path,
            workspace,
            [start_event(workspace), {"event": "task", "text": "t"}, text_message("assistant", "done")],
        )
        with path.open("a", encoding="utf-8") as handle:
            handle.write('{"event": "message", "role": "assist')  # power died here
        session = load_session(path, workspace)
        assert session.messages[-1]["content"][0]["text"] == "done"

    def test_new_command_clears_earlier_conversation(
        self, tmp_path: Path, workspace: Path
    ) -> None:
        events = [
            start_event(workspace),
            {"event": "task", "text": "old task"},
            text_message("assistant", "old answer"),
            {"event": "command", "command": "/new"},
            {"event": "task", "text": "new task"},
            text_message("assistant", "new answer"),
        ]
        session = load_session(write_log(tmp_path, workspace, events), workspace)
        assert session.messages[0]["content"] == "new task"
        assert len(session.messages) == 2

    def test_compact_keeps_only_the_summary_pair(self, tmp_path: Path, workspace: Path) -> None:
        events = [
            start_event(workspace),
            {"event": "task", "text": "big task"},
            text_message("assistant", "lots of work"),
            # compaction logs the replacement pair, then the compact event
            {"event": "message", "role": "user", "content": "[compacted] summary"},
            {"event": "message", "role": "assistant", "content": "Understood."},
            {"event": "compact", "tokens_before": 800_000},
            text_message("assistant", "continuing"),
        ]
        session = load_session(write_log(tmp_path, workspace, events), workspace)
        assert [m["content"] for m in session.messages[:2]] == [
            "[compacted] summary",
            "Understood.",
        ]
        assert len(session.messages) == 3

    def test_unserializable_block_stops_collection_there(
        self, tmp_path: Path, workspace: Path
    ) -> None:
        events = [
            start_event(workspace),
            {"event": "task", "text": "t"},
            text_message("assistant", "good"),
            {"event": "message", "role": "assistant", "content": ["<UnknownBlock repr>"]},
            text_message("assistant", "after corruption"),
        ]
        session = load_session(write_log(tmp_path, workspace, events), workspace)
        assert session.messages[-1]["content"][0]["text"] == "good"

    def test_default_transcript_is_refused(self, tmp_path: Path, workspace: Path) -> None:
        events = [start_event(workspace, log_full=False), {"event": "task", "text": "t"}]
        with pytest.raises(ResumeError, match="log-full"):
            load_session(write_log(tmp_path, workspace, events), workspace)

    def test_other_workspace_is_refused(self, tmp_path: Path, workspace: Path) -> None:
        elsewhere = tmp_path / "elsewhere"
        events = [start_event(elsewhere), {"event": "task", "text": "t"}]
        with pytest.raises(ResumeError, match="workspace"):
            load_session(write_log(tmp_path, workspace, events), workspace)

    def test_non_transcript_file_is_refused(self, tmp_path: Path, workspace: Path) -> None:
        path = tmp_path / "notes.jsonl"
        path.write_text('{"event": "something-else"}\n', encoding="utf-8")
        with pytest.raises(ResumeError, match="not an acode transcript"):
            load_session(path, workspace)

    def test_nothing_left_after_trimming_is_refused(
        self, tmp_path: Path, workspace: Path
    ) -> None:
        events = [start_event(workspace), {"event": "task", "text": "never answered"}]
        with pytest.raises(ResumeError, match="no resumable conversation"):
            load_session(write_log(tmp_path, workspace, events), workspace)


class TestFindLatestLog:
    def test_picks_newest_matching_and_skips_exclusions(
        self, tmp_path: Path, workspace: Path
    ) -> None:
        logs = tmp_path / "logs"
        old = write_log(logs, workspace, [start_event(workspace)], stamp="20260817-090000")
        new = write_log(logs, workspace, [start_event(workspace)], stamp="20260819-090000")
        current = write_log(logs, workspace, [start_event(workspace)], stamp="20260819-235959")
        assert find_latest_log(workspace, logs, exclude=current) == new
        assert find_latest_log(workspace, logs) == current
        assert find_latest_log(workspace, logs, exclude=None) != old

    def test_same_basename_other_workspace_does_not_match(self, tmp_path: Path) -> None:
        logs = tmp_path / "logs"
        ws_a = tmp_path / "one" / "repo"
        ws_b = tmp_path / "two" / "repo"
        ws_a.mkdir(parents=True)
        ws_b.mkdir(parents=True)
        write_log(logs, ws_b, [start_event(ws_b)], stamp="20260819-120000")
        assert find_latest_log(ws_a, logs) is None

    def test_resolve_last_with_no_logs_is_refused(self, tmp_path: Path, workspace: Path) -> None:
        with pytest.raises(ResumeError, match="no previous transcript"):
            resolve_resume("last", workspace, tmp_path / "empty-logs")


class TestResumeNote:
    def test_pair_keeps_roles_alternating(self) -> None:
        pair = resume_note_messages()
        assert [m["role"] for m in pair] == ["user", "assistant"]
        assert "may have changed" in pair[0]["content"]


class TestEndToEnd:
    def test_real_session_roundtrips_through_the_writer(
        self, tmp_path: Path, workspace: Path
    ) -> None:
        """Run a fake session through TrackingUI's full transcript, then
        resume from the file it wrote and compare conversations."""
        logs = tmp_path / "logs"
        transcript = Transcript.open(workspace, logs, full=True)
        assert transcript is not None
        transcript.log("session_start", workspace=str(workspace), model="m", log_full=True)
        ui = TrackingUI(SessionUsage(), transcript)
        client = FakeModelClient(
            [
                tool_turn(ToolUseBlock("t1", "read_file", {"path": "README.md"})),
                text_turn("It is a sample project."),
            ]
        )
        transcript.log("task", text="what is this repo?")
        messages: list[dict[str, Any]] = [{"role": "user", "content": "what is this repo?"}]
        run_task(
            client,
            messages,
            root=workspace,
            ui=ui,
            approve=lambda name, preview: True,
            command_timeout=5.0,
        )
        transcript.close()

        session = load_session(transcript.path, workspace)
        expected = [
            {"role": m["role"], "content": serialize_content(m["content"])} for m in messages
        ]
        assert session.messages == expected
        assert session.dropped == 0
