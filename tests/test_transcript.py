from __future__ import annotations

import json
from pathlib import Path

import pytest

from acode.cli import SessionUsage, TrackingUI, make_approver
from acode.transcript import CLIP_CHARS, Transcript, clip, clip_args
from tests.test_agent import RecordingUI  # noqa: F401  (imported for parity checks)
from tests.test_cli import GateRecorder, stats


def read_events(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture
def transcript(tmp_path: Path) -> Transcript:
    result = Transcript.open(tmp_path / "myproject", tmp_path / "logs")
    assert result is not None
    return result


class TestTranscript:
    def test_creates_named_log_file(self, transcript: Transcript, tmp_path: Path) -> None:
        assert transcript.path.parent == tmp_path / "logs"
        assert transcript.path.name.endswith("-myproject.jsonl")

    def test_events_are_timestamped_json_lines(self, transcript: Transcript) -> None:
        transcript.log("session_start", model="m")
        transcript.log("task", text="do the thing")
        transcript.close()
        events = read_events(transcript.path)
        assert [e["event"] for e in events] == ["session_start", "task"]
        assert events[0]["model"] == "m"
        assert all("ts" in e for e in events)

    def test_streamed_text_becomes_one_event(self, transcript: Transcript) -> None:
        transcript.text("Hel")
        transcript.text("lo.")
        transcript.log("tool_call", tool="read_file")
        transcript.close()
        events = read_events(transcript.path)
        assert events[0] == {**events[0], "event": "assistant_text", "text": "Hello."}
        assert events[1]["event"] == "tool_call"

    def test_pending_text_flushes_on_close(self, transcript: Transcript) -> None:
        transcript.text("trailing words")
        transcript.close()
        [event] = read_events(transcript.path)
        assert event["event"] == "assistant_text"
        assert event["text"] == "trailing words"

    def test_close_is_idempotent_and_write_after_close_is_a_noop(
        self, transcript: Transcript
    ) -> None:
        transcript.log("session_start")
        transcript.close()
        transcript.close()
        transcript.log("task", text="late")
        assert [e["event"] for e in read_events(transcript.path)] == ["session_start"]

    def test_open_returns_none_when_directory_is_unwritable(self, tmp_path: Path) -> None:
        blocker = tmp_path / "blocker"
        blocker.write_text("a file where a directory should go")
        assert Transcript.open(tmp_path, blocker / "logs") is None


class TestClipping:
    def test_short_text_untouched(self) -> None:
        assert clip("short") == "short"

    def test_long_text_clipped_with_marker(self) -> None:
        clipped = clip("x" * (CLIP_CHARS + 500))
        assert len(clipped) < CLIP_CHARS + 100
        assert "[clipped 500 chars]" in clipped

    def test_clip_args_only_touches_strings(self) -> None:
        args = {"path": "a.txt", "content": "y" * (CLIP_CHARS + 1), "count": 7}
        clipped = clip_args(args)
        assert clipped["path"] == "a.txt"
        assert clipped["count"] == 7
        assert "[clipped" in clipped["content"]


class TestTrackingUITranscript:
    def test_ui_events_land_in_the_log(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        transcript = Transcript.open(tmp_path, tmp_path / "logs")
        assert transcript is not None
        ui = TrackingUI(SessionUsage(), transcript)
        ui.on_text("Working on it. ")
        ui.on_tool_call("read_file", {"path": "a.txt"})
        ui.on_tool_result("file contents", is_error=False)
        ui.on_turn_stats(stats(input_tokens=5, output_tokens=9))
        ui.warn("heads up")
        transcript.close()
        capsys.readouterr()  # swallow the rendered output
        events = read_events(transcript.path)
        assert [e["event"] for e in events] == [
            "assistant_text",
            "tool_call",
            "tool_result",
            "usage",
            "warning",
        ]
        assert events[1]["args"] == {"path": "a.txt"}
        assert events[3]["output"] == 9

    def test_transcript_is_optional(self, capsys: pytest.CaptureFixture[str]) -> None:
        ui = TrackingUI(SessionUsage())
        ui.on_text("no log attached")
        ui.on_turn_stats(stats(input_tokens=1))
        capsys.readouterr()


class TestApprovalLogging:
    def approver_events(
        self, tmp_path: Path, *, trust_edits: bool, answer: bool
    ) -> tuple[GateRecorder, Transcript]:
        transcript = Transcript.open(tmp_path, tmp_path / "logs")
        assert transcript is not None
        return GateRecorder(answer=answer), transcript

    def test_approved_and_denied_outcomes_logged(self, tmp_path: Path) -> None:
        ui, transcript = self.approver_events(tmp_path, trust_edits=False, answer=True)
        approve = make_approver(ui, trust_edits=False, transcript=transcript)
        approve("run_command", "$ pytest")
        ui.answer = False
        approve("write_file", "new file: a.txt")
        transcript.close()
        events = read_events(transcript.path)
        assert [(e["tool"], e["outcome"]) for e in events] == [
            ("run_command", "approved"),
            ("write_file", "denied"),
        ]
        assert events[0]["preview"] == "$ pytest"

    def test_auto_approved_outcome_logged(self, tmp_path: Path) -> None:
        ui, transcript = self.approver_events(tmp_path, trust_edits=True, answer=False)
        approve = make_approver(ui, trust_edits=True, transcript=transcript)
        approve("edit_file", "diff text")
        transcript.close()
        [event] = read_events(transcript.path)
        assert event["outcome"] == "auto_approved"
        assert event["preview"] == "diff text"
