from __future__ import annotations

import sys
from typing import Any

import pytest

from acode import agent
from acode.agent import TurnStats
from acode.cli import (
    DIRECT_DEFAULT_MODEL,
    GATEWAY_DEFAULT_MODEL,
    SessionUsage,
    TrackingUI,
    _handle_command,
    make_approver,
    resolve_model,
)
from tests.test_agent import FakeModelClient, FakeResponse, FakeTurn, text_turn


def stats(
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read: int = 0,
    cache_creation: int = 0,
) -> TurnStats:
    return TurnStats(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read,
        cache_creation_tokens=cache_creation,
    )


class RecordingConsole:
    def __init__(self) -> None:
        self.infos: list[str] = []
        self.warnings: list[str] = []

    def info(self, message: str) -> None:
        self.infos.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)


class TestResolveModel:
    def test_direct_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
        assert resolve_model(None) == DIRECT_DEFAULT_MODEL

    def test_gateway_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gw.example.com/serving")
        assert resolve_model(None) == GATEWAY_DEFAULT_MODEL

    def test_explicit_flag_wins_on_either_route(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gw.example.com/serving")
        assert resolve_model("claude-opus-5") == "claude-opus-5"
        monkeypatch.delenv("ANTHROPIC_BASE_URL")
        assert resolve_model("claude-opus-5") == "claude-opus-5"


class TestTlsHttpClient:
    @pytest.fixture(autouse=True)
    def clean_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ACODE_CA_BUNDLE", raising=False)
        monkeypatch.delenv("CHUNKER_CA_BUNDLE", raising=False)

    @pytest.fixture
    def captured_verify(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
        """Capture the verify= argument instead of building a real TLS context."""
        captured: dict[str, Any] = {}

        class FakeClient:
            def __init__(self, *, verify: Any) -> None:
                captured["verify"] = verify

        monkeypatch.setattr(agent.httpx, "Client", FakeClient)
        return captured

    def test_acode_ca_bundle(
        self, monkeypatch: pytest.MonkeyPatch, captured_verify: dict[str, Any]
    ) -> None:
        monkeypatch.setenv("ACODE_CA_BUNDLE", "/etc/ssl/private-ca.pem")
        assert agent._tls_http_client() is not None
        assert captured_verify["verify"] == "/etc/ssl/private-ca.pem"

    def test_chunker_ca_bundle_fallback(
        self, monkeypatch: pytest.MonkeyPatch, captured_verify: dict[str, Any]
    ) -> None:
        monkeypatch.setenv("CHUNKER_CA_BUNDLE", "/etc/ssl/shared-ca.pem")
        assert agent._tls_http_client() is not None
        assert captured_verify["verify"] == "/etc/ssl/shared-ca.pem"

    def test_acode_bundle_wins_over_chunker(
        self, monkeypatch: pytest.MonkeyPatch, captured_verify: dict[str, Any]
    ) -> None:
        monkeypatch.setenv("ACODE_CA_BUNDLE", "/etc/ssl/acode-ca.pem")
        monkeypatch.setenv("CHUNKER_CA_BUNDLE", "/etc/ssl/shared-ca.pem")
        agent._tls_http_client()
        assert captured_verify["verify"] == "/etc/ssl/acode-ca.pem"

    def test_no_configuration_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Force `import truststore` to fail even if the package is installed.
        monkeypatch.setitem(sys.modules, "truststore", None)
        assert agent._tls_http_client() is None


class TestSessionUsage:
    def test_accumulates_across_requests(self) -> None:
        usage = SessionUsage()
        usage.add(stats(input_tokens=10, output_tokens=5, cache_read=100, cache_creation=50))
        usage.add(stats(input_tokens=3, output_tokens=2, cache_read=200, cache_creation=0))
        assert usage.input_tokens == 13
        assert usage.output_tokens == 7
        assert usage.cache_read_tokens == 300
        assert usage.cache_creation_tokens == 50

    def test_cost_estimate_for_known_model(self) -> None:
        usage = SessionUsage()
        usage.add(
            stats(
                input_tokens=1_000_000,
                output_tokens=1_000_000,
                cache_read=1_000_000,
                cache_creation=1_000_000,
            )
        )
        # sonnet-5 rates: 3.00 in + 15.00 out + 3.75 cache write + 0.30 cache read
        assert usage.estimated_cost_usd("claude-sonnet-5") == pytest.approx(22.05)

    def test_prefix_match_covers_gateway_names(self) -> None:
        usage = SessionUsage()
        usage.add(stats(output_tokens=1_000_000))
        assert usage.estimated_cost_usd("databricks-claude-sonnet-4-6") == pytest.approx(15.0)

    def test_unknown_model_has_no_estimate(self) -> None:
        usage = SessionUsage()
        usage.add(stats(input_tokens=100))
        assert usage.estimated_cost_usd("some-other-model") is None
        assert "no pricing known" in usage.summary("some-other-model")

    def test_summary_contains_totals_and_cost(self) -> None:
        usage = SessionUsage()
        usage.add(stats(input_tokens=1_500, output_tokens=200, cache_read=3_000))
        summary = usage.summary("claude-sonnet-5")
        assert "in 1,500" in summary
        assert "out 200" in summary
        assert "cache read 3,000" in summary
        assert "est. $" in summary


class TestTrackingUI:
    def test_stats_are_accumulated_and_still_rendered(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        usage = SessionUsage()
        ui = TrackingUI(usage)
        ui.on_turn_stats(stats(input_tokens=7, output_tokens=8))
        assert usage.input_tokens == 7
        assert "[tokens:" in capsys.readouterr().out


class GateRecorder:
    """Stand-in for ConsoleUI's approval surface."""

    def __init__(self, answer: bool = False) -> None:
        self.answer = answer
        self.asked: list[tuple[str, str]] = []
        self.auto_approved: list[tuple[str, str]] = []

    def ask_approval(self, name: str, preview: str) -> bool:
        self.asked.append((name, preview))
        return self.answer

    def on_auto_approved(self, name: str, preview: str) -> None:
        self.auto_approved.append((name, preview))


class TestMakeApprover:
    def test_default_gates_everything(self) -> None:
        ui = GateRecorder(answer=True)
        approve = make_approver(ui, trust_edits=False)
        assert approve("edit_file", "diff") is True
        assert approve("run_command", "$ ls") is True
        assert [name for name, _ in ui.asked] == ["edit_file", "run_command"]
        assert ui.auto_approved == []

    def test_trust_edits_auto_approves_file_tools(self) -> None:
        ui = GateRecorder(answer=False)  # would deny if ever asked
        approve = make_approver(ui, trust_edits=True)
        assert approve("edit_file", "diff") is True
        assert approve("write_file", "content") is True
        assert ui.asked == []
        assert [name for name, _ in ui.auto_approved] == ["edit_file", "write_file"]

    def test_trust_edits_still_gates_commands(self) -> None:
        ui = GateRecorder(answer=False)
        approve = make_approver(ui, trust_edits=True)
        assert approve("run_command", "$ rm -rf build") is False
        assert ui.asked == [("run_command", "$ rm -rf build")]
        assert ui.auto_approved == []


class TestHandleCommand:
    def test_new_clears_conversation(self) -> None:
        console = RecordingConsole()
        messages: list[dict[str, object]] = [{"role": "user", "content": "old task"}]
        handled = _handle_command(
            "/new", messages=messages, usage=SessionUsage(), model="m", ui=console
        )
        assert handled is True
        assert messages == []
        assert any("cleared" in m for m in console.infos)

    def test_cost_prints_summary(self) -> None:
        console = RecordingConsole()
        usage = SessionUsage()
        usage.add(stats(input_tokens=42))
        handled = _handle_command(
            "/cost", messages=[], usage=usage, model="claude-sonnet-5", ui=console
        )
        assert handled is True
        assert any("session usage" in m and "in 42" in m for m in console.infos)

    def test_unknown_command_warns(self) -> None:
        console = RecordingConsole()
        handled = _handle_command(
            "/teleport", messages=[], usage=SessionUsage(), model="m", ui=console
        )
        assert handled is True
        assert any("/teleport" in w for w in console.warnings)

    def test_plain_text_is_not_a_command(self) -> None:
        console = RecordingConsole()
        handled = _handle_command(
            "fix the bug", messages=[], usage=SessionUsage(), model="m", ui=console
        )
        assert handled is False


class TestContextTracking:
    def test_context_follows_last_request_plus_output(self) -> None:
        usage = SessionUsage()
        usage.add(stats(input_tokens=10, output_tokens=5, cache_read=100, cache_creation=50))
        assert usage.context_tokens == 165
        usage.add(stats(input_tokens=200, output_tokens=30))
        assert usage.context_tokens == 230  # replaced, not accumulated


class CompactConsole(RecordingConsole):
    """RecordingConsole extended with the UI surface compaction touches."""

    def __init__(self) -> None:
        super().__init__()
        self.errors: list[str] = []
        self.messages: list[dict[str, object]] = []

    def on_turn_start(self) -> None: ...

    def on_turn_stats(self, stats: TurnStats) -> None: ...

    def on_message(self, message: dict[str, object]) -> None:
        self.messages.append(message)

    def error(self, message: str) -> None:
        self.errors.append(message)


class TestCompactCommand:
    def test_compact_replaces_conversation(self) -> None:
        console = CompactConsole()
        client = FakeModelClient([text_turn("everything is done; nothing pending")])
        messages: list[dict[str, object]] = [
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "done"},
        ]
        usage = SessionUsage()
        usage.add(stats(input_tokens=500, output_tokens=20))
        handled = _handle_command(
            "/compact", messages=messages, usage=usage, model="m", ui=console, client=client
        )
        assert handled is True
        assert len(messages) == 2
        assert "everything is done" in str(messages[0]["content"])
        assert usage.context_tokens == 0
        assert any("compacted" in m for m in console.infos)

    def test_compact_with_empty_conversation(self) -> None:
        console = CompactConsole()
        client = FakeModelClient([])
        handled = _handle_command(
            "/compact", messages=[], usage=SessionUsage(), model="m", ui=console, client=client
        )
        assert handled is True
        assert client.requests == []
        assert any("Nothing to compact" in m for m in console.infos)

    def test_failed_compaction_changes_nothing(self) -> None:
        console = CompactConsole()
        client = FakeModelClient([FakeTurn(FakeResponse([], "end_turn"))])
        messages: list[dict[str, object]] = [{"role": "user", "content": "task"}]
        usage = SessionUsage()
        usage.add(stats(input_tokens=500, output_tokens=20))
        _handle_command(
            "/compact", messages=messages, usage=usage, model="m", ui=console, client=client
        )
        assert messages == [{"role": "user", "content": "task"}]
        assert usage.context_tokens == 520  # not reset — nothing was freed
        assert any("no summary" in w for w in console.warnings)

    def test_new_resets_context_tokens(self) -> None:
        usage = SessionUsage()
        usage.add(stats(input_tokens=500, output_tokens=20))
        _handle_command(
            "/new", messages=[{"role": "user", "content": "x"}], usage=usage,
            model="m", ui=RecordingConsole(),
        )
        assert usage.context_tokens == 0

    def test_cost_reports_context(self) -> None:
        console = RecordingConsole()
        usage = SessionUsage()
        usage.add(stats(input_tokens=500, output_tokens=20))
        _handle_command("/cost", messages=[], usage=usage, model="m", ui=console)
        assert any("context: ~520" in m for m in console.infos)


class TestHelpCommand:
    def test_help_lists_every_command(self) -> None:
        console = RecordingConsole()
        handled = _handle_command(
            "/help", messages=[], usage=SessionUsage(), model="m", ui=console
        )
        assert handled is True
        listed = "\n".join(console.infos)
        for name in ("/new", "/compact", "/cost", "/help", "/quit"):
            assert name in listed

    def test_unknown_command_points_at_help(self) -> None:
        console = RecordingConsole()
        _handle_command("/teleport", messages=[], usage=SessionUsage(), model="m", ui=console)
        assert any("/help" in w for w in console.warnings)


class HandoffConsole(CompactConsole):
    """CompactConsole that also answers the exit confirmation."""

    def __init__(self, answer: bool) -> None:
        super().__init__()
        self.answer = answer
        self.confirmations: list[str] = []

    def confirm(self, prompt: str) -> bool:
        self.confirmations.append(prompt)
        return self.answer


class TestOfferHandoff:
    def messages(self) -> list[dict[str, object]]:
        return [
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "done"},
        ]

    def test_yes_saves_the_note(self, tmp_path, monkeypatch) -> None:
        from acode import cli, handoff

        store = tmp_path / "store"
        monkeypatch.setattr(cli, "save_handoff", lambda root, note: handoff.save_handoff(root, note, store))
        console = HandoffConsole(answer=True)
        client = FakeModelClient([text_turn("was fixing the parser; next: run tests")])
        cli._offer_handoff(
            client, self.messages(), root=tmp_path, ui=console, model="m", transcript=None
        )
        assert console.confirmations  # the user was asked
        loaded = handoff.load_handoff(tmp_path, store)
        assert loaded is not None and "fixing the parser" in loaded[1]
        assert any("saved" in m for m in console.infos)

    def test_no_makes_no_request(self, tmp_path) -> None:
        from acode import cli

        console = HandoffConsole(answer=False)
        client = FakeModelClient([])
        cli._offer_handoff(
            client, self.messages(), root=tmp_path, ui=console, model="m", transcript=None
        )
        assert client.requests == []

    def test_empty_conversation_is_not_asked(self, tmp_path) -> None:
        from acode import cli

        console = HandoffConsole(answer=True)
        cli._offer_handoff(
            FakeModelClient([]), [], root=tmp_path, ui=console, model="m", transcript=None
        )
        assert console.confirmations == []

    def test_no_summary_saves_nothing(self, tmp_path, monkeypatch) -> None:
        from acode import cli, handoff

        store = tmp_path / "store"
        monkeypatch.setattr(cli, "save_handoff", lambda root, note: handoff.save_handoff(root, note, store))
        console = HandoffConsole(answer=True)
        client = FakeModelClient([FakeTurn(FakeResponse([], "end_turn"))])
        cli._offer_handoff(
            client, self.messages(), root=tmp_path, ui=console, model="m", transcript=None
        )
        assert handoff.load_handoff(tmp_path, store) is None
        assert any("nothing saved" in w for w in console.warnings)
