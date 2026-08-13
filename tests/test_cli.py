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
    resolve_model,
)


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
