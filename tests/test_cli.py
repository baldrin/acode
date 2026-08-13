from __future__ import annotations

import sys
from typing import Any

import pytest

from acode import agent
from acode.cli import (
    DIRECT_DEFAULT_MODEL,
    GATEWAY_DEFAULT_MODEL,
    resolve_model,
)


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
