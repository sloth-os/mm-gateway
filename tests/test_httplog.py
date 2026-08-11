"""Tests for HTTP request/response logging (curl format + masked headers).

Covers the goal: log inbound request headers+body (curl format), outbound
response body, backend request/response headers+body, with sensitive header
*values* masked while header *names* are kept.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from mm_gateway.observability import httplog
from mm_gateway.observability.httplog import (
    mask_authorization,
    mask_headers,
    to_curl,
)


class _FakeLogger:
    """Records structured log calls as dicts so tests can assert on them.

    Replaces ``httplog.log`` for the duration of a test — every
    ``frontend_*``/``backend_*``/``_backend_*_hook`` helper looks up the
    module-level ``log`` at call time, so patching the module global captures
    them all without touching structlog configuration.
    """

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def _record(self, event: str, **kw: Any) -> None:
        self.events.append({"event": event, **kw})

    def info(self, event: str, **kw: Any) -> None:
        self._record(event, **kw)

    def warning(self, event: str, **kw: Any) -> None:
        self._record(event, level="warning", **kw)

    def error(self, event: str, **kw: Any) -> None:
        self._record(event, level="error", **kw)

    def debug(self, event: str, **kw: Any) -> None:
        self._record(event, level="debug", **kw)

    def find(self, event: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e.get("event") == event]


@pytest.fixture
def fake_log(monkeypatch: pytest.MonkeyPatch) -> _FakeLogger:
    fake = _FakeLogger()
    monkeypatch.setattr(httplog, "log", fake)
    return fake


# -- masking ----------------------------------------------------------------- #


def test_mask_keeps_header_name_masks_value() -> None:
    out = mask_headers({"Authorization": "Bearer sk-secret-1234567890"})
    assert "Authorization" in out  # name kept
    assert "sk-secret-1234567890" not in out["Authorization"]  # value masked
    assert out["Authorization"].startswith("Bearer ")


def test_mask_case_insensitive_and_substring_telltales() -> None:
    out = mask_headers({
        "X-API-KEY": "abc1234567890",
        "x-auth-token": "tok-1234567890",
        "Cookie": "session=xyz",
        "Content-Type": "application/json",  # not sensitive — kept verbatim
    })
    assert out["Content-Type"] == "application/json"
    assert "abc1234567890" not in out["X-API-KEY"]
    assert "tok-1234567890" not in out["x-auth-token"]
    assert "session=xyz" not in out["Cookie"]


def test_mask_authorization_short_value_fully_masked() -> None:
    assert mask_authorization("abc") == "****"
    assert mask_authorization("") == ""


# -- curl format ------------------------------------------------------------- #


def test_to_curl_renders_method_url_headers_body() -> None:
    curl = to_curl(
        "POST", "https://up.test/v1/images",
        {"Authorization": "Bearer sk-abcdefgh1234567890", "Content-Type": "application/json"},
        b'{"model":"x","prompt":"a cat"}',
    )
    assert curl.startswith("curl -X POST ")
    assert "https://up.test/v1/images" in curl
    # sensitive value masked, not present verbatim
    assert "sk-abcdefgh1234567890" not in curl
    # body included as --data
    assert "--data" in curl
    assert "a cat" in curl


def test_to_curl_omits_data_for_empty_body() -> None:
    curl = to_curl("GET", "https://up.test/health", {}, None)
    assert "--data" not in curl


# -- frontend middleware (inbound request + outbound response) ---------------- #


def test_frontend_request_logged_in_curl_format_with_masked_headers(
    client: TestClient, fake_log: _FakeLogger,
) -> None:
    r = client.post(
        "/v1/images",
        json={
            "model": "fake-image-1",
            "input": [{"type": "text", "text": "a cat"}],
            "parameters": {"output_count": 1},
        },
        headers={"x-request-id": "req-1", "authorization": "Bearer super-secret-key-9999"},
    )
    assert r.status_code == 202, r.text

    reqs = fake_log.find("frontend_request")
    resps = fake_log.find("frontend_response")
    assert reqs, "frontend_request not logged"
    assert resps, "frontend_response not logged"

    req = reqs[0]
    assert req["curl"].startswith("curl -X POST ")  # curl format
    assert "/v1/images" in req["curl"]
    assert "a cat" in req.get("body", "")  # request body logged
    # sensitive header value masked, not leaked anywhere
    assert "super-secret-key-9999" not in req["curl"]
    assert "super-secret-key-9999" not in str(req.get("headers", {}))

    resp = resps[0]
    assert resp["status"] == 202
    assert "img_" in resp.get("body", "")  # gateway-owned id logged


async def test_backend_hooks_log_request_and_response(fake_log: _FakeLogger) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True}, headers={"x-trace": "abc"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, event_hooks=httplog.backend_event_hooks()) as c:
        resp = await c.post(
            "https://up.test/v1/generate",
            headers={"Authorization": "Bearer sk-backend-secret-1234567890",
                     "Content-Type": "application/json"},
            json={"prompt": "a dog"},
        )
        assert resp.json() == {"ok": True}

    reqs = fake_log.find("backend_request")
    resps = fake_log.find("backend_response")
    assert reqs, "backend_request not logged"
    assert resps, "backend_response not logged"

    req = reqs[0]
    assert req["curl"].startswith("curl -X POST ")  # backend request in curl format
    assert "https://up.test/v1/generate" in req["curl"]
    assert "a dog" in req.get("body", "")  # backend request body logged
    # sensitive value masked in both the curl dump and the headers dict
    assert "sk-backend-secret-1234567890" not in req["curl"]
    assert "sk-backend-secret-1234567890" not in str(req.get("headers", {}))

    r = resps[0]
    assert r["status"] == 200
    assert "ok" in r.get("body", "")  # backend response body logged
    assert r["headers"].get("x-trace") == "abc"  # backend response headers logged
