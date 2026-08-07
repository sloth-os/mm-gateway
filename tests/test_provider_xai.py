"""Tests for the xAI REST provider (grok-imagine image + video).

The adapter talks to xAI's REST API over plain ``httpx`` (no ``xai_sdk`` gRPC
client), so we mount an ``httpx.MockTransport`` onto the provider's client and
assert the request bodies it builds + the responses it maps back — no network.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from mm_gateway.config import BackendConfig
from mm_gateway.core.exceptions import ProviderNotConfiguredError, ProviderRequestError
from mm_gateway.providers.xai import XAIProvider
from mm_gateway.schemas.image import UnifiedImageRequest
from mm_gateway.schemas.video import UnifiedVideoRequest, image_part, text_part


def _backend(base_url: str | None = None, *, api_key: str = "xai-key") -> BackendConfig:
    kw: dict[str, Any] = {"name": "xai", "type": "xai", "api_key": api_key}
    if base_url is not None:
        kw["base_url"] = base_url
    return BackendConfig(**kw)


def _mount(provider: XAIProvider, handler) -> None:
    """Replace the provider's httpx client with one driven by ``handler``.

    ``handler`` is an ``httpx.MockTransport`` callable: ``(request) -> Response``.
    Each request the adapter makes flows through it, so the test can both
    inspect the outgoing body and script the response.
    """
    provider._client = httpx.AsyncClient(
        base_url=provider._base,
        transport=httpx.MockTransport(handler),
        headers={"authorization": "Bearer xai-key"},
    )


def _body(request: httpx.Request) -> dict[str, Any]:
    return json.loads(request.content) if request.content else {}


# -- construction --------------------------------------------------------- #


def test_requires_api_key() -> None:
    with pytest.raises(ProviderNotConfiguredError):
        XAIProvider(BackendConfig(name="xai", type="xai"))


def test_base_url_default_is_real_api() -> None:
    p = XAIProvider(_backend())
    assert p._base == "https://api.x.ai"


def test_base_url_strips_trailing_v1() -> None:
    # An operator who copied ".../v1" from another provider still gets correct
    # "/v1/..." paths — we build them ourselves.
    assert (
        XAIProvider(_backend("https://ai.xmiaom.com"))._base == "https://ai.xmiaom.com"
    )
    assert (
        XAIProvider(_backend("https://ai.xmiaom.com/v1"))._base
        == "https://ai.xmiaom.com"
    )
    assert (
        XAIProvider(_backend("https://ai.xmiaom.com/v1/"))._base
        == "https://ai.xmiaom.com"
    )


def test_init_attaches_bearer_auth_header() -> None:
    # The adapter must construct its own Bearer header (securitySchemes.bearerAuth
    # is required on every image/video endpoint). _mount rebuilds the client with a
    # hardcoded header, so the auth-asserting tests verify the helper, not the
    # adapter — pin the real client's headers here so a regression (dropping the
    # header or the "Bearer " prefix) fails this test, not just the live API.
    p = XAIProvider(_backend(api_key="secret-key"))
    assert p._client.headers["authorization"] == "Bearer secret-key"


# -- image --------------------------------------------------------------- #


def test_image_generate_maps_params_and_response() -> None:
    p = XAIProvider(_backend("https://ai.xmiaom.com"))

    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["method"] = request.method
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = _body(request)
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "url": "https://x.test/img.png",
                        "b64_json": None,
                        "mime_type": "image/png",
                    }
                ],
                "usage": {"cost_in_usd_ticks": 50_000_000},  # half a cent -> $0.005
            },
        )

    _mount(p, handler)

    req = UnifiedImageRequest(
        model="grok-imagine-image-lite",
        prompt="a cat",
        n=2,
        response_format="url",
        aspect_ratio="1:1",
        resolution="1k",
        user="u1",
    )
    resp = asyncio.run(p.generate_image(req))

    assert captured["path"] == "/v1/images/generations"
    assert captured["method"] == "POST"
    assert captured["auth"] == "Bearer xai-key"
    b = captured["body"]
    assert b["model"] == "grok-imagine-image-lite"
    assert b["prompt"] == "a cat"
    assert b["n"] == 2 and b["response_format"] == "url"
    assert b["aspect_ratio"] == "1:1" and b["resolution"] == "1k" and b["user"] == "u1"

    assert len(resp.data) == 1
    assert resp.data[0].url == "https://x.test/img.png"
    assert resp.data[0].media_type == "image/png"
    assert resp.provider == "xai" and resp.model == "grok-imagine-image-lite"
    assert resp.usage is not None
    assert resp.usage.cost == pytest.approx(0.005)


def test_image_generate_without_usage() -> None:
    # Spec: GeneratedImageResponse.usage is nullable (oneOf null|MediaUsage),
    # only `data` is required. A 200 that omits usage must not crash.
    p = XAIProvider(_backend("https://ai.xmiaom.com"))

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = _body(request)
        return httpx.Response(
            200,
            json={
                "data": [{"url": "https://x.test/img.png"}],
            },
        )

    captured: dict[str, Any] = {}
    _mount(p, handler)

    resp = asyncio.run(
        p.generate_image(
            UnifiedImageRequest(model="grok-imagine-image-lite", prompt="a cat")
        )
    )
    assert resp.usage is None
    assert resp.data[0].url == "https://x.test/img.png"


def test_image_generate_passes_extra_fields() -> None:
    # body.update(request.extra) is the only route to spec-defined but unmapped
    # fields (e.g. storage_options). Pin it so a regression can't silently drop them.
    p = XAIProvider(_backend("https://ai.xmiaom.com"))

    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = _body(request)
        return httpx.Response(200, json={"data": [{"url": "https://x.test/i.png"}]})

    _mount(p, handler)
    asyncio.run(
        p.generate_image(
            UnifiedImageRequest(
                model="m", prompt="x", extra={"storage_options": {"bucket": "mm"}}
            )
        )
    )
    assert captured["body"]["storage_options"] == {"bucket": "mm"}


def test_image_generate_clamps_n_to_spec_max() -> None:
    # xAI caps n at 10 (GenerateImageRequest.n maximum: 10); the unified schema
    # allows up to 16, so the adapter must clamp rather than forward 11..16 and
    # eat a 422->502.
    p = XAIProvider(_backend("https://ai.xmiaom.com"))

    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = _body(request)
        return httpx.Response(200, json={"data": [{"url": "https://x.test/i.png"}]})

    _mount(p, handler)
    asyncio.run(
        p.generate_image(
            UnifiedImageRequest(model="grok-imagine-image", prompt="x", n=16)
        )
    )
    assert captured["body"]["n"] == 10


def test_image_generate_auth_header() -> None:
    # Guard against a regression in the adapter's own Bearer construction
    # (xai.py builds the client with the auth header; verify it actually flows).
    p = XAIProvider(_backend())
    auth: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        auth["header"] = request.headers.get("authorization")
        return httpx.Response(200, json={"data": [{"url": "https://x.test/i.png"}]})

    _mount(p, handler)
    asyncio.run(p.generate_image(UnifiedImageRequest(model="m", prompt="x")))
    assert auth["header"] == "Bearer xai-key"


def test_image_generate_propagates_upstream_error() -> None:
    p = XAIProvider(_backend())

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="unauthorized")

    _mount(p, handler)
    with pytest.raises(ProviderRequestError):
        asyncio.run(p.generate_image(UnifiedImageRequest(model="m", prompt="x")))


# -- video create -------------------------------------------------------- #


def test_video_create_t2v_maps_body() -> None:
    p = XAIProvider(_backend())

    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = _body(request)
        captured["path"] = request.url.path
        return httpx.Response(200, json={"request_id": "req-123"})

    _mount(p, handler)

    req = UnifiedVideoRequest(
        model="grok-imagine-video",
        content=[text_part("a cat playing")],
        duration=11.9,
        ratio="16:9",
        resolution="720p",
    )
    task = asyncio.run(p.create_video_task(req))

    assert captured["path"] == "/v1/videos/generations"
    b = captured["body"]
    assert b["model"] == "grok-imagine-video"
    assert b["prompt"] == "a cat playing"
    # Truncated toward zero to an int (int(11.9) == 11), not rounded or forwarded
    # as a float — xAI's duration is an int32. A bare `== 11` would also pass for
    # the float 11.9 in JSON, so assert the deserialized value is an int, not float.
    assert b["duration"] == 11  # truncated to int
    assert isinstance(b["duration"], int) and not isinstance(b["duration"], bool)
    assert b["resolution"] == "720p"
    assert task.task_id == "req-123"
    assert task.status == "pending" and task.provider == "xai"


def test_video_create_i2v_first_image_and_references() -> None:
    p = XAIProvider(_backend())

    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = _body(request)
        captured["auth"] = request.headers.get("authorization")
        captured["path"] = request.url.path
        return httpx.Response(200, json={"request_id": "req-1"})

    _mount(p, handler)

    req = UnifiedVideoRequest(
        model="grok-imagine-video",
        content=[
            text_part("animate"),
            image_part("https://x.test/first.png", "first_frame"),
            image_part("https://x.test/r1.png", "reference_image"),
            image_part("https://x.test/r2.png", "reference_image"),
        ],
        extra={"storage_options": {"bucket": "mm"}},
    )
    asyncio.run(p.create_video_task(req))
    b = captured["body"]
    assert captured["path"] == "/v1/videos/generations"
    assert captured["auth"] == "Bearer xai-key"
    assert b["image"] == {"url": "https://x.test/first.png"}
    assert b["reference_images"] == [
        {"url": "https://x.test/r1.png"},
        {"url": "https://x.test/r2.png"},
    ]
    assert b["storage_options"] == {"bucket": "mm"}  # extra passthrough


def test_video_create_no_request_id_raises() -> None:
    p = XAIProvider(_backend())

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})  # missing request_id

    _mount(p, handler)
    with pytest.raises(ProviderRequestError):
        asyncio.run(
            p.create_video_task(
                UnifiedVideoRequest(model="m", content=[text_part("x")])
            )
        )


def test_video_create_propagates_upstream_error() -> None:
    p = XAIProvider(_backend())

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    _mount(p, handler)
    with pytest.raises(ProviderRequestError):
        asyncio.run(
            p.create_video_task(
                UnifiedVideoRequest(model="m", content=[text_part("x")])
            )
        )


# -- video poll ---------------------------------------------------------- #


def _poll_handler(responder):
    """Build a MockTransport handler that asserts the GET path then delegates."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/v1/videos/req-1"
        return responder(request)

    return handler


def test_poll_pending_202_is_running() -> None:
    p = XAIProvider(_backend())
    _mount(p, _poll_handler(lambda r: httpx.Response(202)))  # no body expected
    t = asyncio.run(p.get_video_task("req-1"))
    assert t.status == "running" and t.task_id == "req-1"


def test_poll_done_returns_video_and_usage() -> None:
    p = XAIProvider(_backend())

    def resp(_r: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "status": "done",
                "model": "grok-imagine-video",
                "progress": 100,
                "video": {"url": "https://x.test/v.mp4", "duration": 8},
                "usage": {"cost_in_usd_ticks": 100_000_000},  # 1 cent -> $0.01
            },
        )

    _mount(p, _poll_handler(resp))
    t = asyncio.run(p.get_video_task("req-1"))
    assert t.status == "succeeded"
    assert t.model == "grok-imagine-video"  # read from the upstream poll body
    assert t.video_urls == ["https://x.test/v.mp4"]
    assert t.usage is not None and t.usage.cost == pytest.approx(0.01)


def test_poll_done_with_empty_url_sets_error() -> None:
    # Spec: when moderation is violated xAI returns status="done" with an empty
    # video url — a completed-but-unusable result. Surface an error rather than a
    # bare silent success with no media and no explanation.
    p = XAIProvider(_backend())

    def resp(_r: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "status": "done",
                "model": "grok-imagine-video",
                "video": {"url": None, "respect_moderation": False, "duration": 8},
            },
        )

    _mount(p, _poll_handler(resp))
    t = asyncio.run(p.get_video_task("req-1"))
    assert t.status == "succeeded"
    assert t.video_urls == []
    assert t.error is not None and "no URL" in t.error


def test_poll_429_maps_to_429_not_502() -> None:
    # The poll error path must forward _map_status so a rate limit surfaces as
    # 429 (backoff-able) to the client, matching the create path + openrouter.
    p = XAIProvider(_backend())
    _mount(p, _poll_handler(lambda r: httpx.Response(429, text="slow down")))
    with pytest.raises(ProviderRequestError) as excinfo:
        asyncio.run(p.get_video_task("req-1"))
    assert excinfo.value.status_code == 429


def test_poll_failed_carries_error() -> None:
    p = XAIProvider(_backend())

    def resp(_r: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "status": "failed",
                "error": {"code": "RATE_LIMIT", "message": "too fast"},
            },
        )

    _mount(p, _poll_handler(resp))
    t = asyncio.run(p.get_video_task("req-1"))
    assert t.status == "failed"
    assert "RATE_LIMIT" in (t.error or "") and "too fast" in (t.error or "")


def test_poll_unknown_status_is_running() -> None:
    p = XAIProvider(_backend())

    def resp(_r: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "warming_up"})

    _mount(p, _poll_handler(resp))
    t = asyncio.run(p.get_video_task("req-1"))
    assert t.status == "running"


def test_poll_propagates_upstream_error() -> None:
    p = XAIProvider(_backend())
    _mount(p, _poll_handler(lambda r: httpx.Response(404, text="not found")))
    with pytest.raises(ProviderRequestError):
        asyncio.run(p.get_video_task("req-1"))
