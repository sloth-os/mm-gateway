"""Tests for the MiniMax provider's video surface (H3).

The adapter talks to MiniMax over ``httpx``; we mount an ``httpx.MockTransport``
onto the provider's video client (``_client_video``) and assert the request
bodies it builds + the responses it maps back — no network. The music surface is
covered in ``test_providers_music.py``.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from mm_gateway.config import BackendConfig
from mm_gateway.core.exceptions import (
    ProviderNotConfiguredError,
    ProviderRequestError,
    TaskFailedError,
)
from mm_gateway.providers.minimax import MiniMaxProvider
from mm_gateway.schemas.video import (
    UnifiedVideoRequest,
    audio_part,
    image_part,
    text_part,
    video_part,
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _backend(
    *,
    api_key: str = "k",
    base_url: str | None = None,
    video_base_url: str | None = None,
) -> BackendConfig:
    extra: dict[str, Any] = {}
    if video_base_url is not None:
        extra["video_base_url"] = video_base_url
    return BackendConfig(
        name="minimax", type="minimax", api_key=api_key, base_url=base_url, extra=extra
    )


def _mount_video(provider: MiniMaxProvider, handler, *, base_url: str) -> None:
    """Replace the provider's video httpx client with a MockTransport-driven one."""
    provider._client_video = httpx.AsyncClient(
        base_url=base_url,
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer k", "Content-Type": "application/json"},
    )


def _body(request: httpx.Request) -> dict[str, Any]:
    return json.loads(request.content) if request.content else {}


def _t2v(**kw: Any) -> UnifiedVideoRequest:
    return UnifiedVideoRequest(
        model="MiniMax-H3", content=[text_part("a cat playing")], **kw
    )


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #


def test_video_requires_api_key() -> None:
    with pytest.raises(ProviderNotConfiguredError):
        MiniMaxProvider(BackendConfig(name="minimax", type="minimax"))


def test_video_client_uses_split_base_url() -> None:
    """The video client targets ``extra["video_base_url"]`` when it differs from
    the music ``base_url`` (the ``*_VIDEO_BASE_URL`` endpoint)."""
    p = MiniMaxProvider(
        _backend(base_url="https://music.test", video_base_url="https://video.test")
    )
    assert str(p._client.base_url).rstrip("/") == "https://music.test"
    assert str(p._client_video.base_url).rstrip("/") == "https://video.test"


def test_video_client_collapses_when_not_split() -> None:
    """Without a separate video base, both clients share the music base."""
    p = MiniMaxProvider(_backend(base_url="https://api.minimax.io"))
    assert str(p._client.base_url).rstrip("/") == "https://api.minimax.io"
    assert str(p._client_video.base_url).rstrip("/") == "https://api.minimax.io"


# --------------------------------------------------------------------------- #
# Create — body building + content pass-through
# --------------------------------------------------------------------------- #


def test_create_t2v_passes_content_duration_resolution_ratio() -> None:
    p = MiniMaxProvider(_backend())
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = _body(request)
        return httpx.Response(200, json={"task_id": "mm-task-1"})

    _mount_video(p, handler, base_url="https://api.minimax.io")

    task = asyncio.run(
        p.create_video_task(_t2v(duration=6, resolution="768P", ratio="16:9"))
    )
    assert task.task_id == "mm-task-1" and task.status == "pending"
    assert captured["path"] == "/v2/video_generation"
    b = captured["body"]
    assert b["model"] == "MiniMax-H3"
    # The H3 content[] shape is the unified content shape — parts pass straight through.
    assert b["content"] == [{"type": "text", "text": "a cat playing"}]
    assert b["duration"] == 6
    assert b["resolution"] == "768P"  # verbatim H3 token, not derived from w/h
    assert b["ratio"] == "16:9"


def test_create_ratio_derived_from_dimensions_when_unset() -> None:
    p = MiniMaxProvider(_backend())
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = _body(request)
        return httpx.Response(200, json={"task_id": "t"})

    _mount_video(p, handler, base_url="https://api.minimax.io")
    asyncio.run(p.create_video_task(_t2v(width=1920, height=1080)))
    assert captured["body"]["ratio"] == "16:9"  # gcd-derived
    assert "resolution" not in captured["body"]  # not set, not derived


def test_create_i2v_first_and_last_frame() -> None:
    p = MiniMaxProvider(_backend())
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = _body(request)
        return httpx.Response(200, json={"task_id": "t"})

    _mount_video(p, handler, base_url="https://api.minimax.io")
    req = UnifiedVideoRequest(
        model="MiniMax-H3",
        content=[
            text_part("animate"),
            image_part("https://x.test/first.png", "first_frame"),
            image_part("https://x.test/last.png", "last_frame"),
        ],
    )
    asyncio.run(p.create_video_task(req))
    content = captured["body"]["content"]
    assert [c["type"] for c in content] == ["text", "image_url", "image_url"]
    assert (
        content[1]["role"] == "first_frame"
        and content[1]["image_url"]["url"] == "https://x.test/first.png"
    )
    assert (
        content[2]["role"] == "last_frame"
        and content[2]["image_url"]["url"] == "https://x.test/last.png"
    )


def test_create_reference_image_video_audio_parts_pass_through() -> None:
    p = MiniMaxProvider(_backend())
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = _body(request)
        return httpx.Response(200, json={"task_id": "t"})

    _mount_video(p, handler, base_url="https://api.minimax.io")
    req = UnifiedVideoRequest(
        model="MiniMax-H3",
        content=[
            text_part("follow the refs"),
            image_part("https://x.test/r.png", "reference_image"),
            video_part("https://x.test/rv.mp4"),
            audio_part("https://x.test/ra.mp3"),
        ],
    )
    asyncio.run(p.create_video_task(req))
    content = captured["body"]["content"]
    by_type = {c["type"]: c for c in content}
    assert by_type["image_url"]["role"] == "reference_image"
    assert by_type["video_url"]["role"] == "reference_video"
    assert by_type["video_url"]["video_url"]["url"] == "https://x.test/rv.mp4"
    assert by_type["audio_url"]["role"] == "reference_audio"
    assert by_type["audio_url"]["audio_url"]["url"] == "https://x.test/ra.mp3"


def test_create_forwards_extra_knobs() -> None:
    """Provider-specific knobs stashed in ``request.extra`` pass straight through."""
    p = MiniMaxProvider(_backend())
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = _body(request)
        return httpx.Response(200, json={"task_id": "t"})

    _mount_video(p, handler, base_url="https://api.minimax.io")
    asyncio.run(
        p.create_video_task(_t2v(extra={"prompt_optimizer": True, "watermark": False}))
    )
    b = captured["body"]
    assert b["prompt_optimizer"] is True and b["watermark"] is False


# --------------------------------------------------------------------------- #
# Create — error paths
# --------------------------------------------------------------------------- #


def test_create_no_task_id_raises() -> None:
    p = MiniMaxProvider(_backend())

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unrelated": "shape"})

    _mount_video(p, handler, base_url="https://api.minimax.io")
    with pytest.raises(ProviderRequestError) as exc:
        asyncio.run(p.create_video_task(_t2v()))
    assert exc.value.status_code == 502


def test_create_base_resp_error_fails_task() -> None:
    p = MiniMaxProvider(_backend())

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"base_resp": {"status_code": 1004, "status_msg": "bad params"}}
        )

    _mount_video(p, handler, base_url="https://api.minimax.io")
    with pytest.raises(TaskFailedError) as exc:
        asyncio.run(p.create_video_task(_t2v()))
    assert "bad params" in str(exc.value)


def test_create_http_error_maps_status() -> None:
    p = MiniMaxProvider(_backend())

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="unauthorized")

    _mount_video(p, handler, base_url="https://api.minimax.io")
    with pytest.raises(ProviderRequestError) as exc:
        asyncio.run(p.create_video_task(_t2v()))
    # 401/403 -> 502 per _map_status.
    assert exc.value.status_code == 502


# --------------------------------------------------------------------------- #
# Poll — status mapping + output extraction
# --------------------------------------------------------------------------- #


def _poll_handler(
    status: str,
    *,
    content: dict[str, Any] | None = None,
    error: Any = None,
    model: str = "MiniMax-H3",
) -> Any:
    task: dict[str, Any] = {"status": status, "model": model}
    if content is not None:
        task["content"] = content
    if error is not None:
        task["error"] = error
    return httpx.Response(200, json={"task": task})


def test_poll_queued_is_pending() -> None:
    p = MiniMaxProvider(_backend())
    _mount_video(
        p, lambda r: _poll_handler("queued"), base_url="https://api.minimax.io"
    )
    t = asyncio.run(p.get_video_task("mm-task-1"))
    assert t.status == "pending" and not t.video_urls


def test_poll_running_then_processing_map_to_running() -> None:
    p = MiniMaxProvider(_backend())
    _mount_video(
        p, lambda r: _poll_handler("running"), base_url="https://api.minimax.io"
    )
    assert asyncio.run(p.get_video_task("t")).status == "running"
    p2 = MiniMaxProvider(_backend())
    _mount_video(
        p2, lambda r: _poll_handler("processing"), base_url="https://api.minimax.io"
    )
    assert asyncio.run(p2.get_video_task("t")).status == "running"


def test_poll_unknown_status_defaults_to_running() -> None:
    p = MiniMaxProvider(_backend())
    _mount_video(
        p, lambda r: _poll_handler("weird-new-state"), base_url="https://api.minimax.io"
    )
    assert asyncio.run(p.get_video_task("t")).status == "running"


def test_poll_succeeded_reads_url_and_cover() -> None:
    p = MiniMaxProvider(_backend())
    _mount_video(
        p,
        lambda r: _poll_handler(
            "succeeded",
            content={
                "url": "https://x.test/v.mp4",
                "cover_url": "https://x.test/c.jpg",
            },
        ),
        base_url="https://api.minimax.io",
    )
    t = asyncio.run(p.get_video_task("mm-task-1"))
    assert t.status == "succeeded"
    assert t.video_urls == ["https://x.test/v.mp4"]
    assert t.cover_url == "https://x.test/c.jpg"
    assert t.model == "MiniMax-H3"


def test_poll_succeeded_with_no_url_is_failed() -> None:
    p = MiniMaxProvider(_backend())
    _mount_video(
        p,
        lambda r: _poll_handler("succeeded", content={}),
        base_url="https://api.minimax.io",
    )
    t = asyncio.run(p.get_video_task("t"))
    assert t.status == "failed"
    assert t.error and "no content url" in t.error


def test_poll_failed_carries_error_message() -> None:
    p = MiniMaxProvider(_backend())
    _mount_video(
        p,
        lambda r: _poll_handler(
            "failed", error={"code": "RATE_LIMIT", "message": "too fast"}
        ),
        base_url="https://api.minimax.io",
    )
    t = asyncio.run(p.get_video_task("t"))
    assert t.status == "failed"
    assert "too fast" in (t.error or "")


def test_poll_cancelled() -> None:
    p = MiniMaxProvider(_backend())
    _mount_video(
        p, lambda r: _poll_handler("cancelled"), base_url="https://api.minimax.io"
    )
    assert asyncio.run(p.get_video_task("t")).status == "cancelled"


def test_poll_no_task_raises() -> None:
    p = MiniMaxProvider(_backend())

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"base_resp": {"status_code": 1005, "status_msg": "not found"}}
        )

    _mount_video(p, handler, base_url="https://api.minimax.io")
    with pytest.raises(ProviderRequestError) as exc:
        asyncio.run(p.get_video_task("nope"))
    assert exc.value.status_code == 502


def test_poll_http_error_maps_status() -> None:
    p = MiniMaxProvider(_backend())

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="upstream down")

    _mount_video(p, handler, base_url="https://api.minimax.io")
    with pytest.raises(ProviderRequestError) as exc:
        asyncio.run(p.get_video_task("t"))
    assert exc.value.status_code == 502  # >=500 -> 502


def test_poll_path_includes_task_id() -> None:
    p = MiniMaxProvider(_backend())
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        return _poll_handler("running")

    _mount_video(p, handler, base_url="https://api.minimax.io")
    asyncio.run(p.get_video_task("mm-task-42"))
    assert captured["path"] == "/v2/query/video_generation/mm-task-42"
