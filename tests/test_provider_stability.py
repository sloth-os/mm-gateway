"""Tests for the Stability AI REST provider (SD3/SDXL image + SVD video).

The adapter talks to Stability's REST API over plain ``httpx``; we assert the
construction-time sync/async URL split and that image vs video calls land on
the right client. No network calls — we inspect ``base_url`` directly and use
``httpx.MockTransport`` for the SVD call.
"""

from __future__ import annotations

import asyncio
import base64
from typing import Any

import httpx
import pytest

from mm_gateway.config import BackendConfig
from mm_gateway.core.exceptions import ProviderNotConfiguredError, ProviderRequestError
from mm_gateway.providers.stability import _BASE, StabilityProvider
from mm_gateway.schemas.video import UnifiedVideoRequest, text_part


def _backend(base_url: str | None = None, *, extra: dict[str, Any] | None = None,
             api_key: str = "stability-key") -> BackendConfig:
    kw: dict[str, Any] = {"name": "stability", "type": "stability", "api_key": api_key}
    if base_url is not None:
        kw["base_url"] = base_url
    if extra:
        kw["extra"] = extra
    return BackendConfig(**kw)


# -- construction -------------------------------------------------------- #


def test_requires_api_key() -> None:
    with pytest.raises(ProviderNotConfiguredError):
        StabilityProvider(BackendConfig(name="stability", type="stability"))


def test_split_sync_async_base_urls() -> None:
    """Image (SD) routes at ``base_url`` (sync), video (SVD) at
    ``extra["video_base_url"]`` (async, recorded by ``config.py`` when it
    differs)."""
    p = StabilityProvider(_backend(
        "https://image.test", extra={"video_base_url": "https://video.test"},
    ))
    assert str(p._client.base_url).rstrip("/") == "https://image.test"
    assert str(p._client_video.base_url).rstrip("/") == "https://video.test"
    # Both clients carry the Bearer auth header.
    assert p._client_video.headers["authorization"] == "Bearer stability-key"


def test_single_base_url_when_not_split() -> None:
    """Without a separate video base, both clients share the image base."""
    p = StabilityProvider(_backend("https://stability.test"))
    assert str(p._client.base_url) == str(p._client_video.base_url)


def test_default_base_url_is_real_api() -> None:
    p = StabilityProvider(_backend())
    assert str(p._client.base_url).rstrip("/") == _BASE


# -- SVD video poll lands on the video client ----------------------------- #


def test_svd_uses_video_client() -> None:
    """The blocking SVD call runs on the *video* client, not the image one."""
    from mm_gateway.schemas.video import image_part
    p = StabilityProvider(_backend(
        "https://image.test", extra={"video_base_url": "https://video.test"},
    ))
    seen_hosts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_hosts.append(str(request.url.host))
        return httpx.Response(200, json={"video": base64.b64encode(b"x").decode()})

    p._client_video = httpx.AsyncClient(
        base_url="https://video.test", timeout=300,
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer stability-key"},
    )
    # create_video_task mints a synthetic id; the poll runs the blocking SVD
    # call on the video client. A data: URI init image avoids a network fetch.
    task = asyncio.run(p.create_video_task(UnifiedVideoRequest(
        model="stable-video-diffusion",
        content=[text_part("x"),
                 image_part("data:image/png;base64," + base64.b64encode(b"i").decode(),
                            "first_frame")],
        motion_intensity=160,
        guidance_scale=2.5,
        output_format="webm",
    )))
    from mm_gateway.providers.stability import _VIDEO_TASKS

    assert _VIDEO_TASKS[task.task_id]["motion_bucket_id"] == 160
    assert _VIDEO_TASKS[task.task_id]["cfg_scale"] == 2.5
    assert _VIDEO_TASKS[task.task_id]["output_format"] == "webm"
    out = asyncio.run(p.get_video_task(task.task_id))
    assert out.status == "succeeded"
    assert seen_hosts == ["video.test"]


def test_svd_missing_input_image_raises() -> None:
    p = StabilityProvider(_backend())
    with pytest.raises(ProviderRequestError):
        asyncio.run(p.create_video_task(UnifiedVideoRequest(
            model="stable-video-diffusion", content=[text_part("x")],
        )))
