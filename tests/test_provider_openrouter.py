"""Tests for the OpenRouter REST provider (unified image + video).

OpenRouter is itself a normalising router, so the adapter is a thin
passthrough over ``httpx``. We mount an ``httpx.MockTransport`` onto the
provider's image and video clients and assert the request bodies it builds +
the responses it maps back — no network.

The construction-split tests assert the sync/async URL split: image lands on
``base_url``, video on ``extra["video_base_url"]`` (when set).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from mm_gateway.config import BackendConfig
from mm_gateway.core.exceptions import ProviderNotConfiguredError, ProviderRequestError
from mm_gateway.providers.openrouter import OpenRouterProvider, _BASE
from mm_gateway.schemas.image import UnifiedImageRequest, text_part as image_text_part
from mm_gateway.schemas.video import UnifiedVideoRequest, text_part


def _backend(base_url: str | None = None, *, extra: dict[str, Any] | None = None,
             api_key: str = "or-key") -> BackendConfig:
    kw: dict[str, Any] = {"name": "openrouter", "type": "openrouter", "api_key": api_key}
    if base_url is not None:
        kw["base_url"] = base_url
    if extra:
        kw["extra"] = extra
    return BackendConfig(**kw)


def _mount(provider: OpenRouterProvider, handler) -> None:
    """Point both the image and video clients at one mock transport."""
    client = httpx.AsyncClient(
        base_url=str(provider._client.base_url),
        transport=httpx.MockTransport(handler),
        headers=provider._client.headers,
    )
    provider._client = client
    provider._client_video = client


def _body(request: httpx.Request) -> dict[str, Any]:
    return json.loads(request.content) if request.content else {}


# -- construction -------------------------------------------------------- #


def test_requires_api_key() -> None:
    with pytest.raises(ProviderNotConfiguredError):
        OpenRouterProvider(BackendConfig(name="openrouter", type="openrouter"))


def test_split_sync_async_base_urls() -> None:
    """Image routes at ``base_url`` (sync), video at ``extra["video_base_url"]``
    (async, recorded by ``config.py`` when it differs)."""
    p = OpenRouterProvider(_backend(
        "https://image.test", extra={"video_base_url": "https://video.test"},
    ))
    assert str(p._client.base_url).rstrip("/") == "https://image.test"
    assert str(p._client_video.base_url).rstrip("/") == "https://video.test"
    # Both clients carry the Bearer auth header.
    assert p._client_video.headers["authorization"] == "Bearer or-key"


def test_single_base_url_when_not_split() -> None:
    """Without a separate video base, both clients share the image base."""
    p = OpenRouterProvider(_backend("https://openrouter.test"))
    assert str(p._client.base_url) == str(p._client_video.base_url)


def test_default_base_url_is_real_api() -> None:
    p = OpenRouterProvider(_backend())
    assert str(p._client.base_url).rstrip("/") == _BASE
