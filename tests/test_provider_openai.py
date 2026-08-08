"""Tests for the OpenAI provider (DALL·E/GPT-Image + Sora video).

The provider builds two ``AsyncOpenAI`` instances (image + video) honoring the
sync/async URL split. We capture constructor args by monkeypatching
``AsyncOpenAI`` — no network.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from mm_gateway.config import BackendConfig
from mm_gateway.core.exceptions import ProviderNotConfiguredError
from mm_gateway.providers import openai as openai_mod
from mm_gateway.providers.openai import OpenAIProvider


def _backend(base_url: str | None = None, *, extra: dict[str, Any] | None = None,
             api_key: str = "sk-test") -> BackendConfig:
    kw: dict[str, Any] = {"name": "openai", "type": "openai", "api_key": api_key}
    if base_url is not None:
        kw["base_url"] = base_url
    if extra:
        kw["extra"] = extra
    return BackendConfig(**kw)


# -- construction -------------------------------------------------------- #


def test_requires_api_key() -> None:
    with pytest.raises(ProviderNotConfiguredError):
        OpenAIProvider(BackendConfig(name="openai", type="openai"))


def test_split_sync_async_base_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Image (DALL·E/GPT-Image) lands on ``base_url`` (sync), video (Sora) on
    ``extra["video_base_url"]`` (async, recorded by ``config.py`` when it
    differs)."""
    captured: list[dict[str, Any]] = []

    class CapturingAsyncOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured.append(kwargs)

    monkeypatch.setattr(openai_mod, "AsyncOpenAI", CapturingAsyncOpenAI)
    p = OpenAIProvider(_backend(
        "https://image.test", extra={"video_base_url": "https://video.test"},
    ))
    assert p._client is not p._client_video  # two distinct clients
    assert [c["base_url"] for c in captured] == ["https://image.test", "https://video.test"]
    # Both clients carry the same key.
    assert all(c["api_key"] == "sk-test" for c in captured)


def test_single_base_url_when_not_split(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a separate video base, both clients share the image base."""
    captured: list[dict[str, Any]] = []

    class CapturingAsyncOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured.append(kwargs)

    monkeypatch.setattr(openai_mod, "AsyncOpenAI", CapturingAsyncOpenAI)
    p = OpenAIProvider(_backend("https://openai.test"))
    assert p._client is not p._client_video
    assert [c["base_url"] for c in captured] == ["https://openai.test", "https://openai.test"]


def test_default_base_url_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no base pinned, the SDK default (api.openai.com) applies — i.e.
    ``base_url=None`` is passed through, not a fabricated host."""
    captured: list[dict[str, Any]] = []

    class CapturingAsyncOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured.append(kwargs)

    monkeypatch.setattr(openai_mod, "AsyncOpenAI", CapturingAsyncOpenAI)
    OpenAIProvider(_backend())
    assert [c["base_url"] for c in captured] == [None, None]


def test_video_calls_route_to_video_client() -> None:
    """create/retrieve/download_content all hit ``_client_video``."""
    p = OpenAIProvider(_backend(
        "https://image.test", extra={"video_base_url": "https://video.test"},
    ))

    class FakeVideos:
        async def create(self, **kwargs: Any) -> Any:
            return SimpleNamespace(
                id="sora-1", model="sora-2", status="in_progress",
                error=None, created_at=1, completed_at=None, progress=None,
            )

        async def retrieve(self, task_id: str) -> Any:
            return SimpleNamespace(
                id=task_id, model="sora-2", status="completed",
                error=None, created_at=1, completed_at=2, progress=1.0,
            )

        async def download_content(self, task_id: str) -> Any:
            class Blob:
                async def aread(self) -> bytes:
                    return b"\x00mp4"
            return Blob()

    p._client_video = SimpleNamespace(videos=FakeVideos())  # type: ignore[assignment]
    import base64
    import asyncio
    from mm_gateway.schemas.video import UnifiedVideoRequest, text_part
    task = asyncio.run(p.create_video_task(UnifiedVideoRequest(
        model="sora-2", content=[text_part("a cat")],
    )))
    assert task.status == "running"
    task = asyncio.run(p.get_video_task("sora-1"))
    assert task.status == "succeeded"
    assert task.video_urls == ["data:video/mp4;base64," + base64.b64encode(b"\x00mp4").decode()]

