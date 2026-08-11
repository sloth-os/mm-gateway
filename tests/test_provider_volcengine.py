"""Tests for the Volcengine Ark provider (Seedream image + Seedance video).

The provider talks to ``AsyncArk``; we swap a fake Ark client onto it so no
network calls are made. The provider reads responses via ``getattr``, so plain
``SimpleNamespace`` objects stand in for the SDK's pydantic models.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from mm_gateway.config import BackendConfig
from mm_gateway.core.exceptions import ProviderNotConfiguredError, ProviderRequestError
from mm_gateway.providers import volcengine as volc_mod
from mm_gateway.providers.volcengine import VolcengineProvider
from mm_gateway.schemas.image import UnifiedImageRequest
from mm_gateway.schemas.image import text_part as image_text_part
from mm_gateway.schemas.video import (
    UnifiedVideoRequest,
    audio_part,
    image_part,
    text_part,
    video_part,
)


class FakeTasks:
    def __init__(self) -> None:
        self.create_calls: list[dict[str, Any]] = []
        self.get_calls: list[str] = []
        self._next_results: list[Any] = []

    async def create(self, **kwargs: Any) -> Any:
        self.create_calls.append(kwargs)
        return SimpleNamespace(id="ark-task-1", safety_identifier=None)

    async def get(self, *, task_id: str) -> Any:
        self.get_calls.append(task_id)
        if self._next_results:
            return self._next_results.pop(0)
        return SimpleNamespace(id=task_id, model="doubao-seedance-2-0-260128", status="running")

    def queue(self, result: Any) -> None:
        self._next_results.append(result)


class FakeArk:
    def __init__(self) -> None:
        self.content_generation = SimpleNamespace(tasks=FakeTasks())
        self.images = SimpleNamespace(generate=self._img_generate)
        self.image_calls: list[dict[str, Any]] = []

    async def _img_generate(self, **kwargs: Any) -> Any:
        self.image_calls.append(kwargs)
        return SimpleNamespace(
            data=[SimpleNamespace(url="https://ark.test/img.png", b64_json=None)],
            usage=SimpleNamespace(total_tokens=12),
            created_at=1700000000,
        )


@pytest.fixture
def provider() -> VolcengineProvider:
    p = VolcengineProvider(BackendConfig(name="volcengine", type="volcengine", api_key="ark-key"))
    ark = FakeArk()
    p._ark = ark
    p._ark_video = ark  # same fake serves image + video in the basic tests
    return p


def test_provider_requires_api_key() -> None:
    with pytest.raises(ProviderNotConfiguredError):
        VolcengineProvider(BackendConfig(name="volcengine", type="volcengine"))


def test_split_sync_async_base_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Image (Seedream) and video (Seedance) get separate Ark clients on their
    respective sync/async base URLs (``*_IMAGE_BASE_URL`` / ``*_VIDEO_BASE_URL``,
    the latter carried via ``extra["video_base_url"]`` when it differs)."""
    captured: list[dict[str, Any]] = []

    class CapturingArk:
        def __init__(self, **kwargs: Any) -> None:
            captured.append(kwargs)
            self.content_generation = SimpleNamespace(tasks=FakeTasks())
            self.images = SimpleNamespace(generate=self._img)

        async def _img(self, **kwargs: Any) -> Any:
            return SimpleNamespace(data=[], usage=None, created_at=0)

    monkeypatch.setattr(volc_mod, "AsyncArk", CapturingArk)
    VolcengineProvider(
        BackendConfig(
            name="volcengine",
            type="volcengine",
            api_key="ark-key",
            base_url="https://image.test",
            extra={"video_base_url": "https://video.test"},
        )
    )
    assert [c["base_url"] for c in captured] == ["https://image.test", "https://video.test"]


def test_single_base_url_when_not_split(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a separate video base, both clients share the image base."""
    captured: list[dict[str, Any]] = []

    class CapturingArk:
        def __init__(self, **kwargs: Any) -> None:
            captured.append(kwargs)

    monkeypatch.setattr(volc_mod, "AsyncArk", CapturingArk)
    VolcengineProvider(
        BackendConfig(
            name="volcengine", type="volcengine", api_key="ark-key",
            base_url="https://ark.test",
        )
    )
    assert [c["base_url"] for c in captured] == ["https://ark.test", "https://ark.test"]


# -- image --------------------------------------------------------------- #

def test_image_generate_maps_params_and_response(provider: VolcengineProvider) -> None:
    req = UnifiedImageRequest(model="doubao-seedream-4-0-t2i-250828", content=[image_text_part("a cat")],
                              size="1024x1024", seed=7, guidance_scale=2.5, watermark=True)
    task = asyncio.run(provider.create_image_task(req))
    assert task.status == "pending" and task.task_id
    # The synthetic task runs the blocking call on first poll.
    task = asyncio.run(provider.get_image_task(task.task_id))
    assert task.status == "succeeded"
    assert task.images[0].url == "https://ark.test/img.png"
    assert task.provider == "volcengine"
    assert task.usage is not None and task.usage.total_tokens == 12
    kw = provider._ark.image_calls[0]
    assert kw["model"] == "doubao-seedream-4-0-t2i-250828"
    assert kw["prompt"] == "a cat"
    assert kw["size"] == "1024x1024"
    assert kw["seed"] == 7 and kw["guidance_scale"] == 2.5 and kw["watermark"] is True


def test_image_propagates_sdk_error(provider: VolcengineProvider) -> None:
    async def boom(**kw: Any) -> Any:
        raise RuntimeError("upstream down")
    provider._ark.images.generate = boom
    task = asyncio.run(provider.create_image_task(UnifiedImageRequest(model="m", content=[image_text_part("x")])))
    with pytest.raises(ProviderRequestError):
        asyncio.run(provider.get_image_task(task.task_id))


# -- video create -------------------------------------------------------- #

def test_video_create_t2v_content(provider: VolcengineProvider) -> None:
    req = UnifiedVideoRequest(
        model="doubao-seedance-2-0-260128", content=[text_part("a cat playing")],
        duration=11, ratio="16:9", generate_audio=True, watermark=True,
    )
    task = asyncio.run(provider.create_video_task(req))
    assert task.task_id == "ark-task-1" and task.status == "pending"
    kw = provider._ark.content_generation.tasks.create_calls[0]
    assert kw["model"] == "doubao-seedance-2-0-260128"
    assert kw["content"] == [{"type": "text", "text": "a cat playing"}]
    assert kw["duration"] == 11 and kw["ratio"] == "16:9"
    assert kw["generate_audio"] is True and kw["watermark"] is True


def test_video_create_i2v_first_and_last_frame(provider: VolcengineProvider) -> None:
    req = UnifiedVideoRequest(
        model="doubao-seedance-2-0-260128",
        content=[
            text_part("animate"),
            image_part("https://x.test/first.png", "first_frame"),
            image_part("https://x.test/last.png", "last_frame"),
        ],
    )
    asyncio.run(provider.create_video_task(req))
    content = provider._ark.content_generation.tasks.create_calls[0]["content"]
    types = [c["type"] for c in content]
    assert types == ["text", "image_url", "image_url"]
    assert content[1]["role"] == "first_frame" and content[1]["image_url"]["url"] == "https://x.test/first.png"
    assert content[2]["role"] == "last_frame" and content[2]["image_url"]["url"] == "https://x.test/last.png"


def test_video_create_reference_images_videos_audios(provider: VolcengineProvider) -> None:
    req = UnifiedVideoRequest(
        model="doubao-seedance-2-0-260128",
        content=[
            text_part("follow the ref"),
            image_part("https://x.test/r1.png", "reference_image"),
            image_part("https://x.test/r2.png", "reference_image"),
            video_part("https://x.test/rv.mp4"),
            audio_part("https://x.test/ra.mp3"),
        ],
    )
    asyncio.run(provider.create_video_task(req))
    content = provider._ark.content_generation.tasks.create_calls[0]["content"]
    roles = [(c["type"], c.get("role")) for c in content]
    assert ("image_url", "reference_image") in roles
    assert ("video_url", "reference_video") in roles
    assert ("audio_url", "reference_audio") in roles
    assert sum(1 for t, _ in roles if t == "image_url" and _ == "reference_image") == 2


def test_video_create_passes_seedance_knobs(provider: VolcengineProvider) -> None:
    req = UnifiedVideoRequest(
        model="doubao-seedance-2-0-260128", content=[text_part("x")], seed=42, camera_fixed=True,
        width=1920, height=1080, callback_url="https://x.test/cb", return_last_frame=True,
        frame_count=121,
        extra={"service_tier": "default", "priority": 5},
    )
    asyncio.run(provider.create_video_task(req))
    kw = provider._ark.content_generation.tasks.create_calls[0]
    assert kw["seed"] == 42 and kw["camera_fixed"] is True
    assert kw["resolution"] == "1080p" and kw["ratio"] == "16:9"
    assert kw["callback_url"] == "https://x.test/cb"
    assert kw["return_last_frame"] is True and kw["service_tier"] == "default" and kw["priority"] == 5
    assert kw["frames"] == 121


def test_video_create_propagates_sdk_error(provider: VolcengineProvider) -> None:
    async def boom(**kw: Any) -> Any:
        raise RuntimeError("ark 500")
    provider._ark.content_generation.tasks.create = boom
    with pytest.raises(ProviderRequestError):
        asyncio.run(provider.create_video_task(UnifiedVideoRequest(model="m", content=[text_part("x")])))


# -- video poll ---------------------------------------------------------- #

def _task(status: str, **extra: Any) -> Any:
    return SimpleNamespace(
        id="ark-task-1", model="doubao-seedance-2-0-260128", status=status,
        content=extra.get("content"),
        usage=extra.get("usage"),
        error=extra.get("error"),
        created_at=1700000000, updated_at=1700000060,
    )


def test_poll_running_then_succeeded(provider: VolcengineProvider) -> None:
    tasks = provider._ark.content_generation.tasks
    tasks.queue(_task("running"))
    tasks.queue(_task("succeeded", content=SimpleNamespace(video_url="https://x.test/v.mp4", last_frame_url="https://x.test/lf.png"),
                       usage=SimpleNamespace(completion_tokens=100, total_tokens=120)))
    t1 = asyncio.run(provider.get_video_task("ark-task-1"))
    assert t1.status == "running" and not t1.video_urls
    t2 = asyncio.run(provider.get_video_task("ark-task-1"))
    assert t2.status == "succeeded"
    assert t2.video_urls == ["https://x.test/v.mp4"]
    assert t2.cover_url == "https://x.test/lf.png"
    assert t2.usage is not None and t2.usage.extra["total_tokens"] == 120


def test_poll_queued_is_pending(provider: VolcengineProvider) -> None:
    provider._ark.content_generation.tasks.queue(_task("queued"))
    t = asyncio.run(provider.get_video_task("ark-task-1"))
    assert t.status == "pending"


def test_poll_failed_carries_error(provider: VolcengineProvider) -> None:
    provider._ark.content_generation.tasks.queue(
        _task("failed", error=SimpleNamespace(code="RATE_LIMIT", message="too fast")))
    t = asyncio.run(provider.get_video_task("ark-task-1"))
    assert t.status == "failed"
    assert "RATE_LIMIT" in (t.error or "") and "too fast" in (t.error or "")


def test_poll_cancelled(provider: VolcengineProvider) -> None:
    provider._ark.content_generation.tasks.queue(_task("cancelled"))
    t = asyncio.run(provider.get_video_task("ark-task-1"))
    assert t.status == "cancelled"


def test_poll_propagates_sdk_error(provider: VolcengineProvider) -> None:
    async def boom(*, task_id: str) -> Any:
        raise RuntimeError("ark 404")
    provider._ark.content_generation.tasks.get = boom
    with pytest.raises(ProviderRequestError):
        asyncio.run(provider.get_video_task("nope"))
