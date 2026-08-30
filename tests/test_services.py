"""Tests for the service-layer error mapping (sync wait, provider errors)."""

from __future__ import annotations

import pytest

from mm_gateway.config import BackendConfig, KeyConfig, Settings
from mm_gateway.core.base import ImageProvider, MusicProvider, VideoProvider
from mm_gateway.core.exceptions import GatewayError
from mm_gateway.registry import Registry
from mm_gateway.schemas.image import UnifiedImageRequest, text_part as image_text_part
from mm_gateway.schemas.music import (
    UnifiedMusicRequest,
    UnifiedMusicTask,
    text_part as music_text_part,
)
from mm_gateway.schemas.video import UnifiedVideoRequest, text_part
from mm_gateway.services import ImageService, MusicService, VideoService


class _BoomImage(ImageProvider):
    name = "boom"
    image_models = ["boom-1"]

    def __init__(self):
        super().__init__(BackendConfig(name="boom", type="boom", api_key="k"))

    async def create_image_task(self, request, *, sync=None):
        raise RuntimeError("upstream exploded")

    async def get_image_task(self, task_id):  # pragma: no cover - not reached
        raise RuntimeError("poll exploded")


class _BoomVideo(VideoProvider):
    name = "boomv"
    video_models = ["boom-v1"]

    def __init__(self):
        super().__init__(BackendConfig(name="boomv", type="boomv", api_key="k"))

    async def create_video_task(self, request):
        raise RuntimeError("video upstream exploded")

    async def get_video_task(self, task_id):
        raise RuntimeError("poll exploded")


class _BoomMusic(MusicProvider):
    name = "boom_m"
    music_models = ["boom-m1"]

    def __init__(self):
        super().__init__(BackendConfig(name="boom_m", type="boom_m", api_key="k"))

    async def create_music_task(self, request):
        raise RuntimeError("music upstream exploded")

    async def get_music_task(self, task_id):
        raise RuntimeError("music poll exploded")


_KEY = KeyConfig(id="test", key="")


@pytest.fixture
def settings() -> Settings:
    return Settings(
        backends=[
            BackendConfig(name="boom", type="boom", api_key="k"),
            BackendConfig(name="boomv", type="boomv", api_key="k"),
            BackendConfig(name="boom_m", type="boom_m", api_key="k"),
        ],
        keys=[_KEY],
    )


@pytest.fixture
def registry(settings):
    reg = Registry(settings)
    reg._backends["boom"] = _BoomImage()
    reg._configs["boom"] = settings.backends[0]
    reg._backends["boomv"] = _BoomVideo()
    reg._configs["boomv"] = settings.backends[1]
    reg._backends["boom_m"] = _BoomMusic()
    reg._configs["boom_m"] = settings.backends[2]
    return reg


def test_image_service_wraps_provider_error(registry):
    import asyncio
    svc = ImageService(registry, max_sync_wait=1.0, poll_interval=0.01, sync_default=False)
    req = UnifiedImageRequest(model="boom-1", content=[image_text_part("x")])
    with pytest.raises(GatewayError) as exc_info:
        asyncio.run(svc.create(req, key=_KEY))
    assert exc_info.value.code == "provider_error"
    assert exc_info.value.status_code == 502
    assert exc_info.value.provider == "boom"


def test_video_service_wraps_create_error(registry):
    import asyncio
    svc = VideoService(registry, max_sync_wait=1.0, poll_interval=0.01, sync_default=False)
    req = UnifiedVideoRequest(model="boom-v1", content=[text_part("x")])
    with pytest.raises(GatewayError) as exc_info:
        asyncio.run(svc.create(req, key=_KEY, backend_name="boomv"))
    assert exc_info.value.code == "provider_error"
    assert exc_info.value.provider == "boomv"


def test_image_service_unsupported_feature(registry):
    import asyncio
    svc = ImageService(registry, max_sync_wait=1.0, poll_interval=0.01, sync_default=False)
    # boomv only supports video, not image. Pin it explicitly so resolve returns
    # it, then ImageService rejects it as unsupported.
    req = UnifiedImageRequest(model="boom-v1", content=[image_text_part("x")], provider="boomv")
    with pytest.raises(GatewayError) as exc_info:
        asyncio.run(svc.create(req, key=_KEY, backend_name="boomv"))
    assert exc_info.value.code == "unsupported_feature"


def test_music_service_wraps_create_error(registry):
    import asyncio

    from mm_gateway.schemas.music import text_part as m_text_part
    svc = MusicService(registry, max_sync_wait=1.0, poll_interval=0.01, sync_default=False)
    req = UnifiedMusicRequest(model="boom-m1", content=[m_text_part("x")])
    with pytest.raises(GatewayError) as exc_info:
        asyncio.run(svc.create(req, key=_KEY, backend_name="boom_m"))
    assert exc_info.value.code == "provider_error"
    assert exc_info.value.provider == "boom_m"


def test_music_service_unsupported_feature(registry):
    import asyncio
    svc = MusicService(registry, max_sync_wait=1.0, poll_interval=0.01, sync_default=False)
    # boomv only supports video, not music. Pin it explicitly so resolve returns
    # it, then MusicService rejects it as unsupported.
    req = UnifiedMusicRequest(model="boom-v1")
    with pytest.raises(GatewayError) as exc_info:
        asyncio.run(svc.create(req, key=_KEY, backend_name="boomv"))
    assert exc_info.value.code == "unsupported_feature"


async def test_music_service_status_uses_cached_snapshot_during_slow_provider_poll():
    """Integration: service create -> monitor -> non-blocking service status."""
    import asyncio

    class SlowMusic(MusicProvider):
        name = "slow_m"
        music_models = ["slow-m1"]

        def __init__(self):
            super().__init__(BackendConfig(name=self.name, type=self.name, api_key="k"))
            self.poll_started = asyncio.Event()
            self.release = asyncio.Event()

        async def create_music_task(self, request):
            return UnifiedMusicTask(
                task_id="slow-task", provider=self.name, model=request.model,
                status="pending",
            )

        async def get_music_task(self, task_id):
            self.poll_started.set()
            await self.release.wait()
            return UnifiedMusicTask(
                task_id=task_id, provider=self.name, model="slow-m1",
                status="succeeded", audio_b64="AAAA",
            )

    settings = Settings(
        backends=[BackendConfig(name="slow_m", type="slow_m", api_key="k")],
        keys=[_KEY],
    )
    reg = Registry(settings)
    provider = SlowMusic()
    reg._backends[provider.name] = provider
    reg._configs[provider.name] = settings.backends[0]
    svc = MusicService(
        reg, max_sync_wait=1.0, poll_interval=0.001, sync_default=False,
    )

    created = await svc.create(
        UnifiedMusicRequest(model="slow-m1", content=[music_text_part("slow song")]),
        key=_KEY,
        backend_name=provider.name,
        wait=False,
    )
    await asyncio.wait_for(provider.poll_started.wait(), timeout=0.1)

    # The provider poll is still blocked, but get() reads the supervisor cache.
    current = await asyncio.wait_for(
        svc.get(created.task_id, backend_name=provider.name), timeout=0.05,
    )
    assert current.status == "pending"

    provider.release.set()
    completed = await svc._supervisor.wait_for_terminal(
        created.task_id, provider=provider.name, timeout=0.1,
    )
    assert completed is not None
    assert completed.status == "succeeded"
    await svc.aclose()
