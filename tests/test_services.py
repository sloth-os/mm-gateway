"""Tests for the service-layer error mapping (sync wait, provider errors)."""

from __future__ import annotations

import pytest

from mm_gateway.config import BackendConfig, KeyConfig, Settings
from mm_gateway.core.base import ImageProvider, MusicProvider, VideoProvider
from mm_gateway.core.exceptions import GatewayError
from mm_gateway.registry import Registry
from mm_gateway.schemas.image import UnifiedImageRequest, text_part as image_text_part
from mm_gateway.schemas.music import UnifiedMusicRequest
from mm_gateway.schemas.video import UnifiedVideoRequest, text_part
from mm_gateway.services import ImageService, MusicService, VideoService


class _BoomImage(ImageProvider):
    name = "boom"
    image_models = ["boom-1"]

    def __init__(self):
        super().__init__(BackendConfig(name="boom", type="boom", api_key="k"))

    async def create_image_task(self, request):
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
