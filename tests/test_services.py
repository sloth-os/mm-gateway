"""Tests for the service-layer error mapping (sync wait, provider errors)."""

from __future__ import annotations

import pytest

from mm_gateway.config import BackendConfig, KeyConfig, Settings
from mm_gateway.core.base import ImageProvider, VideoProvider
from mm_gateway.core.exceptions import GatewayError
from mm_gateway.registry import Registry
from mm_gateway.schemas.image import UnifiedImageRequest
from mm_gateway.schemas.video import UnifiedVideoRequest, text_part
from mm_gateway.services import ImageService, VideoService


class _BoomImage(ImageProvider):
    name = "boom"
    image_models = ["boom-1"]

    def __init__(self):
        super().__init__(BackendConfig(name="boom", type="boom", api_key="k"))

    async def generate_image(self, request):
        raise RuntimeError("upstream exploded")


class _BoomVideo(VideoProvider):
    name = "boomv"
    video_models = ["boom-v1"]

    def __init__(self):
        super().__init__(BackendConfig(name="boomv", type="boomv", api_key="k"))

    async def create_video_task(self, request):
        raise RuntimeError("video upstream exploded")

    async def get_video_task(self, task_id):
        raise RuntimeError("poll exploded")


_KEY = KeyConfig(id="test", key="")


@pytest.fixture
def settings() -> Settings:
    return Settings(
        backends=[
            BackendConfig(name="boom", type="boom", api_key="k"),
            BackendConfig(name="boomv", type="boomv", api_key="k"),
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
    return reg


def test_image_service_wraps_provider_error(registry):
    import asyncio
    svc = ImageService(registry)
    req = UnifiedImageRequest(model="boom-1", prompt="x")
    with pytest.raises(GatewayError) as exc_info:
        asyncio.run(svc.generate(req, key=_KEY))
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
    svc = ImageService(registry)
    # boomv only supports video, not image. Pin it explicitly so resolve returns
    # it, then ImageService rejects it as unsupported.
    req = UnifiedImageRequest(model="boom-v1", prompt="x", provider="boomv")
    with pytest.raises(GatewayError) as exc_info:
        asyncio.run(svc.generate(req, key=_KEY, backend_name="boomv"))
    assert exc_info.value.code == "unsupported_feature"
