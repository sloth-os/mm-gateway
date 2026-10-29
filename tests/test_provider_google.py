"""Tests for the Google provider (Imagen image + Veo video + Lyria music).

The provider builds two ``genai.Client`` instances (image + video) honoring the
sync/async URL split, plus a music REST base. We capture constructor args by
monkeypatching ``genai.Client`` — no network.
"""

from __future__ import annotations

from typing import Any

import pytest

from mm_gateway.config import BackendConfig
from mm_gateway.core.exceptions import ProviderNotConfiguredError
from mm_gateway.providers import google as google_mod
from mm_gateway.providers.google import GoogleProvider, _GLM_BASE


def _backend(base_url: str | None = None, *, extra: dict[str, Any] | None = None,
             api_key: str = "g-key") -> BackendConfig:
    kw: dict[str, Any] = {"name": "google", "type": "google", "api_key": api_key}
    if base_url is not None:
        kw["base_url"] = base_url
    if extra:
        kw["extra"] = extra
    return BackendConfig(**kw)


def _http_options_base(kwargs: dict[str, Any]) -> str | None:
    """Pull the base_url out of an HttpOptions instance (or None)."""
    opts = kwargs.get("http_options")
    if opts is None:
        return None
    return getattr(opts, "base_url", None)


# -- construction -------------------------------------------------------- #


def test_requires_api_key() -> None:
    with pytest.raises(ProviderNotConfiguredError):
        GoogleProvider(BackendConfig(name="google", type="google"))


def test_split_sync_async_base_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Image (Imagen/generate_content) lands on ``base_url`` (sync), video (Veo)
    on ``extra["video_base_url"]`` (async, recorded by ``config.py`` when it
    differs)."""
    captured: list[dict[str, Any]] = []

    class CapturingClient:
        def __init__(self, **kwargs: Any) -> None:
            captured.append(kwargs)

    monkeypatch.setattr(google_mod, "genai", type("G", (), {"Client": CapturingClient}))
    p = GoogleProvider(_backend(
        "https://image.test", extra={"video_base_url": "https://video.test"},
    ))
    assert p._client is not p._client_video
    assert [_http_options_base(c) for c in captured] == ["https://image.test", "https://video.test"]
    assert all(c["api_key"] == "g-key" for c in captured)


def test_single_base_url_when_not_split(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a separate video base, both clients share the image base."""
    captured: list[dict[str, Any]] = []

    class CapturingClient:
        def __init__(self, **kwargs: Any) -> None:
            captured.append(kwargs)

    monkeypatch.setattr(google_mod, "genai", type("G", (), {"Client": CapturingClient}))
    p = GoogleProvider(_backend("https://google.test"))
    assert p._client is not p._client_video
    assert [_http_options_base(c) for c in captured] == ["https://google.test", "https://google.test"]


def test_default_base_url_omits_host_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no base pinned, the SDK default host applies — i.e. the injected
    ``http_options`` carries a ``None`` base_url (no host override). The
    ``http_options`` is still passed so the backend-logging httpx client is
    injected regardless of whether a host is pinned."""
    captured: list[dict[str, Any]] = []

    class CapturingClient:
        def __init__(self, **kwargs: Any) -> None:
            captured.append(kwargs)

    monkeypatch.setattr(google_mod, "genai", type("G", (), {"Client": CapturingClient}))
    GoogleProvider(_backend())
    assert all("http_options" in c for c in captured)
    assert all(_http_options_base(c) is None for c in captured)


def test_music_base_prefers_music_base_url() -> None:
    """Lyria's REST base follows ``music_base_url`` > ``base_url`` > SDK default."""
    p = GoogleProvider(_backend(
        "https://image.test",
        extra={"video_base_url": "https://video.test",
               "music_base_url": "https://music.test"},
    ))
    assert p._music_base == "https://music.test"
    # Falls back to the image base when no music base is pinned.
    p2 = GoogleProvider(_backend("https://image.test"))
    assert p2._music_base == "https://image.test"
    # Falls back to the SDK default host when nothing is pinned.
    p3 = GoogleProvider(_backend())
    assert p3._music_base == _GLM_BASE


# -- Gemini Omni video (Interactions) ------------------------------------ #

import asyncio  # noqa: E402
import base64  # noqa: E402
import json  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import httpx  # noqa: E402

from mm_gateway.schemas.image import UnifiedImageRequest  # noqa: E402
from mm_gateway.schemas.image import image_part as img_image  # noqa: E402
from mm_gateway.schemas.image import text_part as img_text  # noqa: E402
from mm_gateway.schemas.video import (  # noqa: E402
    UnifiedVideoRequest,
    audio_part,
    image_part,
    text_part,
    video_part,
)


def _omni_provider(handler) -> GoogleProvider:
    p = GoogleProvider(_backend())
    p._client_interactions = httpx.AsyncClient(
        base_url=_GLM_BASE, transport=httpx.MockTransport(handler), headers={"x-goog-api-key": "g-key"})
    return p


def test_omni_create_posts_a_background_interaction() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"id": "v1_abc", "status": "in_progress"})

    p = _omni_provider(handler)
    task = asyncio.run(p.create_video_task(UnifiedVideoRequest(
        model="gemini-omni-1.1-flash",
        content=[text_part("<FIRST_FRAME> <LAST_FRAME> a door opens"),
                 image_part("data:image/png;base64,AAAA", role="last_frame"),
                 image_part("https://example.test/first.jpg", role="first_frame"),
                 audio_part("https://example.test/voice.mp3")],
        duration=14, width=1920, height=1080, negative_prompt="blur",
    )))
    assert task.task_id == "interactions/v1_abc" and task.status == "running"
    request = seen[0]
    assert request.method == "POST" and request.url.path == "/v1beta/interactions"
    assert request.headers["x-goog-api-key"] == "g-key"
    body = json.loads(request.content)
    assert body["model"] == "gemini-omni-1.1-flash"
    assert body["background"] is True and body["store"] is True
    # Frames come first (first, then last); Omni takes no audio input and no negative prompt.
    assert body["input"] == [
        {"type": "text", "text": "<FIRST_FRAME> <LAST_FRAME> a door opens"},
        {"type": "image", "uri": "https://example.test/first.jpg"},
        {"type": "image", "data": "AAAA", "mime_type": "image/png"},
    ]
    assert body["response_format"] == {"type": "video", "aspect_ratio": "16:9", "resolution": "1080p",
                                       "duration": "10s"}
    assert body["generation_config"] == {"video_config": {"task": "image_to_video"}}
    assert "negative_prompt" not in json.dumps(body)


def test_omni_reference_request_and_unsupported_ratio() -> None:
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "v1_ref"})

    p = _omni_provider(handler)
    task = asyncio.run(p.create_video_task(UnifiedVideoRequest(
        model="gemini-omni-1.1-flash", ratio="1:1",
        content=[text_part("<IMAGE_REF_0> dances"), image_part("https://example.test/a.png", role="reference_image"),
                 video_part("https://example.test/clip.mp4")],
    )))
    assert task.status == "pending"
    body = bodies[0]
    # 1:1 is not an Omni ratio, so the model's default applies.
    assert body["response_format"] == {"type": "video"}
    assert body["generation_config"] == {"video_config": {"task": "reference_to_video"}}
    assert [part["type"] for part in body["input"]] == ["text", "image", "video"]


def test_omni_poll_returns_the_inline_video() -> None:
    video = base64.b64encode(b"mp4-bytes").decode()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET" and request.url.path == "/v1beta/interactions/v1_abc"
        return httpx.Response(200, json={
            "id": "v1_abc", "status": "completed", "model": "gemini-omni-1.1-flash",
            "steps": [{"type": "model_output", "content": [
                {"type": "text", "text": "Here is your video."},
                {"type": "video", "data": video, "mime_type": "video/mp4"},
            ]}],
        })

    task = asyncio.run(_omni_provider(handler).get_video_task("interactions/v1_abc"))
    assert task.status == "succeeded"
    assert task.model == "gemini-omni-1.1-flash"
    assert task.video_urls == [f"data:video/mp4;base64,{video}"]


def test_omni_poll_reports_failures() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "v1_bad", "status": "failed",
                                         "errors": [{"code": "SAFETY", "message": "blocked"}]})

    task = asyncio.run(_omni_provider(handler).get_video_task("interactions/v1_bad"))
    assert task.status == "failed" and task.error == "SAFETY: blocked"

    def empty(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "v1_empty", "status": "completed", "steps": []})

    task = asyncio.run(_omni_provider(empty).get_video_task("interactions/v1_empty"))
    assert task.status == "failed" and "no video" in task.error


def test_veo_tasks_keep_the_operations_path(monkeypatch: pytest.MonkeyPatch) -> None:
    p = GoogleProvider(_backend())

    async def get(op):
        return SimpleNamespace(done=False, response=None, error=None)

    monkeypatch.setattr(p._client_video.aio.operations, "get", get)
    task = asyncio.run(p.get_video_task("models/veo-3.1-generate-preview/operations/op1"))
    assert task.status == "running"


# -- Gemini image inputs ------------------------------------------------- #


def test_gemini_image_sends_input_images_and_image_size(monkeypatch: pytest.MonkeyPatch) -> None:
    p = GoogleProvider(_backend())
    calls: list[dict[str, Any]] = []

    async def generate_content(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(candidates=[])

    monkeypatch.setattr(p._client.aio.models, "generate_content", generate_content)
    request = UnifiedImageRequest(
        model="gemini-nano-banana-2.1", resolution="2k", aspect_ratio="4:1",
        content=[img_text("put the cat on the sofa"),
                 img_image(data=base64.b64encode(b"cat").decode(), mime_type="image/jpeg"),
                 img_image("data:image/png;base64," + base64.b64encode(b"sofa").decode()),
                 img_image("https://example.test/room.webp")],
    )
    asyncio.run(p._generate_content_image(request))
    contents = calls[0]["contents"]
    assert contents[0] == "put the cat on the sofa"
    assert contents[1].inline_data.data == b"cat" and contents[1].inline_data.mime_type == "image/jpeg"
    assert contents[2].inline_data.data == b"sofa" and contents[2].inline_data.mime_type == "image/png"
    assert contents[3].file_data.file_uri == "https://example.test/room.webp"
    image_config = calls[0]["config"].image_config
    assert image_config.image_size == "2K" and image_config.aspect_ratio == "4:1"

    # A text-only request keeps sending the bare prompt.
    asyncio.run(p._generate_content_image(UnifiedImageRequest(
        model="gemini-3-pro-image", content=[img_text("a lighthouse")])))
    assert calls[1]["contents"] == "a lighthouse"


# -- Veo input images ---------------------------------------------------- #


def test_veo_sends_frames_and_reference_images_as_sdk_images(monkeypatch: pytest.MonkeyPatch) -> None:
    p = GoogleProvider(_backend())
    calls: list[dict[str, Any]] = []

    async def generate_videos(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(name="models/veo-3.1-generate-001/operations/op9", response=None)

    monkeypatch.setattr(p._client_video.aio.models, "generate_videos", generate_videos)
    png = base64.b64encode(b"first").decode()
    task = asyncio.run(p.create_video_task(UnifiedVideoRequest(
        model="veo-3.1-generate-001", duration=8,
        content=[text_part("a fox"), image_part(f"data:image/png;base64,{png}"),
                 image_part("gs://bucket/last.jpg", role="last_frame"),
                 image_part("gs://bucket/fox.png", role="reference_image")],
    )))
    assert task.task_id == "models/veo-3.1-generate-001/operations/op9"
    source, config = calls[0]["source"], calls[0]["config"]
    assert source.image.image_bytes == b"first" and source.image.mime_type == "image/png"
    assert config.last_frame.gcs_uri == "gs://bucket/last.jpg"
    assert config.last_frame.mime_type == "image/jpeg"
    [reference] = config.reference_images
    assert reference.image.gcs_uri == "gs://bucket/fox.png"
    assert reference.reference_type == "ASSET"


def test_veo_fetches_http_images(monkeypatch: pytest.MonkeyPatch) -> None:
    p = GoogleProvider(_backend())

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://example.test/frame.webp"
        return httpx.Response(200, content=b"webp-bytes", headers={"content-type": "image/webp"})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient",
                        lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    image = asyncio.run(p._video_image("https://example.test/frame.webp"))
    assert image.image_bytes == b"webp-bytes" and image.mime_type == "image/webp"
