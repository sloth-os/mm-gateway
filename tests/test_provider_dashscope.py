"""Tests for the DashScope provider (Wanx image + Wan video).

The provider talks to the async AIO SDK classes (``AioImageSynthesis`` /
``AioVideoSynthesis``); we swap fakes onto the module so no network is made.
Responses are read via attribute access, so plain ``SimpleNamespace`` objects
stand in for the SDK's DictMixin models.

Regression note: ``generate_image`` must not touch ``request.prompt_extend`` —
``UnifiedImageRequest`` has no such field, and the old code raised
``AttributeError`` on every image call.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from mm_gateway.config import BackendConfig
from mm_gateway.core.exceptions import ProviderNotConfiguredError, ProviderRequestError
from mm_gateway.providers import dashscope as dashscope_mod
from mm_gateway.providers.dashscope import DashScopeProvider
from mm_gateway.schemas.image import UnifiedImageRequest
from mm_gateway.schemas.video import UnifiedVideoRequest, image_part, text_part


class FakeImageSynthesis:
    def __init__(self) -> None:
        self.call_kwargs: dict[str, Any] | None = None
        self.wait_input: Any = None

    async def async_call(self, **kwargs: Any) -> Any:
        self.call_kwargs = kwargs
        return SimpleNamespace(output=SimpleNamespace(task_id="img-task-1"))

    async def wait(self, resp: Any, wait_timeout: int = 300) -> Any:
        self.wait_input = resp
        return SimpleNamespace(
            status_code=200,
            output=SimpleNamespace(
                task_status="SUCCEEDED",
                results=[SimpleNamespace(url="https://dashscope.test/img.png")],
            ),
        )


class FakeVideoSynthesis:
    def __init__(self) -> None:
        self.call_kwargs: dict[str, Any] | None = None
        self.fetch_calls: list[str] = []
        self._next: list[Any] = []

    async def async_call(self, **kwargs: Any) -> Any:
        self.call_kwargs = kwargs
        return SimpleNamespace(output=SimpleNamespace(task_id="vid-task-1"))

    async def fetch(self, task_id: str) -> Any:
        self.fetch_calls.append(task_id)
        if self._next:
            return self._next.pop(0)
        return SimpleNamespace(
            output=SimpleNamespace(task_status="RUNNING", model="wanx2.1-t2v-turbo")
        )

    def queue(self, result: Any) -> None:
        self._next.append(result)


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> DashScopeProvider:
    img = FakeImageSynthesis()
    vid = FakeVideoSynthesis()
    monkeypatch.setattr(dashscope_mod, "AioImageSynthesis", img)
    monkeypatch.setattr(dashscope_mod, "AioVideoSynthesis", vid)
    p = DashScopeProvider(
        BackendConfig(name="dashscope", type="dashscope", api_key="ds-key")
    )
    p._fake_image = img  # type: ignore[attr-defined]
    p._fake_video = vid  # type: ignore[attr-defined]
    return p


def test_provider_requires_api_key() -> None:
    with pytest.raises(ProviderNotConfiguredError):
        DashScopeProvider(BackendConfig(name="dashscope", type="dashscope"))


def test_image_generate_maps_params_and_response(provider: DashScopeProvider) -> None:
    req = UnifiedImageRequest(
        model="wanx2.1-t2i-turbo",
        prompt="a cat",
        size="1024x1024",
        seed=7,
        n=1,
        negative_prompt="blurry",
    )
    resp = asyncio.run(provider.generate_image(req))
    kw = provider._fake_image.call_kwargs  # type: ignore[attr-defined]
    assert kw["model"] == "wanx2.1-t2i-turbo"
    assert kw["prompt"] == "a cat"
    assert kw["size"] == "1024*1024"  # 'x' -> '*'
    assert kw["seed"] == 7 and kw["n"] == 1 and kw["negative_prompt"] == "blurry"
    # prompt_extend is provider-specific; it is NOT a field on UnifiedImageRequest
    # and must not be read as a direct attribute — it only passes via `extra`.
    assert "prompt_extend" not in kw
    assert resp.data[0].url == "https://dashscope.test/img.png"
    assert resp.provider == "dashscope" and resp.model == "wanx2.1-t2i-turbo"


def test_image_generate_passes_prompt_extend_via_extra(
    provider: DashScopeProvider,
) -> None:
    req = UnifiedImageRequest(
        model="wanx2.1-t2i-turbo", prompt="a cat", extra={"prompt_extend": True}
    )
    asyncio.run(provider.generate_image(req))
    kw = provider._fake_image.call_kwargs  # type: ignore[attr-defined]
    assert kw["prompt_extend"] is True


def test_image_submit_no_task_id_is_clean_error(provider: DashScopeProvider) -> None:
    """A non-DashScope-shaped submit response (e.g. base_url pointed at an
    OpenAI-compatible endpoint) leaves ``output`` unset. This must surface as a
    clear ProviderRequestError — not an opaque AttributeError on
    ``resp.output.task_id`` (the original CI failure)."""
    async def empty(**kw: Any) -> Any:
        # Mirrors the SDK response shape when the upstream returns an error or a
        # non-task body: top-level fields set, output missing/None.
        return SimpleNamespace(status_code=400, code="Bad Request", message="no task", output=None)

    provider._fake_image.async_call = empty  # type: ignore[attr-defined]
    with pytest.raises(ProviderRequestError) as ei:
        asyncio.run(provider.generate_image(UnifiedImageRequest(model="m", prompt="x")))
    msg = str(ei.value)
    assert "no task_id" in msg
    assert "Bad Request" in msg  # the upstream code/message are included for debugging


def test_image_propagates_sdk_error(provider: DashScopeProvider) -> None:
    async def boom(**kw: Any) -> Any:
        raise RuntimeError("upstream down")

    provider._fake_image.async_call = boom  # type: ignore[attr-defined]
    with pytest.raises(ProviderRequestError):
        asyncio.run(provider.generate_image(UnifiedImageRequest(model="m", prompt="x")))


def test_video_create_maps_params_and_response(provider: DashScopeProvider) -> None:
    req = UnifiedVideoRequest(
        model="wanx2.1-t2v-turbo",
        content=[text_part("a cat playing"), image_part("https://x.test/f.png")],
        ratio="16:9",
        duration=5,
        seed=42,
        prompt_extend=True,
    )
    task = asyncio.run(provider.create_video_task(req))
    kw = provider._fake_video.call_kwargs  # type: ignore[attr-defined]
    assert (
        task.task_id == "vid-task-1"
        and task.status == "pending"
        and task.model == "wanx2.1-t2v-turbo"
    )
    assert kw["prompt"] == "a cat playing"
    assert kw["img_url"] == "https://x.test/f.png"
    assert kw["ratio"] == "16:9" and kw["duration"] == 5 and kw["seed"] == 42
    assert kw["prompt_extend"] is True


def test_video_poll_reads_model_from_response(provider: DashScopeProvider) -> None:
    provider._fake_video.queue(  # type: ignore[attr-defined]
        SimpleNamespace(
            output=SimpleNamespace(
                task_status="SUCCEEDED",
                video_url="https://dashscope.test/v.mp4",
                model="wanx2.1-t2v-plus",
            )
        )
    )
    task = asyncio.run(provider.get_video_task("vid-task-1"))
    assert task.status == "succeeded"
    assert task.model == "wanx2.1-t2v-plus"
    assert task.video_urls == ["https://dashscope.test/v.mp4"]


def test_video_poll_failed(provider: DashScopeProvider) -> None:
    provider._fake_video.queue(  # type: ignore[attr-defined]
        SimpleNamespace(output=SimpleNamespace(task_status="FAILED", message="boom"))
    )
    task = asyncio.run(provider.get_video_task("vid-task-1"))
    assert task.status == "failed" and task.error == "boom"
