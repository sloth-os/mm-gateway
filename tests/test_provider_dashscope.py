"""Tests for the DashScope provider (Wanx/Qwen-Image image + Wan video).

The provider talks to the async AIO SDK classes (``AioImageSynthesis`` /
``AioVideoSynthesis``); we swap fakes onto the module so no network is made.
Responses are read via attribute access, so plain ``SimpleNamespace`` objects
stand in for the SDK's DictMixin models.

Image generation tries the **native async task path** first
(``async_call`` submits a real task id, ``wait`` blocks until SUCCEEDED) — the
unrestricted route that serves every model including ``qwen-image-2.0-pro``.
When the backend rejects async submission (403 / "does not support asynchronous
calls") the adapter falls back to ``sync_call`` (the headerless inline path).
Both are wrapped as a synthetic in-memory task by ``SyncImageTaskMixin``:
``create_image_task`` mints a gateway-local ``img-`` id and returns ``pending``;
the first ``get_image_task`` poll runs the blocking generation and moves the
task to ``succeeded``/``failed``. Video stays on the native ``async_call`` /
``fetch`` flow.

Regression note: ``_generate_image`` must not touch ``request.prompt_extend`` —
``UnifiedImageRequest`` has no such field, and the old code raised
``AttributeError`` on every image call.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from mm_gateway.config import BackendConfig
from mm_gateway.core.exceptions import (
    ProviderNotConfiguredError,
    ProviderRequestError,
    TaskFailedError,
)
from mm_gateway.providers import dashscope as dashscope_mod
from mm_gateway.providers.dashscope import DashScopeProvider
from mm_gateway.schemas.image import UnifiedImageRequest
from mm_gateway.schemas.image import text_part as image_text_part
from mm_gateway.schemas.video import UnifiedVideoRequest, image_part, text_part


def _succeeded(url: str = "https://dashscope.test/img.png") -> Any:
    return SimpleNamespace(
        status_code=200,
        code=None,
        message=None,
        output=SimpleNamespace(
            task_status="SUCCEEDED",
            results=[SimpleNamespace(url=url)],
        ),
    )


class FakeImageSynthesis:
    """Fake for the image path. Native async (``async_call``+``wait``) is the
    default; ``sync_call`` is the fallback used when async is rejected.

    ``reject_async`` makes ``async_call`` return a 403 AccessDenied body (the
    shape third-party gateways / keys-without-async-entitlement return), so the
    adapter falls back to ``sync_call``. ``raise_async`` makes ``async_call``
    raise instead (some SDK builds raise rather than returning an error body).
    """

    def __init__(self) -> None:
        self.async_kwargs: dict[str, Any] | None = None
        self.sync_kwargs: dict[str, Any] | None = None
        self.wait_kwargs: dict[str, Any] | None = None
        self.async_calls: int = 0
        self.sync_calls: int = 0
        self.wait_calls: list[str] = []
        self.reject_async: bool = False
        self.raise_async: str | None = None
        self._wait_result: Any = _succeeded()

    async def async_call(self, **kwargs: Any) -> Any:
        self.async_kwargs = kwargs
        self.async_calls += 1
        if self.raise_async is not None:
            raise RuntimeError(self.raise_async)
        if self.reject_async:
            return SimpleNamespace(
                status_code=403,
                code="AccessDenied",
                message="current user api does not support asynchronous calls",
                output=None,
            )
        return SimpleNamespace(status_code=200, output=SimpleNamespace(task_id="ds-task-1"))

    async def wait(self, task_id: str, **kwargs: Any) -> Any:
        self.wait_kwargs = kwargs
        self.wait_calls.append(task_id)
        return self._wait_result

    async def sync_call(self, **kwargs: Any) -> Any:
        self.sync_kwargs = kwargs
        self.sync_calls += 1
        return self._wait_result


class FakeVideoSynthesis:
    def __init__(self) -> None:
        self.call_kwargs: dict[str, Any] | None = None
        self.fetch_calls: list[str] = []
        self._next: list[Any] = []

    async def async_call(self, **kwargs: Any) -> Any:
        self.call_kwargs = kwargs
        return SimpleNamespace(output=SimpleNamespace(task_id="vid-task-1"))

    async def fetch(self, task_id: str, **kwargs: Any) -> Any:
        self.fetch_kwargs = kwargs
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


def test_image_generate_native_async_path(provider: DashScopeProvider) -> None:
    """Default path: native async_call + wait, no sync_call fallback."""
    req = UnifiedImageRequest(
        model="wanx2.1-t2i-turbo",
        content=[image_text_part("a cat")],
        size="1024x1024",
        seed=7,
        n=1,
        negative_prompt="blurry",
    )
    # create_image_task mints a gateway-local id and returns pending; it does
    # NOT call the SDK yet — the generation runs on first poll.
    task = asyncio.run(provider.create_image_task(req))
    assert task.status == "pending" and task.task_id.startswith("img-")
    assert provider._fake_image.async_calls == 0  # type: ignore[attr-defined]
    # The first poll runs async_call + wait (native async) and completes.
    task = asyncio.run(provider.get_image_task(task.task_id))
    img = provider._fake_image  # type: ignore[attr-defined]
    kw = img.async_kwargs
    assert kw["model"] == "wanx2.1-t2i-turbo"
    assert kw["prompt"] == "a cat"
    assert kw["size"] == "1024*1024"  # 'x' -> '*'
    assert kw["seed"] == 7 and kw["n"] == 1 and kw["negative_prompt"] == "blurry"
    # prompt_extend is provider-specific; it is NOT a field on UnifiedImageRequest
    # and must not be read as a direct attribute — it only passes via `extra`.
    assert "prompt_extend" not in kw
    assert task.images[0].url == "https://dashscope.test/img.png"
    assert task.provider == "dashscope" and task.model == "wanx2.1-t2i-turbo"
    assert img.async_calls == 1 and img.wait_calls == ["ds-task-1"]
    assert img.sync_calls == 0  # no fallback
    # A second poll returns the cached terminal task without re-calling.
    asyncio.run(provider.get_image_task(task.task_id))
    assert img.async_calls == 1 and img.sync_calls == 0  # type: ignore[attr-defined]


def test_image_generate_qwen_image_uses_native_async(
    provider: DashScopeProvider,
) -> None:
    """qwen-image-2.0-pro is served via the native async path (sync_call's
    docstring restricts it to wan2.2-t2i-*)."""
    req = UnifiedImageRequest(
        model="qwen-image-2.0-pro", content=[image_text_part("a cat")]
    )
    task = asyncio.run(provider.create_image_task(req))
    task = asyncio.run(provider.get_image_task(task.task_id))
    img = provider._fake_image  # type: ignore[attr-defined]
    assert img.async_calls == 1 and img.sync_calls == 0
    assert img.async_kwargs["model"] == "qwen-image-2.0-pro"  # type: ignore[attr-defined]
    assert task.images[0].url == "https://dashscope.test/img.png"


def test_image_generate_passes_prompt_extend_via_extra(
    provider: DashScopeProvider,
) -> None:
    req = UnifiedImageRequest(
        model="wanx2.1-t2i-turbo",
        content=[image_text_part("a cat")],
        extra={"prompt_extend": True},
    )
    task = asyncio.run(provider.create_image_task(req))
    asyncio.run(provider.get_image_task(task.task_id))
    kw = provider._fake_image.async_kwargs  # type: ignore[attr-defined]
    assert kw["prompt_extend"] is True


def test_image_falls_back_to_sync_when_async_rejected(
    provider: DashScopeProvider,
) -> None:
    """When the backend rejects async submission (403 AccessDenied "does not
    support asynchronous calls") the adapter falls back to sync_call."""
    img = provider._fake_image  # type: ignore[attr-defined]
    img.reject_async = True
    req = UnifiedImageRequest(
        model="wanx2.1-t2i-turbo", content=[image_text_part("a cat")]
    )
    task = asyncio.run(provider.create_image_task(req))
    task = asyncio.run(provider.get_image_task(task.task_id))
    # async_call was attempted and rejected; sync_call completed the work.
    assert img.async_calls == 1 and img.sync_calls == 1
    assert img.sync_kwargs["model"] == "wanx2.1-t2i-turbo"  # type: ignore[attr-defined]
    assert task.status == "succeeded"
    assert task.images[0].url == "https://dashscope.test/img.png"


def test_image_falls_back_to_sync_when_async_raises_rejection(
    provider: DashScopeProvider,
) -> None:
    """An async rejection that *raises* (rather than returning an error body)
    also triggers the sync fallback."""
    img = provider._fake_image  # type: ignore[attr-defined]
    img.raise_async = "AccessDenied: does not support asynchronous calls"
    req = UnifiedImageRequest(
        model="wanx2.1-t2i-turbo", content=[image_text_part("a cat")]
    )
    task = asyncio.run(provider.create_image_task(req))
    task = asyncio.run(provider.get_image_task(task.task_id))
    assert img.async_calls == 1 and img.sync_calls == 1
    assert task.status == "succeeded"


def test_image_failed_status_is_clean_error(provider: DashScopeProvider) -> None:
    """A wait whose response is not SUCCEEDED (e.g. an upstream error or a
    non-DashScope-shaped body when base_url is pointed elsewhere) surfaces as a
    clean TaskFailedError — not an opaque AttributeError on
    ``resp.output.results`` (the original CI failure)."""

    img = provider._fake_image  # type: ignore[attr-defined]
    img._wait_result = SimpleNamespace(
        status_code=400, code="Bad Request", message="no task", output=None
    )
    task = asyncio.run(
        provider.create_image_task(
            UnifiedImageRequest(model="m", content=[image_text_part("x")])
        )
    )
    with pytest.raises(TaskFailedError) as ei:
        asyncio.run(provider.get_image_task(task.task_id))
    msg = str(ei.value)
    # The upstream code/message are preserved in the message for debugging.
    assert "dashscope image task" in msg
    assert "Bad Request" in msg


def test_image_propagates_sdk_error(provider: DashScopeProvider) -> None:
    """A non-rejection error from wait surfaces as ProviderRequestError."""

    async def boom_wait(task_id: str, **kw: Any) -> Any:
        raise RuntimeError("upstream down")

    img = provider._fake_image  # type: ignore[attr-defined]
    img.wait = boom_wait  # type: ignore[method-assign]
    task = asyncio.run(
        provider.create_image_task(
            UnifiedImageRequest(model="m", content=[image_text_part("x")])
        )
    )
    with pytest.raises(ProviderRequestError):
        asyncio.run(provider.get_image_task(task.task_id))


def test_image_propagates_non_rejection_submit_error(
    provider: DashScopeProvider,
) -> None:
    """A submit error that is NOT an async rejection surfaces as
    ProviderRequestError (no silent fallback)."""
    img = provider._fake_image  # type: ignore[attr-defined]
    img.raise_async = "model not found"  # no rejection markers
    task = asyncio.run(
        provider.create_image_task(
            UnifiedImageRequest(model="m", content=[image_text_part("x")])
        )
    )
    with pytest.raises(ProviderRequestError):
        asyncio.run(provider.get_image_task(task.task_id))


def test_image_poll_unknown_task_is_404(provider: DashScopeProvider) -> None:
    with pytest.raises(ProviderRequestError) as ei:
        asyncio.run(provider.get_image_task("no-such-task"))
    assert "404" in str(ei.value) or "not found" in str(ei.value)


def test_split_sync_async_base_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Image calls route at the image (sync) base URL, video calls at the
    video (async) base URL — selected per-call via ``base_address=``, not via
    the racy module-global ``dashscope.base_http_api_url``.

    Mirrors ``config.py``'s resolution: ``base_url`` (image/sync) +
    ``extra["video_base_url"]`` (async) recorded when it differs from the image
    one.
    """
    img = FakeImageSynthesis()
    vid = FakeVideoSynthesis()
    monkeypatch.setattr(dashscope_mod, "AioImageSynthesis", img)
    monkeypatch.setattr(dashscope_mod, "AioVideoSynthesis", vid)
    p = DashScopeProvider(
        BackendConfig(
            name="dashscope",
            type="dashscope",
            api_key="ds-key",
            base_url="https://image.test",
            extra={"video_base_url": "https://video.test"},
        )
    )
    # Image native-async path: async_call + wait both carry the image base.
    itask = asyncio.run(
        p.create_image_task(
            UnifiedImageRequest(model="wanx2.1-t2i-turbo", content=[image_text_part("x")])
        )
    )
    asyncio.run(p.get_image_task(itask.task_id))
    assert img.async_kwargs["base_address"] == "https://image.test"
    assert img.wait_kwargs["base_address"] == "https://image.test"
    # Fallback sync path also carries the image base.
    img.reject_async = True
    itask2 = asyncio.run(
        p.create_image_task(
            UnifiedImageRequest(model="wanx2.1-t2i-turbo", content=[image_text_part("x")])
        )
    )
    asyncio.run(p.get_image_task(itask2.task_id))
    assert img.sync_kwargs["base_address"] == "https://image.test"
    # Video async_call + fetch carry the video base.
    vtask = asyncio.run(
        p.create_video_task(
            UnifiedVideoRequest(model="wanx2.1-t2v-turbo", content=[text_part("x")])
        )
    )
    asyncio.run(p.get_video_task(vtask.task_id))
    assert vid.call_kwargs["base_address"] == "https://video.test"
    assert vid.fetch_kwargs["base_address"] == "https://video.test"


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
