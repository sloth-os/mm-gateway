"""Tests for the DashScope provider (Wanx/Qwen-Image image + Wan video).

The provider talks to the async AIO SDK classes (``AioImageSynthesis`` /
``AioImageGeneration`` / ``AioVideoSynthesis``); we swap fakes onto the module
so no network is made. Responses are read via attribute access, so plain
``SimpleNamespace`` objects stand in for the SDK's DictMixin models.

DashScope serves two image surfaces, and the adapter routes by model family:

* **Wan2.x-image generation** (``wan2.7-image`` / ``wan2.6-image``) lives on the
  **multimodal-generation** surface (``AioImageGeneration``) — a messages-shaped
  body, with finished images in ``output.choices[].message.content[]``.
* **Wanx2.1 / Qwen-Image synthesis** (``wanx2.1-t2i-*``, ``qwen-image-2.0-pro``)
  stays on the **image-synthesis** surface (``AioImageSynthesis``) — a
  prompt-shaped body, with finished images in ``output.results[].url``.

Both surfaces try the **native async task path** by default (``async_call``
submits a real task id, ``wait`` blocks until SUCCEEDED) — the unrestricted
route that serves every model including ``qwen-image-2.0-pro``. When the
backend rejects async submission (403 / "does not support asynchronous calls")
the adapter falls back to the synchronous inline path (``sync_call`` on the
synthesis surface, ``call`` on the generation surface — both headerless). When
the front end asks to block (``sync=True`` from ``wait=true`` or the
``image_sync_default`` setting) the adapter uses the synchronous inline path
*directly*, skipping async submit/wait — important on third-party
DashScope-compatible gateways whose async task poll endpoint is broken. Both
are wrapped as a synthetic in-memory task by ``SyncImageTaskMixin``:
``create_image_task`` mints a gateway-local ``img-`` id and returns
``pending``; the first ``get_image_task`` poll runs the blocking generation and
moves the task to ``succeeded``/``failed``. Video stays on the native
``async_call`` / ``fetch`` flow, which is unaffected.

Regression note: ``_generate_image`` must not touch ``request.prompt_extend`` —
``UnifiedImageRequest`` has no such field, and the old code raised
``AttributeError`` on every image call.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import httpx
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
from mm_gateway.schemas.image import image_part as image_image_part
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
        return SimpleNamespace(
            status_code=200, output=SimpleNamespace(task_id="ds-task-1")
        )

    async def wait(self, task_id: str, **kwargs: Any) -> Any:
        self.wait_kwargs = kwargs
        self.wait_calls.append(task_id)
        return self._wait_result

    async def sync_call(self, **kwargs: Any) -> Any:
        self.sync_kwargs = kwargs
        self.sync_calls += 1
        return self._wait_result


def _generation_succeeded(url: str = "https://dashscope.test/gen.png") -> Any:
    """A successful multimodal-generation response — image in choices[].content."""
    return SimpleNamespace(
        status_code=200,
        code=None,
        message=None,
        output=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=[{"image": url}],
                    ),
                ),
            ],
        ),
    )


class FakeImageGeneration:
    """Fake for the multimodal-generation surface (``AioImageGeneration``).

    Mirrors ``FakeImageSynthesis`` but on the messages-shaped surface:
    ``async_call``+``wait`` is the native path, ``call`` is the sync fallback,
    and the finished image rides ``output.choices[].message.content[]`` as a
    ``{"image": url}`` item (matching the SDK's ``ImageGenerationResponse``).
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
        self._wait_result: Any = _generation_succeeded()

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
        return SimpleNamespace(
            status_code=200, output=SimpleNamespace(task_id="gen-task-1")
        )

    async def wait(self, task_id: str, **kwargs: Any) -> Any:
        self.wait_kwargs = kwargs
        self.wait_calls.append(task_id)
        return self._wait_result

    async def call(self, **kwargs: Any) -> Any:
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
    gen = FakeImageGeneration()
    vid = FakeVideoSynthesis()
    monkeypatch.setattr(dashscope_mod, "AioImageSynthesis", img)
    monkeypatch.setattr(dashscope_mod, "AioImageGeneration", gen)
    monkeypatch.setattr(dashscope_mod, "AioVideoSynthesis", vid)
    p = DashScopeProvider(
        BackendConfig(name="dashscope", type="dashscope", api_key="ds-key")
    )
    p._fake_image = img  # type: ignore[attr-defined]
    p._fake_image_gen = gen  # type: ignore[attr-defined]
    p._fake_video = vid  # type: ignore[attr-defined]
    return p


def _mount_video(provider: DashScopeProvider, handler) -> list[str]:
    """Replace the provider's video-poll httpx client with a ``MockTransport``
    driven by ``handler``; return the list of polled URLs (captured in order).

    The video *poll* is hand-rolled over httpx (see ``get_video_task``): it no
    longer goes through the SDK's ``AioVideoSynthesis.fetch``, so video-poll
    tests script the response here rather than queueing ``_fake_video`` results.
    The client carries no base_url — the adapter builds the full slashless URL.
    """
    captured: list[str] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        captured.append(str(request.url))
        return handler(request)

    provider._client_video = httpx.AsyncClient(transport=httpx.MockTransport(_handler))
    return captured


def test_provider_requires_api_key() -> None:
    with pytest.raises(ProviderNotConfiguredError):
        DashScopeProvider(BackendConfig(name="dashscope", type="dashscope"))


def test_image_generate_native_async_path(provider: DashScopeProvider) -> None:
    """Default path: native async_call + wait, no sync_call fallback."""
    req = UnifiedImageRequest(
        model="wanx2.1-t2i-turbo",
        content=[image_text_part("a cat")],
        width=1024,
        height=1024,
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


def test_image_sync_intent_uses_sync_inline(provider: DashScopeProvider) -> None:
    """A ``sync=True`` create goes straight to the synchronous inline
    ``sync_call`` (``BaseAioApi.call``, no ``X-DashScope-Async`` header) and
    never touches the native async submit/wait path.

    This is the ``wait=true`` path: the front end asked to block, so the adapter
    must use DashScope's synchronous API — not submit an async task and poll it
    (which is the path that breaks on third-party gateways whose task poll
    endpoint returns 500 SYSTEM_ERROR).
    """
    img = provider._fake_image  # type: ignore[attr-defined]
    req = UnifiedImageRequest(
        model="wanx2.1-t2i-turbo", content=[image_text_part("a cat")]
    )
    task = asyncio.run(provider.create_image_task(req, sync=True))
    assert task.status == "pending" and img.async_calls == 0 and img.sync_calls == 0
    task = asyncio.run(provider.get_image_task(task.task_id))
    # sync_call used directly; native async submit/wait never attempted.
    assert img.async_calls == 0 and img.wait_calls == []
    assert img.sync_calls == 1
    assert img.sync_kwargs["model"] == "wanx2.1-t2i-turbo"
    assert img.sync_kwargs["base_address"] is None  # default image base (no override)
    assert task.status == "succeeded"
    assert task.images[0].url == "https://dashscope.test/img.png"


def test_image_sync_intent_is_native_async_when_false(
    provider: DashScopeProvider,
) -> None:
    """``sync=False`` (or absent) keeps the native async task path — existing
    behavior preserved for clients that want a task id back to poll themselves."""
    img = provider._fake_image  # type: ignore[attr-defined]
    req = UnifiedImageRequest(
        model="wanx2.1-t2i-turbo", content=[image_text_part("a cat")]
    )
    task = asyncio.run(provider.create_image_task(req, sync=False))
    task = asyncio.run(provider.get_image_task(task.task_id))
    assert img.async_calls == 1 and img.wait_calls == ["ds-task-1"]
    assert img.sync_calls == 0


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


# --------------------------------------------------------------------------- #
# Wan2.x-image generation path (multimodal-generation surface)
# --------------------------------------------------------------------------- #


def test_generation_routes_wan27_image_to_generation_surface(
    provider: DashScopeProvider,
) -> None:
    """``wan2.7-image`` lives on the multimodal-generation surface, not the
    synthesis surface — it must hit ``AioImageGeneration`` and never
    ``AioImageSynthesis`` (the regression: routing it through synthesis returns
    ``400 InvalidParameter: url error``)."""
    req = UnifiedImageRequest(model="wan2.7-image", content=[image_text_part("a cat")])
    task = asyncio.run(provider.create_image_task(req))
    task = asyncio.run(provider.get_image_task(task.task_id))
    gen = provider._fake_image_gen  # type: ignore[attr-defined]
    syn = provider._fake_image  # type: ignore[attr-defined]
    # Generation surface used; synthesis surface untouched.
    assert gen.async_calls == 1 and gen.sync_calls == 0
    assert syn.async_calls == 0 and syn.sync_calls == 0
    assert task.status == "succeeded"
    assert task.images[0].url == "https://dashscope.test/gen.png"
    assert task.model == "wan2.7-image" and task.provider == "dashscope"


def test_generation_native_async_builds_messages_shaped_kwargs(
    provider: DashScopeProvider,
) -> None:
    """The generation surface takes a messages body: text parts become
    ``{"text": ...}``` items and image parts become ``{"image": url}`` items
    inside ``messages=[{role:"user", content:[...]}]``. The generation knobs
    (size, n, seed, negative_prompt, watermark) ride as top-level kwargs."""
    req = UnifiedImageRequest(
        model="wan2.7-image",
        content=[
            image_text_part("a cat"),
            image_image_part("https://x.test/ref.png"),
        ],
        size="2K",
        n=2,
        seed=9,
        negative_prompt="blurry",
        watermark=False,
    )
    task = asyncio.run(provider.create_image_task(req))
    asyncio.run(provider.get_image_task(task.task_id))
    gen = provider._fake_image_gen  # type: ignore[attr-defined]
    kw = gen.async_kwargs
    assert kw["model"] == "wan2.7-image"
    assert kw["messages"] == [
        {
            "role": "user",
            "content": [
                {"text": "a cat"},
                {"image": "https://x.test/ref.png"},
            ],
        }
    ]
    # size passes through verbatim (these models accept tokens like "2K").
    assert kw["size"] == "2K"
    assert kw["n"] == 2 and kw["seed"] == 9
    assert kw["negative_prompt"] == "blurry"
    assert kw["watermark"] is False
    # wait was driven against the returned task id, on the generation surface.
    assert gen.wait_calls == ["gen-task-1"]


def test_generation_image_extracted_from_choices_content(
    provider: DashScopeProvider,
) -> None:
    """The finished image rides ``output.choices[].message.content[]`` as a
    ``{"image": url}`` item (matching the SDK's ``ImageGenerationResponse``), not
    ``output.results[].url``."""
    gen = provider._fake_image_gen  # type: ignore[attr-defined]
    gen._wait_result = _generation_succeeded("https://dashscope.test/a.png")
    req = UnifiedImageRequest(model="wan2.7-image", content=[image_text_part("x")])
    task = asyncio.run(provider.create_image_task(req))
    task = asyncio.run(provider.get_image_task(task.task_id))
    assert task.images[0].url == "https://dashscope.test/a.png"


def test_generation_falls_back_to_sync_when_async_rejected(
    provider: DashScopeProvider,
) -> None:
    """When the backend rejects async submission (403 AccessDenied "does not
    support asynchronous calls") the generation path falls back to the sync
    inline ``call`` (``BaseAioApi.call``, no ``X-DashScope-Async`` header)."""
    gen = provider._fake_image_gen  # type: ignore[attr-defined]
    gen.reject_async = True
    req = UnifiedImageRequest(model="wan2.7-image", content=[image_text_part("a cat")])
    task = asyncio.run(provider.create_image_task(req))
    task = asyncio.run(provider.get_image_task(task.task_id))
    assert gen.async_calls == 1 and gen.sync_calls == 1
    assert gen.sync_kwargs["model"] == "wan2.7-image"
    assert gen.sync_kwargs["messages"] == [
        {"role": "user", "content": [{"text": "a cat"}]}
    ]
    assert task.status == "succeeded"
    assert task.images[0].url == "https://dashscope.test/gen.png"


def test_generation_falls_back_to_sync_when_async_raises_rejection(
    provider: DashScopeProvider,
) -> None:
    """An async rejection that *raises* (rather than returning an error body)
    also triggers the sync fallback on the generation surface."""
    gen = provider._fake_image_gen  # type: ignore[attr-defined]
    gen.raise_async = "AccessDenied: does not support asynchronous calls"
    req = UnifiedImageRequest(model="wan2.7-image", content=[image_text_part("a cat")])
    task = asyncio.run(provider.create_image_task(req))
    task = asyncio.run(provider.get_image_task(task.task_id))
    assert gen.async_calls == 1 and gen.sync_calls == 1
    assert task.status == "succeeded"


def test_generation_sync_intent_uses_sync_inline(
    provider: DashScopeProvider,
) -> None:
    """A ``sync=True`` create on the generation surface goes straight to
    ``AioImageGeneration.call`` (the synchronous inline multimodal-generation
    path) and never touches async submit/wait."""
    gen = provider._fake_image_gen  # type: ignore[attr-defined]
    req = UnifiedImageRequest(model="wan2.7-image", content=[image_text_part("a cat")])
    task = asyncio.run(provider.create_image_task(req, sync=True))
    task = asyncio.run(provider.get_image_task(task.task_id))
    assert gen.async_calls == 0 and gen.wait_calls == []
    assert gen.sync_calls == 1
    assert gen.sync_kwargs["model"] == "wan2.7-image"
    assert gen.sync_kwargs["messages"] == [
        {"role": "user", "content": [{"text": "a cat"}]}
    ]
    assert task.status == "succeeded"
    assert task.images[0].url == "https://dashscope.test/gen.png"


def test_generation_failed_status_is_clean_error(provider: DashScopeProvider) -> None:
    """A non-200 response or no image content surfaces as a clean
    ``TaskFailedError`` — not an opaque AttributeError (the original CI
    failure mode for misrouted calls)."""
    gen = provider._fake_image_gen  # type: ignore[attr-defined]
    gen._wait_result = SimpleNamespace(
        status_code=400, code="Bad Request", message="no image", output=None
    )
    req = UnifiedImageRequest(model="wan2.7-image", content=[image_text_part("x")])
    task = asyncio.run(provider.create_image_task(req))
    with pytest.raises(TaskFailedError) as ei:
        asyncio.run(provider.get_image_task(task.task_id))
    msg = str(ei.value)
    assert "dashscope image task" in msg
    assert "Bad Request" in msg


def test_generation_empty_choices_is_failed(provider: DashScopeProvider) -> None:
    """A 200 response with no image items surfaces as a clean TaskFailedError."""
    gen = provider._fake_image_gen  # type: ignore[attr-defined]
    gen._wait_result = SimpleNamespace(
        status_code=200,
        code=None,
        message=None,
        output=SimpleNamespace(choices=[]),
    )
    req = UnifiedImageRequest(model="wan2.7-image", content=[image_text_part("x")])
    task = asyncio.run(provider.create_image_task(req))
    with pytest.raises(TaskFailedError):
        asyncio.run(provider.get_image_task(task.task_id))


def test_generation_propagates_non_rejection_submit_error(
    provider: DashScopeProvider,
) -> None:
    """A submit error that is NOT an async rejection surfaces as
    ProviderRequestError (no silent fallback) on the generation surface too."""
    gen = provider._fake_image_gen  # type: ignore[attr-defined]
    gen.raise_async = "model not found"  # no rejection markers
    req = UnifiedImageRequest(model="wan2.7-image", content=[image_text_part("x")])
    task = asyncio.run(provider.create_image_task(req))
    with pytest.raises(ProviderRequestError):
        asyncio.run(provider.get_image_task(task.task_id))


def test_generation_routes_other_image_models_to_synthesis(
    provider: DashScopeProvider,
) -> None:
    """Routing guard: ``wanx2.1-t2i-*`` (``*-t2i-*``) and ``qwen-image-*``
    (not ``*-image``) stay on the synthesis surface — they must NOT touch
    ``AioImageGeneration``."""
    for model in ("wanx2.1-t2i-turbo", "qwen-image-2.0-pro"):
        req = UnifiedImageRequest(model=model, content=[image_text_part("a cat")])
        asyncio.run(
            provider.get_image_task(
                asyncio.run(provider.create_image_task(req)).task_id
            )
        )
    syn = provider._fake_image  # type: ignore[attr-defined]
    gen = provider._fake_image_gen  # type: ignore[attr-defined]
    assert syn.async_calls == 2  # both models went through synthesis
    assert gen.async_calls == 0 and gen.sync_calls == 0  # never the generation surface


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
            UnifiedImageRequest(
                model="wanx2.1-t2i-turbo", content=[image_text_part("x")]
            )
        )
    )
    asyncio.run(p.get_image_task(itask.task_id))
    assert img.async_kwargs["base_address"] == "https://image.test"
    assert img.wait_kwargs["base_address"] == "https://image.test"
    # Fallback sync path also carries the image base.
    img.reject_async = True
    itask2 = asyncio.run(
        p.create_image_task(
            UnifiedImageRequest(
                model="wanx2.1-t2i-turbo", content=[image_text_part("x")]
            )
        )
    )
    asyncio.run(p.get_image_task(itask2.task_id))
    assert img.sync_kwargs["base_address"] == "https://image.test"
    # Video async_call carries the video base; the poll is hand-rolled over
    # httpx (no SDK fetch) and must hit the video base too — slashless.
    vtask = asyncio.run(
        p.create_video_task(
            UnifiedVideoRequest(model="wanx2.1-t2v-turbo", content=[text_part("x")])
        )
    )
    polled = _mount_video(
        p,
        lambda req: httpx.Response(200, json={"output": {"task_status": "SUCCEEDED"}}),
    )
    asyncio.run(p.get_video_task(vtask.task_id))
    assert vid.call_kwargs["base_address"] == "https://video.test"
    assert polled, "video poll issued no request"
    assert polled[0] == "https://video.test/tasks/vid-task-1"
    assert not polled[0].endswith("/"), f"trailing slash re-introduced: {polled[0]}"


def test_poll_routes_through_image_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: after an async image task is created, polling for its
    result must hit ``DASHSCOPE_IMAGE_BASE_URL`` (the proxy), not the SDK's
    hardcoded ``https://dashscope.aliyuncs.com/api/v1/tasks/`` default.

    The root cause was that the SDK's ``AioImageGeneration.wait`` /
    ``AioImageSynthesis.wait`` / ``AioVideoSynthesis.fetch`` overrides called
    ``super().wait(task, api_key, workspace=..., wait_timeout=...)`` /
    ``super().fetch(task, api_key=..., workspace=...)`` with no ``**kwargs``, so
    a per-call ``base_address=`` never reached ``AsyncAioTaskGetMixin._get`` and
    the poll fell back to the module-global ``dashscope.base_http_api_url``. With
    split sync/async URLs the proxy key is invalid at the real host (401
    InvalidApiKey), so generation always failed after submit succeeded.

    This drives the *real* patched SDK classes (not the fakes, which replace
    them) and asserts the single poll request URL is the proxy.
    """
    from dashscope.api_entities.dashscope_response import DashScopeAPIResponse
    from dashscope.api_entities.http_request import HttpRequest

    captured: list[str] = []

    async def fake_aio_call(self):  # type: ignore[no-untyped-def]
        captured.append(self.url)
        return DashScopeAPIResponse(
            request_id="r",
            status_code=200,
            code=None,
            message=None,
            output={
                "task_status": "SUCCEEDED",
                "task_id": "t",
                "results": [{"url": "https://x/y.png"}],
                "video_url": "https://x/v.mp4",
                "model": "m",
            },
        )

    monkeypatch.setattr(HttpRequest, "aio_call", fake_aio_call)

    base = "https://ai.ctaigw.cn/v1"
    asyncio.run(
        dashscope_mod.AioImageSynthesis.wait("syn-1", api_key="k", base_address=base)
    )
    asyncio.run(
        dashscope_mod.AioImageGeneration.wait("gen-1", api_key="k", base_address=base)
    )
    asyncio.run(
        dashscope_mod.AioVideoSynthesis.fetch("vid-1", api_key="k", base_address=base)
    )

    assert captured, "no poll request was issued"
    assert all(u.startswith(base) for u in captured), (
        f"polls bypassed the proxy base URL: {captured}"
    )
    assert not any("dashscope.aliyuncs.com" in u for u in captured), (
        f"polls hit the hardcoded default host instead of the proxy: {captured}"
    )
    assert all("/tasks/" in u for u in captured), (
        f"polls did not hit the tasks endpoint: {captured}"
    )


def test_base_url_forwarding_install_is_idempotent() -> None:
    """``_install_base_url_forwarding`` must be a no-op on a second call.

    The marker guard reads the raw classmethod object from the class dict
    (not ``cls.wait``, which triggers the classmethod descriptor and hides the
    marker). Re-calling the installer must leave the bound ``wait``/``fetch``
    wrappers untouched — otherwise the "Bound at import time, once" contract is
    violated and a hot-reload / explicit re-call would silently re-patch.
    """
    surfaces = (
        dashscope_mod.AioImageGeneration,
        dashscope_mod.AioImageSynthesis,
        dashscope_mod.AioVideoSynthesis,
    )
    before = [(cls.__dict__.get("wait"), cls.__dict__.get("fetch")) for cls in surfaces]
    dashscope_mod._install_base_url_forwarding()
    dashscope_mod._install_base_url_forwarding()
    after = [(cls.__dict__.get("wait"), cls.__dict__.get("fetch")) for cls in surfaces]
    assert before == after, (
        "idempotency guard failed: _install_base_url_forwarding() re-patched "
        f"an already-patched surface instead of skipping (before={before}, "
        f"after={after})"
    )
    assert all(getattr(w, "_mm_gateway_forwarding", False) for w, _ in after), (
        "forwarding marker missing from an installed wait wrapper"
    )


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
    _mount_video(
        provider,
        lambda req: httpx.Response(
            200,
            json={
                "output": {
                    "task_status": "SUCCEEDED",
                    "video_url": "https://dashscope.test/v.mp4",
                    "model": "wanx2.1-t2v-plus",
                },
            },
        ),
    )
    task = asyncio.run(provider.get_video_task("vid-task-1"))
    assert task.status == "succeeded"
    assert task.model == "wanx2.1-t2v-plus"
    assert task.video_urls == ["https://dashscope.test/v.mp4"]


def test_video_poll_failed(provider: DashScopeProvider) -> None:
    _mount_video(
        provider,
        lambda req: httpx.Response(
            200,
            json={
                "output": {"task_status": "FAILED", "message": "boom"},
            },
        ),
    )
    task = asyncio.run(provider.get_video_task("vid-task-1"))
    assert task.status == "failed" and task.error == "boom"


def test_video_poll_slashed_url_does_not_500(provider: DashScopeProvider) -> None:
    """Regression for the CI E2E failure (run 31263106466, job
    93117291883): the proxy ``ai.ctaigw.cn`` returns ``500 SYSTEM_ERROR`` for
    ``GET /v1/tasks/{id}/`` (trailing slash) but 200 for the slashless form.
    The SDK's ``_build_api_request`` re-appended the slash, so every poll 500'd
    upstream; ``from_api_response`` then omitted ``output`` and the adapter
    crashed on ``status.output.task_status`` (502 to the frontend). The poll now
    goes out slashless over httpx.
    """
    captured = _mount_video(
        provider,
        lambda req: httpx.Response(
            200,
            json={
                "output": {"task_status": "SUCCEEDED", "video_url": "https://x/v.mp4"},
            },
        ),
    )
    asyncio.run(provider.get_video_task("c39ebb62"))
    assert captured, "no poll request issued"
    assert not captured[0].endswith("/"), f"trailing slash re-introduced: {captured[0]}"
    assert "/tasks/c39ebb62" in captured[0]


def test_video_poll_upstream_500_yields_clean_502(
    provider: DashScopeProvider,
) -> None:
    """When the upstream poll returns non-200 (the 500 SYSTEM_ERROR shape),
    the adapter must raise a clean ``ProviderRequestError`` rather than
    dereference a missing ``output`` and crash with ``AttributeError``.
    """
    _mount_video(
        provider,
        lambda req: httpx.Response(
            500, json={"code": "SYSTEM_ERROR", "message": "internal"}
        ),
    )
    with pytest.raises(ProviderRequestError) as exc:
        asyncio.run(provider.get_video_task("vid-task-1"))
    assert "HTTP 500" in str(exc.value)


def test_video_poll_missing_output_yields_clean_502(
    provider: DashScopeProvider,
) -> None:
    """A 200 with no ``output`` block (an upstream error envelope) must surface
    as a ``ProviderRequestError``, not crash on ``None.task_status``.
    """
    _mount_video(
        provider,
        lambda req: httpx.Response(
            200,
            json={
                "code": "AccessDenied",
                "message": "no output for you",
            },
        ),
    )
    with pytest.raises(ProviderRequestError) as exc:
        asyncio.run(provider.get_video_task("vid-task-1"))
    assert "no output" in str(exc.value)


def test_video_poll_non_json_body_yields_clean_502(
    provider: DashScopeProvider,
) -> None:
    """A non-JSON upstream body must not crash ``resp.json()``; it surfaces as a
    ``ProviderRequestError``."""
    _mount_video(
        provider,
        lambda req: httpx.Response(
            200, content=b"<html>bad</html>", headers={"content-type": "text/html"}
        ),
    )
    with pytest.raises(ProviderRequestError) as exc:
        asyncio.run(provider.get_video_task("vid-task-1"))
    assert "non-JSON" in str(exc.value)


def test_video_poll_maps_usage(provider: DashScopeProvider) -> None:
    _mount_video(
        provider,
        lambda req: httpx.Response(
            200,
            json={
                "output": {
                    "task_status": "SUCCEEDED",
                    "video_url": "https://x/v.mp4",
                    "model": "wanx2.1-t2v-turbo",
                },
                "usage": {"video_count": 1, "video_duration": 5, "video_ratio": "16:9"},
            },
        ),
    )
    task = asyncio.run(provider.get_video_task("vid-task-1"))
    assert task.status == "succeeded"
    assert task.usage is not None
    assert task.usage.video_count == 1 and task.usage.video_duration == 5
    assert task.usage.extra == {"video_ratio": "16:9"}
