"""DashScope provider — Wanx/Wan + Qwen-Image (image) and Wan (video).

Image generation uses DashScope's **native async task API** directly when the
backend supports it — ``AioImageSynthesis.async_call`` submits a task (returns a
real ``task_id``) and ``AioImageSynthesis.wait`` blocks until the images are
ready. This is the unrestricted path: it serves every model including
``qwen-image-2.0-pro``, which the synchronous inline path does not advertise.

Some backends reject async task submission — third-party DashScope-compatible
gateways and API keys without the async entitlement return
``403 AccessDenied: current user api does not support asynchronous calls``. When
that happens the adapter falls back to ``AioImageSynthesis.sync_call`` (the
headerless inline path, ``BaseAioApi.call``), which returns finished images in a
single call. That is the "implement in this project" case: the synchronous call
is wrapped as a synthetic in-memory task by
:class:`~mm_gateway.providers._sync_image.SyncImageTaskMixin` so the gateway's
create→poll surface still holds.

Both paths are wrapped as a synthetic task by ``SyncImageTaskMixin``:
``create_image_task`` mints a gateway-local ``img-`` id and returns ``pending``;
the first ``get_image_task`` poll runs the blocking generation (native async
``wait``, or the ``sync_call`` fallback) and moves the task to
``succeeded``/``failed``. Video stays on the native ``async_call`` / ``fetch``
flow, which is unaffected.

Sync vs async base URL: the SDK honours a per-call ``base_address=`` kwarg
(verified in the installed ``dashscope`` source — ``_build_api_request`` pops
it, falling back to the module global only when unset), so the adapter routes
image calls at ``backend.base_url`` (the ``*_IMAGE_BASE_URL`` sync endpoint,
resolved modality-first by ``config.py``) and video calls at
``backend.extra["video_base_url"]`` (the ``*_VIDEO_BASE_URL`` async endpoint,
recorded by ``config.py`` when it differs from the image one). This replaces
the prior module-global ``dashscope.base_http_api_url`` mutation, which was
racy across concurrent backends and routed video polls at the image host when
the two endpoints differed. The ``X-DashScope-Async: enable`` header set by
``async_call`` selects the async task mode; ``sync_call`` (``BaseAioApi.call``)
omits it.
"""

from __future__ import annotations

import time
from typing import Any

import dashscope
from dashscope.aigc.image_synthesis import AioImageSynthesis
from dashscope.aigc.video_synthesis import AioVideoSynthesis

from mm_gateway.core.base import ImageProvider, VideoProvider
from mm_gateway.core.exceptions import (
    ProviderNotConfiguredError,
    ProviderRequestError,
    TaskFailedError,
)
from mm_gateway.observability.logging import get_logger
from mm_gateway.providers._sync_image import SyncImageTaskMixin
from mm_gateway.schemas.image import (
    ImageData,
    ImageUsage,
    UnifiedImageRequest,
    UnifiedImageResponse,
)
from mm_gateway.schemas.video import UnifiedVideoRequest, UnifiedVideoTask

log = get_logger("provider.dashscope")

_STATUS_MAP = {
    "PENDING": "pending",
    "RUNNING": "running",
    "SUSPENDED": "running",
    "SUCCEEDED": "succeeded",
    "FAILED": "failed",
    "CANCELED": "cancelled",
    "UNKNOWN": "failed",
}

# Markers that the upstream rejected the *async* task path specifically (not a
# generation failure). On these we fall back to the synchronous inline path.
_ASYNC_REJECTION_MARKERS = ("asynchronous", "accessdenied", "does not support")


def _is_async_rejection(status_code: Any, code: Any, message: Any) -> bool:
    """True iff the error indicates async task submission is unsupported."""
    text = " ".join(str(v).lower() for v in (code, message) if v)
    if any(m in text for m in _ASYNC_REJECTION_MARKERS):
        return True
    return status_code == 403


class _AsyncNotSupported(Exception):
    """Internal sentinel: the backend rejected async task submission."""


class DashScopeProvider(SyncImageTaskMixin, ImageProvider, VideoProvider):
    name = "dashscope"
    image_models = [
        "wanx2.1-t2i-turbo",
        "wanx2.1-t2i-plus",
        "wanx2.1-t2i-flash",
        # Qwen-Image — the native async task path (async_call + wait) is the
        # unrestricted route; sync_call's docstring restricts it to wan2.2-t2i-*.
        "qwen-image-2.0-pro",
    ]
    video_models = [
        "wanx2.1-t2v-turbo",
        "wanx2.1-i2v-turbo",
        "wanx2.1-t2v-plus",
        "wanx2.1-i2v-plus",
    ]

    def __init__(self, backend):
        super().__init__(backend)
        if not backend.api_key:
            raise ProviderNotConfiguredError("dashscope")
        self._api_key = backend.api_key
        # Per-call URL selection honors the sync/async split resolved by
        # ``config.py``: ``base_url`` is the image (sync) endpoint
        # (``*_IMAGE_BASE_URL`` preferred); ``extra["video_base_url"]`` is the
        # async endpoint (``*_VIDEO_BASE_URL`` when it differs from the image
        # one). We pass ``base_address=`` on every SDK call instead of mutating
        # the module-global ``dashscope.base_http_api_url`` — which is racy
        # across concurrent backends and would route video polls at the image
        # host when the two differ.
        dashscope.api_key = backend.api_key
        self._image_base = backend.base_url or None
        self._video_base = backend.extra.get("video_base_url") or backend.base_url or None

    @staticmethod
    def _image_kwargs(request: UnifiedImageRequest) -> dict[str, Any]:
        """Build the kwargs shared by async_call/sync_call for image gen."""
        kwargs: dict[str, Any] = {
            "model": request.model,
            "prompt": request.prompt() or "",
        }
        if request.n:
            kwargs["n"] = request.n
        if request.size:
            kwargs["size"] = request.size.replace("x", "*")
        if request.seed is not None:
            kwargs["seed"] = request.seed
        if request.negative_prompt:
            kwargs["negative_prompt"] = request.negative_prompt
        kwargs.update(request.extra)
        return kwargs

    async def _image_native_async(self, kwargs: dict[str, Any]) -> Any:
        """Submit an image task and block until it finishes (native async).

        Raises ``_AsyncNotSupported`` if the backend rejects async task
        submission (403 / "does not support asynchronous calls"); other errors
        surface as ``TaskFailedError`` / ``ProviderRequestError``.
        """
        try:
            create = await AioImageSynthesis.async_call(
                api_key=self._api_key, base_address=self._image_base, **kwargs
            )
        except _AsyncNotSupported:
            raise
        except Exception as exc:  # noqa: BLE001
            if _is_async_rejection(None, type(exc).__name__, str(exc)):
                raise _AsyncNotSupported() from exc
            raise ProviderRequestError(
                f"dashscope image submit failed: {exc}", provider="dashscope"
            ) from exc

        # async_call may return an error response instead of raising.
        status_code = getattr(create, "status_code", 200)
        output = getattr(create, "output", None)
        task_id = getattr(output, "task_id", None) if output else None
        if status_code != 200 or not task_id:
            code = getattr(create, "code", None)
            message = getattr(create, "message", None)
            if _is_async_rejection(status_code, code, message):
                raise _AsyncNotSupported()
            raise TaskFailedError(
                f"dashscope image submit failed: status_code={status_code} "
                f"code={code} message={message}",
                provider="dashscope",
            )

        try:
            resp = await AioImageSynthesis.wait(
                task_id, api_key=self._api_key, base_address=self._image_base
            )
        except Exception as exc:  # noqa: BLE001
            raise ProviderRequestError(
                f"dashscope image wait failed: {exc}", provider="dashscope"
            ) from exc
        return resp

    async def _image_sync_inline(self, kwargs: dict[str, Any]) -> Any:
        """Synchronous inline image call — no ``X-DashScope-Async`` header.

        ``sync_call`` (``BaseAioApi.call``) returns the finished images in
        ``output.results`` in a single call. Used as the fallback when the
        backend rejects async task submission.
        """
        try:
            return await AioImageSynthesis.sync_call(
                api_key=self._api_key, base_address=self._image_base, **kwargs
            )
        except Exception as exc:  # noqa: BLE001
            raise ProviderRequestError(
                f"dashscope image failed: {exc}", provider="dashscope"
            ) from exc

    async def _generate_image(
        self, request: UnifiedImageRequest
    ) -> UnifiedImageResponse:
        """Generate one image set: native async first, sync inline on rejection.

        ``size`` uses DashScope's ``"*"`` separator (``1024*1024``). Both paths
        return ``ImageSynthesisResponse`` with ``output.task_status`` ==
        ``SUCCEEDED`` and ``output.results[].url``; a non-SUCCEEDED status or
        missing results surfaces as a clean ``TaskFailedError``.
        """
        kwargs = self._image_kwargs(request)
        try:
            resp = await self._image_native_async(kwargs)
        except _AsyncNotSupported:
            log.info("dashscope_image_async_unsupported_fallback_sync",
                     model=request.model)
            resp = await self._image_sync_inline(kwargs)

        status_code = getattr(resp, "status_code", None)
        output = getattr(resp, "output", None)
        task_status = getattr(output, "task_status", None) if output else None
        results = getattr(output, "results", None) if output else None
        if status_code != 200 or task_status != "SUCCEEDED" or not results:
            code = getattr(resp, "code", None)
            message = getattr(resp, "message", None)
            err = (
                f"dashscope image task: status_code={status_code} "
                f"task_status={task_status} code={code} message={message}"
            )
            raise TaskFailedError(err, provider="dashscope")
        data = [ImageData(url=getattr(r, "url", None)) for r in results]
        return UnifiedImageResponse(
            created=int(time.time()),
            data=data,
            model=request.model,
            provider=self.name,
            usage=ImageUsage(),
        )

    async def create_video_task(self, request: UnifiedVideoRequest) -> UnifiedVideoTask:
        kwargs: dict[str, Any] = {
            "model": request.model,
            "prompt": request.prompt() or "",
        }
        if request.size:
            kwargs["size"] = request.size.replace("x", "*")
        if request.ratio:
            kwargs["ratio"] = request.ratio
        if request.duration is not None:
            kwargs["duration"] = int(request.duration)
        if request.seed is not None:
            kwargs["seed"] = request.seed
        if request.prompt_extend is not None:
            kwargs["prompt_extend"] = request.prompt_extend
        if request.first_image():
            kwargs["img_url"] = request.first_image()
        kwargs.update(request.extra)

        try:
            resp = await AioVideoSynthesis.async_call(
                api_key=self._api_key, base_address=self._video_base, **kwargs
            )
        except Exception as exc:
            raise ProviderRequestError(
                f"dashscope video submit failed: {exc}", provider="dashscope"
            ) from exc

        task_id = resp.output.task_id
        return UnifiedVideoTask(
            task_id=task_id, provider=self.name, model=request.model, status="pending"
        )

    async def get_video_task(self, task_id: str) -> UnifiedVideoTask:
        try:
            status = await AioVideoSynthesis.fetch(
                task_id, api_key=self._api_key, base_address=self._video_base
            )
        except Exception as exc:
            raise ProviderRequestError(
                f"dashscope video poll failed: {exc}", provider="dashscope"
            ) from exc
        st = _STATUS_MAP.get(status.output.task_status, "running")
        model = getattr(status.output, "model", "") or ""
        task = UnifiedVideoTask(
            task_id=task_id, provider=self.name, model=model, status=st
        )  # type: ignore[arg-type]
        if st == "succeeded":
            url = getattr(status.output, "video_url", None)
            if url:
                task.video_urls = [url]
            if getattr(status, "usage", None):
                task.usage = None  # populated if present
        elif st in ("failed", "cancelled", "expired"):
            task.error = (
                getattr(status.output, "message", None) or status.output.task_status
            )
        return task
