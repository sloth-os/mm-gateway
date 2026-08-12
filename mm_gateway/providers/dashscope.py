"""DashScope provider — Wanx/Wan + Qwen-Image (image) and Wan (video).

DashScope serves two distinct image surfaces, and the adapter routes by model
family:

* **Wan2.x-image generation** (``wan2.7-image`` / ``wan2.6-image`` / ...) lives
  on the **multimodal-generation** surface — a messages-shaped body posted to
  ``/services/aigc/multimodal-generation/generation``. The SDK exposes it as
  ``AioImageGeneration``: ``call`` (sync, inline) or ``async_call`` (submit a
  real task id) + ``fetch``/``wait``. The finished images come back as
  ``output.choices[].message.content[]`` items with ``{"image": "<url>"}``.
  This is the surface the working ``ai.ctaigw.cn`` example targets — routing a
  ``wan2.7-image`` call through the older synthesis surface returns
  ``400 InvalidParameter: url error, please check url`` because that path is
  not wired for the model.
* **Wanx2.1 / Qwen-Image synthesis** (``wanx2.1-t2i-*``, ``qwen-image-2.0-pro``)
  stays on the **image-synthesis** surface (``/services/aigc/text2image/...``)
  via ``AioImageSynthesis``.

Image generation uses DashScope's **native async task API** by default —
``async_call`` submits a task (returns a real ``task_id``) and ``wait`` blocks
until the images are ready. This is the unrestricted path: it serves every
model including ``qwen-image-2.0-pro``, which the synchronous inline path does
not advertise.

When the front end asks to block (``wait=true`` or the
``image_sync_default`` setting), the adapter instead uses the **synchronous
inline API** directly — ``AioImageSynthesis.sync_call`` (``BaseAioApi.call``,
no ``X-DashScope-Async`` header) on the synthesis surface, or
``AioImageGeneration.call`` on the multimodal-generation surface. Both return
finished images in a single request with no task poll. This matters on
third-party DashScope-compatible gateways whose async task *poll* endpoint is
broken (``GET /v1/tasks/{id}/`` returns ``500 SYSTEM_ERROR``) even though
async submit succeeds: there the native async path submits fine but can never
resolve, so a ``wait=true`` request would hang to the sync deadline and fail.
The sync inline path skips the poll entirely.

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
``succeeded``/``failed``. Video submits via the SDK's native ``async_call`` (the
``X-DashScope-Async`` task path) but polls by hand over httpx — see
``get_video_task`` for why the SDK's ``fetch`` is bypassed.

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

import asyncio
import collections.abc
import json
import time
from typing import Any, ClassVar

import dashscope
import httpx
from dashscope.aigc.image_generation import AioImageGeneration
from dashscope.aigc.image_synthesis import AioImageSynthesis
from dashscope.aigc.video_synthesis import AioVideoSynthesis
from dashscope.api_entities.dashscope_response import (
    DashScopeAPIResponse,
    ImageGenerationResponse,
    ImageSynthesisResponse,
    VideoSynthesisResponse,
)
from dashscope.api_entities.http_request import HttpRequest
from dashscope.client.base_api import BaseAsyncAioApi

from mm_gateway.core.base import ImageProvider, VideoProvider
from mm_gateway.core.exceptions import (
    ProviderNotConfiguredError,
    ProviderRequestError,
    ProviderTimeoutError,
    TaskFailedError,
)
from mm_gateway.observability.httplog import (
    backend_event_hooks,
    log_backend_request,
    log_backend_response,
)
from mm_gateway.observability.logging import get_logger
from mm_gateway.providers._dimensions import aspect_ratio, pixel_size
from mm_gateway.providers._http import _map_status
from mm_gateway.providers._sync_image import SyncImageTaskMixin
from mm_gateway.schemas.image import (
    ImageData,
    ImageUsage,
    UnifiedImageRequest,
    UnifiedImageResponse,
)
from mm_gateway.schemas.video import UnifiedVideoRequest, UnifiedVideoTask, VideoUsage

log = get_logger("provider.dashscope")


# -- Backend request/response logging --------------------------------------- #
# DashScope's SDK speaks aiohttp internally and exposes no httpx injection
# point, so we can't attach httpx event hooks like the other providers. Every
# async/sync call, wait and fetch funnels through ``HttpRequest.aio_call``
# (``BaseAsyncAioApi.async_call``/``.call`` and ``AsyncAioTaskGetMixin`` both
# end in ``await request.aio_call()``), so monkeypatching that one method is
# the single chokepoint that covers all five call types the adapter uses.
# Idempotent: a marker on the wrapper prevents double-wrapping on re-import.


def _dashscope_request_content(request: HttpRequest) -> bytes | None:
    """Best-effort backend request body for logging.

    Only POST bodies are rendered (GET task queries carry their id in the URL).
    ``get_aiohttp_payload`` resolves the input via ``InputResolver.__next__``,
    which is re-callable for our dict/list inputs, so calling it here before the
    real send is safe. Multipart form payloads aren't JSON-renderable and are
    skipped.
    """
    data = getattr(request, "data", None)
    if data is None or str(request.method).upper() != "POST":
        return None
    try:
        is_form, obj = data.get_aiohttp_payload()
    except Exception:  # noqa: BLE001
        return None
    if is_form:
        return None
    try:
        return json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
    except Exception:  # noqa: BLE001
        return None


def _dashscope_log_response(resp: Any, url: str) -> None:
    """Log a backend response from a ``DashScopeAPIResponse``."""
    status = getattr(resp, "status_code", None)
    headers = getattr(resp, "headers", None) or {}
    body_obj = {
        "request_id": getattr(resp, "request_id", None),
        "code": getattr(resp, "code", None),
        "message": getattr(resp, "message", None),
        "output": getattr(resp, "output", None),
        "usage": getattr(resp, "usage", None),
    }
    try:
        content = json.dumps(body_obj, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001
        content = str(body_obj)
    log_backend_response(status, url, headers, content)


def _install_dashscope_logging() -> None:
    if getattr(HttpRequest.aio_call, "_mm_gateway_logged", False):
        return
    orig_aio_call = HttpRequest.aio_call

    async def _logged_aio_call(self):  # type: ignore[no-untyped-def]
        content = _dashscope_request_content(self)
        log_backend_request(str(self.method), self.url, self.headers, content)
        result = await orig_aio_call(self)
        if isinstance(result, collections.abc.AsyncGenerator):
            # Streaming (SSE) — log each yielded response as it arrives.
            async def _logged_stream():
                async for item in result:
                    _dashscope_log_response(item, self.url)
                    yield item

            return _logged_stream()
        _dashscope_log_response(result, self.url)
        return result

    _logged_aio_call._mm_gateway_logged = True  # type: ignore[attr-defined]
    HttpRequest.aio_call = _logged_aio_call


_install_dashscope_logging()


# -- Patch wait/fetch to forward base_address -------------------------------- #
# DashScope's async task methods (``wait``/``fetch``) drop ``base_address``:
# ``AioImageGeneration.wait``/``AioImageSynthesis.wait``/``AioVideoSynthesis.fetch``
# call ``super().wait(task, api_key, workspace=..., wait_timeout=...)`` /
# ``super().fetch(task, api_key=..., workspace=...)`` with no ``**kwargs``, so
# the proxy base URL never reaches ``AsyncAioTaskGetMixin._get``. Every poll
# falls back to the module-global ``dashscope.base_http_api_url`` — the real
# ``dashscope.aliyuncs.com`` host — even when ``async_call``/``call`` submitted
# the task through a proxy. With split sync/async base URLs the poll then hits
# the wrong host: the proxy key is invalid there (401 InvalidApiKey) and the
# task is unreachable (404), so generation always fails after submit succeeds.
# We re-bind each surface's ``wait``/``fetch`` to forward ``base_address`` and
# re-run ``from_api_response`` to keep the typed response class the adapter
# reads. Bound at import time, once, via a marker.

_BASE_WAIT = BaseAsyncAioApi.wait.__func__
_BASE_FETCH = BaseAsyncAioApi.fetch.__func__


def _make_forwarding_wait(resp_cls):
    @classmethod  # type: ignore[misc]
    async def wait(cls_, task, api_key=None, workspace=None, wait_timeout=-1, **kwargs):
        r = await _BASE_WAIT(
            cls_,
            task,
            api_key,
            workspace=workspace,
            wait_timeout=wait_timeout,
            **kwargs,
        )
        return resp_cls.from_api_response(r)

    return wait


def _make_forwarding_fetch(resp_cls):
    @classmethod  # type: ignore[misc]
    async def fetch(cls_, task, api_key=None, workspace=None, **kwargs):
        r = await _BASE_FETCH(
            cls_,
            task,
            api_key=api_key,
            workspace=workspace,
            **kwargs,
        )
        return resp_cls.from_api_response(r)

    return fetch


def _install_base_url_forwarding() -> None:
    for cls, resp_cls in (
        (AioImageGeneration, ImageGenerationResponse),
        (AioImageSynthesis, ImageSynthesisResponse),
        (AioVideoSynthesis, VideoSynthesisResponse),
    ):
        # Read the marker off the raw classmethod object stored in the class
        # dict. Going through ``cls.wait`` would trigger the classmethod
        # descriptor, which returns a bound method whose attribute lookup
        # delegates to ``__func__`` — not the wrapper the marker is set on —
        # so the guard would never see it and would re-patch every call.
        existing = cls.__dict__.get("wait")
        if getattr(existing, "_mm_gateway_forwarding", False):
            continue
        wait = _make_forwarding_wait(resp_cls)
        fetch = _make_forwarding_fetch(resp_cls)
        wait._mm_gateway_forwarding = True  # type: ignore[attr-defined]
        cls.wait = wait
        cls.fetch = fetch


_install_base_url_forwarding()


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
    # DashScope exposes both a synchronous inline API (``sync_call`` /
    # ``AioImageGeneration.call``) and a native async task API (``async_call`` +
    # ``wait``). The mixin threads the front-end's resolved ``sync`` intent to
    # ``_generate_image`` so a ``wait=true`` request hits the synchronous inline
    # path — important on third-party DashScope-compatible gateways whose async
    # task poll endpoint is broken (returns 500 SYSTEM_ERROR on the very poll
    # the gateway needs to resolve).
    _sync_aware_image: ClassVar[bool] = True
    image_models: ClassVar[list[str]] = [
        "wanx2.1-t2i-turbo",
        "wanx2.1-t2i-plus",
        "wanx2.1-t2i-flash",
        # Qwen-Image — the native async task path (async_call + wait) is the
        # unrestricted route; sync_call's docstring restricts it to wan2.2-t2i-*.
        "qwen-image-2.0-pro",
        # Wan2.x-image generation family — served on the multimodal-generation
        # surface (messages-shaped), not the image-synthesis surface. Operators
        # may also pin other ``*-image`` slugs via DASHSCOPE_IMAGE_MODEL.
        "wan2.7-image",
        "wan2.6-image",
    ]
    video_models: ClassVar[list[str]] = [
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
        self._video_base = (
            backend.extra.get("video_base_url") or backend.base_url or None
        )
        # Dedicated httpx clients for the task *poll*. The SDK's
        # ``AioImageGeneration.wait``/``AioImageSynthesis.wait``/
        # ``AioVideoSynthesis.fetch`` are bypassed here (see
        # ``_poll_task``/``get_video_task``): ``_build_api_request`` re-appends a
        # trailing slash to the task URL (``/v1/tasks/{id}/``) after
        # ``_normalization_url``/``join_url`` strip it; the third-party proxy
        # ``ai.ctaigw.cn`` answers ``GET /v1/tasks/{id}`` with 200 but
        # ``GET /v1/tasks/{id}/`` with ``500 SYSTEM_ERROR``. The clients carry
        # no ``base_url`` — the full slashless URL is built per call so httpx's
        # relative-URL path-joining cannot silently drop the ``/v1`` segment.
        # The auth header rides on each client.
        self._client_image = httpx.AsyncClient(
            timeout=300.0,
            headers={"Authorization": f"Bearer {self._api_key}"},
            event_hooks=backend_event_hooks(),
        )
        self._client_video = httpx.AsyncClient(
            timeout=300.0,
            headers={"Authorization": f"Bearer {self._api_key}"},
            event_hooks=backend_event_hooks(),
        )

    @staticmethod
    def _is_generation_model(model: str) -> bool:
        """True iff the model is served on the multimodal-generation surface
        (messages-shaped ``AioImageGeneration``) rather than the
        image-synthesis surface (``AioImageSynthesis``).

        The synthesis surface only serves wanx2.1-t2i-* / qwen-image-*; the
        Wan2.x-image generation family (``wan2.7-image``, ``wan2.6-image``,
        and operator-pinned ``wan*-image`` slugs) lives on the generation
        surface, and routing them through synthesis returns
        ``400 InvalidParameter: url error``.
        """
        m = (model or "").lower()
        return m.endswith("-image") and "t2i" not in m

    @staticmethod
    def _image_kwargs(request: UnifiedImageRequest) -> dict[str, Any]:
        """Build the kwargs shared by async_call/sync_call for image gen."""
        kwargs: dict[str, Any] = {
            "model": request.model,
            "prompt": request.prompt() or "",
        }
        if request.n:
            kwargs["n"] = request.n
        if size := pixel_size(request, "*"):
            kwargs["size"] = size
        if request.seed is not None:
            kwargs["seed"] = request.seed
        if request.negative_prompt:
            kwargs["negative_prompt"] = request.negative_prompt
        kwargs.update(request.extra)
        return kwargs

    @staticmethod
    def _image_generation_kwargs(request: UnifiedImageRequest) -> dict[str, Any]:
        """Build the kwargs for the multimodal-generation image surface.

        The Wan2.x-image family takes a messages-shaped input
        (``messages=[{role:"user", content:[{text:...}|{image:url}]}]``) and
        the generation knobs ride as top-level kwargs the SDK folds into
        ``parameters`` (size, n, seed, watermark, thinking_mode, ...). ``size``
        is passed through verbatim — these models accept resolution tokens like
        ``"2K"`` as well as ``"1024*1024"``.
        """
        content: list[dict[str, Any]] = []
        for part in request.content:
            p = part.root
            if getattr(p, "type", None) == "text" and getattr(p, "text", ""):
                content.append({"text": p.text})
            elif getattr(p, "type", None) == "image" and getattr(p, "url", None):
                content.append({"image": p.url})
        if not content:
            content = [{"text": request.prompt() or ""}]

        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": [{"role": "user", "content": content}],
        }
        if size := pixel_size(request, "*"):
            kwargs["size"] = size
        if request.n:
            kwargs["n"] = request.n
        if request.seed is not None:
            kwargs["seed"] = request.seed
        if request.negative_prompt:
            kwargs["negative_prompt"] = request.negative_prompt
        if request.watermark is not None:
            kwargs["watermark"] = request.watermark
        kwargs.update(request.extra)
        return kwargs

    @staticmethod
    def _extract_generation_images(resp: Any) -> list[ImageData]:
        """Pull image URLs out of ``output.choices[].message.content[]``."""
        urls: list[str] = []
        output = getattr(resp, "output", None)
        for choice in getattr(output, "choices", None) or []:
            msg = getattr(choice, "message", None)
            for item in getattr(msg, "content", None) or []:
                if isinstance(item, dict):
                    url = item.get("image")
                else:
                    url = getattr(item, "image", None)
                if url:
                    urls.append(url)
        return [ImageData(url=u) for u in urls]

    async def _poll_task(
        self,
        task_id: str,
        client: httpx.AsyncClient,
        base: str | None,
        resp_cls: type,
    ) -> Any:
        """Poll a DashScope async task by hand over httpx until it is terminal.

        Mirrors ``get_video_task``'s slashless GET (the SDK's
        ``_build_api_request`` re-appends a trailing slash to
        ``/v1/tasks/{id}`` which the third-party proxy rejects with
        ``500 SYSTEM_ERROR``), but — unlike the single-shot video poll — loops
        with the SDK ``wait`` backoff until the task reaches a terminal state.
        The image path runs the full submit→block generation inside the first
        ``SyncImageTaskMixin`` poll, so it must return a finished response, not
        a ``running`` snapshot. The raw JSON is wrapped in a
        ``DashScopeAPIResponse`` and materialized through
        ``resp_cls.from_api_response`` so the existing extractors
        (``_generate_image`` reads ``output.results``; ``_generate_image_generation``
        reads ``output.choices[].message.content[]``) see the same typed shape
        the SDK ``wait`` returned.
        """
        url = f"{(base or 'https://dashscope.aliyuncs.com/api/v1').rstrip('/')}/tasks/{task_id}"
        # Match BaseAsyncAioApi.wait: start at 1s, double every 3 polls, cap 5s.
        wait_seconds = 1.0
        max_wait_seconds = 5.0
        step = 0
        while True:
            step += 1
            try:
                resp = await client.get(url)
            except httpx.TimeoutException as exc:
                raise ProviderTimeoutError(
                    f"dashscope task poll timed out: {exc}", provider="dashscope"
                ) from exc
            except httpx.HTTPError as exc:
                raise ProviderRequestError(
                    f"dashscope task poll transport error: {exc}",
                    provider="dashscope",
                ) from exc

            if resp.status_code >= 400:
                raise ProviderRequestError(
                    f"dashscope task poll returned HTTP {resp.status_code}",
                    provider="dashscope",
                    status_code=_map_status(resp.status_code),
                    details={
                        "upstream_status": resp.status_code,
                        "upstream_body": resp.text[:1000],
                    },
                )

            try:
                data = resp.json()
            except ValueError as exc:
                raise ProviderRequestError(
                    f"dashscope task poll returned non-JSON body: {exc}",
                    provider="dashscope",
                    details={
                        "upstream_status": resp.status_code,
                        "upstream_body": resp.text[:1000],
                    },
                ) from exc

            output = data.get("output") if isinstance(data, dict) else None
            if isinstance(output, dict):
                task_status = output.get("task_status")
                st = _STATUS_MAP.get(task_status, "running")
            else:
                # No ``output`` block (e.g. an upstream error envelope) — mirror
                # the SDK ``wait`` which returns the response as-is when output
                # is None; the caller surfaces the non-SUCCEEDED result cleanly.
                st = "running"
            if st in ("succeeded", "failed", "cancelled", "expired"):
                break
            if wait_seconds < max_wait_seconds and step % 3 == 0:
                wait_seconds = min(wait_seconds * 2, max_wait_seconds)
            await asyncio.sleep(wait_seconds)

        api_resp = DashScopeAPIResponse(
            status_code=resp.status_code,
            request_id=data.get("request_id") if isinstance(data, dict) else None,
            code=data.get("code") if isinstance(data, dict) else None,
            message=data.get("message") if isinstance(data, dict) else None,
            output=output,
            usage=data.get("usage") if isinstance(data, dict) else None,
            headers=dict(resp.headers),
        )
        return resp_cls.from_api_response(api_resp)

    async def _image_generation_native_async(self, kwargs: dict[str, Any]) -> Any:
        """Submit a multimodal-generation task and block until it finishes.

        Mirrors ``_image_native_async`` but on the generation surface: the
        finished images ride ``output.choices[].message.content[]``.
        """
        try:
            create = await AioImageGeneration.async_call(
                api_key=self._api_key, base_address=self._image_base, **kwargs
            )
        except _AsyncNotSupported:
            raise
        except Exception as exc:
            if _is_async_rejection(None, type(exc).__name__, str(exc)):
                raise _AsyncNotSupported() from exc
            raise ProviderRequestError(
                f"dashscope image submit failed: {exc}", provider="dashscope"
            ) from exc

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
            resp = await self._poll_task(
                task_id, self._client_image, self._image_base, ImageGenerationResponse
            )
        except (ProviderTimeoutError, ProviderRequestError):
            raise
        except Exception as exc:
            raise ProviderRequestError(
                f"dashscope image wait failed: {exc}", provider="dashscope"
            ) from exc
        return resp

    async def _image_generation_sync_inline(self, kwargs: dict[str, Any]) -> Any:
        """Synchronous multimodal-generation call — finished images inline."""
        try:
            return await AioImageGeneration.call(
                api_key=self._api_key, base_address=self._image_base, **kwargs
            )
        except Exception as exc:
            raise ProviderRequestError(
                f"dashscope image failed: {exc}", provider="dashscope"
            ) from exc

    async def _generate_image_generation(
        self,
        request: UnifiedImageRequest,
        *,
        sync: bool | None = None,
    ) -> UnifiedImageResponse:
        """Generate via the multimodal-generation surface (Wan2.x-image family).

        ``sync=True`` goes straight to the synchronous inline call (the path
        that returns finished images in one request, no task poll) — the front
        end asked to block, and on some third-party DashScope-compatible
        gateways the async task poll endpoint itself is broken. ``sync`` absent
        or ``False`` keeps the native async task path first, falling back to
        sync inline only when the backend rejects async submission. A non-200
        response or no image content surfaces as a clean ``TaskFailedError``.
        """
        kwargs = self._image_generation_kwargs(request)
        if sync:
            resp = await self._image_generation_sync_inline(kwargs)
        else:
            try:
                resp = await self._image_generation_native_async(kwargs)
            except _AsyncNotSupported:
                log.info(
                    "dashscope_image_async_unsupported_fallback_sync",
                    model=request.model,
                )
                resp = await self._image_generation_sync_inline(kwargs)

        status_code = getattr(resp, "status_code", None)
        data = self._extract_generation_images(resp)
        if status_code != 200 or not data:
            code = getattr(resp, "code", None)
            message = getattr(resp, "message", None)
            err = (
                f"dashscope image task: status_code={status_code} "
                f"code={code} message={message}"
            )
            raise TaskFailedError(err, provider="dashscope")
        return UnifiedImageResponse(
            created=int(time.time()),
            data=data,
            model=request.model,
            provider=self.name,
            usage=ImageUsage(),
        )

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
        except Exception as exc:
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
            resp = await self._poll_task(
                task_id, self._client_image, self._image_base, ImageSynthesisResponse
            )
        except (ProviderTimeoutError, ProviderRequestError):
            raise
        except Exception as exc:
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
        except Exception as exc:
            raise ProviderRequestError(
                f"dashscope image failed: {exc}", provider="dashscope"
            ) from exc

    async def _generate_image(
        self,
        request: UnifiedImageRequest,
        *,
        sync: bool | None = None,
    ) -> UnifiedImageResponse:
        """Generate one image set, routed by model family to its surface.

        Wan2.x-image generation models (``*-image`` excluding ``*-t2i-*``) go to
        the multimodal-generation surface; everything else (wanx2.1-t2i-* /
        qwen-image-*) goes to the image-synthesis surface. ``sync=True`` selects
        the synchronous inline call on either surface (the front end asked to
        block); absent/``False`` keeps the native async task path first,
        falling back to sync inline when the backend rejects async submission.
        """
        if self._is_generation_model(request.model):
            return await self._generate_image_generation(request, sync=sync)

        # Synthesis path. ``size`` uses DashScope's ``"*"`` separator
        # (``1024*1024``). Both paths return ``ImageSynthesisResponse`` with
        # ``output.task_status`` == ``SUCCEEDED`` and ``output.results[].url``;
        # a non-SUCCEEDED status or missing results surfaces as a clean
        # ``TaskFailedError``.
        kwargs = self._image_kwargs(request)
        if sync:
            resp = await self._image_sync_inline(kwargs)
        else:
            try:
                resp = await self._image_native_async(kwargs)
            except _AsyncNotSupported:
                log.info(
                    "dashscope_image_async_unsupported_fallback_sync",
                    model=request.model,
                )
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
        if size := pixel_size(request, "*"):
            kwargs["size"] = size
        if ratio := aspect_ratio(request):
            kwargs["ratio"] = ratio
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
        # Poll by hand over httpx rather than via ``AioVideoSynthesis.fetch``.
        # The SDK's ``_build_api_request`` re-appends a trailing slash to the
        # task URL (``/v1/tasks/{id}/``) after ``_normalization_url``/``join_url``
        # strip it; the third-party DashScope-compatible proxy
        # ``ai.ctaigw.cn`` answers ``GET /v1/tasks/{id}`` with 200 but
        # ``GET /v1/tasks/{id}/`` with ``500 SYSTEM_ERROR``. A slashed URL
        # therefore makes every poll 500 upstream; ``from_api_response`` then
        # omits ``output`` on the non-200, so ``status.output.task_status``
        # raises ``AttributeError`` and surfaces as a 502 here. Issuing the GET
        # ourselves (no trailing slash) matches the working curl example and
        # lets us defend against non-200 / missing-output bodies cleanly.
        base = (self._video_base or "https://dashscope.aliyuncs.com/api/v1").rstrip("/")
        url = f"{base}/tasks/{task_id}"
        try:
            resp = await self._client_video.get(url)
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                f"dashscope video poll timed out: {exc}", provider="dashscope"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderRequestError(
                f"dashscope video poll transport error: {exc}", provider="dashscope"
            ) from exc

        if resp.status_code >= 400:
            raise ProviderRequestError(
                f"dashscope video poll returned HTTP {resp.status_code}",
                provider="dashscope",
                status_code=_map_status(resp.status_code),
                details={
                    "upstream_status": resp.status_code,
                    "upstream_body": resp.text[:1000],
                },
            )

        try:
            data = resp.json()
        except ValueError as exc:
            raise ProviderRequestError(
                f"dashscope video poll returned non-JSON body: {exc}",
                provider="dashscope",
                details={
                    "upstream_status": resp.status_code,
                    "upstream_body": resp.text[:1000],
                },
            ) from exc

        output = data.get("output") if isinstance(data, dict) else None
        if not isinstance(output, dict):
            # No ``output`` block (e.g. an upstream error envelope) — surface it
            # instead of dereferencing None and crashing.
            code = data.get("code") if isinstance(data, dict) else None
            message = data.get("message") if isinstance(data, dict) else None
            raise ProviderRequestError(
                f"dashscope video poll returned no output: code={code} message={message}",
                provider="dashscope",
                details={
                    "upstream_status": resp.status_code,
                    "upstream_body": str(data)[:1000],
                },
            )

        task_status = output.get("task_status")
        st = _STATUS_MAP.get(task_status, "running")
        task = UnifiedVideoTask(
            task_id=task_id,
            provider=self.name,
            model=output.get("model") or "",
            status=st,
            raw=data,
        )  # type: ignore[arg-type]
        if st == "succeeded":
            video_url = output.get("video_url")
            if video_url:
                task.video_urls = [video_url]
            usage = data.get("usage") if isinstance(data.get("usage"), dict) else None
            if usage:
                task.usage = VideoUsage(
                    video_count=usage.get("video_count"),
                    video_duration=usage.get("video_duration"),
                    extra={
                        k: v
                        for k, v in usage.items()
                        if k not in ("video_count", "video_duration")
                    },
                )
        elif st in ("failed", "cancelled", "expired"):
            task.error = output.get("message") or task_status or st
        return task
