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

from mm_gateway.config import _is_socks_proxy
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
from mm_gateway.providers._http import _map_status, proxy_kwargs
from mm_gateway.providers._sync_image import SyncImageTaskMixin
from mm_gateway.schemas.image import (
    ImageData,
    ImageUsage,
    UnifiedImageRequest,
    UnifiedImageResponse,
)
from mm_gateway.schemas.video import (
    UnifiedVideoRequest,
    UnifiedVideoTask,
    VideoAudioPart,
    VideoImagePart,
    VideoUsage,
    VideoVideoPart,
)

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


# -- Outbound proxy for the aiohttp SDK path -------------------------------- #
# Unlike the other providers (which build an httpx client we can hand a ``proxy=``),
# DashScope's SDK speaks aiohttp internally with no httpx injection point. Its
# shared session uses ``trust_env=True``, so an operator can already steer HTTP
# traffic through env vars — but that is global and silent, and it does not cover
# SOCKS (aiohttp honours only HTTP(S) env proxies). To route a configured
# outbound proxy (per-backend override wins, else the global the registry folded
# into ``extra["outbound_proxy"]``) we build a dedicated aiohttp ClientSession
# whose connector is an ``aiohttp_socks.ProxyConnector`` (HTTP CONNECT and
# SOCKS4/4a/5/5h), and hand it to the SDK via the ``session=`` kwarg:
# ``_build_api_request`` forwards ``session`` to the ``HttpRequest`` as its
# ``_external_aio_session``, which the request then uses in place of the shared
# session. When no proxy is configured, ``None`` is passed and the SDK falls back
# to its shared session (preserving the prior env-var behaviour).

def _aiohttp_proxy_connector(proxy_url: str):
    """Build an aiohttp connector that routes upstream traffic through a proxy.

    ``aiohttp_socks.ProxyConnector.from_url`` accepts ``http://`` and the
    ``socks4/4a/5/5h`` schemes (tunnelling an HTTPS upstream via CONNECT for the
    HTTP proxy type) and forwards its ``**kwargs`` to ``TCPConnector``, so the
    SDK's certifi-backed SSL context is passed for upstream verification. It
    raises ``ValueError`` on an ``https://`` proxy URL (a TLS link to the proxy
    itself is not supported by aiohttp_socks), which is surfaced as a clear
    error rather than a silent misconfiguration.
    """
    try:
        from aiohttp_socks import ProxyConnector
    except ImportError as exc:  # pragma: no cover - config-time guard
        # Unlike the httpx/websockets paths (where HTTP CONNECT is built in),
        # the dashscope SDK speaks aiohttp, and an explicit outbound proxy of
        # *any* scheme (HTTP or SOCKS) is routed through an aiohttp_socks
        # ProxyConnector — so the package is needed here for HTTP proxies too,
        # not only SOCKS. Surface a clear error rather than an opaque ImportError
        # at first use.
        raise ProviderRequestError(
            f"outbound proxy {proxy_url!r} on the dashscope (aiohttp) path "
            "requires the 'aiohttp-socks' package (install mm-gateway[socks]).",
            provider="dashscope",
        ) from exc
    from dashscope.api_entities.aio_session import get_ssl_context
    try:
        return ProxyConnector.from_url(proxy_url, ssl=get_ssl_context())
    except ValueError as exc:  # https:// proxy URL
        raise ProviderRequestError(
            f"outbound proxy {proxy_url!r} (https scheme) is not supported on "
            "the dashscope (aiohttp) path; use http:// or socks5://.",
            provider="dashscope",
        ) from exc


async def _build_proxy_aio_session(proxy_url: str):
    """A dedicated ``aiohttp.ClientSession`` routed through ``proxy_url``.

    ``trust_env`` is False so the explicit connector is authoritative and the
    ambient ``HTTP_PROXY``/``HTTPS_PROXY`` env vars do not double-apply.
    """
    import aiohttp
    return aiohttp.ClientSession(
        connector=_aiohttp_proxy_connector(proxy_url), trust_env=False,
    )


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


# Wan 3.0 and later take the native video-synthesis body: typed ``input.media`` items (``first_frame``,
# ``last_frame``, up to 10 ``reference_image``, 5 ``reference_video`` and 5 ``reference_audio``) and ``parameters``
# (``resolution`` tier, ``ratio``, ``duration`` 2-30 s, ``audio``, ``seed``, ``prompt_extend``, ``watermark``). A
# first or last frame cannot be combined with references, so a request that carries both keeps its frames and drops
# the references. There is no negative prompt
# (https://www.alibabacloud.com/help/en/model-studio/wan3-video-generation-api-reference).
# HappyHorse 1.1 and wan2.7-i2v take the same native body with their own media types and parameters
# (https://www.alibabacloud.com/help/en/model-studio/happyhorse-api-reference/,
# https://www.alibabacloud.com/help/en/model-studio/image-to-video-general-api-reference).
_NATIVE_MEDIA_PREFIXES = ("wan3", "happyhorse", "wan2.7-i2v")
_WAN3_RESOLUTIONS = (("480P", 480), ("720P", 720), ("1080P", 1080))
_WAN3_RATIOS = frozenset({"adaptive", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"})
_WAN3_MAX = {"reference_image": 10, "reference_video": 5, "reference_audio": 5}
WAN3_MIN_DURATION, WAN3_MAX_DURATION = 2, 30
_SEED_MAX = 2147483647
# The new T2I protocol (wan2.6-t2i) is served on the messages-shaped generation surface
# (https://help.aliyun.com/en/model-studio/text-to-image-v2-api-reference).
_GENERATION_T2I_PREFIXES = ("wan2.6-t2i",)
# Image-to-video takes the first frame as img_url and a resolution tier, not a size (the output follows the image's
# ratio); wan2.7-t2v takes a resolution tier and a ratio instead of a size
# (https://help.aliyun.com/en/model-studio/image-to-video-api-reference,
# https://help.aliyun.com/en/model-studio/text-to-video-api-reference).
_RESOLUTION_TIER_PREFIXES = ("wan2.7-t2v",)
# HappyHorse: -t2v takes the prompt alone, -i2v exactly one first_frame (the output keeps its ratio, so no ratio
# parameter) and -r2v 1-9 reference_image items cited as [Image N]; 3-15 s; no audio or prompt_extend parameter.
_HAPPYHORSE_RATIOS = frozenset({"16:9", "9:16", "1:1", "4:3", "3:4", "4:5", "5:4", "9:21", "21:9"})
HAPPYHORSE_MIN_DURATION, HAPPYHORSE_MAX_DURATION = 3, 15
_HAPPYHORSE_MAX_REFERENCES = 9
# wan2.7-i2v: first_frame, last_frame and driving_audio (the audio drives lip-sync and timing); 720P/1080P; 2-15 s.
_WAN27_I2V_RESOLUTIONS = (("720P", 720), ("1080P", 1080))
WAN27_I2V_MIN_DURATION, WAN27_I2V_MAX_DURATION = 2, 15


def native_media_model(model: str | None) -> bool:
    """Models that take the native ``input.media`` body rather than the SDK's ``img_url``."""
    return bool(model) and model.lower().startswith(_NATIVE_MEDIA_PREFIXES)


def resolution_tier_model(model: str | None) -> bool:
    """SDK-path video models sized by a ``resolution`` tier (480P/720P/1080P) rather than a ``size``."""
    m = (model or "").lower()
    return "-i2v" in m or m.startswith(_RESOLUTION_TIER_PREFIXES)


def _wan3_resolution(request: UnifiedVideoRequest, tiers: tuple[tuple[str, int], ...] = _WAN3_RESOLUTIONS) -> str | None:
    if request.resolution:
        tier = request.resolution.strip().upper()
        return tier if tier.endswith("P") else f"{tier}P"
    if request.width is None or request.height is None:
        return None
    short = min(request.width, request.height)
    return min(tiers, key=lambda t: abs(t[1] - short))[0]


def _native_parameters(request: UnifiedVideoRequest, low: int, high: int, *,
                       resolutions: tuple[tuple[str, int], ...] = _WAN3_RESOLUTIONS) -> dict[str, Any]:
    """The ``resolution``, ``duration`` (clamped to ``[low, high]``), ``seed`` and ``watermark`` parameters."""
    parameters: dict[str, Any] = {}
    if resolution := _wan3_resolution(request, resolutions):
        parameters["resolution"] = resolution
    if request.duration is not None and request.duration > 0:
        parameters["duration"] = max(low, min(high, int(round(request.duration))))
    if request.seed is not None and request.seed >= 0:
        parameters["seed"] = request.seed % (_SEED_MAX + 1)
    if request.watermark is not None:
        parameters["watermark"] = request.watermark
    return parameters


def wan3_body(request: UnifiedVideoRequest) -> dict[str, Any]:
    """The native Wan 3.x request body of ``request``."""
    frames: list[dict[str, str]] = []
    references: list[dict[str, str]] = []
    for part in request.content:
        item = part.root
        if isinstance(item, VideoImagePart):
            entry = {"type": item.role, "url": item.image_url.url}
            if item.role in ("first_frame", "last_frame"):
                if not any(f["type"] == item.role for f in frames):
                    frames.append(entry)
            else:
                references.append(entry)
        elif isinstance(item, VideoVideoPart):
            references.append({"type": "reference_video", "url": item.video_url.url})
        elif isinstance(item, VideoAudioPart):
            references.append({"type": "reference_audio", "url": item.audio_url.url})
    if frames:
        if references:
            log.info(
                "dashscope_wan3_references_dropped_for_frames",
                model=request.model,
                dropped=len(references),
            )
        media = frames
    else:
        counts: dict[str, int] = {}
        media = []
        for ref in references:
            counts[ref["type"]] = counts.get(ref["type"], 0) + 1
            if counts[ref["type"]] <= _WAN3_MAX[ref["type"]]:
                media.append(ref)
    parameters: dict[str, Any] = {}
    if resolution := _wan3_resolution(request):
        parameters["resolution"] = resolution
    ratio = aspect_ratio(request)
    if ratio in _WAN3_RATIOS:
        parameters["ratio"] = ratio
    if request.duration is not None:
        parameters["duration"] = (
            -1
            if request.duration < 0
            else max(WAN3_MIN_DURATION, min(WAN3_MAX_DURATION, int(round(request.duration))))
        )
    if request.generate_audio is not None:
        parameters["audio"] = bool(request.generate_audio)
    if request.seed is not None:
        parameters["seed"] = request.seed if request.seed == -1 else request.seed % (_SEED_MAX + 1)
    if request.prompt_extend is not None:
        parameters["prompt_extend"] = request.prompt_extend
    if request.watermark is not None:
        parameters["watermark"] = request.watermark
    parameters.update(request.extra)
    payload: dict[str, Any] = {"prompt": request.prompt() or ""}
    if media:
        payload["media"] = media
    return {"model": request.model, "input": payload, "parameters": parameters}


def happyhorse_body(request: UnifiedVideoRequest) -> dict[str, Any]:
    """The native HappyHorse request body of ``request`` (the model id's suffix picks the mode)."""
    model = request.model.lower()
    images = [p.root.image_url.url for p in request.content if isinstance(p.root, VideoImagePart)]
    media: list[dict[str, str]] = []
    if model.endswith("-i2v") and (first := request.first_image()):
        media = [{"type": "first_frame", "url": first}]
    elif model.endswith("-r2v"):
        # Every image of a reference-to-video request is a reference, cited in the prompt as [Image N].
        media = [{"type": "reference_image", "url": url} for url in images[:_HAPPYHORSE_MAX_REFERENCES]]
    parameters = _native_parameters(request, HAPPYHORSE_MIN_DURATION, HAPPYHORSE_MAX_DURATION)
    if not model.endswith("-i2v") and (ratio := aspect_ratio(request)) in _HAPPYHORSE_RATIOS:
        parameters["ratio"] = ratio
    parameters.update(request.extra)
    payload: dict[str, Any] = {"prompt": request.prompt() or ""}
    if media:
        payload["media"] = media
    return {"model": request.model, "input": payload, "parameters": parameters}


def wan27_i2v_body(request: UnifiedVideoRequest) -> dict[str, Any]:
    """The native wan2.7-i2v request body: a first frame, an optional last frame and an optional driving audio.

    Reference images and videos have no wan2.7-i2v counterpart and are dropped (``first_clip`` continuation is not
    a gateway role); the output keeps the frame's ratio.
    """
    media: list[dict[str, str]] = []
    for role in ("first_frame", "last_frame"):
        url = next((p.root.image_url.url for p in request.content
                    if isinstance(p.root, VideoImagePart) and p.root.role == role), None)
        if url:
            media.append({"type": role, "url": url})
    if audios := request.reference_audios():
        media.append({"type": "driving_audio", "url": audios[0]})
    parameters = _native_parameters(request, WAN27_I2V_MIN_DURATION, WAN27_I2V_MAX_DURATION,
                                    resolutions=_WAN27_I2V_RESOLUTIONS)
    if request.prompt_extend is not None:
        parameters["prompt_extend"] = request.prompt_extend
    parameters.update(request.extra)
    payload: dict[str, Any] = {"prompt": request.prompt() or ""}
    if request.negative_prompt:
        payload["negative_prompt"] = request.negative_prompt
    if media:
        payload["media"] = media
    return {"model": request.model, "input": payload, "parameters": parameters}


def native_video_body(request: UnifiedVideoRequest) -> dict[str, Any]:
    """The native body for ``request.model``: HappyHorse, wan2.7-i2v or Wan 3.x."""
    model = request.model.lower()
    if model.startswith("happyhorse"):
        return happyhorse_body(request)
    if model.startswith("wan2.7-i2v"):
        return wan27_i2v_body(request)
    return wan3_body(request)


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
        "wan2.2-t2i-flash",
        "wan2.2-t2i-plus",
        "wan2.5-t2i-preview",
        # Qwen-Image — the native async task path (async_call + wait) is the
        # unrestricted route; sync_call's docstring restricts it to wan2.2-t2i-*.
        "qwen-image-3.0-pro",
        "qwen-image-3.0",
        "qwen-image-2.0-pro",
        "qwen-image-2.0",
        # Wan2.x-image generation family and wan2.6-t2i (the new T2I protocol) —
        # served on the multimodal-generation surface (messages-shaped), not the
        # image-synthesis surface. Operators may also pin other ``*-image``
        # slugs via DASHSCOPE_IMAGE_MODEL.
        "wan2.7-image-pro",
        "wan2.7-image",
        "wan2.6-image",
        "wan2.6-t2i",
    ]
    video_models: ClassVar[list[str]] = [
        # Wan 3.x, HappyHorse and wan2.7-i2v — the native ``input.media`` body (see ``native_video_body``).
        "wan3.0-video",
        "wan3.0-video-prime",
        "happyhorse-1.1-t2v",
        "happyhorse-1.1-i2v",
        "happyhorse-1.1-r2v",
        "wan2.7-i2v",
        # Text-to-video: wan2.7 takes a resolution tier and ratio, the legacy
        # protocol (wan2.6 and earlier) an exact size.
        "wan2.7-t2v",
        "wan2.6-t2v",
        "wan2.5-t2v-preview",
        "wan2.2-t2v-plus",
        "wanx2.1-t2v-turbo",
        "wanx2.1-t2v-plus",
        # Image-to-video: img_url plus a resolution tier.
        "wan2.6-i2v",
        "wan2.6-i2v-flash",
        "wan2.5-i2v-preview",
        "wan2.2-i2v-flash",
        "wan2.2-i2v-plus",
        "wanx2.1-i2v-turbo",
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
        # Resolved effective outbound proxy (backend override, else the global
        # the registry folded in). Routed through the aiohttp SDK session below
        # and the httpx poll clients (both must honour the same proxy or the
        # submit and the poll would leave through different egresses).
        self._proxy_url = backend.extra.get("outbound_proxy")
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
        pkwargs = proxy_kwargs(self._proxy_url)
        self._client_image = httpx.AsyncClient(
            timeout=300.0,
            headers={"Authorization": f"Bearer {self._api_key}"},
            event_hooks=backend_event_hooks(),
            **pkwargs,
        )
        self._client_video = httpx.AsyncClient(
            timeout=300.0,
            headers={"Authorization": f"Bearer {self._api_key}"},
            event_hooks=backend_event_hooks(),
            **pkwargs,
        )

    async def _aio_session(self):
        """A cached proxy-carrying aiohttp session for the SDK path, or None.

        Built lazily on first use (aiohttp binds the session to the running
        loop, so it cannot be created at ``__init__`` time). When no outbound
        proxy is configured this returns ``None`` and the SDK falls back to its
        shared session (which honours env-var proxies via ``trust_env``). The
        session is reused for the provider's lifetime so connection pooling is
        preserved; a closed session (e.g. after an event-loop teardown) is
        rebuilt transparently.
        """
        if not self._proxy_url:
            return None
        sess = getattr(self, "_aio_proxy_session", None)
        if sess is None or sess.closed:
            sess = await _build_proxy_aio_session(self._proxy_url)
            self._aio_proxy_session = sess
        return sess

    @staticmethod
    def _is_generation_model(model: str) -> bool:
        """True iff the model is served on the multimodal-generation surface
        (messages-shaped ``AioImageGeneration``) rather than the
        image-synthesis surface (``AioImageSynthesis``).

        The synthesis surface only serves the legacy T2I protocol (wan2.5-t2i
        and earlier) / qwen-image-*; the Wan2.x-image generation family
        (``wan2.7-image``, ``wan2.7-image-pro``, ``wan2.6-image``, and
        operator-pinned ``wan*-image`` slugs) and the new T2I protocol
        (``wan2.6-t2i``) live on the generation surface, and routing them
        through synthesis returns ``400 InvalidParameter: url error``.
        """
        m = (model or "").lower()
        if "t2i" in m:
            return m.startswith(_GENERATION_T2I_PREFIXES)
        return m.endswith("-image") or (m.startswith("wan") and "-image-" in m)

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
                api_key=self._api_key, base_address=self._image_base,
                session=await self._aio_session(), **kwargs
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
                api_key=self._api_key, base_address=self._image_base,
                session=await self._aio_session(), **kwargs
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
                api_key=self._api_key, base_address=self._image_base,
                session=await self._aio_session(), **kwargs
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
                api_key=self._api_key, base_address=self._image_base,
                session=await self._aio_session(), **kwargs
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
        if native_media_model(request.model):
            return await self._create_native_video_task(request)
        kwargs: dict[str, Any] = {
            "model": request.model,
            "prompt": request.prompt() or "",
        }
        if resolution_tier_model(request.model):
            if resolution := _wan3_resolution(request):
                kwargs["resolution"] = resolution
            if "-i2v" not in request.model.lower() and (ratio := aspect_ratio(request)):
                kwargs["ratio"] = ratio
        else:
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
                api_key=self._api_key, base_address=self._video_base,
                session=await self._aio_session(), **kwargs
            )
        except Exception as exc:
            raise ProviderRequestError(
                f"dashscope video submit failed: {exc}", provider="dashscope"
            ) from exc

        task_id = getattr(getattr(resp, "output", None), "task_id", None)
        if not task_id:
            # A rejected submit (an auth error, a wrong API root answering 503 with an
            # empty envelope) comes back without ``output``: say what the upstream
            # answered instead of failing on the missing task id.
            status = getattr(resp, "status_code", None)
            code = getattr(resp, "code", None)
            message = getattr(resp, "message", None)
            raise ProviderRequestError(
                f"dashscope video submit returned HTTP {status}: code={code} message={message}",
                provider="dashscope",
                status_code=_map_status(status) if isinstance(status, int) else 502,
                details={
                    "upstream_status": status,
                    "upstream_code": code,
                    "upstream_message": message,
                },
            )
        return UnifiedVideoTask(
            task_id=task_id, provider=self.name, model=request.model, status="pending"
        )

    async def _create_native_video_task(self, request: UnifiedVideoRequest) -> UnifiedVideoTask:
        """Submit a Wan 3.x, HappyHorse or wan2.7-i2v task with the native body (``input.media`` + ``parameters``).

        The SDK's ``AioVideoSynthesis`` only knows ``img_url``: last frames, reference images, videos and audio, the
        ``audio`` switch and the resolution tier never reached the model. The poll stays the hand-rolled
        ``get_video_task``.
        """
        body = native_video_body(request)
        base = (self._video_base or "https://dashscope.aliyuncs.com/api/v1").rstrip("/")
        url = f"{base}/services/aigc/video-generation/video-synthesis"
        try:
            resp = await self._client_video.post(url, json=body, headers={"X-DashScope-Async": "enable"})
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(f"dashscope video submit timed out: {exc}", provider="dashscope") from exc
        except httpx.HTTPError as exc:
            raise ProviderRequestError(
                f"dashscope video submit failed: {exc}", provider="dashscope"
            ) from exc
        try:
            data = resp.json()
        except ValueError:
            data = {}
        output = data.get("output") if isinstance(data, dict) else None
        task_id = output.get("task_id") if isinstance(output, dict) else None
        if resp.status_code >= 400 or not task_id:
            code = data.get("code") if isinstance(data, dict) else None
            message = data.get("message") if isinstance(data, dict) else None
            raise ProviderRequestError(
                f"dashscope video submit returned HTTP {resp.status_code}: code={code} message={message}",
                provider="dashscope",
                status_code=_map_status(resp.status_code) if resp.status_code >= 400 else 502,
                details={
                    "upstream_status": resp.status_code,
                    "upstream_code": code,
                    "upstream_message": message,
                },
            )
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
