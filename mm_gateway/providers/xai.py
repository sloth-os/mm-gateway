"""xAI provider — Grok Imagine image and video via the xAI REST API.

xAI publishes an OpenAPI spec for image/video generation at
``https://api.x.ai/api-docs/openapi.json``. This adapter speaks that REST API
directly over HTTP (image: ``POST /v1/images/generations``; video:
``POST /v1/videos/generations`` + ``GET /v1/videos/{request_id}``) rather than
the gRPC ``xai_sdk`` client, so an operator can point the gateway at any
xAI-compatible HTTP endpoint (the real ``api.x.ai`` or a proxy) via
``XAI_BASE_URL`` — the same 1:1 env contract every other REST adapter honours.

The ``xai_sdk`` package is no longer imported here.
"""

from __future__ import annotations

import time
from typing import Any, ClassVar

import httpx

from mm_gateway.core.base import ImageProvider, VideoProvider
from mm_gateway.core.exceptions import (
    ProviderNotConfiguredError,
    ProviderRequestError,
    ProviderTimeoutError,
)
from mm_gateway.observability.logging import get_logger
from mm_gateway.providers._http import _map_status, make_client, request_json
from mm_gateway.providers._sync_image import SyncImageTaskMixin
from mm_gateway.schemas.image import (
    ImageData,
    ImageUsage,
    UnifiedImageRequest,
    UnifiedImageResponse,
)
from mm_gateway.schemas.video import UnifiedVideoRequest, UnifiedVideoTask, VideoUsage

log = get_logger("provider.xai")

# MediaUsage.cost_in_usd_ticks: one USD cent = 100,000,000 ticks,
# so one US dollar = 10,000,000,000 ticks.
_TICKS_PER_USD = 10_000_000_000


def _cost_from_ticks(ticks: Any) -> float | None:
    try:
        return float(ticks) / _TICKS_PER_USD
    except (TypeError, ValueError):
        return None


class XAIProvider(SyncImageTaskMixin, ImageProvider, VideoProvider):
    name = "xai"
    image_models: ClassVar[list[str]] = [
        "grok-imagine-image",
        "grok-imagine-image-pro",
        "grok-imagine-image-quality",
    ]
    video_models: ClassVar[list[str]] = [
        "grok-imagine-video",
        "grok-imagine-video-1.5-preview",
    ]

    def __init__(self, backend):
        super().__init__(backend)
        if not backend.api_key:
            raise ProviderNotConfiguredError("xai")
        # Normalise the base URL: xAI's paths are all "/v1/...". A bare host
        # (the real https://api.x.ai, or a proxy like https://ai.xmiaom.com)
        # is what the spec assumes; strip a trailing "/v1" an operator may have
        # copied from another provider so we always build "/v1/..." ourselves.
        base = (backend.base_url or "https://api.x.ai").rstrip("/")
        base = base.removesuffix("/v1")
        self._base = base
        self._client = make_client(
            self._base,
            timeout=180.0,
            headers={"authorization": f"Bearer {backend.api_key}"},
        )

    async def _generate_image(
        self, request: UnifiedImageRequest
    ) -> UnifiedImageResponse:
        body: dict[str, Any] = {"model": request.model, "prompt": request.prompt() or ""}
        if request.n:
            # xAI caps n at 10 (GenerateImageRequest.n maximum: 10); clamp so an
            # out-of-range request is honoured up to the cap rather than 422->502.
            body["n"] = min(request.n, 10)
        if request.response_format:
            body["response_format"] = request.response_format
        if request.aspect_ratio:
            body["aspect_ratio"] = request.aspect_ratio
        if request.resolution:
            body["resolution"] = request.resolution
        if request.user:
            body["user"] = request.user
        body.update(request.extra)

        data = await request_json(
            self._client,
            "POST",
            "/v1/images/generations",
            provider="xai",
            json=body,
        )

        items = [
            ImageData(
                url=d.get("url"),
                b64_json=d.get("b64_json"),
                media_type=d.get("mime_type"),
            )
            for d in (data.get("data") or [])
        ]
        usage = None
        if data.get("usage"):
            usage = ImageUsage(
                cost=_cost_from_ticks(data["usage"].get("cost_in_usd_ticks"))
            )
        return UnifiedImageResponse(
            created=int(time.time()),
            data=items,
            model=request.model,
            provider=self.name,
            usage=usage,
        )

    async def create_video_task(self, request: UnifiedVideoRequest) -> UnifiedVideoTask:
        body: dict[str, Any] = {
            "model": request.model,
            "prompt": request.prompt() or "",
        }
        if request.duration is not None:
            body["duration"] = int(request.duration)
        if request.ratio:
            body["aspect_ratio"] = request.ratio
        if request.resolution:
            body["resolution"] = request.resolution
        if request.first_image():
            body["image"] = {"url": request.first_image()}
        if request.reference_images():
            body["reference_images"] = [{"url": u} for u in request.reference_images()]
        body.update(request.extra)

        data = await request_json(
            self._client,
            "POST",
            "/v1/videos/generations",
            provider="xai",
            json=body,
        )
        request_id = data.get("request_id")
        if not request_id:
            raise ProviderRequestError(
                "xai video create returned no request_id", provider="xai"
            )
        return UnifiedVideoTask(
            task_id=request_id,
            provider=self.name,
            model=request.model,
            status="pending",
        )

    # GET /v1/videos/{id} returns a flat VideoResponse: {status, model,
    # progress, usage, video, error}. status is "pending" (HTTP 202, no body)
    # | "done" | "failed"; the `video`/`usage`/`error` siblings sit at the same
    # level as `status` (no nested `response` envelope).
    _STATUS_MAP: ClassVar[dict[str, str]] = {
        "pending": "pending",
        "done": "succeeded",
        "failed": "failed",
    }

    async def get_video_task(self, task_id: str) -> UnifiedVideoTask:
        # Poll by hand rather than via request_json: a 202 means "still in
        # progress" and may carry no JSON body, which resp.json() cannot parse.
        try:
            resp = await self._client.get(f"/v1/videos/{task_id}")
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                f"xai video poll timed out: {exc}", provider="xai"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderRequestError(
                f"xai video poll transport error: {exc}", provider="xai"
            ) from exc

        if resp.status_code == 202:
            return UnifiedVideoTask(
                task_id=task_id,
                provider=self.name,
                model="",
                status="running",
            )
        if resp.status_code >= 400:
            raise ProviderRequestError(
                f"xai video poll returned HTTP {resp.status_code}",
                provider="xai",
                status_code=_map_status(resp.status_code),
                details={
                    "upstream_status": resp.status_code,
                    "upstream_body": resp.text[:1000],
                },
            )

        data = resp.json()
        status = self._STATUS_MAP.get(str(data.get("status")), "running")
        # `model` is a flat sibling of `status` (present once the task is no
        # longer pending; the spec notes it is omitted when status is failed,
        # so fall back to "" rather than echoing the create-time model).
        task = UnifiedVideoTask(
            task_id=task_id,
            provider=self.name,
            model=data.get("model") or "",
            status=status,  # type: ignore[arg-type]
            raw=data,
        )
        if status == "succeeded":
            # `video`/`usage` are flat siblings of `status` (no `response` key).
            video = data.get("video") or {}
            if video.get("url"):
                task.video_urls = [video["url"]]
            else:
                # Spec: a "done" task with an empty video URL means moderation
                # was violated (respect_moderation=false, url empty). The task
                # is complete but produced no usable media — surface that as an
                # error instead of a bare silent success with no explanation.
                task.error = "video generation completed with no URL (moderation?)"
            usage = data.get("usage")
            if usage:
                task.usage = VideoUsage(
                    cost=_cost_from_ticks(usage.get("cost_in_usd_ticks"))
                )
        elif status == "failed":
            err = data.get("error")
            if isinstance(err, dict):
                task.error = f"{err.get('code', 'unknown')}: {err.get('message', '')}"
            else:
                task.error = "video generation failed"
        return task
