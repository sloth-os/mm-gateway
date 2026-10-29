"""Google provider — Gemini/Imagen (image), Gemini Omni and Veo (video) and Lyria (music).

Image and Veo video go through the ``google-genai`` SDK (``client.aio.models`` and
``client.aio.operations``). Music is served by the **Lyria 3** Interactions
API (``POST /v1beta/interactions``), spoken directly over httpx against the
same ``generativelanguage.googleapis.com`` host the SDK uses, authenticated
with the ``x-goog-api-key`` header. Lyria is synchronous — a single
Interactions call returns the audio inline — so, like ElevenLabs/MiniMax, we
wrap it as a synthetic in-memory task for the gateway's uniform poll surface.
Gemini Omni video uses the same surface as a background interaction: the create
returns its id and the poll ``GET``s it (see :mod:`mm_gateway.providers._omni`).
"""

from __future__ import annotations

import time
import uuid
from typing import Any, ClassVar

import httpx
from google import genai
from google.genai import types

from mm_gateway.core.base import ImageProvider, MusicProvider, VideoProvider
from mm_gateway.core.exceptions import (
    ProviderNotConfiguredError,
    ProviderRequestError,
    TaskFailedError,
)
from mm_gateway.observability.httplog import backend_event_hooks
from mm_gateway.observability.logging import get_logger
from mm_gateway.providers._genai_media import GenaiImageVideoMixin
from mm_gateway.providers._http import make_client, proxy_kwargs, request_json
from mm_gateway.providers._lyria import (
    extract_lyria_output,
    lyria_body,
    lyria_media_type,
    lyria_request_mime,
)
from mm_gateway.providers._omni import TASK_PREFIX, omni_body, omni_model, omni_task
from mm_gateway.providers._sync_image import SyncImageTaskMixin
from mm_gateway.schemas.music import MusicUsage, UnifiedMusicRequest, UnifiedMusicTask
from mm_gateway.schemas.video import UnifiedVideoRequest, UnifiedVideoTask

log = get_logger("provider.google")

# In-memory store for the synchronous Lyria "tasks". Single-process only.
_MUSIC_TASKS: dict[str, dict[str, Any]] = {}

# Host the google-genai SDK targets by default; overridden by base_url when set.
_GLM_BASE = "https://generativelanguage.googleapis.com"


class GoogleProvider(SyncImageTaskMixin, GenaiImageVideoMixin, ImageProvider, VideoProvider, MusicProvider):
    name = "google"
    image_models: ClassVar[list[str]] = [
        "gemini-nano-banana-2.1",
        "gemini-3-pro-image",
        "gemini-3.1-flash-image",
        "gemini-3.1-flash-lite-image",
        "gemini-2.5-flash-image",
        "imagen-4.0-generate-001",
        "imagen-3.0-generate-001",
    ]
    video_models: ClassVar[list[str]] = [
        # Gemini Omni Flash is served by the Interactions API (see ``_omni``); the Veo ids left the Gemini API.
        "gemini-omni-1.1-flash",
        "veo-3.1-generate-preview",
        "veo-3.0-generate-001",
        "veo-2.0-generate-001",
    ]
    music_models: ClassVar[list[str]] = ["lyria-3.5", "lyria-3-pro-preview", "lyria-3-clip-preview"]

    def __init__(self, backend):
        super().__init__(backend)
        if not backend.api_key:
            raise ProviderNotConfiguredError("google")
        # Per-modality genai clients honor the sync/async URL split resolved by
        # ``config.py``: image (Imagen/generate_content) uses ``base_url`` (the
        # ``*_IMAGE_BASE_URL`` sync endpoint); video (Veo) uses
        # ``extra["video_base_url"]`` (the ``*_VIDEO_BASE_URL`` async endpoint)
        # when it differs from the image one. The real
        # generativelanguage.googleapis.com serves both at one host, so the two
        # clients collapse unless an operator pins them apart.
        image_base = backend.base_url or None
        video_base = backend.extra.get("video_base_url") or image_base
        # Resolved effective outbound proxy (backend override, else the global
        # the registry folded in); shared by the genai SDK client and the Lyria
        # REST client.
        self._proxy_url = backend.extra.get("outbound_proxy")
        self._client = self._build_genai_client(backend.api_key, image_base, self._proxy_url)
        self._client_video = self._build_genai_client(backend.api_key, video_base, self._proxy_url)
        # Lyria REST surface. Prefer a music-specific base_url if the operator
        # split Google's modalities; otherwise the same host the SDK uses.
        self._music_base = (backend.extra.get("music_base_url")
                            or backend.base_url or _GLM_BASE).rstrip("/")
        self._api_key = backend.api_key
        # Gemini Omni video: the Interactions REST surface on the video host.
        self._client_interactions = make_client(
            (video_base or _GLM_BASE).rstrip("/"), timeout=120.0,
            headers={"x-goog-api-key": backend.api_key}, proxy_url=self._proxy_url,
        )

    @staticmethod
    def _build_genai_client(api_key: str, base_url: str | None, proxy_url: str | None = None):
        kwargs: dict[str, Any] = {"api_key": api_key}
        http_kwargs: dict[str, Any] = {
            # Inject an httpx client whose event hooks log the backend request/
            # response (curl format + masked sensitive headers).
            "httpxAsyncClient": httpx.AsyncClient(event_hooks=backend_event_hooks(),
                                                  **proxy_kwargs(proxy_url)),
        }
        kwargs["http_options"] = types.HttpOptions(base_url=base_url, **http_kwargs) if base_url \
            else types.HttpOptions(**http_kwargs)
        return genai.Client(**kwargs)

    # -- Gemini Omni video (Interactions) ---------------------------------- #

    async def create_video_task(self, request: UnifiedVideoRequest) -> UnifiedVideoTask:
        if not omni_model(request.model):
            return await super().create_video_task(request)
        data = await request_json(self._client_interactions, "POST", "/v1beta/interactions",
                                  provider=self.name, json=omni_body(request))
        if not data.get("id"):
            raise ProviderRequestError("google interaction create returned no id", provider=self.name,
                                       details={"upstream_body": str(data)[:500]})
        return omni_task(f"{TASK_PREFIX}{data['id']}", self.name, request.model, data)

    async def get_video_task(self, task_id: str) -> UnifiedVideoTask:
        if not task_id.startswith(TASK_PREFIX):
            return await super().get_video_task(task_id)
        data = await request_json(self._client_interactions, "GET",
                                  f"/v1beta/interactions/{task_id.removeprefix(TASK_PREFIX)}",
                                  provider=self.name)
        return omni_task(task_id, self.name, str(data.get("model") or ""), data)

    # -- Lyria music ------------------------------------------------------- #

    async def create_music_task(self, request: UnifiedMusicRequest) -> UnifiedMusicTask:
        prompt = request.generation_prompt()
        if not prompt and not request.lyrics:
            raise ProviderRequestError(
                "google lyria music requires a prompt (text part)", provider="google",
                status_code=400,
            )
        task_id = f"lyria-{uuid.uuid4().hex}"
        _MUSIC_TASKS[task_id] = {
            "model": request.model or "lyria-3.5",
            "request": request,
            "status": "pending",
            "created_at": int(time.time()),
        }
        return UnifiedMusicTask(
            task_id=task_id, provider=self.name, model=request.model, status="pending",
            created_at=_MUSIC_TASKS[task_id]["created_at"],
        )

    async def get_music_task(self, task_id: str) -> UnifiedMusicTask:
        rec = _MUSIC_TASKS.get(task_id)
        if rec is None:
            raise ProviderRequestError(
                f"google music task {task_id} not found", provider="google", status_code=404,
            )
        if rec["status"] in ("succeeded", "failed"):
            return UnifiedMusicTask(
                task_id=task_id, provider=self.name, model=rec["model"], status=rec["status"],
                audio_b64=rec.get("audio_b64"), audio_media_type=rec.get("audio_media_type"),
                lyrics=rec.get("lyrics"), error=rec.get("error"),
                created_at=rec["created_at"], completed_at=rec.get("completed_at"),
                usage=rec.get("usage"),
            )
        # Run the blocking Lyria call now.
        rec["status"] = "running"
        request: UnifiedMusicRequest = rec["request"]
        try:
            body = lyria_body(request)
            # Lyria 3 Interactions API: POST /v1beta/interactions — the model
            # travels in the request body, not the path, and there is no
            # :predictInteractions method (that path 404s). Authenticated with
            # the x-goog-api-key header the google-genai SDK uses (rather than
            # the ?key= query form, which the gateway's curl logger would not
            # mask outside CI).
            url = f"{self._music_base}/v1beta/interactions"
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": self._api_key,
            }
            async with httpx.AsyncClient(timeout=240.0, event_hooks=backend_event_hooks(),
                                         **proxy_kwargs(self._proxy_url)) as c:
                resp = await c.post(url, json=body, headers=headers)
        except httpx.HTTPError as exc:
            rec["status"] = "failed"; rec["error"] = str(exc)
            raise ProviderRequestError(
                f"google music transport error: {exc}", provider="google", status_code=502
            ) from exc
        if resp.status_code >= 400:
            rec["status"] = "failed"; rec["error"] = resp.text[:500]
            raise ProviderRequestError(
                f"google music returned HTTP {resp.status_code}", provider="google",
                status_code=502, details={"upstream_body": resp.text[:1000]},
            )
        data = resp.json()
        audio_b64, lyrics = extract_lyria_output(data)
        if not audio_b64:
            rec["status"] = "failed"
            rec["error"] = "no audio in lyria response"
            raise TaskFailedError("google lyria returned no audio", provider="google")
        media_type = lyria_media_type(request.audio_format)
        rec["status"] = "succeeded"
        rec["completed_at"] = int(time.time())
        rec["audio_b64"] = audio_b64
        rec["audio_media_type"] = media_type
        if lyrics:
            rec["lyrics"] = lyrics
        if request.duration is not None:
            rec["usage"] = MusicUsage(duration=int(request.duration))
        return UnifiedMusicTask(
            task_id=task_id, provider=self.name, model=rec["model"], status="succeeded",
            audio_b64=audio_b64, audio_media_type=media_type, lyrics=rec.get("lyrics"),
            created_at=rec["created_at"], completed_at=rec["completed_at"],
            usage=rec.get("usage"),
        )
