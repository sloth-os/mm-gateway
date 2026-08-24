"""Google Vertex AI provider — the Gemini Enterprise Agent Platform surface.

Vertex AI exposes the **same** generative models as the AI Studio (Gemini
Developer) surface — Imagen (image), Veo (video) and Lyria (music) — and is
reached through the **same** ``google-genai`` SDK, just with the client built
in Vertex mode. The SDK's own docstring calls Vertex the "Gemini Enterprise
Agent Platform". So this adapter reuses the AI Studio adapter's Imagen/Veo
logic (:class:`~mm_gateway.providers._genai_media.GenaiImageVideoMixin`) and
the shared Lyria request/response helpers
(:mod:`~mm_gateway.providers._lyria`) verbatim; only the ``genai.Client``
construction differs.

Authentication is **Application Default Credentials via a service-account JSON
key** (no API key). Two ways to supply the key, resolved in this order:

* ``backend.extra["credentials_json"]`` — the **raw JSON content** of an SA key.
  Used by CI, where a GitHub secret holds the file contents verbatim. Loaded
  with ``google.auth.load_credentials_from_dict`` (no temp file written).
* ``backend.extra["credentials_file"]`` — a **filesystem path** to an SA key
  JSON. Used by YAML config deployments. Loaded with
  ``google.auth.load_credentials_from_file``.
* Otherwise the SDK's own ADC resolution (``google.auth.default``) is used, so a
  deployment that has run ``gcloud auth application-default login`` (or runs on
  GCE/GKE/Workload Identity, or sets ``GOOGLE_APPLICATION_CREDENTIALS``) works
  with no explicit key — the key file is discovered from the environment.

In every case the resulting ``Credentials`` are passed straight to
``genai.Client(credentials=...)``. The **project** comes from the explicit
``VERTEX_PROJECT`` (or the SA key's own ``project_id`` when one is loaded);
``VERTEX_LOCATION`` optionally pins the region (e.g. ``us-central1``), which
selects the ``https://{location}-aiplatform.googleapis.com`` endpoint.

A location is **not required**: when none is pinned, the client defaults to
``"global"`` (the ``https://aiplatform.googleapis.com`` endpoint with no region
prefix). The global endpoint is the one Lyria 3 requires on Vertex — Lyria 3
only serves from the global location (regional requests return an internal
error) — so defaulting to ``global`` makes the music modality work out of the
box, and the image/video models that also accept the global endpoint continue
to work there. Operators who prefer a region can still pin it.

**Music (Lyria)** on Vertex goes through the SDK's Interactions surface
(``client.aio.interactions.create()`` — REST ``POST /v1beta/interactions``),
the same wire shape the AI Studio adapter uses, authenticated with the ADC
bearer token rather than an ``x-goog-api-key``. Lyria is synchronous — a
single Interactions call returns the audio inline — so, like the AI Studio
adapter, we wrap it as a synthetic in-memory task for the gateway's uniform
poll surface.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, ClassVar

import google.auth
import httpx
from google import genai
from google.auth.exceptions import DefaultCredentialsError
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
from mm_gateway.providers._lyria import (
    extract_lyria_output,
    lyria_body,
    lyria_media_type,
)
from mm_gateway.providers._sync_image import SyncImageTaskMixin
from mm_gateway.schemas.music import MusicUsage, UnifiedMusicRequest, UnifiedMusicTask

log = get_logger("provider.vertex")

# Scopes Vertex requests when it resolves ADC itself; matched by SA keys too.
_VERTEX_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

# Lyria 3 on Vertex is served only from the global location — regional requests
# return an internal error (see google-genai issue #2533: "Lyria 3 only supports
# global"). So when an operator pins no region, default the client there.
_DEFAULT_LOCATION = "global"

# Per-client request ceilings (ms), threaded through HttpOptions.timeout so the
# SDK passes them to httpx as per-request timeouts. Without one the SDK resolves
# a None per-request timeout and httpx clamps it to its 5s default, which is far
# too short for generation: Lyria is synchronous and a full song takes well past
# 5s; Veo polls a long-running operation; even Imagen can run a few seconds. The
# music ceiling matches the AI Studio adapter's 240s httpx budget (which itself
# stays under the e2e client's 300s poll budget).
_IMAGE_TIMEOUT_MS = 300_000   # 5 min — Imagen generate_content.
_VIDEO_TIMEOUT_MS = 900_000   # 15 min — Veo long-running video generation.
_MUSIC_TIMEOUT_MS = 240_000   # 4 min — Lyria synchronous Interactions call.

# In-memory store for the synchronous Lyria "tasks". Single-process only.
_MUSIC_TASKS: dict[str, dict[str, Any]] = {}


class VertexProvider(
    SyncImageTaskMixin, GenaiImageVideoMixin, ImageProvider, VideoProvider, MusicProvider
):
    name = "vertex"
    # Vertex serves the same model ids as AI Studio (Imagen/Veo/Lyria).
    image_models: ClassVar[list[str]] = [
        "imagen-4.0-generate-001",
        "imagen-3.0-generate-001",
        "gemini-2.5-flash-image",
    ]
    video_models: ClassVar[list[str]] = [
        "veo-2.0-generate-001",
        "veo-3.0-generate-001",
        "veo-3.1-generate-preview",
    ]
    music_models: ClassVar[list[str]] = ["lyria-3"]

    def __init__(self, backend):
        super().__init__(backend)
        credentials, project, location = self._resolve_credentials(backend)
        if credentials is None:
            raise ProviderNotConfiguredError(
                "vertex",
                "Vertex requires a service-account JSON key (credentials_json or "
                "credentials_file) or application default credentials.",
            )
        # A region is optional: the global endpoint (location="global") is the
        # one Lyria 3 requires, and the image/video models accept it too.
        location = location or _DEFAULT_LOCATION
        image_base = backend.base_url or None
        video_base = backend.extra.get("video_base_url") or image_base
        music_base = backend.extra.get("music_base_url") or image_base
        self._client = self._build_vertex_client(credentials, project, location, image_base, _IMAGE_TIMEOUT_MS)
        self._client_video = self._build_vertex_client(credentials, project, location, video_base, _VIDEO_TIMEOUT_MS)
        # Lyria (Interactions) gets its own client so an operator can pin a
        # music-specific base_url; it defaults to the same global client. Its
        # timeout matches the AI Studio adapter's httpx budget (Lyria is
        # synchronous but a full song can take well over the SDK/httpx 5s
        # default — without it the Interactions call times out client-side).
        self._client_music = self._build_vertex_client(credentials, project, location, music_base, _MUSIC_TIMEOUT_MS)

    @staticmethod
    def _resolve_credentials(backend) -> tuple[Any, str | None, str | None]:
        """Resolve ADC credentials, project, and location from the backend.

        Returns ``(credentials, project, location)``. ``credentials`` is None
        when no SA key and no ambient ADC are available. ``project`` is the
        explicit VERTEX_PROJECT or the SA key's project_id; ``location`` is
        VERTEX_LOCATION (may be None — the client then defaults to "global").
        """
        project = backend.extra.get("project") or None
        location = backend.extra.get("location") or None
        cred_json = backend.extra.get("credentials_json")
        cred_file = backend.extra.get("credentials_file")
        try:
            if cred_json:
                creds, pid = google.auth.load_credentials_from_dict(
                    json.loads(cred_json) if isinstance(cred_json, str) else cred_json,
                    scopes=_VERTEX_SCOPES,
                )
            elif cred_file:
                creds, pid = google.auth.load_credentials_from_file(cred_file, scopes=_VERTEX_SCOPES)
            else:
                # Ambient ADC: GOOGLE_APPLICATION_CREDENTIALS env or the
                # metadata server (GCE/GKE/Workload Identity).
                creds, pid = google.auth.default(scopes=_VERTEX_SCOPES)
        except (DefaultCredentialsError, ValueError, OSError) as exc:
            log.info("vertex_adc_unavailable", error=str(exc))
            return None, project, location
        return creds, project or pid, location

    @staticmethod
    def _build_vertex_client(credentials, project, location, base_url, timeout_ms):
        kwargs: dict[str, Any] = {
            "vertexai": True,
            "credentials": credentials,
            "project": project,
            "location": location,
        }
        # Inject an httpx client whose event hooks log the backend request/
        # response (curl format + masked sensitive headers), matching the AI
        # Studio adapter. base_url overrides the endpoint when an operator pins
        # a regional host via VERTEX_*_BASE_URL. ``timeout`` (ms) is what the SDK
        # threads down to httpx as the per-request timeout; the SDK resolves
        # ``None`` to a None per-request timeout, which httpx clamps to its 5s
        # default — far too short for Lyria's synchronous generation — so each
        # client pins a ceiling (matching the AI Studio adapter's 240s budget).
        http_kwargs: dict[str, Any] = {
            "httpxAsyncClient": httpx.AsyncClient(event_hooks=backend_event_hooks()),
        }
        if base_url:
            kwargs["http_options"] = types.HttpOptions(base_url=base_url, timeout=timeout_ms, **http_kwargs)
        else:
            kwargs["http_options"] = types.HttpOptions(timeout=timeout_ms, **http_kwargs)
        return genai.Client(**kwargs)

    # -- Lyria music ------------------------------------------------------- #

    async def create_music_task(self, request: UnifiedMusicRequest) -> UnifiedMusicTask:
        prompt = request.generation_prompt()
        if not prompt and not request.lyrics:
            raise ProviderRequestError(
                "vertex lyria music requires a prompt (text part)", provider="vertex",
                status_code=400,
            )
        task_id = f"lyria-{uuid.uuid4().hex}"
        _MUSIC_TASKS[task_id] = {
            "model": request.model or "lyria-3",
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
                f"vertex music task {task_id} not found", provider="vertex", status_code=404,
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
        body = lyria_body(request)
        try:
            # Vertex serves Lyria through the SDK's Interactions surface
            # (client.aio.interactions.create → POST /v1beta/interactions),
            # authenticated with the ADC bearer token the SDK injects. The body
            # shape is the same one the AI Studio adapter sends.
            interaction = await self._client_music.aio.interactions.create(**body)
        except Exception as exc:
            rec["status"] = "failed"; rec["error"] = str(exc)
            raise ProviderRequestError(
                f"vertex music request failed: {exc}", provider="vertex", status_code=502
            ) from exc
        audio_b64, lyrics = extract_lyria_output(interaction)
        if not audio_b64:
            rec["status"] = "failed"
            rec["error"] = "no audio in lyria response"
            raise TaskFailedError("vertex lyria returned no audio", provider="vertex")
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
