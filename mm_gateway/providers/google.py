"""Google provider — Imagen (image), Veo (video) and Lyria (music).

Image and video go through the ``google-genai`` SDK (``client.aio.models`` and
``client.aio.operations``). Music is served by the **Lyria 3** Interactions
API (``POST /v1beta/interactions``), spoken directly over httpx against the
same ``generativelanguage.googleapis.com`` host the SDK uses, authenticated
with the ``x-goog-api-key`` header. Lyria is synchronous — a single
Interactions call returns the audio inline — so, like ElevenLabs/MiniMax, we
wrap it as a synthetic in-memory task for the gateway's uniform poll surface.
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
from mm_gateway.providers._sync_image import SyncImageTaskMixin
from mm_gateway.schemas.music import MusicUsage, UnifiedMusicRequest, UnifiedMusicTask

log = get_logger("provider.google")

# In-memory store for the synchronous Lyria "tasks". Single-process only.
_MUSIC_TASKS: dict[str, dict[str, Any]] = {}

# Host the google-genai SDK targets by default; overridden by base_url when set.
_GLM_BASE = "https://generativelanguage.googleapis.com"


class GoogleProvider(SyncImageTaskMixin, GenaiImageVideoMixin, ImageProvider, VideoProvider, MusicProvider):
    name = "google"
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
        self._client = self._build_genai_client(backend.api_key, image_base)
        self._client_video = self._build_genai_client(backend.api_key, video_base)
        # Lyria REST surface. Prefer a music-specific base_url if the operator
        # split Google's modalities; otherwise the same host the SDK uses.
        self._music_base = (backend.extra.get("music_base_url")
                            or backend.base_url or _GLM_BASE).rstrip("/")
        self._api_key = backend.api_key

    @staticmethod
    def _build_genai_client(api_key: str, base_url: str | None):
        kwargs: dict[str, Any] = {"api_key": api_key}
        http_kwargs: dict[str, Any] = {
            # Inject an httpx client whose event hooks log the backend request/
            # response (curl format + masked sensitive headers).
            "httpxAsyncClient": httpx.AsyncClient(event_hooks=backend_event_hooks()),
        }
        kwargs["http_options"] = types.HttpOptions(base_url=base_url, **http_kwargs) if base_url \
            else types.HttpOptions(**http_kwargs)
        return genai.Client(**kwargs)

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
            body = self._lyria_body(request)
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
            async with httpx.AsyncClient(timeout=240.0, event_hooks=backend_event_hooks()) as c:
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
        audio_b64, lyrics = _extract_lyria_output(data)
        if not audio_b64:
            rec["status"] = "failed"
            rec["error"] = "no audio in lyria response"
            raise TaskFailedError("google lyria returned no audio", provider="google")
        media_type = _lyria_media_type(request.audio_format)
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

    def _lyria_body(self, request: UnifiedMusicRequest) -> dict[str, Any]:
        # The canonical inputs map onto Lyria parts. Provider-specific wire
        # names stay here rather than leaking into the public REST schema.
        parts: list[dict[str, Any]] = []
        if prompt := request.generation_prompt():
            parts.append({"type": "text", "text": prompt})
        if request.lyrics:
            parts.append({"type": "text", "text": f"Lyrics:\n{request.lyrics}"})
        for p in request.content:
            root = p.root
            if hasattr(root, "image_url"):
                url = root.image_url.url
                if url.startswith("data:"):
                    header, _, data = url.partition(",")
                    parts.append({
                        "type": "image",
                        "mime_type": header[5:].split(";", 1)[0] or "image/png",
                        "data": data,
                    })
                else:
                    parts.append({
                        "type": "image_url",
                        "image_url": {"url": url},
                        "role": getattr(root, "role", "reference_image"),
                    })
            elif hasattr(root, "audio_url"):
                url = root.audio_url.url
                if url.startswith("data:"):
                    header, _, data = url.partition(",")
                    parts.append({
                        "type": "audio",
                        "mime_type": header[5:].split(";", 1)[0] or "audio/mpeg",
                        "data": data,
                        "role": getattr(root, "role", "reference_audio"),
                    })
                else:
                    parts.append({
                        "type": "audio_url",
                        "audio_url": {"url": url},
                        "role": getattr(root, "role", "reference_audio"),
                    })
        for img in request.extra.get("images", []) or []:
            parts.append({"type": "image", "mime_type": img.get("mime_type", "image/jpeg"),
                          "data": img.get("data")})
        if not parts:
            parts.append({"type": "text", "text": request.lyrics or ""})
        body: dict[str, Any] = {"model": request.model, "input": parts}
        # The Interactions request takes these as top-level fields (not nested in
        # a ``config`` key). ``response_format`` is the AudioResponseFormat
        # envelope: a ``"type": "audio"`` discriminator plus a ``mime_type``
        # drawn from the SDK's output enum (audio/wav, audio/mp3, ...). Omit it
        # entirely when no format is pinned: Lyria's default output is MP3.
        mime = _lyria_request_mime(request.audio_format)
        if mime:
            body["response_format"] = {"type": "audio", "mime_type": mime}
        # ``generation_config`` carries the generation knobs the SDK recognises
        # (``seed``). Other provider-specific knobs ride through best-effort
        # inside ``generation_config`` — unknown fields are ignored upstream.
        generation_config: dict[str, Any] = {}
        if request.negative_prompt:
            generation_config["negative_prompt"] = request.negative_prompt
        if request.seed is not None:
            generation_config["seed"] = request.seed
        if request.guidance_scale is not None:
            generation_config["guidance_scale"] = request.guidance_scale
        if request.n is not None:
            generation_config["number_of_outputs"] = request.n
        generation_config.update(request.extra.get("lyria_config") or {})
        if generation_config:
            body["generation_config"] = generation_config
        return body


def _lyria_request_mime(audio_format: str | None) -> str | None:
    """SDK-enum value for the Interactions ``response_format.mime_type``.

    The output enum (``AudioResponseFormatMimeType``) is
    ``audio/mp3``/``audio/wav``/``audio/ogg_opus``/``audio/l16``/``audio/alaw``/
    ``audio/mulaw`` — note ``audio/mp3``, NOT ``audio/mpeg`` (that lives in the
    broader *input* audio-parts enum and is not accepted for the output format).
    Returns ``None`` to omit ``response_format`` entirely: Lyria emits MP3 by
    default when no envelope is sent, so a bare MP3 request need not pin it.
    """
    if not audio_format:
        return None
    if audio_format == "mp3":
        return "audio/mp3"
    if audio_format == "wav":
        return "audio/wav"
    if audio_format == "ogg_opus":
        return "audio/ogg_opus"
    return f"audio/{audio_format}"


def _lyria_media_type(audio_format: str | None) -> str:
    """Client-facing MIME of the inline audio the Lyria call returns.

    Matches the gateway convention every other music provider uses (minimax /
    elevenlabs / udioapi / mureka all map mp3 -> ``audio/mpeg``, and
    ``rest.py``'s music default is ``audio/mpeg``). The default is MP3 because
    Lyria emits MP3 unless ``response_format`` requests WAV.
    """
    if not audio_format or audio_format == "mp3":
        return "audio/mpeg"
    if audio_format == "wav":
        return "audio/wav"
    if audio_format == "ogg_opus":
        return "audio/ogg"
    return f"audio/{audio_format}"


def _extract_lyria_output(data: dict[str, Any]) -> tuple[str | None, str | None]:
    """Pull the inline audio (base64) and any text/lyrics out of a Lyria
    response. The shape is ``steps[].content[]`` blocks: audio blocks carry
    ``{type:"audio", data, mime_type}``, text blocks ``{type:"text", text}``."""
    audio_b64: str | None = None
    lyrics: str | None = None
    steps = data.get("steps") or data.get("model_output") or []
    if isinstance(steps, dict):
        steps = [steps]
    for step in steps:
        content = (step or {}).get("content") or (step or {}).get("model_output") or []
        if isinstance(content, dict):
            content = [content]
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "audio" and block.get("data") and not audio_b64:
                audio_b64 = block["data"]
            elif btype == "text" and block.get("text") and not lyrics:
                lyrics = block["text"]
    # Some envelopes surface the audio at the top level instead.
    if not audio_b64:
        audio_b64 = data.get("output_audio") or data.get("audio")
    if not lyrics:
        lyrics = data.get("output_text") or data.get("text")
    return audio_b64, lyrics
