"""Google provider — Imagen (image), Veo (video) and Lyria (music).

Image and video go through the ``google-genai`` SDK (``client.aio.models`` and
``client.aio.operations``). Music is served by the **Lyria 3** Interactions
API, which the SDK does not yet expose ergonomically; this adapter speaks that
one REST surface directly over httpx against the same
``generativelanguage.googleapis.com`` host the SDK uses, with the API key as a
``?key=`` query parameter. Lyria is synchronous — a single ``predictInteractions``
call returns the audio inline — so, like ElevenLabs/MiniMax, we wrap it as a
synthetic in-memory task for the gateway's uniform poll surface.
"""

from __future__ import annotations

import asyncio
import base64
import time
import uuid
from typing import Any, ClassVar

import httpx
from google import genai
from google.genai import types

from mm_gateway.core.base import ImageProvider, MusicProvider, VideoProvider
from mm_gateway.core.exceptions import ProviderNotConfiguredError, ProviderRequestError, TaskFailedError
from mm_gateway.observability.logging import get_logger
from mm_gateway.providers._sync_image import SyncImageTaskMixin
from mm_gateway.schemas.image import ImageData, ImageUsage, UnifiedImageRequest, UnifiedImageResponse
from mm_gateway.schemas.music import MusicUsage, UnifiedMusicRequest, UnifiedMusicTask
from mm_gateway.schemas.video import UnifiedVideoRequest, UnifiedVideoTask

log = get_logger("provider.google")

# In-memory store for the synchronous Lyria "tasks". Single-process only.
_MUSIC_TASKS: dict[str, dict[str, Any]] = {}

# Host the google-genai SDK targets by default; overridden by base_url when set.
_GLM_BASE = "https://generativelanguage.googleapis.com"


class GoogleProvider(SyncImageTaskMixin, ImageProvider, VideoProvider, MusicProvider):
    name = "google"
    image_models = ["imagen-4.0-generate-001", "imagen-3.0-generate-001", "gemini-2.5-flash-image"]
    video_models = ["veo-2.0-generate-001", "veo-3.0-generate-001", "veo-3.1-generate-preview"]
    music_models: ClassVar[list[str]] = ["lyria-3"]

    def __init__(self, backend):
        super().__init__(backend)
        if not backend.api_key:
            raise ProviderNotConfiguredError("google")
        kwargs: dict[str, Any] = {"api_key": backend.api_key}
        if backend.base_url:
            kwargs["http_options"] = types.HttpOptions(base_url=backend.base_url)
        self._client = genai.Client(**kwargs)
        # Lyria REST surface. Prefer a music-specific base_url if the operator
        # split Google's modalities; otherwise the same host the SDK uses.
        self._music_base = (backend.extra.get("music_base_url")
                            or backend.base_url or _GLM_BASE).rstrip("/")
        self._api_key = backend.api_key

    async def _generate_image(self, request: UnifiedImageRequest) -> UnifiedImageResponse:
        model = request.model
        try:
            if model.startswith("gemini"):
                return await self._generate_content_image(request)
            return await self._generate_imagen(request)
        except Exception as exc:  # noqa: BLE001
            raise ProviderRequestError(f"google image failed: {exc}", provider="google") from exc

    async def _generate_imagen(self, request: UnifiedImageRequest) -> UnifiedImageResponse:
        cfg_fields: dict[str, Any] = {}
        if request.n:
            cfg_fields["number_of_images"] = request.n
        if request.aspect_ratio:
            cfg_fields["aspect_ratio"] = request.aspect_ratio
        if request.negative_prompt:
            cfg_fields["negative_prompt"] = request.negative_prompt
        if request.seed is not None:
            cfg_fields["seed"] = request.seed
        if request.guidance_scale is not None:
            cfg_fields["guidance_scale"] = request.guidance_scale
        if request.output_format:
            cfg_fields["output_mime_type"] = f"image/{request.output_format}"
        cfg_fields.update(request.extra)
        config = types.GenerateImagesConfig(**cfg_fields)

        resp = await self._client.aio.models.generate_images(
            model=request.model, prompt=request.prompt() or "", config=config
        )
        data: list[ImageData] = []
        for gen in resp.generated_images or []:
            img = gen.image
            b64 = base64.b64encode(img.image_bytes).decode() if img.image_bytes else None
            data.append(ImageData(b64_json=b64, media_type=getattr(img, "mime_type", None) or "image/png"))
        return UnifiedImageResponse(
            created=int(time.time()), data=data, model=request.model,
            provider=self.name, usage=ImageUsage(),
        )

    async def _generate_content_image(self, request: UnifiedImageRequest) -> UnifiedImageResponse:
        image_config = types.ImageConfig()
        if request.aspect_ratio:
            image_config.aspect_ratio = request.aspect_ratio
        config = types.GenerateContentConfig(response_modalities=["IMAGE"], image_config=image_config)
        resp = await self._client.aio.models.generate_content(
            model=request.model, contents=request.prompt() or "", config=config
        )
        data: list[ImageData] = []
        for cand in resp.candidates or []:
            for part in (cand.content.parts or []):
                if part.inline_data and part.inline_data.data:
                    b64 = base64.b64encode(part.inline_data.data).decode()
                    data.append(ImageData(b64_json=b64, media_type=part.inline_data.mime_type or "image/png"))
        return UnifiedImageResponse(
            created=int(time.time()), data=data, model=request.model,
            provider=self.name, usage=ImageUsage(),
        )

    async def create_video_task(self, request: UnifiedVideoRequest) -> UnifiedVideoTask:
        source_fields: dict[str, Any] = {}
        if request.prompt():
            source_fields["prompt"] = request.prompt()
        if request.first_image():
            source_fields["image"] = types.Image(image_uri=request.first_image())
        source = types.GenerateVideosSource(**source_fields)

        cfg: dict[str, Any] = {}
        if request.ratio:
            cfg["aspect_ratio"] = request.ratio
        if request.resolution:
            cfg["resolution"] = request.resolution
        if request.duration is not None:
            cfg["duration_seconds"] = int(request.duration)
        if request.seed is not None:
            cfg["seed"] = request.seed
        if request.generate_audio is not None:
            cfg["generate_audio"] = request.generate_audio
        if request.negative_prompt:
            cfg["negative_prompt"] = request.negative_prompt
        if request.last_image():
            cfg["last_frame"] = types.Image(image_uri=request.last_image())
        cfg.update(request.extra)
        config = types.GenerateVideosConfig(**cfg)

        try:
            op = await self._client.aio.models.generate_videos(
                model=request.model, source=source, config=config
            )
        except Exception as exc:  # noqa: BLE001
            raise ProviderRequestError(f"google video create failed: {exc}", provider="google") from exc
        return UnifiedVideoTask(
            task_id=op.name or op.response and "unknown",
            provider=self.name, model=request.model, status="pending",
            raw={"operation_name": op.name},
        )

    async def get_video_task(self, task_id: str) -> UnifiedVideoTask:
        # Reconstruct a handle for the LRO so we can poll by name.
        op = types.Operation(name=task_id)  # type: ignore[arg-type]
        try:
            op = await self._client.aio.operations.get(op)
        except Exception as exc:  # noqa: BLE001
            raise ProviderRequestError(f"google video poll failed: {exc}", provider="google") from exc
        status: str = "running" if not op.done else "succeeded"
        task = UnifiedVideoTask(task_id=task_id, provider=self.name, model="", status=status)  # type: ignore[arg-type]
        if op.done and op.response and getattr(op.response, "generated_videos", None):
            urls: list[str] = []
            for v in op.response.generated_videos:
                if v.video and v.video.uri:
                    urls.append(v.video.uri)
            task.video_urls = urls
        elif op.done and getattr(op, "error", None):
            task.status = "failed"
            task.error = str(op.error)
        return task

    # -- Lyria music ------------------------------------------------------- #

    async def create_music_task(self, request: UnifiedMusicRequest) -> UnifiedMusicTask:
        prompt = request.prompt()
        if not prompt:
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
            url = (f"{self._music_base}/v1beta/models/{rec['model']}:predictInteractions"
                   f"?key={self._api_key}")
            async with httpx.AsyncClient(timeout=240.0) as c:
                resp = await c.post(url, json=body, headers={"Content-Type": "application/json"})
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
        rec["status"] = "succeeded"
        rec["completed_at"] = int(time.time())
        rec["audio_b64"] = audio_b64
        rec["audio_media_type"] = "audio/wav"
        if lyrics:
            rec["lyrics"] = lyrics
        if request.duration is not None:
            rec["usage"] = MusicUsage(duration=int(request.duration))
        return UnifiedMusicTask(
            task_id=task_id, provider=self.name, model=rec["model"], status="succeeded",
            audio_b64=audio_b64, audio_media_type="audio/wav", lyrics=rec.get("lyrics"),
            created_at=rec["created_at"], completed_at=rec["completed_at"],
            usage=rec.get("usage"),
        )

    def _lyria_body(self, request: UnifiedMusicRequest) -> dict[str, Any]:
        # The unified content[] maps directly onto Lyria's input parts: text ->
        # {type:"text", text}; inline reference images (extra["images"]) ->
        # {type:"image", mime_type, data}.
        parts: list[dict[str, Any]] = []
        for p in request.content:
            root = p.root
            if hasattr(root, "text"):
                parts.append({"type": "text", "text": root.text})
        for img in request.extra.get("images", []) or []:
            parts.append({"type": "image", "mime_type": img.get("mime_type", "image/jpeg"),
                          "data": img.get("data")})
        if not parts:
            parts.append({"type": "text", "text": request.prompt() or ""})
        body: dict[str, Any] = {"model": request.model, "input": parts}
        config: dict[str, Any] = {"response_modalities": ["AUDIO"]}
        if request.audio_format:
            config["response_format"] = {"type": request.audio_format}
        if request.negative_prompt:
            config["negative_prompt"] = request.negative_prompt
        if request.seed is not None:
            config["seed"] = request.seed
        if request.guidance_scale is not None:
            config["guidance_scale"] = request.guidance_scale
        if request.n is not None:
            config["number_of_outputs"] = request.n
        config.update(request.extra.get("lyria_config") or {})
        body["config"] = config
        return body


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
