"""ElevenLabs music provider — uses the official ``elevenlabs`` SDK.

ElevenLabs' ``client.music.compose`` is synchronous: it streams the generated
audio bytes back from a single ``POST /v1/music`` (an async generator). There
is no task id to poll. To give the gateway a uniform task-based surface across
all music providers, this adapter mints a gateway-local id at create time,
records the request in an in-memory store (like the Stability SVD video
adapter), and runs the blocking SDK call on the first poll — the synthetic
task moves ``pending -> running -> succeeded`` as the stream completes.
"""

from __future__ import annotations

import base64
import time
import uuid
from typing import Any, ClassVar

import httpx

from mm_gateway.core.base import MusicProvider
from mm_gateway.core.exceptions import (
    ProviderNotConfiguredError,
    ProviderRequestError,
    TaskFailedError,
)
from mm_gateway.observability.httplog import backend_event_hooks
from mm_gateway.observability.logging import get_logger
from mm_gateway.providers._http import proxy_kwargs
from mm_gateway.schemas.music import MusicUsage, UnifiedMusicRequest, UnifiedMusicTask

log = get_logger("provider.elevenlabs")

# In-memory store for the synthetic music "tasks". Single-process only; a real
# deployment would use a durable task store — see tasks/store.py.
_MUSIC_TASKS: dict[str, dict[str, Any]] = {}

# SDK output-format enum accepts 'auto' or 'codec_samplerate_bitrate' strings
# like 'mp3_44100_128'. We forward the unified audio_quality ('44100_128') with
# the codec prefix derived from audio_format ('mp3' default).
_DEFAULT_MODEL = "music_v2"
_CODEC_BY_FORMAT = {"wav": "wav", "mp3": "mp3", "ogg": "ogg", "aac": "aac"}


class ElevenLabsProvider(MusicProvider):
    name = "elevenlabs"
    music_models: ClassVar[list[str]] = ["music_v1", "music_v2"]

    def __init__(self, backend):
        super().__init__(backend)
        if not backend.api_key:
            raise ProviderNotConfiguredError("elevenlabs")
        # Lazy import: the SDK is an optional dependency for non-elevenlabs
        # deployments. Importing at construction time (only when configured)
        # keeps unconfigured gateways from requiring it.
        from elevenlabs import AsyncElevenLabs

        self._api_key = backend.api_key
        self._client = AsyncElevenLabs(
            api_key=backend.api_key,
            base_url=backend.base_url or None,
            timeout=240.0,
            httpx_client=httpx.AsyncClient(event_hooks=backend_event_hooks(),
                                            **proxy_kwargs(backend.extra.get("outbound_proxy"))),
        )

    @staticmethod
    def _output_format(request: UnifiedMusicRequest) -> str | None:
        """Map unified knobs to an ElevenLabs output_format string.

        ElevenLabs wants ``codec_samplerate_bitrate`` (e.g. 'mp3_44100_128') or
        'auto'. If the caller gave an explicit audio_quality we trust it;
        otherwise default to 'auto' so the SDK picks the right format for the
        model (mp3_44100_128 for v1, mp3_48000_192 for v2).
        """
        if request.sample_rate_hz is not None or request.bitrate_kbps is not None:
            codec = _CODEC_BY_FORMAT.get((request.audio_format or "mp3").lower(), "mp3")
            sample_rate = request.sample_rate_hz or 44100
            bitrate = request.bitrate_kbps or 128
            return f"{codec}_{sample_rate}_{bitrate}"
        if request.audio_quality:
            codec = _CODEC_BY_FORMAT.get((request.audio_format or "mp3").lower(), "mp3")
            return f"{codec}_{request.audio_quality}"
        if request.audio_format:
            # A bare codec like 'wav' with no quality hint -> 'auto' (SDK picks).
            return "auto"
        return None

    async def create_music_task(self, request: UnifiedMusicRequest) -> UnifiedMusicTask:
        prompt = request.generation_prompt()
        if not prompt and not request.lyrics:
            raise ProviderRequestError(
                "elevenlabs music requires a prompt (text part)", provider="elevenlabs",
                status_code=400,
            )
        task_id = f"el-{uuid.uuid4().hex}"
        _MUSIC_TASKS[task_id] = {
            "model": request.model or _DEFAULT_MODEL,
            "prompt": prompt or request.lyrics,
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
                f"elevenlabs task {task_id} not found", provider="elevenlabs", status_code=404,
            )
        if rec["status"] in ("succeeded", "failed"):
            return UnifiedMusicTask(
                task_id=task_id, provider=self.name, model=rec["model"], status=rec["status"],
                audio_b64=rec.get("audio_b64"), audio_media_type=rec.get("audio_media_type"),
                error=rec.get("error"), created_at=rec["created_at"],
                completed_at=rec.get("completed_at"),
            )
        # Run the blocking SDK stream now.
        rec["status"] = "running"
        request: UnifiedMusicRequest = rec["request"]
        try:
            audio_bytes = await self._compose(request, rec["model"])
        except Exception as exc:
            rec["status"] = "failed"
            rec["error"] = str(exc)
            raise ProviderRequestError(
                f"elevenlabs music failed: {exc}", provider="elevenlabs", status_code=502
            ) from exc
        if not audio_bytes:
            rec["status"] = "failed"
            rec["error"] = "no audio returned"
            raise TaskFailedError("elevenlabs music returned no audio", provider="elevenlabs")
        rec["status"] = "succeeded"
        rec["completed_at"] = int(time.time())
        media_type = self._media_type(request)
        rec["audio_media_type"] = media_type
        rec["audio_b64"] = base64.b64encode(audio_bytes).decode()
        return UnifiedMusicTask(
            task_id=task_id, provider=self.name, model=rec["model"], status="succeeded",
            audio_b64=rec["audio_b64"], audio_media_type=media_type,
            created_at=rec["created_at"], completed_at=rec["completed_at"],
            usage=MusicUsage(duration=request.duration),
        )

    async def _compose(self, request: UnifiedMusicRequest, model: str) -> bytes:
        prompt = request.generation_prompt() or ""
        if request.lyrics:
            prompt = f"{prompt}\n\nLyrics:\n{request.lyrics}".strip()
        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "model_id": model,
        }
        fmt = self._output_format(request)
        if fmt:
            kwargs["output_format"] = fmt
        if request.duration is not None:
            # ElevenLabs takes milliseconds in [3000, 600000].
            kwargs["music_length_ms"] = max(3000, min(600000, int(request.duration * 1000)))
        if request.seed is not None:
            kwargs["seed"] = request.seed
        if request.is_instrumental is not None:
            kwargs["force_instrumental"] = request.is_instrumental
        if request.respect_section_durations is not None:
            kwargs["respect_sections_durations"] = request.respect_section_durations
        if request.provenance is not None:
            kwargs["sign_with_c_2_pa"] = request.provenance
        # Pass any provider-specific knobs the caller stashed in extra through
        # the SDK (e.g. finetune_id, store_for_inpainting).
        for k in ("finetune_id", "respect_sections_durations",
                  "store_for_inpainting", "sign_with_c_2_pa"):
            if k in request.extra:
                kwargs[k] = request.extra[k]
        chunks = []
        # ``compose`` is an async generator (an async iterator of bytes) that
        # wraps an async context manager internally. Iterate it directly to
        # drain the stream.
        async for chunk in self._client.music.compose(**kwargs):
            chunks.append(chunk)
        return b"".join(chunks)

    @staticmethod
    def _media_type(request: UnifiedMusicRequest) -> str:
        fmt = (request.audio_format or "mp3").lower()
        return {"mp3": "audio/mpeg", "wav": "audio/wav", "ogg": "audio/ogg",
                "aac": "audio/aac"}.get(fmt, "audio/mpeg")
