"""MiniMax music provider — REST API over ``https://api.minimax.io``.

``POST /v1/music_generation`` is synchronous: a single blocking call returns
``data.status`` 1 (in progress) or 2 (completed) with the audio inline — hex
encoded by default, or a 24h URL when ``output_format`` is ``url``. There is no
job id to poll. As with the ElevenLabs / Stability-SVD adapters, we mint a
gateway-local task id at create time, record the request in an in-memory store,
and run the blocking call on the first poll — the synthetic task moves
``pending -> running -> succeeded`` as the call completes.

Docs: https://platform.minimax.io/docs/api-reference/music-generation
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
from mm_gateway.observability.logging import get_logger
from mm_gateway.providers._http import _map_status
from mm_gateway.schemas.music import MusicUsage, UnifiedMusicRequest, UnifiedMusicTask

log = get_logger("provider.minimax")

_BASE = "https://api.minimax.io"

# In-memory store for the synchronous music "tasks". Single-process only.
_MUSIC_TASKS: dict[str, dict[str, Any]] = {}

# MiniMax audio_setting.format values -> MIME types for the inline bytes.
_MIME_BY_FORMAT = {"mp3": "audio/mpeg", "wav": "audio/wav", "pcm": "audio/pcm"}


class MiniMaxProvider(MusicProvider):
    name = "minimax"
    music_models: ClassVar[list[str]] = ["music-3.0", "music-2.6", "music-cover"]

    def __init__(self, backend):
        super().__init__(backend)
        if not backend.api_key:
            raise ProviderNotConfiguredError("minimax")
        self._api_key = backend.api_key
        self._client = httpx.AsyncClient(
            base_url=backend.base_url or _BASE,
            timeout=300,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )

    async def create_music_task(self, request: UnifiedMusicRequest) -> UnifiedMusicTask:
        prompt = request.prompt()
        # Non-instrumental, non-cover models require lyrics; instrumental/cover
        # require a prompt. Accept whichever the caller supplied and let the
        # upstream reject an inconsistent combination.
        if not prompt and not request.extra.get("lyrics"):
            raise ProviderRequestError(
                "minimax music requires a prompt or lyrics (text part)",
                provider="minimax", status_code=400,
            )
        task_id = f"mm-{uuid.uuid4().hex}"
        _MUSIC_TASKS[task_id] = {
            "model": request.model or "music-3.0",
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
                f"minimax task {task_id} not found", provider="minimax", status_code=404,
            )
        if rec["status"] in ("succeeded", "failed"):
            return UnifiedMusicTask(
                task_id=task_id, provider=self.name, model=rec["model"], status=rec["status"],
                audio_urls=rec.get("audio_urls", []), audio_b64=rec.get("audio_b64"),
                audio_media_type=rec.get("audio_media_type"),
                error=rec.get("error"), created_at=rec["created_at"],
                completed_at=rec.get("completed_at"), usage=rec.get("usage"),
            )
        # Run the blocking call now.
        rec["status"] = "running"
        request: UnifiedMusicRequest = rec["request"]
        try:
            body = self._build_body(request, rec["model"])
            resp = await self._client.post("/v1/music_generation", json=body)
        except httpx.HTTPError as exc:
            rec["status"] = "failed"; rec["error"] = str(exc)
            raise ProviderRequestError(
                f"minimax transport error: {exc}", provider="minimax", status_code=502
            ) from exc
        if resp.status_code >= 400:
            rec["status"] = "failed"; rec["error"] = resp.text[:500]
            raise ProviderRequestError(
                f"minimax returned HTTP {resp.status_code}", provider="minimax",
                status_code=_map_status(resp.status_code),
                details={"upstream_status": resp.status_code, "upstream_body": resp.text[:1000]},
            )
        data = resp.json()
        base_resp = data.get("base_resp") or {}
        status_code = base_resp.get("status_code", 0)
        if status_code != 0:
            rec["status"] = "failed"
            rec["error"] = base_resp.get("status_msg") or f"minimax status {status_code}"
            raise TaskFailedError(
                f"minimax music failed: {rec['error']}", provider="minimax"
            )
        d = data.get("data") or {}
        if d.get("status") != 2:
            # Still in progress (status 1). Leave the synthetic task running so a
            # subsequent poll re-issues the call — MiniMax is nominally sync, but
            # a 1 here means "try again".
            rec["status"] = "running"
            return UnifiedMusicTask(
                task_id=task_id, provider=self.name, model=rec["model"], status="running",
                created_at=rec["created_at"],
            )
        audio = d.get("audio")
        if not audio:
            rec["status"] = "failed"; rec["error"] = "no audio in response"
            raise TaskFailedError("minimax music returned no audio", provider="minimax")
        rec["status"] = "succeeded"
        rec["completed_at"] = int(time.time())
        fmt = (request.audio_format or "mp3").lower()
        media_type = _MIME_BY_FORMAT.get(fmt, "audio/mpeg")
        # output_format 'url' -> audio is a URL; 'hex' (default) -> hex-encoded bytes.
        if body.get("output_format") == "url" and audio.startswith("http"):
            rec["audio_urls"] = [audio]
            rec["audio_media_type"] = media_type
        else:
            try:
                raw = bytes.fromhex(audio)
            except ValueError:
                raw = audio.encode() if isinstance(audio, str) else bytes(audio)
            rec["audio_b64"] = base64.b64encode(raw).decode()
            rec["audio_media_type"] = media_type
        extra_info = data.get("extra_info") or {}
        if extra_info.get("music_duration") is not None:
            rec["usage"] = MusicUsage(duration=int(extra_info["music_duration"]))
        return UnifiedMusicTask(
            task_id=task_id, provider=self.name, model=rec["model"], status="succeeded",
            audio_urls=rec.get("audio_urls", []), audio_b64=rec.get("audio_b64"),
            audio_media_type=media_type, created_at=rec["created_at"],
            completed_at=rec["completed_at"], usage=rec.get("usage"),
        )

    def _build_body(self, request: UnifiedMusicRequest, model: str) -> dict[str, Any]:
        body: dict[str, Any] = {"model": model}
        prompt = request.prompt()
        if prompt:
            body["prompt"] = prompt
        # Lyrics may arrive as a dedicated text part tagged in extra, or as the
        # prompt itself for non-instrumental generation.
        lyrics = request.extra.get("lyrics")
        if lyrics:
            body["lyrics"] = lyrics
        elif prompt and request.is_instrumental is not True:
            # Treat the prompt as lyrics when no separate lyrics were given and
            # the caller didn't ask for instrumental (which needs a description).
            body["lyrics"] = prompt
            body.pop("prompt", None)
        if request.is_instrumental is not None:
            body["is_instrumental"] = request.is_instrumental
        # Prefer a URL (avoids a large hex round-trip and 24h expiry is fine for
        # the gateway's sync-poll window); fall back to hex if the caller asked.
        body["output_format"] = "url" if (request.audio_format or "").lower() != "hex" else "hex"
        audio_setting: dict[str, Any] = {}
        if request.audio_quality:
            # audio_quality is 'samplerate_bitrate' e.g. '44100_128000'.
            parts = request.audio_quality.split("_")
            if len(parts) == 2 and parts[0].isdigit():
                audio_setting["sample_rate"] = int(parts[0])
                audio_setting["bitrate"] = int(parts[1])
        fmt = (request.audio_format or "").lower()
        if fmt in ("mp3", "wav", "pcm"):
            audio_setting["format"] = fmt
        if audio_setting:
            body["audio_setting"] = audio_setting
        # Reference audio for cover models.
        refs = request.reference_audios()
        if refs:
            body["audio_url"] = refs[0]
        # Forward provider-specific knobs the caller stashed in extra.
        for k in ("stream", "lyrics_optimizer", "audio_base64", "cover_feature_id"):
            if k in request.extra:
                body[k] = request.extra[k]
        return body
