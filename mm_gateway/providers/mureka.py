"""Mureka music provider — REST API on ``https://platform.mureka.ai``.

Two-phase async flow:

* ``POST /v1/song/generate`` -> a JSON object carrying a ``task_id``.
* ``GET  /v1/song/query/{task_id}`` -> ``{status, audio_url, ...}``.

The official vitepress-openapi docs render their schemas client-side, so the
exact field set isn't statically documented; this adapter follows the endpoints
the docs name verbatim and is tolerant of the response shape (it reads the task
id and audio URL from the common candidate field names). It is easy to tighten
once an operator confirms the live response.

Docs: https://platform.mureka.ai/docs/api/operations/post-v1-song-generate.html
"""

from __future__ import annotations

import time
from typing import Any, ClassVar

from mm_gateway.core.base import MusicProvider
from mm_gateway.core.exceptions import (
    ProviderNotConfiguredError,
    ProviderRequestError,
)
from mm_gateway.observability.logging import get_logger
from mm_gateway.providers._http import make_client, request_json
from mm_gateway.schemas.music import MusicUsage, UnifiedMusicRequest, UnifiedMusicTask

log = get_logger("provider.mureka")

_BASE = "https://platform.mureka.ai"

# Candidate response field names for the task id and audio url. Mureka's query
# response carries the result at the top level and/or nested; try each in turn.
_TASK_ID_FIELDS = ("task_id", "taskId", "id")
_AUDIO_FIELDS = ("audio_url", "audioUrl", "url")
# Query status values the docs enumerate -> unified lifecycle.
_STATUS_MAP = {
    "queued": "pending",
    "pending": "pending",
    "running": "running",
    "processing": "running",
    "succeeded": "succeeded",
    "success": "succeeded",
    "completed": "succeeded",
    "failed": "failed",
    "error": "failed",
}


class MurekaProvider(MusicProvider):
    name = "mureka"
    music_models: ClassVar[list[str]] = ["mureka-song-1", "mureka-song-1.5"]

    def __init__(self, backend):
        super().__init__(backend)
        if not backend.api_key:
            raise ProviderNotConfiguredError("mureka")
        self._client = make_client(
            backend.base_url or _BASE,
            timeout=180.0,
            headers={"Authorization": f"Bearer {backend.api_key}"},
        )

    async def create_music_task(self, request: UnifiedMusicRequest) -> UnifiedMusicTask:
        prompt = request.prompt()
        if not prompt and not request.lyrics:
            raise ProviderRequestError(
                "mureka music requires a prompt (text part)", provider="mureka", status_code=400,
            )
        body = self._build_body(request)
        data = await request_json(self._client, "POST", "/v1/song/generate", provider="mureka", json=body)
        task_id = _first(data, _TASK_ID_FIELDS) or _first(data.get("data") or {}, _TASK_ID_FIELDS)
        if not task_id:
            raise ProviderRequestError(
                "mureka create returned no task_id", provider="mureka",
                details={"upstream_body": str(data)[:500]},
            )
        return UnifiedMusicTask(
            task_id=str(task_id), provider=self.name, model=request.model, status="pending",
            created_at=int(time.time()),
        )

    async def get_music_task(self, task_id: str) -> UnifiedMusicTask:
        data = await request_json(
            self._client, "GET", f"/v1/song/query/{task_id}", provider="mureka",
        )
        status_raw = str(data.get("status") or data.get("task_status") or "").lower()
        status = _STATUS_MAP.get(status_raw, "running")
        task = UnifiedMusicTask(
            task_id=task_id, provider=self.name, model="", status=status,  # type: ignore[arg-type]
            raw=data, created_at=int(time.time()),
        )
        if status == "succeeded":
            audio = _first(data, _AUDIO_FIELDS)
            if audio:
                task.audio_urls = [audio]
                task.audio_media_type = "audio/mpeg"
            else:
                # Completed without a URL — moderation or upstream glitch.
                task.status = "failed"
                task.error = data.get("fail_message") or "mureka task completed with no audio URL"
            dur = data.get("duration") or (data.get("extra") or {}).get("duration")
            if dur not in (None, "", 0):
                try:
                    task.usage = MusicUsage(duration=int(float(dur)))
                except (TypeError, ValueError):
                    pass
        elif status == "failed":
            task.error = data.get("fail_message") or data.get("error") or "mureka task failed"
        return task

    def _build_body(self, request: UnifiedMusicRequest) -> dict[str, Any]:
        body: dict[str, Any] = {"model": request.model}
        prompt = request.prompt() or ""
        # Mureka's "Lyrics to song" path takes lyrics; a bare description goes to
        # "Prompt to song". We send lyrics when the text looks like lyrics
        # (multi-line or structure tags), otherwise prompt.
        if request.lyrics:
            body["lyrics"] = request.lyrics
            body["prompt"] = prompt
        elif "\n" in prompt or prompt.startswith("["):
            body["lyrics"] = prompt
        else:
            body["prompt"] = prompt
        if request.title:
            body["title"] = request.title
        if request.negative_prompt:
            body["tags"] = request.negative_prompt
        elif request.style:
            body["tags"] = request.style
        if request.is_instrumental is not None:
            body["instrumental"] = request.is_instrumental
        if request.bpm is not None:
            body["bpm"] = request.bpm
        if request.duration is not None:
            body["duration"] = int(request.duration)
        if request.seed is not None:
            body["seed"] = request.seed
        if request.voice:
            body["voice_id"] = request.voice
        audio_config: dict[str, Any] = {}
        if request.audio_format:
            audio_config["format"] = request.audio_format
        if request.sample_rate_hz is not None:
            audio_config["sample_rate"] = request.sample_rate_hz
        if request.bitrate_kbps is not None:
            audio_config["bitrate"] = request.bitrate_kbps * 1000
        if audio_config:
            body["audio_config"] = audio_config
        # Forward any provider-specific knobs.
        for k in ("model_name", "audio_config", "voice_id", "seed"):
            if k in request.extra:
                body[k] = request.extra[k]
        return body


def _first(d: Any, fields: tuple[str, ...]) -> str | None:
    if not isinstance(d, dict):
        return None
    for f in fields:
        v = d.get(f)
        if v:
            return str(v)
    return None
