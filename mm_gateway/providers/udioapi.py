"""udioapi.pro music provider — REST API.

Two-phase async flow mirroring Suno's shape:

* ``POST /api/v2/generate`` -> ``{"workId": ...}`` (the task id).
* ``GET  /api/v2/feed?workId=...`` -> ``data.response_data[]`` tracks, each with
  ``status`` in ``{"text", "first", "complete"}`` and an ``audio_url`` once
  ``complete``. ``fail_message`` / ``error_message`` signal moderation failure.

The unified request's text parts drive the two mutually-exclusive modes: a
single text part with no ``style``/``title`` is treated as an inspiration
description (``gpt_description_prompt``); ``style``/``title`` in ``extra`` (or a
``negative_prompt``) switch it to custom mode (``prompt`` + ``style`` +
``title`` + ``tags``).

Docs: https://udioapi.pro/docs/v2-generate , /docs/v2-feed
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

log = get_logger("provider.udioapi")

_BASE = "https://udioapi.pro"

# feed status -> unified lifecycle. "text"/"first" are intermediate stages.
_STAGE_TO_STATUS = {"text": "running", "first": "running", "complete": "succeeded"}


class UdioApiProvider(MusicProvider):
    name = "udioapi"
    music_models: ClassVar[list[str]] = [
        "chirp-v4-0", "chirp-v4-5", "chirp-v4-5-plus", "chirp-v5", "chirp-v5-5",
    ]

    def __init__(self, backend):
        super().__init__(backend)
        if not backend.api_key:
            raise ProviderNotConfiguredError("udioapi")
        self._client = make_client(
            backend.base_url or _BASE,
            timeout=180.0,
            headers={"Authorization": f"Bearer {backend.api_key}"},
            proxy_url=backend.extra.get("outbound_proxy"),
        )

    async def create_music_task(self, request: UnifiedMusicRequest) -> UnifiedMusicTask:
        prompt = request.prompt()
        if not prompt and not request.lyrics:
            raise ProviderRequestError(
                "udioapi music requires a prompt (text part)", provider="udioapi", status_code=400,
            )
        body = self._build_body(request)
        data = await request_json(self._client, "POST", "/api/v2/generate", provider="udioapi", json=body)
        work_id = data.get("workId") or (data.get("data") or {}).get("task_id")
        if not work_id:
            raise ProviderRequestError(
                "udioapi create returned no workId", provider="udioapi",
                details={"upstream_body": str(data)[:500]},
            )
        return UnifiedMusicTask(
            task_id=str(work_id), provider=self.name, model=request.model, status="pending",
            created_at=int(time.time()),
        )

    async def get_music_task(self, task_id: str) -> UnifiedMusicTask:
        data = await request_json(
            self._client, "GET", "/api/v2/feed", provider="udioapi", params={"workId": task_id},
        )
        d = data.get("data") or {}
        # Top-level status type e.g. "SUCCESS" / "FAIL".
        top_type = str(d.get("type") or "").upper()
        tracks = d.get("response_data") or []
        task = UnifiedMusicTask(
            task_id=task_id, provider=self.name, model="", status="running", raw=data,
            created_at=int(time.time()),
        )
        if not tracks:
            # No tracks yet — still queued.
            return task
        # Pick the first track that has reached 'complete', else the last track.
        complete = [t for t in tracks if str(t.get("status")) == "complete"]
        track = complete[0] if complete else tracks[-1]
        stage = str(track.get("status"))
        fail = track.get("fail_message") or track.get("error_message")
        if fail:
            task.status = "failed"
            task.error = str(fail)
            return task
        if top_type == "FAIL":
            task.status = "failed"
            task.error = "udioapi task failed"
            return task
        task.status = _STAGE_TO_STATUS.get(stage, "running")  # type: ignore[assignment]
        if task.status == "succeeded":
            url = track.get("audio_url")
            if url:
                task.audio_urls = [url]
                task.audio_media_type = "audio/mpeg"
            else:
                # 'complete' with no URL — moderation likely blocked it.
                task.status = "failed"
                task.error = "udioapi task complete with no audio URL (moderation?)"
            dur = track.get("duration")
            if dur not in (None, "", 0):
                try:
                    task.usage = MusicUsage(duration=int(float(dur)))
                except (TypeError, ValueError):
                    pass
        return task

    def _build_body(self, request: UnifiedMusicRequest) -> dict[str, Any]:
        prompt = request.prompt() or ""
        body: dict[str, Any] = {}
        style = request.style
        title = request.title
        if request.lyrics:
            prompt = "\n".join(part for part in (prompt, request.lyrics) if part)
        # Custom mode when style/title/negative_prompt(tags) are present; else
        # inspiration mode (gpt_description_prompt).
        if style or title or request.negative_prompt:
            body["prompt"] = prompt
            if style:
                body["style"] = style
            if title:
                body["title"] = title
            if request.negative_prompt:
                body["tags"] = request.negative_prompt
        else:
            body["gpt_description_prompt"] = prompt
        if request.model:
            body["model"] = request.model
        if request.is_instrumental is not None:
            body["make_instrumental"] = request.is_instrumental
        if request.vocal_gender:
            body["gender"] = request.vocal_gender
        if request.style_strength is not None:
            body["style_weight"] = request.style_strength
        if request.novelty is not None:
            body["weirdness_constraint"] = request.novelty
        if request.reference_audio_strength is not None:
            body["audio_weight"] = request.reference_audio_strength
        # Forward provider-specific knobs.
        for k in ("gender", "style_weight", "weirdness_constraint", "audio_weight"):
            if k in request.extra:
                body[k] = request.extra[k]
        return body
