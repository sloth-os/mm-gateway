"""ACE-Step 1.5 music provider — REST API.

Two-phase async flow plus a binary download:

* ``POST /release_task`` -> ``{data: {task_id, status: "queued", ...}}``.
* ``POST /query_result`` (body ``{"task_id_list": [...]}``) -> ``data[]`` with
  ``status`` (0 queued/running, 1 succeeded, 2 failed) and a ``result`` JSON
  string whose parsed objects carry ``file`` (a path like ``/v1/audio?path=...``
  to fetch the audio binary).
* ``GET /v1/audio?path=...`` -> raw audio bytes.

Auth is either a ``ai_token`` body field or an ``Authorization: Bearer`` header;
this adapter sends the header when an API key is configured.

Docs: https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/API.md
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from typing import Any, ClassVar

from mm_gateway.core.base import MusicProvider
from mm_gateway.core.exceptions import (
    ProviderNotConfiguredError,
    ProviderRequestError,
    ProviderTimeoutError,
)
from mm_gateway.observability.logging import get_logger
from mm_gateway.providers._http import _map_status, make_client, request_json
from mm_gateway.schemas.music import MusicUsage, UnifiedMusicRequest, UnifiedMusicTask

log = get_logger("provider.acestep")

# query_result.data[].status -> unified lifecycle.
_STATUS_MAP = {0: "running", 1: "succeeded", 2: "failed"}


class AceStepProvider(MusicProvider):
    name = "acestep"
    music_models: ClassVar[list[str]] = [
        "acestep-v15-turbo",
        "acestep-v15-xl-turbo",
        "acestep-v15-base",
        "acestep-v15-turbo-shift3",
        "ace-step-1.5",
    ]
    # /release_task retry tuning (transient upstream 502/503/504 + timeouts).
    _create_max_attempts: ClassVar[int] = 3
    _create_backoff_base: ClassVar[float] = 0.5

    def __init__(self, backend):
        super().__init__(backend)
        if not backend.base_url:
            # ACE-Step is self-hosted; there is no default cloud host, so an
            # operator must point the gateway at their server via
            # ACESTEP_BASE_URL. Surface this clearly rather than failing per
            # request with an opaque "missing absolute URL" httpx error.
            raise ProviderNotConfiguredError("acestep", "ACE-Step requires ACESTEP_BASE_URL to be set.")
        headers: dict[str, str] = {}
        if backend.api_key:
            headers["Authorization"] = f"Bearer {backend.api_key}"
        self._client = make_client(backend.base_url, timeout=300.0, headers=headers)

    async def create_music_task(self, request: UnifiedMusicRequest) -> UnifiedMusicTask:
        prompt = request.generation_prompt()
        if not prompt and not request.lyrics:
            raise ProviderRequestError(
                "acestep music requires a prompt (text part)", provider="acestep", status_code=400,
            )
        body = self._build_body(request)
        data = await self._post_release_task(body)
        d = data.get("data") or {}
        task_id = d.get("task_id")
        if not task_id:
            raise ProviderRequestError(
                "acestep create returned no task_id", provider="acestep",
                details={"upstream_body": str(data)[:500]},
            )
        return UnifiedMusicTask(
            task_id=str(task_id), provider=self.name, model=request.model, status="pending",
            created_at=int(time.time()),
        )

    async def _post_release_task(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST /release_task with a bounded retry on transient upstream failures.

        ACE-Step's create is a cheap enqueue that should return immediately with a
        ``task_id``. The hosted front end (Cloudflare in front of ``api.acemusic.ai``)
        can intermittently return ``504 Gateway Timeout`` with ``Retry-After`` when the
        origin briefly fails to answer — a transient condition, not a request-shape
        error (the body/headers match the API docs). Failing the whole generation on
        the first such blip is brittle, so retry a couple of times with a short
        backoff before surfacing the error. A 504 returns no ``task_id``, so a retry
        cannot collide with a task we already created from the gateway's view.
        """
        max_attempts = self._create_max_attempts
        last_exc: Exception | None = None
        for attempt in range(max_attempts):
            try:
                return await request_json(
                    self._client, "POST", "/release_task", provider="acestep", json=body,
                )
            except ProviderTimeoutError as exc:
                last_exc = exc
            except ProviderRequestError as exc:
                # Only retry transient server-side failures (5xx), not 4xx client
                # errors (auth/validation/quota), which won't change on retry.
                if (exc.details or {}).get("upstream_status") not in (502, 503, 504):
                    raise
                last_exc = exc
            # Brief backoff; do not honor a long Retry-After (observed ~120s) so a
            # down origin cannot stall the request for minutes.
            if attempt < max_attempts - 1:
                await asyncio.sleep(self._create_backoff_base * (2 ** attempt))
        assert last_exc is not None
        raise last_exc

    async def get_music_task(self, task_id: str) -> UnifiedMusicTask:
        data = await request_json(
            self._client, "POST", "/query_result", provider="acestep",
            json={"task_id_list": [task_id]},
        )
        items = data.get("data") or []
        # Find this task's entry.
        entry = next((it for it in items if str(it.get("task_id")) == task_id), None)
        if entry is None and items:
            entry = items[0]
        if entry is None:
            return UnifiedMusicTask(
                task_id=task_id, provider=self.name, model="", status="running",
                raw=data, created_at=int(time.time()),
            )
        status = _STATUS_MAP.get(int(entry.get("status", 0)), "running")  # type: ignore[arg-type]
        task = UnifiedMusicTask(
            task_id=task_id, provider=self.name, model="", status=status,  # type: ignore[arg-type]
            raw=data, created_at=int(time.time()),
        )
        if status == "failed":
            task.error = "acestep task failed"
            return task
        if status != "succeeded":
            return task
        result = entry.get("result")
        parsed: list[dict[str, Any]] = []
        if isinstance(result, str):
            try:
                parsed = json.loads(result)
            except json.JSONDecodeError:
                parsed = []
        elif isinstance(result, list):
            parsed = result
        elif isinstance(result, dict):
            parsed = [result]
        if not parsed:
            task.status = "failed"
            task.error = "acestep task succeeded with no result"
            return task
        file_path = parsed[0].get("file")
        if not file_path:
            task.status = "failed"
            task.error = "acestep result has no audio file"
            return task
        # The `file` value is a path like "/v1/audio?path=..."; fetch the bytes
        # so the Lyria response carries inline base64 audio.
        try:
            audio_bytes = await self._fetch_audio(file_path)
        except ProviderRequestError:
            # Fall back to exposing the path as a URL so a client can fetch it.
            task.audio_urls = [self._absolute(file_path)]
            task.audio_media_type = _media_type(request=None, fmt=None)
            return task
        media_type = _media_type(request=None, fmt=None)
        task.audio_b64 = base64.b64encode(audio_bytes).decode()
        task.audio_media_type = media_type
        metas = (parsed[0].get("metas") or {})
        if metas.get("duration") is not None:
            try:
                task.usage = MusicUsage(duration=int(float(metas["duration"])))
            except (TypeError, ValueError):
                pass
        if parsed[0].get("lyrics"):
            task.lyrics = parsed[0]["lyrics"]
        return task

    async def _fetch_audio(self, path: str) -> bytes:
        url = path if path.startswith("http") else self._absolute(path)
        try:
            resp = await self._client.get(url)
        except Exception as exc:
            raise ProviderRequestError(f"acestep audio fetch error: {exc}", provider="acestep") from exc
        if resp.status_code >= 400:
            raise ProviderRequestError(
                f"acestep audio fetch returned HTTP {resp.status_code}", provider="acestep",
                status_code=_map_status(resp.status_code),
            )
        return resp.content

    def _absolute(self, path: str) -> str:
        base = str(self._client.base_url).rstrip("/")
        return path if path.startswith("http") else base + ("" if path.startswith("/") else "/") + path

    def _build_body(self, request: UnifiedMusicRequest) -> dict[str, Any]:
        prompt = request.generation_prompt() or ""
        body: dict[str, Any] = {"prompt": prompt}
        if request.lyrics:
            body["lyrics"] = request.lyrics
        if request.vocal_language:
            body["vocal_language"] = request.vocal_language
        fmt = (request.audio_format or "").lower()
        if fmt in ("flac", "mp3", "opus", "aac", "wav", "wav32"):
            body["audio_format"] = fmt
        if request.duration is not None:
            body["audio_duration"] = float(request.duration)
        if request.bpm is not None:
            body["bpm"] = request.bpm
        key_scale = request.key_scale or " ".join(
            part for part in (request.key, request.scale) if part
        )
        if key_scale:
            body["key_scale"] = key_scale
        if request.time_signature:
            body["time_signature"] = request.time_signature
        if request.guidance_scale is not None:
            body["guidance_scale"] = request.guidance_scale
        if request.seed is not None:
            body["seed"] = request.seed
            body["use_random_seed"] = False
        if request.model:
            body["model"] = request.model
        if request.inference_steps is not None:
            body["inference_steps"] = request.inference_steps
        if request.n is not None:
            body["batch_size"] = request.n
        if continuation := request.continuation_audio():
            body["reference_audio_path"] = continuation
        elif references := request.reference_audios():
            body["reference_audio_path"] = references[0]
        # Forward provider-specific knobs.
        for k in ("thinking", "sample_mode", "sample_query", "use_format",
                  "inference_steps", "batch_size", "task_type", "reference_audio_path"):
            if k in request.extra:
                body[k] = request.extra[k]
        return body


def _media_type(request, fmt) -> str:
    f = (fmt or (request.audio_format if request else None) or "mp3").lower()
    return {"flac": "audio/flac", "mp3": "audio/mpeg", "opus": "audio/ogg",
            "aac": "audio/aac", "wav": "audio/wav", "wav32": "audio/wav"}.get(f, "audio/mpeg")


# Re-exported so the timeout import isn't dropped by linters that flag unused
# imports; request_json surfaces timeouts already.
__all__ = ["AceStepProvider", "ProviderTimeoutError"]
