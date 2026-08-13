"""MiniMax provider — music + video (H3) over ``https://api.minimax.io``.

Music (``POST /v1/music_generation``) is synchronous: a single blocking call
returns ``data.status`` 1 (in progress) or 2 (completed) with the audio inline —
a 24h URL by default (``output_format`` ``url``), or hex-encoded bytes when the
caller passes ``audio_format == "hex"``. There is no job id to poll. As with the
ElevenLabs / Stability-SVD adapters, we mint a gateway-local task id at create
time, record the request in an in-memory store, and run the blocking call on the
first poll — the synthetic task moves ``pending -> running -> succeeded`` as the
call completes.

Video (``MiniMax-H3``) is a genuine two-phase async task API:
``POST /v2/video_generation`` → ``{task_id}``, then
``GET /v2/query/video_generation/{task_id}`` → ``{task:{status, content:{url},
...}}``. The H3 ``content[]`` shape (typed ``text`` / ``image_url``-with-role /
``video_url`` / ``audio_url`` parts) *is* the unified video schema's content
shape, so the parts pass straight through. Statuses ``queued``/``running``/
``processing`` map to ``pending``/``running``; ``succeeded``/``failed``/
``cancelled`` are terminal.

Docs:
- https://platform.minimax.io/docs/api-reference/music-generation
- https://platform.minimax.io/docs/guides/video-generation
"""

from __future__ import annotations

import base64
import time
import uuid
from typing import Any, ClassVar

import httpx

from mm_gateway.core.base import MusicProvider, VideoProvider
from mm_gateway.core.exceptions import (
    ProviderNotConfiguredError,
    ProviderRequestError,
    TaskFailedError,
)
from mm_gateway.observability.httplog import backend_event_hooks
from mm_gateway.observability.logging import get_logger
from mm_gateway.providers._dimensions import aspect_ratio
from mm_gateway.providers._http import _map_status
from mm_gateway.schemas.music import MusicUsage, UnifiedMusicRequest, UnifiedMusicTask
from mm_gateway.schemas.video import UnifiedVideoRequest, UnifiedVideoTask

log = get_logger("provider.minimax")

_BASE = "https://api.minimax.io"

# In-memory store for the synchronous music "tasks". Single-process only.
_MUSIC_TASKS: dict[str, dict[str, Any]] = {}

# MiniMax audio_setting.format values -> MIME types for the inline bytes.
_MIME_BY_FORMAT = {"mp3": "audio/mpeg", "wav": "audio/wav", "pcm": "audio/pcm"}

# MiniMax video task statuses -> unified lifecycle. The H3 query response
# carries ``task.status``; non-terminal values stay pending/running until a
# later poll reaches a terminal state.
_VIDEO_STATUS_MAP = {
    "queued": "pending",
    "running": "running",
    "processing": "running",
    "succeeded": "succeeded",
    "failed": "failed",
    "cancelled": "cancelled",
}


class MiniMaxProvider(MusicProvider, VideoProvider):
    name = "minimax"
    music_models: ClassVar[list[str]] = ["music-3.0", "music-2.6", "music-cover"]
    video_models: ClassVar[list[str]] = ["MiniMax-H3"]

    def __init__(self, backend):
        super().__init__(backend)
        if not backend.api_key:
            raise ProviderNotConfiguredError("minimax")
        self._api_key = backend.api_key
        # Per-modality clients honor the sync/async URL split resolved by
        # ``config.py``: music (sync ``/v1/music_generation``) uses ``base_url``
        # (the ``*_MUSIC_BASE_URL`` endpoint, preferred); video (async
        # ``/v2/video_generation``) uses ``extra["video_base_url"]`` (the
        # ``*_VIDEO_BASE_URL`` endpoint) when it differs. The real
        # api.minimax.io serves both at one host, so the two clients collapse
        # unless an operator pins them apart.
        music_base = backend.base_url or _BASE
        video_base = backend.extra.get("video_base_url") or music_base
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        self._client = httpx.AsyncClient(
            base_url=music_base, timeout=300, headers=headers,
            event_hooks=backend_event_hooks(),
        )
        self._client_video = httpx.AsyncClient(
            base_url=video_base, timeout=300, headers=headers,
            event_hooks=backend_event_hooks(),
        )

    async def create_music_task(self, request: UnifiedMusicRequest) -> UnifiedMusicTask:
        prompt = request.generation_prompt()
        # Non-instrumental, non-cover models require lyrics; instrumental/cover
        # require a prompt. Accept whichever the caller supplied and let the
        # upstream reject an inconsistent combination.
        if not prompt and not request.lyrics:
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
        # output_format 'url' (default) -> audio is a URL; 'hex' (opt-in) -> hex-encoded bytes.
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
        lyrics = request.lyrics
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
        if request.sample_rate_hz is not None:
            audio_setting["sample_rate"] = request.sample_rate_hz
        if request.bitrate_kbps is not None:
            audio_setting["bitrate"] = request.bitrate_kbps * 1000
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
            if refs[0].startswith("data:"):
                body["audio_base64"] = refs[0].partition(",")[2]
            else:
                body["audio_url"] = refs[0]
        if request.enhance_lyrics is not None:
            body["lyrics_optimizer"] = request.enhance_lyrics
        # Forward provider-specific knobs the caller stashed in extra.
        for k in ("stream", "lyrics_optimizer", "audio_base64", "cover_feature_id"):
            if k in request.extra:
                body[k] = request.extra[k]
        return body

    # -- Video (MiniMax-H3) ------------------------------------------------- #

    async def create_video_task(self, request: UnifiedVideoRequest) -> UnifiedVideoTask:
        body = self._build_video_body(request)
        try:
            resp = await self._client_video.post("/v2/video_generation", json=body)
        except httpx.HTTPError as exc:
            raise ProviderRequestError(
                f"minimax video transport error: {exc}", provider="minimax", status_code=502
            ) from exc
        if resp.status_code >= 400:
            raise ProviderRequestError(
                f"minimax video returned HTTP {resp.status_code}", provider="minimax",
                status_code=_map_status(resp.status_code),
                details={"upstream_status": resp.status_code, "upstream_body": resp.text[:1000]},
            )
        data = resp.json()
        # The video v2 surface returns ``{task_id}`` on success; some error
        # envelopes still carry the music-style ``base_resp`` block, so honour
        # it when present before demanding a task id.
        base_resp = data.get("base_resp") or {}
        if base_resp.get("status_code", 0) != 0:
            raise TaskFailedError(
                f"minimax video create failed: {base_resp.get('status_msg') or base_resp.get('status_code')}",
                provider="minimax",
            )
        task_id = data.get("task_id")
        if not task_id:
            raise ProviderRequestError(
                "minimax video create returned no task_id", provider="minimax",
                status_code=502, details={"upstream_body": str(data)[:1000]},
            )
        return UnifiedVideoTask(
            task_id=task_id, provider=self.name, model=request.model, status="pending",
        )

    async def get_video_task(self, task_id: str) -> UnifiedVideoTask:
        try:
            resp = await self._client_video.get(f"/v2/query/video_generation/{task_id}")
        except httpx.HTTPError as exc:
            raise ProviderRequestError(
                f"minimax video poll transport error: {exc}", provider="minimax", status_code=502
            ) from exc
        if resp.status_code >= 400:
            raise ProviderRequestError(
                f"minimax video poll returned HTTP {resp.status_code}", provider="minimax",
                status_code=_map_status(resp.status_code),
                details={"upstream_status": resp.status_code, "upstream_body": resp.text[:1000]},
            )
        data = resp.json()
        task = data.get("task") if isinstance(data, dict) else None
        if not isinstance(task, dict):
            base_resp = data.get("base_resp") or {} if isinstance(data, dict) else {}
            raise ProviderRequestError(
                f"minimax video poll returned no task: {base_resp.get('status_msg') or data}",
                provider="minimax", status_code=502,
                details={"upstream_body": str(data)[:1000]},
            )
        raw_status = task.get("status")
        status = _VIDEO_STATUS_MAP.get(raw_status, "running")
        out = UnifiedVideoTask(
            task_id=task_id, provider=self.name, model=task.get("model") or "",
            status=status,  # type: ignore[arg-type]
            raw=data,
        )
        if status == "succeeded":
            content = task.get("content") or {}
            url = content.get("url") if isinstance(content, dict) else None
            if url:
                out.video_urls = [url]
            cover = content.get("cover_url") if isinstance(content, dict) else None
            if cover:
                out.cover_url = cover
            if not out.video_urls:
                # Succeeded but no URL — treat as a failed generation.
                out.status = "failed"  # type: ignore[arg-type]
                out.error = "minimax video succeeded with no content url"
        elif status in ("failed", "cancelled"):
            err = task.get("error")
            if isinstance(err, dict):
                out.error = str(err.get("message") or err.get("code") or err)
            else:
                out.error = str(err) if err else (raw_status or status)
        return out

    def _build_video_body(self, request: UnifiedVideoRequest) -> dict[str, Any]:
        # The H3 ``content[]`` shape is the unified video content shape, so the
        # typed parts pass straight through as dicts (text / image_url-with-role
        # / video_url / audio_url).
        body: dict[str, Any] = {
            "model": request.model,
            "content": [p.model_dump(exclude_none=True) for p in request.content],
        }
        if request.duration is not None:
            body["duration"] = int(request.duration)
        # MiniMax resolution tokens are provider-specific ("768P", "2K"); pass
        # an explicit ``resolution`` through verbatim rather than deriving one
        # from width/height (which would not map to a valid H3 token).
        if request.resolution:
            body["resolution"] = request.resolution
        if ratio := aspect_ratio(request):
            body["ratio"] = ratio
        # Forward provider-specific knobs the caller stashed in extra (e.g.
        # ``prompt_optimizer``, ``watermark``); the unified knobs not confirmed
        # by the H3 docs are deliberately not forwarded to avoid 400s.
        body.update(request.extra)
        return body
