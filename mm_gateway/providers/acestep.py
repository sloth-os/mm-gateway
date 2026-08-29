"""ACE-Step 1.5 music provider — REST API.

The same ACE-Step server speaks two modes (selected by ``api_mode`` in the
official ``acestep.sh`` config); this adapter mirrors that:

* **native** (default for self-hosted servers, e.g. ``http://127.0.0.1:8001``):
  two-phase async plus a binary download —

  * ``POST /release_task`` -> ``{data: {task_id, status: "queued", ...}}``.
  * ``POST /query_result`` (body ``{"task_id_list": [...]}``) -> ``data[]`` with
    ``status`` (0 queued/running, 1 succeeded, 2 failed) and a ``result`` JSON
    string whose parsed objects carry ``file`` (a path like
    ``/v1/audio?path=...`` to fetch the audio binary).
  * ``GET /v1/audio?path=...`` -> raw audio bytes.

* **completion** (used for the hosted cloud front end ``api.acemusic.ai``):
  OpenAI-style single-phase —

  * ``POST /v1/chat/completions`` returns the final result inline; audio is a
    ``data:audio/mpeg;base64,...`` URL at
    ``choices[0].message.audio[].audio_url.url``. There is no job id to poll, so
    we mint a gateway-local task id at create time and run the blocking call on
    the first poll (the synthetic-task pattern shared with the MiniMax adapter).

The cloud host fronts native ``/release_task`` with Cloudflare, and the origin
behind it is chronically unreachable — every CI run gets ``504 Gateway
Timeout`` (``retry-after: 120``) even with a valid key and the create-retry
below. The official script and the sibling ``speak`` repo both use completion
mode against ``api.acemusic.ai``, so ``auto`` (the default) picks completion when
the base URL host ends with ``acemusic.ai`` and native otherwise. Force either
with ``backend.extra["acestep_api_mode"]`` = ``native`` | ``completion``.

Auth is an ``Authorization: Bearer`` header when an API key is configured.

Docs: https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/API.md
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
import uuid
from typing import Any, ClassVar

from mm_gateway.core.base import MusicProvider
from mm_gateway.core.exceptions import (
    ProviderNotConfiguredError,
    ProviderRequestError,
    ProviderTimeoutError,
    TaskFailedError,
)
from mm_gateway.observability.logging import get_logger
from mm_gateway.providers._http import _map_status, make_client, request_json
from mm_gateway.schemas.music import MusicUsage, UnifiedMusicRequest, UnifiedMusicTask

log = get_logger("provider.acestep")

# query_result.data[].status -> unified lifecycle.
_STATUS_MAP = {0: "running", 1: "succeeded", 2: "failed"}

# Model id used for completion mode when the caller did not supply one. The
# cloud host requires the ``acemusic/`` prefix (matching the official script and
# speak's acemusic client); the bare ids we expose for native self-hosted
# servers do not.
_DEFAULT_COMPLETION_MODEL = "acemusic/acestep-v15-turbo"

# In-memory store for the synchronous completion-mode "tasks". Single-process,
# like the MiniMax adapter's store.
_COMPLETION_TASKS: dict[str, dict[str, Any]] = {}


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
        self._mode = self._resolve_mode(backend)
        self._client = make_client(backend.base_url, timeout=300.0, headers=headers,
                                   proxy_url=backend.extra.get("outbound_proxy"))

    @staticmethod
    def _resolve_mode(backend) -> str:
        """Pick native vs completion.

        ``auto`` (default) selects completion for the hosted cloud front end
        (``api.acemusic.ai``) — the origin behind Cloudflare is chronically
        unreachable on native ``/release_task`` — and native everywhere else.
        An explicit ``backend.extra["acestep_api_mode"]`` overrides either way.
        """
        explicit = str((backend.extra or {}).get("acestep_api_mode", "")).strip().lower()
        if explicit in ("native", "completion"):
            return explicit
        base_url = backend.base_url or ""
        host = str(base_url).split("://", 1)[-1].split("/", 1)[0].lower()
        return "completion" if host.endswith("acemusic.ai") else "native"

    async def create_music_task(self, request: UnifiedMusicRequest) -> UnifiedMusicTask:
        prompt = request.generation_prompt()
        if not prompt and not request.lyrics:
            raise ProviderRequestError(
                "acestep music requires a prompt (text part)", provider="acestep", status_code=400,
            )
        if self._mode == "completion":
            return self._create_completion_task(request)
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

    def _create_completion_task(self, request: UnifiedMusicRequest) -> UnifiedMusicTask:
        """Mint a gateway-local task id; the blocking call runs on first poll."""
        task_id = f"acestep-{uuid.uuid4().hex}"
        _COMPLETION_TASKS[task_id] = {
            "model": request.model or "",
            "request": request,
            "status": "pending",
            "created_at": int(time.time()),
        }
        return UnifiedMusicTask(
            task_id=task_id, provider=self.name, model=request.model, status="pending",
            created_at=_COMPLETION_TASKS[task_id]["created_at"],
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
        if self._mode == "completion":
            return await self._get_completion_task(task_id)
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
        """Native ``/release_task`` body, matching the official ``acestep.sh``.

        Defaults that the official script always sends (``thinking``,
        ``use_format``, ``use_cot_caption``, ``use_cot_language`` = true,
        ``use_random_seed`` = true) are set up-front and only flipped/overridden
        when the request or ``extra`` says otherwise. ``use_random_seed`` becomes
        false when an explicit ``seed`` is given.
        """
        prompt = request.generation_prompt() or ""
        body: dict[str, Any] = {
            "prompt": prompt,
            "thinking": True,
            "use_format": True,
            "use_cot_caption": True,
            "use_cot_language": True,
            "use_random_seed": True,
        }
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
        src_audio = request.continuation_audio() or (request.reference_audios()[:1] or [None])
        if src_audio:
            # Official field name is ``src_audio_path``; when source audio is
            # supplied the task is a cover/repaint, so default task_type too.
            body["src_audio_path"] = src_audio[0] if isinstance(src_audio, list) else src_audio
            body.setdefault("task_type", "cover")
        # Forward provider-specific knobs (override the defaults above).
        for k in ("thinking", "sample_mode", "sample_query", "use_format",
                  "use_cot_caption", "use_cot_language", "inference_steps",
                  "batch_size", "task_type", "src_audio_path",
                  "audio_cover_strength", "repainting_start", "repainting_end"):
            if k in request.extra:
                body[k] = request.extra[k]
        return body

    # ------------------------------------------------------------------ #
    # completion mode — POST /v1/chat/completions, single-phase
    # ------------------------------------------------------------------ #

    async def _get_completion_task(self, task_id: str) -> UnifiedMusicTask:
        rec = _COMPLETION_TASKS.get(task_id)
        if rec is None:
            raise ProviderRequestError(
                f"acestep task {task_id} not found", provider="acestep", status_code=404,
            )
        if rec["status"] in ("succeeded", "failed"):
            return UnifiedMusicTask(
                task_id=task_id, provider=self.name, model=rec["model"], status=rec["status"],
                audio_urls=rec.get("audio_urls", []), audio_b64=rec.get("audio_b64"),
                audio_media_type=rec.get("audio_media_type"),
                error=rec.get("error"), created_at=rec["created_at"],
                completed_at=rec.get("completed_at"), usage=rec.get("usage"),
                lyrics=rec.get("lyrics"), raw=rec.get("raw"),
            )
        rec["status"] = "running"
        request: UnifiedMusicRequest = rec["request"]
        try:
            body = self._build_completion_body(request, rec["model"])
            resp = await self._client.post("/v1/chat/completions", json=body)
        except Exception as exc:
            rec["status"] = "failed"; rec["error"] = str(exc)
            raise ProviderRequestError(
                f"acestep completion transport error: {exc}", provider="acestep", status_code=502
            ) from exc
        if resp.status_code >= 400:
            rec["status"] = "failed"; rec["error"] = resp.text[:500]
            raise ProviderRequestError(
                f"acestep completion returned HTTP {resp.status_code}", provider="acestep",
                status_code=_map_status(resp.status_code),
                details={"upstream_status": resp.status_code, "upstream_body": resp.text[:1000]},
            )
        try:
            data = resp.json()
        except Exception as exc:
            rec["status"] = "failed"; rec["error"] = "non-JSON response"
            raise TaskFailedError(
                f"acestep completion returned non-JSON: {exc}", provider="acestep"
            ) from exc
        rec["raw"] = data
        job_id = str(data.get("id") or data.get("task_id") or task_id)
        choices = data.get("choices") or []
        choice = choices[0] if choices else {}
        finish_reason = str(choice.get("finish_reason") or "")
        if finish_reason == "error":
            rec["status"] = "failed"
            rec["error"] = _completion_error(data) or "acestep completion reported error"
            return UnifiedMusicTask(
                task_id=task_id, provider=self.name, model=rec["model"], status="failed",
                error=rec["error"], created_at=rec["created_at"], raw=data,
            )
        message = choice.get("message") or {}
        audio_urls = _completion_audio_urls(message)
        if not audio_urls:
            err = _completion_error(data)
            rec["status"] = "failed"
            rec["error"] = err or "acestep completion returned no audio"
            return UnifiedMusicTask(
                task_id=task_id, provider=self.name, model=rec["model"], status="failed",
                error=rec["error"], created_at=rec["created_at"], raw=data,
            )
        rec["status"] = "succeeded"
        rec["completed_at"] = int(time.time())
        rec["audio_urls"] = audio_urls
        fmt = (request.audio_format or "mp3").lower()
        rec["audio_media_type"] = _media_type_for(fmt)
        audio_b64: str | None = None
        for url in audio_urls:
            data_url = _data_url_b64(url)
            if data_url:
                audio_b64 = data_url
                break
        rec["audio_b64"] = audio_b64
        content_text = message.get("content")
        if isinstance(content_text, str) and content_text.strip():
            rec["lyrics"] = content_text.strip()
        usage = _completion_usage(data, message)
        if usage is not None:
            rec["usage"] = usage
        return UnifiedMusicTask(
            task_id=job_id, provider=self.name, model=rec["model"], status="succeeded",
            audio_urls=audio_urls, audio_b64=audio_b64,
            audio_media_type=rec["audio_media_type"], created_at=rec["created_at"],
            completed_at=rec["completed_at"], usage=rec.get("usage"),
            lyrics=rec.get("lyrics"), raw=data,
        )

    def _build_completion_body(self, request: UnifiedMusicRequest, model: str) -> dict[str, Any]:
        """``/v1/chat/completions`` body, matching the official ``acestep.sh``.

        ``model`` must carry the ``acemusic/`` prefix on the cloud host; add it
        when the caller passed a bare id. Content is a string for text-only
        requests, or a parts array (text + input_audio) when source audio is
        given.
        """
        model = (model or request.model or _DEFAULT_COMPLETION_MODEL).strip()
        if "/" not in model:
            model = f"acemusic/{model}"
        content = self._completion_message(request)
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "stream": False,
            "thinking": True,
            "use_format": True,
            "use_cot_caption": True,
            "use_cot_language": True,
        }
        audio_config: dict[str, Any] = {
            "format": (request.audio_format or "mp3").lower(),
        }
        if request.vocal_language:
            audio_config["vocal_language"] = request.vocal_language
        if request.duration is not None:
            audio_config["duration"] = float(request.duration)
        if request.bpm is not None:
            audio_config["bpm"] = request.bpm
        key_scale = request.key_scale or " ".join(
            part for part in (request.key, request.scale) if part
        )
        if key_scale:
            audio_config["key_scale"] = key_scale
        if request.time_signature:
            audio_config["time_signature"] = request.time_signature
        body["audio_config"] = audio_config
        if request.guidance_scale is not None:
            body["guidance_scale"] = request.guidance_scale
        if request.seed is not None:
            body["seed"] = request.seed
        if request.n is not None:
            body["batch_size"] = request.n
        # Forward provider-specific knobs (cover/repaint + overrides).
        for k in ("thinking", "use_format", "use_cot_caption", "use_cot_language",
                  "sample_mode", "task_type", "guidance_scale", "seed", "batch_size",
                  "audio_cover_strength", "repainting_start", "repainting_end"):
            if k in request.extra:
                body[k] = request.extra[k]
        return body

    def _completion_message(self, request: UnifiedMusicRequest) -> Any:
        """Build the user ``content``: parts array when source audio, else string."""
        prompt = request.generation_prompt() or ""
        lyrics = request.lyrics or ""
        text_part = (f"<prompt>{prompt}</prompt>" if prompt else "") \
            + (f"\n<lyrics>{lyrics}</lyrics>" if lyrics else "")
        src = request.continuation_audio() or (request.reference_audios()[:1] or [None])
        src_url = src[0] if isinstance(src, list) and src else src
        if src_url:
            parts: list[dict[str, Any]] = []
            if text_part:
                parts.append({"type": "text", "text": text_part})
            data, fmt = _decode_audio_input(src_url)
            parts.append({"type": "input_audio",
                           "input_audio": {"data": data, "format": fmt or "wav"}})
            return parts
        return text_part


def _media_type(request, fmt) -> str:
    f = (fmt or (request.audio_format if request else None) or "mp3").lower()
    return _media_type_for(f)


def _media_type_for(fmt: str) -> str:
    f = (fmt or "mp3").lower()
    return {"flac": "audio/flac", "mp3": "audio/mpeg", "opus": "audio/ogg",
            "aac": "audio/aac", "wav": "audio/wav", "wav32": "audio/wav"}.get(f, "audio/mpeg")


def _decode_audio_input(src: str) -> tuple[str, str]:
    """Return (base64-data, format) for a source audio URL/data-URL/path.

    ``src`` may be a ``data:audio/<fmt>;base64,...`` data URL (split into the
    base64 payload and the format) or an http(s) URL (which the server fetches
    itself — send it back as-is in the ``data`` slot, matching how the official
    script treats a non-data source).
    """
    if isinstance(src, str) and src.startswith("data:"):
        header, _, payload = src.partition(",")
        fmt = "wav"
        # header looks like "data:audio/wav;base64"
        if "/" in header:
            fmt = header.split("/", 1)[1].split(";", 1)[0] or fmt
        if ";base64" in header:
            return payload, fmt
        # Non-base64 data URL — decode then re-encode so the body carries
        # plain base64 as the official script expects.
        try:
            from urllib.parse import unquote_to_bytes  # local import; rare path
            return base64.b64encode(unquote_to_bytes(payload)).decode(), fmt
        except (TypeError, ValueError):
            return payload, fmt
    return str(src), "wav"


def _data_url_b64(url: str) -> str | None:
    """If ``url`` is a ``data:audio/...;base64,...`` URL, return the payload."""
    if isinstance(url, str) and url.startswith("data:") and ";base64," in url:
        return url.split(";base64,", 1)[1] or None
    return None


def _completion_audio_urls(message: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for item in (message.get("audio") or []):
        if not isinstance(item, dict):
            continue
        au = item.get("audio_url") or {}
        url = au.get("url") if isinstance(au, dict) else None
        if isinstance(url, str) and url:
            out.append(url)
    return out


def _completion_error(data: dict[str, Any]) -> str:
    detail = data.get("detail")
    if isinstance(detail, str) and detail.strip():
        return detail.strip()
    err = data.get("error")
    if isinstance(err, dict):
        for k in ("message", "detail", "error"):
            v = err.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    if isinstance(err, str) and err.strip():
        return err.strip()
    return ""


def _completion_usage(data: dict[str, Any], message: dict[str, Any]) -> MusicUsage | None:
    usage = data.get("usage")
    if isinstance(usage, dict):
        for k in ("duration", "audio_duration"):
            v = usage.get(k)
            if v is not None:
                try:
                    return MusicUsage(duration=int(float(v)))
                except (TypeError, ValueError):
                    pass
    metas = (message.get("metas") or {}) if isinstance(message, dict) else {}
    if isinstance(metas, dict) and metas.get("duration") is not None:
        try:
            return MusicUsage(duration=int(float(metas["duration"])))
        except (TypeError, ValueError):
            pass
    return None


# Re-exported so the timeout import isn't dropped by linters that flag unused
# imports; request_json surfaces timeouts already.
__all__ = ["AceStepProvider", "ProviderTimeoutError"]
