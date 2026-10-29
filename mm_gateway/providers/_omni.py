"""Shared Gemini Omni Flash request/response helpers.

Gemini Omni Flash (``gemini-omni-1.1-flash`` on the Gemini API, ``gemini-omni-1.1-flash-preview`` on Vertex AI)
generates video through the Interactions API (``POST /v1beta/interactions``), not ``generate_videos``: text, image
and video ``input`` parts, a ``{"type": "video"}`` ``response_format`` carrying the aspect ratio, resolution and
duration, and, with ``background: true``, a long-running interaction polled with ``GET /v1beta/interactions/{id}``
that returns the MP4 inline as base64 (https://ai.google.dev/gemini-api/docs/omni). The AI Studio adapter speaks
that REST surface over httpx and the Vertex adapter the SDK's ``client.aio.interactions``; both build the body and
read the result here, like :mod:`mm_gateway.providers._lyria`.
"""

from __future__ import annotations

from typing import Any

from mm_gateway.providers._dimensions import aspect_ratio
from mm_gateway.schemas.video import UnifiedVideoRequest, UnifiedVideoTask, VideoImagePart, VideoVideoPart

# Gateway task ids of Omni interactions carry this prefix, so the video poll can tell them from Veo operations.
TASK_PREFIX = "interactions/"
OMNI_MIN_DURATION, OMNI_MAX_DURATION = 3, 10
_RESOLUTIONS = (("360p", 360), ("720p", 720), ("1080p", 1080), ("4k", 2160))
_RATIOS = frozenset({"16:9", "9:16"})
# Interaction status -> unified task status. ``incomplete`` (partial output) and ``budget_exceeded`` are terminal.
_STATUS_MAP = {
    "queued": "pending",
    "in_progress": "running",
    "completed": "succeeded",
    "failed": "failed",
    "cancelled": "cancelled",
    "incomplete": "failed",
    "budget_exceeded": "failed",
    "requires_action": "failed",
}


def omni_model(model: str | None) -> bool:
    """Models that generate video through the Interactions API rather than ``generate_videos``."""
    return bool(model) and model.lower().startswith("gemini-omni")


def _media(kind: str, url: str) -> dict[str, Any]:
    """An Interactions ``image``/``video`` input: inline base64 for a data URL, else a ``uri``."""
    if url.startswith("data:"):
        header, _, data = url.partition(",")
        return {"type": kind, "data": data, "mime_type": header[5:].split(";", 1)[0] or f"{kind}/*"}
    return {"type": kind, "uri": url}


def _resolution(request: UnifiedVideoRequest) -> str | None:
    if request.resolution:
        return request.resolution.strip().lower()
    if request.width is None or request.height is None:
        return None
    short = min(request.width, request.height)
    return min(_RESOLUTIONS, key=lambda t: abs(t[1] - short))[0]


def omni_body(request: UnifiedVideoRequest) -> dict[str, Any]:
    """The Interactions request body of ``request`` (a background interaction).

    Inputs go in the order the prompt's sources refer to them: the first frame, the last frame, the reference
    images, then the videos. Frames make the task ``image_to_video`` and references alone ``reference_to_video``.
    Omni takes no audio input and no negative prompt, so audio parts and ``negative_prompt`` are dropped; the
    sound is described in the prompt.
    """
    first: list[str] = []
    last: list[str] = []
    references: list[str] = []
    videos: list[str] = []
    for part in request.content:
        item = part.root
        if isinstance(item, VideoImagePart):
            if item.role == "first_frame":
                first.append(item.image_url.url)
            elif item.role == "last_frame":
                last.append(item.image_url.url)
            else:
                references.append(item.image_url.url)
        elif isinstance(item, VideoVideoPart):
            videos.append(item.video_url.url)
    parts: list[dict[str, Any]] = []
    if prompt := request.prompt():
        parts.append({"type": "text", "text": prompt})
    parts += [_media("image", url) for url in (*first[:1], *last[:1], *references)]
    parts += [_media("video", url) for url in videos]
    response_format: dict[str, Any] = {"type": "video"}
    if (ratio := aspect_ratio(request)) in _RATIOS:
        response_format["aspect_ratio"] = ratio
    if resolution := _resolution(request):
        response_format["resolution"] = resolution
    if request.duration is not None and request.duration > 0:
        seconds = max(OMNI_MIN_DURATION, min(OMNI_MAX_DURATION, int(round(request.duration))))
        response_format["duration"] = f"{seconds}s"
    body: dict[str, Any] = {
        "model": request.model,
        "input": parts,
        "response_format": response_format,
        "background": True,
        "store": True,
    }
    task = "image_to_video" if first or last else "reference_to_video" if references or videos else None
    if task:
        body["generation_config"] = {"video_config": {"task": task}}
    body.update(request.extra)
    return body


def _as_dict(data: Any) -> dict[str, Any]:
    if hasattr(data, "model_dump"):
        data = data.model_dump(exclude_none=True)
    return data if isinstance(data, dict) else {}


def omni_status(data: Any) -> str:
    """The unified task status of an interaction (``pending`` until it reports one)."""
    status = _as_dict(data).get("status")
    if not status:
        return "pending"
    return _STATUS_MAP.get(str(status), "running")


def omni_error(data: Any) -> str | None:
    """The interaction's error messages, joined (``None`` when it reports none)."""
    errors = _as_dict(data).get("errors") or []
    messages = [
        f"{e.get('code')}: {e.get('message')}" if e.get("code") else str(e.get("message"))
        for e in errors if isinstance(e, dict) and (e.get("message") or e.get("code"))
    ]
    return "; ".join(messages) or None


def omni_video(data: Any) -> str | None:
    """The generated video as a URL: a ``data:`` URL for inline bytes, else the returned ``uri``.

    The REST shape is ``steps[].content[]`` blocks (``{type: "video", data|uri, mime_type}``); the SDK model also
    surfaces the last video as ``output_video``.
    """
    data = _as_dict(data)
    blocks: list[Any] = []
    for step in data.get("steps") or []:
        content = (step or {}).get("content") or []
        blocks.extend(content if isinstance(content, list) else [content])
    if isinstance(data.get("output_video"), dict):
        blocks.append(data["output_video"])
    for block in reversed(blocks):
        if not isinstance(block, dict) or block.get("type", "video") != "video":
            continue
        if block.get("data"):
            return f"data:{block.get('mime_type') or 'video/mp4'};base64,{block['data']}"
        if block.get("uri"):
            return block["uri"]
    return None


def omni_task(task_id: str, provider: str, model: str, data: Any) -> UnifiedVideoTask:
    """The gateway task for an interaction (the create response or a poll)."""
    status = omni_status(data)
    task = UnifiedVideoTask(task_id=task_id, provider=provider, model=model, status=status)  # type: ignore[arg-type]
    if status == "succeeded":
        if url := omni_video(data):
            task.video_urls = [url]
        else:
            task.status = "failed"
            task.error = "gemini omni interaction completed with no video"
    elif status in ("failed", "cancelled"):
        task.error = omni_error(data) or f"gemini omni interaction {_as_dict(data).get('status')}"
    return task


__all__ = [
    "OMNI_MAX_DURATION",
    "OMNI_MIN_DURATION",
    "TASK_PREFIX",
    "omni_body",
    "omni_error",
    "omni_model",
    "omni_status",
    "omni_task",
    "omni_video",
]
