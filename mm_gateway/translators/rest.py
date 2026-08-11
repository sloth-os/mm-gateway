"""Translate the public REST contracts to and from internal unified models."""

from __future__ import annotations

from datetime import UTC, datetime

from mm_gateway.schemas.api import (
    ImageOutput,
    ImageRequest,
    ImageTaskResponse,
    LyricsInput,
    MusicAudioInput,
    MusicImageInput,
    MusicOutput,
    MusicRequest,
    MusicTaskResponse,
    ResourceLinks,
    TaskError,
    TextInput,
    Usage,
    VideoAudioInput,
    VideoImageInput,
    VideoInput,
    VideoOutput,
    VideoRequest,
    VideoTaskResponse,
)
from mm_gateway.schemas.image import UnifiedImageRequest, UnifiedImageTask
from mm_gateway.schemas.image import image_part as image_image_part
from mm_gateway.schemas.image import text_part as image_text_part
from mm_gateway.schemas.music import UnifiedMusicRequest, UnifiedMusicTask
from mm_gateway.schemas.music import audio_part as music_audio_part
from mm_gateway.schemas.music import image_part as music_image_part
from mm_gateway.schemas.music import text_part as music_text_part
from mm_gateway.schemas.video import UnifiedVideoRequest, UnifiedVideoTask, video_part
from mm_gateway.schemas.video import audio_part as video_audio_part
from mm_gateway.schemas.video import image_part as video_image_part
from mm_gateway.schemas.video import text_part as video_text_part
from mm_gateway.tasks.store import TaskRecord


def _data_uri(data: str, mime_type: str) -> str:
    return data if data.startswith("data:") else f"data:{mime_type};base64,{data}"


def _image_source(uri: str):
    if not uri.lower().startswith("data:"):
        return image_image_part(uri)
    header, _, data = uri.partition(",")
    mime_type = header[5:].split(";", 1)[0] or "image/png"
    return image_image_part(data=data, mime_type=mime_type)


def _apply_dimensions(parameters: dict) -> None:
    dimensions = parameters.pop("dimensions", None)
    if dimensions is not None:
        parameters.update(dimensions)


def from_image_request(body: ImageRequest) -> UnifiedImageRequest:
    content = []
    for part in body.input:
        if isinstance(part, TextInput):
            content.append(image_text_part(part.text))
        else:
            content.append(_image_source(part.uri))
    parameters = body.parameters.model_dump(exclude_none=True)
    _apply_dimensions(parameters)
    parameters["n"] = parameters.pop("output_count", None)
    parameters["num_inference_steps"] = parameters.pop("inference_steps", None)
    delivery = parameters.pop("delivery", None)
    if delivery is not None:
        parameters["response_format"] = (
            "url" if delivery == "remote" else "b64_json"
        )
    parameters["output_format"] = parameters.pop("file_format", None)
    parameters["output_compression"] = parameters.pop("compression", None)
    return UnifiedImageRequest(
        model=body.model,
        content=content,
        **{key: value for key, value in parameters.items() if value is not None},
    )


def from_video_request(body: VideoRequest) -> UnifiedVideoRequest:
    content = []
    for part in body.input:
        if isinstance(part, TextInput):
            content.append(video_text_part(part.text))
        elif isinstance(part, VideoImageInput):
            content.append(video_image_part(part.uri, part.role))
        elif isinstance(part, VideoAudioInput):
            content.append(video_audio_part(part.uri, part.role))
        elif isinstance(part, VideoInput):
            content.append(video_part(part.uri, part.role))
    parameters = body.parameters.model_dump(exclude_none=True)
    _apply_dimensions(parameters)
    parameters["duration"] = parameters.pop("duration_seconds", None)
    parameters["generate_audio"] = parameters.pop("include_audio", None)
    camera_motion = parameters.pop("camera_motion", None)
    if camera_motion is not None:
        parameters["camera_fixed"] = camera_motion == "fixed"
    parameters["prompt_extend"] = parameters.pop("enhance_prompt", None)
    parameters["return_last_frame"] = parameters.pop("include_last_frame", None)
    parameters["output_format"] = parameters.pop("file_format", None)
    return UnifiedVideoRequest(
        model=body.model,
        content=content,
        **{key: value for key, value in parameters.items() if value is not None},
    )


def from_music_request(body: MusicRequest) -> UnifiedMusicRequest:
    lyrics_parts = []
    content = []
    for part in body.input:
        if isinstance(part, TextInput):
            content.append(music_text_part(part.text))
        elif isinstance(part, LyricsInput):
            lyrics_parts.append(part.text)
        elif isinstance(part, MusicImageInput):
            content.append(music_image_part(part.uri, part.role))
        elif isinstance(part, MusicAudioInput):
            content.append(music_audio_part(part.uri, part.role))
    parameters = body.parameters.model_dump(exclude_none=True)
    title = parameters.pop("title", None)
    style = parameters.pop("style", None)
    parameters["duration"] = parameters.pop("duration_seconds", None)
    parameters["audio_format"] = parameters.pop("file_format", None)
    parameters["is_instrumental"] = parameters.pop("instrumental", None)
    parameters["n"] = parameters.pop("output_count", None)
    return UnifiedMusicRequest(
        model=body.model,
        content=content,
        lyrics="\n".join(lyrics_parts) or None,
        title=title,
        style=style,
        **{key: value for key, value in parameters.items() if value is not None},
    )


def _datetime(value: float) -> datetime:
    timestamp = float(value)
    if timestamp > 100_000_000_000:
        timestamp /= 1000
    return datetime.fromtimestamp(timestamp, tz=UTC)


def _created_at(
    task: UnifiedImageTask | UnifiedVideoTask | UnifiedMusicTask,
    record: TaskRecord,
) -> datetime:
    # The public resource is created by the gateway. Provider timestamps may use
    # different clocks or be synthesized afresh on every poll, so the task-store
    # timestamp is the stable REST resource timestamp.
    return _datetime(record.created_at)


def _completed_at(task: UnifiedImageTask | UnifiedVideoTask | UnifiedMusicTask
                  ) -> datetime | None:
    return _datetime(task.completed_at) if task.completed_at is not None else None


def _error(task: UnifiedImageTask | UnifiedVideoTask | UnifiedMusicTask
           ) -> TaskError | None:
    if not task.error:
        return None
    return TaskError(code=f"generation_{task.status}", message=task.error)


def to_image_response(task: UnifiedImageTask, record: TaskRecord, *, self_url: str
                      ) -> ImageTaskResponse:
    outputs = [
        ImageOutput(
            uri=item.url
            or _data_uri(item.b64_json or "", item.media_type or "image/png"),
            mime_type=item.media_type,
            revised_prompt=item.revised_prompt,
        )
        for item in task.images
        if item.url or item.b64_json
    ]
    usage = None
    if task.usage:
        usage = Usage(
            cost=task.usage.cost,
            input_tokens=task.usage.input_tokens,
            output_tokens=task.usage.output_tokens,
            total_tokens=task.usage.total_tokens,
            output_count=len(outputs) or None,
        )
    return ImageTaskResponse(
        id=record.task_id,
        model=record.model,
        status=task.status,
        outputs=outputs,
        error=_error(task),
        usage=usage,
        metadata=record.metadata,
        created_at=_created_at(task, record),
        completed_at=_completed_at(task),
        links=ResourceLinks(self_url=self_url),
    )


def to_video_response(task: UnifiedVideoTask, record: TaskRecord, *, self_url: str
                      ) -> VideoTaskResponse:
    outputs = [
        VideoOutput(
            uri=url,
            cover_uri=task.cover_url if index == 0 else None,
        )
        for index, url in enumerate(task.video_urls)
    ]
    usage = None
    if task.usage:
        usage = Usage(
            cost=task.usage.cost,
            output_count=task.usage.video_count or len(outputs) or None,
            duration_seconds=task.usage.video_duration,
        )
    return VideoTaskResponse(
        id=record.task_id,
        model=record.model,
        status=task.status,
        outputs=outputs,
        error=_error(task),
        usage=usage,
        metadata=record.metadata,
        created_at=_created_at(task, record),
        completed_at=_completed_at(task),
        links=ResourceLinks(self_url=self_url),
    )


def to_music_response(task: UnifiedMusicTask, record: TaskRecord, *, self_url: str
                      ) -> MusicTaskResponse:
    outputs = []
    if task.audio_b64:
        outputs.append(MusicOutput(
            uri=_data_uri(task.audio_b64, task.audio_media_type or "audio/mpeg"),
            mime_type=task.audio_media_type,
        ))
    outputs.extend(MusicOutput(uri=url, mime_type=task.audio_media_type)
                   for url in task.audio_urls)
    usage = None
    if task.usage:
        usage = Usage(
            cost=task.usage.cost,
            output_count=len(outputs) or None,
            duration_seconds=task.usage.duration,
        )
    return MusicTaskResponse(
        id=record.task_id,
        model=record.model,
        status=task.status,
        outputs=outputs,
        lyrics=task.lyrics,
        error=_error(task),
        usage=usage,
        metadata=record.metadata,
        created_at=_created_at(task, record),
        completed_at=_completed_at(task),
        links=ResourceLinks(self_url=self_url),
    )


__all__ = [
    "from_image_request",
    "from_music_request",
    "from_video_request",
    "to_image_response",
    "to_music_response",
    "to_video_response",
]
