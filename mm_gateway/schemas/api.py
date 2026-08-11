"""Public REST wire models for the image, video, and music APIs.

Each modality has its own collection and item endpoints, while sharing the same
resource lifecycle and top-level request vocabulary:

``{model, input, parameters, routing, metadata}``

Both the envelope and ``parameters`` are strict. New backend capabilities must
first be expressed as provider-neutral gateway concepts and translated inside
the adapters; upstream-specific option names never cross this boundary.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

TaskStatus = Literal["pending", "running", "succeeded", "failed", "cancelled", "expired"]
Prompt = Annotated[str, Field(min_length=1)]

_STRICT = ConfigDict(extra="forbid")
_RESPONSE = ConfigDict(extra="allow")


# --------------------------------------------------------------------------- #
# Shared request and response fields
# --------------------------------------------------------------------------- #


class ProblemDetail(BaseModel):
    """RFC 9457 problem details with stable gateway extensions."""

    model_config = _RESPONSE

    type: str = Field(..., description="URI identifying the problem type.")
    title: str = Field(..., description="Short, stable summary of the problem type.")
    status: int = Field(..., ge=400, le=599, description="HTTP response status code.")
    detail: str = Field(..., description="Human-readable detail for this occurrence.")
    instance: str | None = Field(None, description="URI reference identifying this occurrence.")
    code: str = Field(..., description="Stable machine-readable gateway error code.")
    request_id: str | None = Field(None, description="Request correlation identifier.")
    errors: list[dict[str, Any]] | None = Field(
        None,
        description="Field-level validation errors, when applicable.",
    )


class TaskError(BaseModel):
    model_config = _RESPONSE

    code: str
    message: str


class ResourceLinks(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    self_url: str = Field(..., alias="self", description="Canonical resource URL.")


class RoutingDirective(BaseModel):
    """Select a server-defined, provider-neutral routing policy."""

    model_config = _STRICT

    profile: str = Field(
        ...,
        min_length=1,
        description=(
            "Gateway-defined routing profile, such as `quality`, `fast`, or "
            "`eu`. It never names a provider or backend."
        ),
    )


class Usage(BaseModel):
    """Provider-neutral usage fields shared by all three modalities."""

    model_config = _RESPONSE

    cost: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    output_count: int | None = None
    duration_seconds: float | None = None


class TextInput(BaseModel):
    model_config = _STRICT

    type: Literal["text"]
    text: Prompt


class ImageInput(BaseModel):
    model_config = _STRICT

    type: Literal["image"]
    url: str | None = Field(None, min_length=1)
    data: str | None = Field(None, min_length=1, description="Base64-encoded image bytes.")
    mime_type: str | None = None

    @model_validator(mode="after")
    def exactly_one_source(self) -> ImageInput:
        if bool(self.url) == bool(self.data):
            raise ValueError("image input requires exactly one of `url` or `data`")
        return self


class AudioInput(BaseModel):
    model_config = _STRICT

    type: Literal["audio"]
    url: str | None = Field(None, min_length=1)
    data: str | None = Field(None, min_length=1, description="Base64-encoded audio bytes.")
    mime_type: str | None = None

    @model_validator(mode="after")
    def exactly_one_source(self) -> AudioInput:
        if bool(self.url) == bool(self.data):
            raise ValueError("audio input requires exactly one of `url` or `data`")
        return self


class VideoInput(BaseModel):
    model_config = _STRICT

    type: Literal["video"]
    url: str | None = Field(None, min_length=1)
    data: str | None = Field(None, min_length=1, description="Base64-encoded video bytes.")
    mime_type: str | None = None
    role: Literal["reference_video"] = "reference_video"

    @model_validator(mode="after")
    def exactly_one_source(self) -> VideoInput:
        if bool(self.url) == bool(self.data):
            raise ValueError("video input requires exactly one of `url` or `data`")
        return self


class VideoImageInput(ImageInput):
    role: Literal["first_frame", "last_frame", "reference_image"] = "first_frame"


class MusicImageInput(ImageInput):
    role: Literal["reference_image"] = "reference_image"


class VideoAudioInput(AudioInput):
    role: Literal["reference_audio"] = "reference_audio"


class MusicAudioInput(AudioInput):
    role: Literal["reference_audio", "continuation_audio"] = "reference_audio"


class LyricsInput(BaseModel):
    model_config = _STRICT

    type: Literal["lyrics"]
    text: Prompt


ImageInputPart = Annotated[TextInput | ImageInput, Field(discriminator="type")]
VideoInputPart = Annotated[
    TextInput | VideoImageInput | VideoAudioInput | VideoInput, Field(discriminator="type")
]
MusicInputPart = Annotated[
    TextInput | LyricsInput | MusicImageInput | MusicAudioInput,
    Field(discriminator="type"),
]

ImageInputList = Annotated[
    list[ImageInputPart],
    Field(min_length=1, description="Non-empty ordered image-generation inputs."),
]
VideoInputList = Annotated[
    list[VideoInputPart],
    Field(min_length=1, description="Non-empty ordered video-generation inputs."),
]
MusicInputList = Annotated[
    list[MusicInputPart],
    Field(min_length=1, description="Non-empty ordered music-generation inputs."),
]


class _RequestBase(BaseModel):
    model_config = _STRICT

    model: str = Field(..., min_length=1, description="Model id returned by GET /v1/models.")
    routing: RoutingDirective | None = None
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Client-owned metadata returned unchanged with the task.",
    )


# --------------------------------------------------------------------------- #
# Image API
# --------------------------------------------------------------------------- #


class ImageParameters(BaseModel):
    model_config = _STRICT

    negative_prompt: str | None = None
    output_count: int | None = Field(None, ge=1, le=16)
    size: str | None = None
    width: int | None = Field(None, ge=1)
    height: int | None = Field(None, ge=1)
    aspect_ratio: str | None = None
    resolution: str | None = None
    quality: str | None = None
    style: str | None = None
    seed: int | None = None
    guidance_scale: float | None = Field(None, ge=0)
    inference_steps: int | None = Field(None, ge=1)
    strength: float | None = Field(None, ge=0, le=1)
    watermark: bool | None = None
    delivery: Literal["url", "base64"] | None = None
    file_format: str | None = None
    compression: int | None = Field(None, ge=0, le=100)
    background: str | None = None

    @model_validator(mode="after")
    def unambiguous_dimensions(self) -> ImageParameters:
        if (self.width is None) != (self.height is None):
            raise ValueError("`width` and `height` must be provided together")
        if self.size is not None and self.width is not None:
            raise ValueError("use either `size` or `width` and `height`, not both")
        return self


class ImageRequest(_RequestBase):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "model": "gateway-image-pro",
                "input": [
                    {"type": "text", "text": "a cyberpunk cat in the rain"}
                ],
                "parameters": {
                    "size": "1024x1024",
                    "quality": "high",
                    "delivery": "url",
                },
                "metadata": {"requester": "design-tool"},
            }
        },
    )

    input: ImageInputList
    parameters: ImageParameters = Field(default_factory=ImageParameters)


class ImageOutput(BaseModel):
    model_config = _RESPONSE

    url: str | None = None
    data: str | None = Field(None, description="Base64-encoded image bytes.")
    mime_type: str | None = None
    revised_prompt: str | None = None

    @model_validator(mode="after")
    def exactly_one_source(self) -> ImageOutput:
        if bool(self.url) == bool(self.data):
            raise ValueError("image output requires exactly one of `url` or `data`")
        return self


class ImageTaskResponse(BaseModel):
    model_config = _RESPONSE

    id: str
    object: Literal["image"] = "image"
    model: str
    status: TaskStatus
    outputs: list[ImageOutput] = Field(default_factory=list)
    error: TaskError | None = None
    usage: Usage | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    completed_at: datetime | None = None
    links: ResourceLinks


# --------------------------------------------------------------------------- #
# Video API
# --------------------------------------------------------------------------- #


class VideoParameters(BaseModel):
    model_config = _STRICT

    negative_prompt: str | None = None
    duration_seconds: float | None = Field(None, gt=0)
    aspect_ratio: str | None = None
    resolution: str | None = None
    size: str | None = None
    width: int | None = Field(None, ge=1)
    height: int | None = Field(None, ge=1)
    fps: int | None = Field(None, ge=1)
    seed: int | None = None
    include_audio: bool | None = None
    camera_motion: Literal["auto", "fixed"] | None = None
    watermark: bool | None = None
    enhance_prompt: bool | None = None
    include_last_frame: bool | None = None
    guidance_scale: float | None = Field(None, ge=0)
    motion_intensity: int | None = Field(None, ge=0, le=255)
    frame_count: int | None = Field(None, ge=1)
    file_format: str | None = None

    @model_validator(mode="after")
    def unambiguous_dimensions(self) -> VideoParameters:
        if (self.width is None) != (self.height is None):
            raise ValueError("`width` and `height` must be provided together")
        if self.size is not None and self.width is not None:
            raise ValueError("use either `size` or `width` and `height`, not both")
        return self


class VideoRequest(_RequestBase):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "model": "gateway-video-pro",
                "input": [
                    {
                        "type": "text",
                        "text": "a cinematic drone shot over mountains",
                    }
                ],
                "parameters": {"duration_seconds": 5, "aspect_ratio": "16:9"},
            }
        },
    )

    input: VideoInputList
    parameters: VideoParameters = Field(default_factory=VideoParameters)


class VideoOutput(BaseModel):
    model_config = _RESPONSE

    url: str
    cover_url: str | None = None
    mime_type: str | None = None


class VideoTaskResponse(BaseModel):
    model_config = _RESPONSE

    id: str
    object: Literal["video"] = "video"
    model: str
    status: TaskStatus
    outputs: list[VideoOutput] = Field(default_factory=list)
    error: TaskError | None = None
    usage: Usage | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    completed_at: datetime | None = None
    links: ResourceLinks


# --------------------------------------------------------------------------- #
# Music API
# --------------------------------------------------------------------------- #


class MusicParameters(BaseModel):
    model_config = _STRICT

    negative_prompt: str | None = None
    title: str | None = None
    style: str | None = None
    duration_seconds: float | None = Field(None, gt=0)
    bpm: int | None = Field(None, ge=1)
    key: str | None = None
    scale: str | None = None
    time_signature: str | None = None
    vocal_language: str | None = None
    file_format: str | None = None
    sample_rate_hz: int | None = Field(None, ge=8000)
    bitrate_kbps: int | None = Field(None, ge=8)
    instrumental: bool | None = None
    seed: int | None = None
    guidance_scale: float | None = Field(None, ge=0)
    output_count: int | None = Field(None, ge=1, le=16)
    enhance_lyrics: bool | None = None
    voice: str | None = None
    vocal_gender: str | None = None
    style_strength: float | None = Field(None, ge=0, le=1)
    novelty: float | None = Field(None, ge=0, le=1)
    reference_audio_strength: float | None = Field(None, ge=0, le=1)
    inference_steps: int | None = Field(None, ge=1)
    respect_section_durations: bool | None = None
    provenance: bool | None = None


class MusicRequest(_RequestBase):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "model": "gateway-music-lyria",
                "input": [
                    {"type": "text", "text": "an upbeat pop song about summer"}
                ],
                "parameters": {
                    "duration_seconds": 30,
                    "bpm": 120,
                    "file_format": "wav",
                },
            }
        },
    )

    input: MusicInputList
    parameters: MusicParameters = Field(default_factory=MusicParameters)


class MusicOutput(BaseModel):
    model_config = _RESPONSE

    url: str | None = None
    data: str | None = Field(None, description="Base64-encoded audio bytes.")
    mime_type: str | None = None

    @model_validator(mode="after")
    def exactly_one_source(self) -> MusicOutput:
        if bool(self.url) == bool(self.data):
            raise ValueError("music output requires exactly one of `url` or `data`")
        return self


class MusicTaskResponse(BaseModel):
    model_config = _RESPONSE

    id: str
    object: Literal["music"] = "music"
    model: str
    status: TaskStatus
    outputs: list[MusicOutput] = Field(default_factory=list)
    lyrics: str | None = None
    error: TaskError | None = None
    usage: Usage | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    completed_at: datetime | None = None
    links: ResourceLinks


# --------------------------------------------------------------------------- #
# Meta API
# --------------------------------------------------------------------------- #


class ModelEntry(BaseModel):
    model_config = _RESPONSE

    id: str
    object: Literal["model"] = "model"
    modality: Literal["image", "video", "music"]


class ModelListResponse(BaseModel):
    model_config = _RESPONSE

    object: Literal["list"] = "list"
    data: list[ModelEntry]


class HealthResponse(BaseModel):
    model_config = _RESPONSE

    status: Literal["ok"] = "ok"


__all__ = [
    "AudioInput",
    "HealthResponse",
    "ImageInput",
    "ImageInputList",
    "ImageOutput",
    "ImageParameters",
    "ImageRequest",
    "ImageTaskResponse",
    "LyricsInput",
    "ModelEntry",
    "ModelListResponse",
    "MusicAudioInput",
    "MusicImageInput",
    "MusicInputList",
    "MusicOutput",
    "MusicParameters",
    "MusicRequest",
    "MusicTaskResponse",
    "ProblemDetail",
    "ResourceLinks",
    "RoutingDirective",
    "TaskError",
    "TaskStatus",
    "TextInput",
    "Usage",
    "VideoAudioInput",
    "VideoImageInput",
    "VideoInput",
    "VideoInputList",
    "VideoOutput",
    "VideoParameters",
    "VideoRequest",
    "VideoTaskResponse",
]
