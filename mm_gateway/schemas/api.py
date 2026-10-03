"""Public REST wire models for media generation and reusable voices.

Each modality has its own collection and item endpoints, while sharing the same
resource lifecycle and top-level request vocabulary:

``{model, input, parameters, routing, metadata}``

Both the envelope and ``parameters`` are strict. New backend capabilities must
first be expressed as provider-neutral gateway concepts and translated inside
the adapters; upstream-specific option names never cross this boundary.
"""

from __future__ import annotations

import base64
import binascii
from datetime import datetime
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator, model_validator

TaskStatus = Literal["pending", "running", "succeeded", "failed", "cancelled", "expired"]
Prompt = Annotated[str, Field(min_length=1)]

_STRICT = ConfigDict(extra="forbid")
_RESPONSE = ConfigDict(extra="allow")


def _validate_media_uri(value: str) -> str:
    if any(character.isspace() for character in value):
        raise ValueError("media URI must not contain whitespace")
    parsed = urlsplit(value)
    if not parsed.scheme:
        raise ValueError("media URI must be absolute")
    if parsed.scheme.lower() == "data":
        metadata, separator, payload = value[5:].partition(",")
        if not separator or ";base64" not in metadata.lower() or not payload:
            raise ValueError("inline media must use a base64 data URI")
    return value


MediaUri = Annotated[
    str,
    Field(
        min_length=1,
        description="Absolute media URI. Inline media uses a base64 data URI.",
        json_schema_extra={"format": "uri"},
    ),
    AfterValidator(_validate_media_uri),
]


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


class BudgetDirective(BaseModel):
    """A client-chosen spend bucket within the key, optionally self-capped."""

    model_config = _STRICT

    scope: str = Field(
        ...,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9._:/-]+$",
        description="Spend bucket name (a project, a customer, a batch).",
    )
    limit_usd: float | None = Field(
        None,
        ge=0,
        description="Self-imposed cap for the scope in USD (the operator's scope cap still applies).",
    )


class RoutingDirective(BaseModel):
    """Steer auto mode: policy, ordering, cost ceiling, fallbacks and budget scope.

    See docs/design/auto-mode.md. Every member is optional.
    """

    model_config = _STRICT

    profile: str | None = Field(
        None,
        min_length=1,
        description=(
            "Gateway-defined routing profile, such as `quality`, `fast`, or "
            "`eu`. It never names a provider or backend."
        ),
    )
    optimize: Literal["balanced", "cost", "latency"] | None = Field(
        None,
        description="How admissible candidates are ordered (default: the gateway's default, `balanced`).",
    )
    max_cost_usd: float | None = Field(
        None,
        gt=0,
        description="Hard per-task ceiling on the estimated cost in USD; unpriced models are excluded.",
    )
    fallback: Literal["none", "same_model", "any"] | None = Field(
        None,
        description=(
            "Pinned models only: `none` (default) tries one backend, `same_model` "
            "every backend/account serving the model, `any` also the replacement "
            "and the auto candidates when the model is retired or unavailable."
        ),
    )
    budget: BudgetDirective | None = None


class Usage(BaseModel):
    """Provider-neutral usage fields shared by all three modalities."""

    model_config = _RESPONSE

    cost: float | None = None
    cost_source: Literal["provider", "estimate"] | None = None
    currency: Literal["USD"] | None = None
    input_characters: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    output_count: int | None = None
    duration_seconds: float | None = None


class BudgetState(BaseModel):
    """The state of one budget (a key's period or a client scope)."""

    model_config = _RESPONSE

    scope: str | None = None
    limit_usd: float | None = None
    spent_usd: float = 0.0
    reserved_usd: float = 0.0
    remaining_usd: float | None = None
    tasks: int | None = None


class RoutingInfo(BaseModel):
    """How auto mode served a task (docs/design/auto-mode.md#fallbacks)."""

    model_config = _RESPONSE

    requested_model: str
    fallback: bool = False
    fallback_reason: str | None = None
    attempts: int = 1
    optimize: str = "balanced"
    estimated_cost: float | None = None
    budget: BudgetState | None = None


class Dimensions(BaseModel):
    """Exact output dimensions in pixels."""

    model_config = _STRICT

    width: int = Field(..., ge=1)
    height: int = Field(..., ge=1)


class TextInput(BaseModel):
    model_config = _STRICT

    type: Literal["text"]
    text: Prompt


class ImageInput(BaseModel):
    model_config = _STRICT

    type: Literal["image"]
    uri: MediaUri


class AudioInput(BaseModel):
    model_config = _STRICT

    type: Literal["audio"]
    uri: MediaUri


class VideoInput(BaseModel):
    model_config = _STRICT

    type: Literal["video"]
    uri: MediaUri
    role: Literal["reference_video"] = "reference_video"


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

    model: str | None = Field(
        None,
        description=(
            "Model id returned by GET /v1/models, or omit / set to `auto` to let "
            "the gateway auto-route to a backend whose limits fit the request's "
            "input (modalities, dimensions, duration, ...)."
        ),
    )
    routing: RoutingDirective | None = None
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Client-owned metadata returned unchanged with the task.",
    )

    @model_validator(mode="after")
    def _normalize_model(self) -> "_RequestBase":
        # Empty string is the same as omission; lowercase "auto" stays canonical.
        if self.model is not None:
            stripped = self.model.strip()
            if stripped == "":
                self.model = None
            elif stripped.lower() == "auto":
                self.model = "auto"
        return self


# --------------------------------------------------------------------------- #
# Image API
# --------------------------------------------------------------------------- #


class ImageParameters(BaseModel):
    model_config = _STRICT

    negative_prompt: str | None = None
    output_count: int | None = Field(None, ge=1, le=16)
    dimensions: Dimensions | None = None
    quality: str | None = None
    style: str | None = None
    seed: int | None = None
    guidance_scale: float | None = Field(None, ge=0)
    inference_steps: int | None = Field(None, ge=1)
    strength: float | None = Field(None, ge=0, le=1)
    watermark: bool | None = None
    delivery: Literal["remote", "inline"] | None = None
    file_format: str | None = None
    compression: int | None = Field(None, ge=0, le=100)
    background: str | None = None

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
                    "dimensions": {"width": 1024, "height": 1024},
                    "quality": "high",
                    "delivery": "remote",
                },
                "metadata": {"requester": "design-tool"},
            }
        },
    )

    input: ImageInputList
    parameters: ImageParameters = Field(default_factory=ImageParameters)


class ImageOutput(BaseModel):
    model_config = _RESPONSE

    uri: MediaUri
    mime_type: str | None = None
    revised_prompt: str | None = None


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
    routing: RoutingInfo | None = None
    links: ResourceLinks


# --------------------------------------------------------------------------- #
# Video API
# --------------------------------------------------------------------------- #


class VideoParameters(BaseModel):
    model_config = _STRICT

    negative_prompt: str | None = None
    duration_seconds: float | None = Field(None, gt=0)
    dimensions: Dimensions | None = None
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
                "parameters": {
                    "duration_seconds": 5,
                    "dimensions": {"width": 1280, "height": 720},
                },
            }
        },
    )

    input: VideoInputList
    parameters: VideoParameters = Field(default_factory=VideoParameters)


class VideoOutput(BaseModel):
    model_config = _RESPONSE

    uri: MediaUri
    cover_uri: MediaUri | None = None
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
    routing: RoutingInfo | None = None
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

    uri: MediaUri
    mime_type: str | None = None


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
    routing: RoutingInfo | None = None
    links: ResourceLinks


# Speech and reusable voice clones use the same envelope and task lifecycle.
AudioFileFormat = Literal["mp3", "wav", "pcm", "flac", "opus", "aac"]
AudioInputList = Annotated[list[TextInput], Field(min_length=1)]


class AudioParameters(BaseModel):
    model_config = _STRICT

    voice: str = Field("default", min_length=1, max_length=128,
                       pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
                       description="A gateway voice id from GET /v1/voices.")
    instructions: str | None = Field(None, min_length=1, max_length=4096)
    language: str | None = Field(None, pattern=r"^[a-z]{2,3}(-[A-Za-z0-9]{2,8})*$")
    speed: float | None = Field(None, ge=0.25, le=4, allow_inf_nan=False)
    file_format: AudioFileFormat | None = None
    sample_rate_hz: int | None = Field(None, ge=8000, le=48000)
    bitrate_kbps: int | None = Field(None, ge=8, le=320)
    delivery: Literal["inline", "remote"] | None = None
    seed: int | None = Field(None, ge=0, le=4294967295)


class AudioRequest(_RequestBase):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": {"model": "gateway-audio-pro",
            "input": [{"type": "text", "text": "Welcome to the gateway."}],
            "parameters": {"voice": "default", "file_format": "mp3"}}},
    )
    input: AudioInputList
    parameters: AudioParameters = Field(default_factory=AudioParameters)

    @model_validator(mode="after")
    def _spoken_text(self) -> "AudioRequest":
        if not any(part.text.strip() for part in self.input):
            raise ValueError("speech input must contain spoken text")
        return self


class AudioOutput(BaseModel):
    model_config = _RESPONSE

    uri: MediaUri
    mime_type: str | None = None
    sample_rate_hz: int | None = None
    channels: int | None = None
    duration_seconds: float | None = None


class AudioTaskResponse(BaseModel):
    model_config = _RESPONSE

    id: str
    object: Literal["audio"] = "audio"
    model: str
    status: TaskStatus
    outputs: list[AudioOutput] = Field(default_factory=list)
    error: TaskError | None = None
    usage: Usage | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    completed_at: datetime | None = None
    routing: RoutingInfo | None = None
    links: ResourceLinks


def _validate_voice_uri(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme == "data":
        header, _, payload = value.partition(",")
        if not header.lower().startswith("data:audio/"):
            raise ValueError("voice samples must be audio data URIs")
        # Bound memory before decoding; each sample is limited to 20 MiB.
        if len(payload) > 4 * ((20 * 1024 * 1024 + 2) // 3):
            raise ValueError("voice sample exceeds 20 MiB")
        try:
            blob = base64.b64decode(payload, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("voice sample contains invalid base64") from exc
        if not blob or len(blob) > 20 * 1024 * 1024:
            raise ValueError("voice sample must contain 1 byte to 20 MiB")
    elif parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("voice samples require an HTTP(S) URL or audio data URI")
    return value


VoiceMediaUri = Annotated[MediaUri, AfterValidator(_validate_voice_uri)]


class VoiceSampleInput(BaseModel):
    model_config = _STRICT

    type: Literal["audio"]
    uri: VoiceMediaUri


VoiceInputList = Annotated[list[VoiceSampleInput], Field(min_length=1, max_length=10)]


class VoiceConsent(BaseModel):
    model_config = _STRICT

    granted: Literal[True] = Field(..., description="The speaker authorized creation and use of this voice.")
    recording_uri: VoiceMediaUri | None = None
    language: str | None = Field(None, pattern=r"^[a-z]{2,3}$")

    @field_validator("granted", mode="before")
    @classmethod
    def _explicit_consent(cls, value: Any) -> bool:
        if value is not True:
            raise ValueError("consent.granted must be true")
        return value

    @model_validator(mode="after")
    def _recording_language(self) -> "VoiceConsent":
        if bool(self.recording_uri) != bool(self.language):
            raise ValueError("consent recording_uri and language must be supplied together")
        return self


class VoiceParameters(BaseModel):
    model_config = _STRICT

    name: str = Field(..., min_length=1, max_length=128, pattern=r".*\S.*")
    description: str | None = Field(None, max_length=1000)
    remove_background_noise: bool | None = None


class VoiceCloneRequest(_RequestBase):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": {
            "input": [{"type": "audio", "uri": "data:audio/wav;base64,c2FtcGxl"}],
            "parameters": {"name": "Narrator"}, "consent": {"granted": True}}},
    )
    input: VoiceInputList
    parameters: VoiceParameters
    consent: VoiceConsent


class VoiceResponse(BaseModel):
    model_config = _RESPONSE

    id: str
    object: Literal["voice"] = "voice"
    kind: Literal["preset", "cloned"] = "cloned"
    name: str
    model: str | None = None
    status: TaskStatus
    verification_required: bool = False
    error: TaskError | None = None
    usage: Usage | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None
    completed_at: datetime | None = None
    routing: RoutingInfo | None = None
    links: ResourceLinks


class VoiceListResponse(BaseModel):
    model_config = _RESPONSE

    object: Literal["list"] = "list"
    data: list[VoiceResponse]


# --------------------------------------------------------------------------- #
# Meta API
# --------------------------------------------------------------------------- #


class ModelEntry(BaseModel):
    model_config = _RESPONSE

    id: str
    object: Literal["model"] = "model"
    modality: Literal["image", "video", "music", "audio"]


class ModelListResponse(BaseModel):
    model_config = _RESPONSE

    object: Literal["list"] = "list"
    data: list[ModelEntry]


class ModelLimitsEntry(BaseModel):
    """A model catalogue entry enriched with its documented input/output limits.

    ``limits`` carries the neutral limits the auto-router uses and that a
    client can consult when crafting a prompt for a specific model. Unknown
    limits are omitted; clients ignore unknown response members.
    """

    model_config = _RESPONSE

    id: str
    object: Literal["model"] = "model"
    modality: Literal["image", "video", "music", "audio"]
    limits: dict[str, Any] = Field(
        default_factory=dict,
        description="Neutral input/output limits (modalities, max prompt, max "
                    "output count, supported sizes/durations, role flags, ...).",
    )


class ModelLimitsListResponse(BaseModel):
    model_config = _RESPONSE

    object: Literal["list"] = "list"
    data: list[ModelLimitsEntry]


class EstimateCandidate(BaseModel):
    model_config = _RESPONSE

    model: str
    estimated_cost: float | None = None
    lifecycle: Literal["active", "deprecated", "retired"] = "active"
    admissible: bool = True
    reason: str | None = Field(
        None,
        description="Why the candidate is not admissible: limits, retired, max_cost, unpriced or budget.",
    )


class EstimateResponse(BaseModel):
    """The routing and cost a create would get, without creating a task."""

    model_config = _RESPONSE

    object: Literal["estimate"] = "estimate"
    modality: Literal["image", "video", "music", "audio"]
    currency: Literal["USD"] = "USD"
    model: str | None = None
    estimated_cost: float | None = None
    candidates: list[EstimateCandidate] = Field(default_factory=list)
    budget: BudgetState | None = None


class UsagePeriod(BaseModel):
    model_config = _RESPONSE

    kind: Literal["day", "month", "total"]
    start: datetime | None = None
    end: datetime | None = None


class ModelSpend(BaseModel):
    model_config = _RESPONSE

    model: str
    modality: Literal["image", "video", "music", "audio"]
    spent_usd: float = 0.0
    tasks: int = 0


class UsageResponse(BaseModel):
    """Spend, reservations and budgets of the authenticated key."""

    model_config = _RESPONSE

    object: Literal["usage"] = "usage"
    currency: Literal["USD"] = "USD"
    period: UsagePeriod
    key: BudgetState
    scopes: list[BudgetState] = Field(default_factory=list)
    models: list[ModelSpend] = Field(default_factory=list)


class HealthResponse(BaseModel):
    model_config = _RESPONSE

    status: Literal["ok"] = "ok"


__all__ = [
    "AudioInput",
    "AudioInputList",
    "AudioOutput",
    "AudioParameters",
    "AudioRequest",
    "AudioTaskResponse",
    "VoiceCloneRequest",
    "VoiceConsent",
    "VoiceInputList",
    "VoiceListResponse",
    "VoiceParameters",
    "VoiceResponse",
    "VoiceSampleInput",
    "BudgetDirective",
    "BudgetState",
    "EstimateCandidate",
    "EstimateResponse",
    "ModelSpend",
    "RoutingInfo",
    "UsagePeriod",
    "UsageResponse",
    "Dimensions",
    "HealthResponse",
    "ImageInput",
    "ImageInputList",
    "ImageOutput",
    "ImageParameters",
    "ImageRequest",
    "ImageTaskResponse",
    "LyricsInput",
    "MediaUri",
    "ModelEntry",
    "ModelLimitsEntry",
    "ModelLimitsListResponse",
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
