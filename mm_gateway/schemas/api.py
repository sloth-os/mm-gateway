"""Front-end wire-shape Pydantic models for the OpenAPI (Swagger) spec.

These mirror the front-end request/response shapes the gateway speaks on the
HTTP boundary -- the Gemini Interactions image shape ({model, input, config}),
the Seedance video shape ({model, content, ...}), and the Gemini Lyria 3 music
shape ({model, input, config}) -- not the unified internal schemas in
schemas/image.py / video.py / music.py.

They exist so FastAPI can emit a complete OpenAPI spec (request bodies, response
bodies, header/query parameters, examples, required/optional markers) from the
routes. The routes still parse raw JSON with request.json() and feed it to the
translators, so these models are deliberately permissive (extra="allow"): unknown
keys pass straight through to the translator's extra bucket, so a typed model
can never reject a body the lenient translator would accept.

Note on responses: the routes return JSONResponse(content=...), so a
response_model declared on a route is consulted only when generating the OpenAPI
schema -- FastAPI does not re-validate or filter a Response object at runtime.
These response models therefore document the exact translator output shapes
(to_gemini_task / to_seedance_task / to_lyria_task) without altering runtime.
"""

from __future__ import annotations

from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, RootModel

# Permissive config: unknown keys pass straight through to the translator's
# extra bucket, so a typed model can never reject a body the translator accepts.
_PERMISSIVE = ConfigDict(extra="allow")

TaskStatus = Literal["pending", "running", "succeeded", "failed", "cancelled", "expired"]


# --------------------------------------------------------------------------- #
# Shared shapes
# --------------------------------------------------------------------------- #


class ErrorBody(BaseModel):
    """The gateway's consistent error envelope (GatewayError.to_dict)."""

    code: str = Field(..., description="Stable machine-readable error code.")
    message: str = Field(..., description="Human-readable error message.")
    provider: str | None = Field(None, description="Backend that produced the error, if any.")
    details: dict[str, Any] | None = Field(None, description="Extra structured detail.")


class ErrorEnvelope(BaseModel):
    """Every non-2xx response is wrapped in {"error": ErrorBody}."""

    error: ErrorBody = Field(..., description="The error descriptor.")


class UsageCost(BaseModel):
    """Per-task usage/cost summary surfaced on poll responses."""

    cost: float | None = Field(None, description="Estimated cost in USD.")


# --------------------------------------------------------------------------- #
# Content parts (the typed blocks inside `input` / `content` arrays)
# --------------------------------------------------------------------------- #


class TextPart(BaseModel):
    model_config = _PERMISSIVE
    type: Literal["text"] = Field("text", description="A text prompt fragment.")
    text: str = Field(..., description="The text content.")


class ImagePart(BaseModel):
    """An image input: either a URL or inline base64 data."""

    model_config = _PERMISSIVE
    type: Literal["image"] = Field("image", description="An image input.")
    url: str | None = Field(None, description="Image URL.")
    data: str | None = Field(None, description="Inline base64-encoded image bytes.")
    mime_type: str | None = Field(None, description="MIME type of inline data.")


class ImageUrlPart(BaseModel):
    """Seedance-style image_url part with an optional role."""

    model_config = _PERMISSIVE
    type: Literal["image_url"] = Field("image_url", description="An image_url input.")
    image_url: dict[str, Any] = Field(..., description='{"url": "..."} envelope.')
    role: str | None = Field(None, description="Role, e.g. first_frame / last_frame.")


class VideoUrlPart(BaseModel):
    model_config = _PERMISSIVE
    type: Literal["video_url"] = Field("video_url", description="A reference video input.")
    video_url: dict[str, Any] = Field(..., description='{"url": "..."} envelope.')
    role: str | None = Field(None, description="Role, e.g. reference_video.")


class AudioUrlPart(BaseModel):
    model_config = _PERMISSIVE
    type: Literal["audio_url"] = Field("audio_url", description="A reference audio input.")
    audio_url: dict[str, Any] = Field(..., description='{"url": "..."} envelope.')
    role: str | None = Field(None, description="Role, e.g. reference_audio.")


class DraftPart(BaseModel):
    model_config = _PERMISSIVE
    type: Literal["draft_task"] = Field("draft_task", description="A draft task reference.")
    draft_task: dict[str, Any] = Field(..., description="Draft task descriptor.")


# A content part is any of the typed blocks above (union kept loose so unknown
# part types are not rejected -- the translators silently drop unknown types).
ContentPart = Union[
    TextPart, ImagePart, ImageUrlPart, VideoUrlPart, AudioUrlPart, DraftPart, dict
]


# --------------------------------------------------------------------------- #
# Routing directive (provider: {tag|backend})
# --------------------------------------------------------------------------- #


class ProviderDirective(BaseModel):
    """Routing directive read from the request body by routing_overrides."""

    model_config = _PERMISSIVE
    tag: str | None = Field(None, description="Pin to a backend by tag label.")
    backend: str | None = Field(None, description="Pin to a backend by name.")


# --------------------------------------------------------------------------- #
# IMAGE -- Gemini Interactions shape
# --------------------------------------------------------------------------- #


class ImageConfig(BaseModel):
    """Knobs accepted inside `config` for an image request (Gemini shape).

    Any unknown key passes through to the provider via the translator's extra.
    """

    model_config = _PERMISSIVE
    negative_prompt: str | None = Field(None, description="Negative prompt.")
    n: int | None = Field(None, ge=1, le=16, description="Number of images.")
    size: str | None = Field(None, description="Output size, e.g. 1024x1024.")
    width: int | None = Field(None, description="Width in pixels.")
    height: int | None = Field(None, description="Height in pixels.")
    aspect_ratio: str | None = Field(None, description="Aspect ratio, e.g. 16:9.")
    resolution: str | None = Field(None, description="Resolution hint.")
    quality: str | None = Field(None, description="Quality preset, e.g. high.")
    style: str | None = Field(None, description="Style preset.")
    seed: int | None = Field(None, description="RNG seed for reproducibility.")
    guidance_scale: float | None = Field(None, description="CFG guidance scale.")
    num_inference_steps: int | None = Field(None, description="Inference step count.")
    strength: float | None = Field(None, description="Denoising strength (img2img).")
    watermark: bool | None = Field(None, description="Apply provider watermark.")
    response_format: dict[str, Any] | str | None = Field(
        None, description='Imagen/Lyria envelope {"type":"url|b64_json","quality":...} or "url"/"b64_json".'
    )
    output_format: str | None = Field(None, description="Output file format.")
    output_compression: int | None = Field(None, description="Output compression 0-100.")
    background: str | None = Field(None, description="Background, e.g. transparent.")
    stream: bool | None = Field(None, description="Stream output.")
    callback_url: str | None = Field(None, description="Async callback webhook.")
    user: str | None = Field(None, description="End-user id for abuse tracking.")


class ImageRequest(BaseModel):
    """POST /v1/images request body -- the Gemini Interactions image shape."""

    model_config = _PERMISSIVE
    model: str = Field(..., description="Model id or gateway alias (e.g. gateway-image-pro).")
    input: Union[str, list[ContentPart]] = Field(
        ...,
        description="A string prompt, or an array of typed parts (text/image).",
    )
    config: ImageConfig | None = Field(None, description="Generation knobs.")
    response_format: str | None = Field(None, description='"url" or "b64_json" output delivery.')
    provider: ProviderDirective | None = Field(None, description="Routing override.")

    model_config = ConfigDict(
        extra="allow", json_schema_extra={
            "example": {
                "model": "gateway-image-pro",
                "input": "a cyberpunk cat in the rain, neon",
                "config": {"n": 1, "size": "1024x1024", "quality": "high", "seed": 42},
                "response_format": "url",
            }
        }
    )


# --------------------------------------------------------------------------- #
# VIDEO -- Seedance shape
# --------------------------------------------------------------------------- #


class VideoRequest(BaseModel):
    """POST /v1/videos request body -- the Seedance (Volcengine Ark) shape."""

    model_config = ConfigDict(
        extra="allow", json_schema_extra={
            "example": {
                "model": "gateway-video-pro",
                "content": [{"type": "text", "text": "a cat playing piano, cinematic"}],
                "ratio": "16:9",
                "duration": 5,
                "seed": 7,
            }
        }
    )
    model: str = Field(..., description="Model id or gateway alias.")
    content: list[ContentPart] = Field(..., description="Typed parts: text, image_url, video_url, audio_url, draft_task.")
    negative_prompt: str | None = Field(None, description="Negative prompt.")
    duration: float | None = Field(None, description="Clip duration in seconds.")
    ratio: str | None = Field(None, description="Aspect ratio, e.g. 16:9.")
    aspect_ratio: str | None = Field(None, description="Alias for ratio.")
    resolution: str | None = Field(None, description="Resolution preset.")
    size: str | None = Field(None, description="Size, e.g. 1920x1080.")
    width: int | None = Field(None, description="Width in pixels.")
    height: int | None = Field(None, description="Height in pixels.")
    fps: int | None = Field(None, description="Frames per second.")
    seed: int | None = Field(None, description="RNG seed.")
    generate_audio: bool | None = Field(None, description="Generate accompanying audio.")
    camera_fixed: bool | None = Field(None, description="Fix the camera.")
    watermark: bool | None = Field(None, description="Apply provider watermark.")
    prompt_extend: bool | None = Field(None, description="Let the provider expand the prompt.")
    callback_url: str | None = Field(None, description="Async callback webhook.")
    return_last_frame: bool | None = Field(None, description="Return the last frame for chaining.")
    provider: ProviderDirective | None = Field(None, description="Routing override.")


# --------------------------------------------------------------------------- #
# MUSIC -- Gemini Lyria 3 shape
# --------------------------------------------------------------------------- #


class MusicResponseFormat(BaseModel):
    """Lyria response_format envelope: {"type": "audio", "quality": ...}."""

    model_config = _PERMISSIVE
    type: str | None = Field(None, description='"audio" selects wav output.')
    quality: str | None = Field(None, description="Audio quality preset.")


class MusicConfig(BaseModel):
    """Knobs accepted inside `config` for a music request (Lyria shape)."""

    model_config = _PERMISSIVE
    negative_prompt: str | None = Field(None, description="Negative prompt.")
    duration: float | None = Field(None, description="Duration in seconds.")
    bpm: float | None = Field(None, description="Tempo in beats per minute.")
    key_scale: str | None = Field(None, description='Musical key, e.g. "C minor".')
    key: str | None = Field(None, description='Tonal key, e.g. "C".')
    scale: str | None = Field(None, description='Scale, e.g. "minor".')
    time_signature: str | None = Field(None, description='Time signature, e.g. "4/4".')
    vocal_language: str | None = Field(None, description="Vocal language.")
    audio_format: str | None = Field(None, description='Output format, e.g. "wav".')
    audio_quality: str | None = Field(None, description="Output quality preset.")
    is_instrumental: bool | None = Field(None, description="Instrumental only (no vocals).")
    generate_audio: bool | None = Field(None, description="Generate audio (vs. structure only).")
    seed: int | None = Field(None, description="RNG seed.")
    guidance_scale: float | None = Field(None, description="Guidance scale.")
    n: int | None = Field(None, ge=1, description="Number of outputs.")
    callback_url: str | None = Field(None, description="Async callback webhook.")
    response_format: MusicResponseFormat | None = Field(None, description="Lyria response envelope.")


class MusicRequest(BaseModel):
    """POST /v1/music request body -- the Gemini Lyria 3 Interactions shape."""

    model_config = ConfigDict(
        extra="allow", json_schema_extra={
            "example": {
                "model": "gateway-music-lyria",
                "input": "an upbeat pop song about summer",
                "config": {"duration": 30, "bpm": 120, "is_instrumental": False, "audio_format": "wav"},
                "response_format": {"type": "audio"},
            }
        }
    )
    model: str = Field(..., description="Model id or gateway alias.")
    input: Union[str, list[ContentPart]] = Field(
        ..., description="A string prompt, or an array of typed parts (text/image)."
    )
    config: MusicConfig | None = Field(None, description="Generation knobs.")
    response_format: MusicResponseFormat | None = Field(None, description="Lyria response envelope.")
    provider: ProviderDirective | None = Field(None, description="Routing override.")


# --------------------------------------------------------------------------- #
# Create-response shape (POST .../create -> {"id": "..."})
# --------------------------------------------------------------------------- #


class CreateResponse(BaseModel):
    """The create endpoints return only the task id."""

    id: str = Field(..., description="Opaque task id used to poll for results.")


# --------------------------------------------------------------------------- #
# Poll-response blocks
# --------------------------------------------------------------------------- #


class ImageBlock(BaseModel):
    """An image result block: a URL or inline base64, with optional mime_type."""

    model_config = _PERMISSIVE
    type: Literal["image"] = Field("image", description="An image result block.")
    url: str | None = Field(None, description="Image URL.")
    data: str | None = Field(None, description="Inline base64-encoded image bytes.")
    mime_type: str | None = Field(None, description="MIME type of inline data.")


class AudioBlock(BaseModel):
    """An audio result block: inline base64 or a URL, with optional mime_type."""

    model_config = _PERMISSIVE
    type: Literal["audio"] = Field("audio", description="An audio result block.")
    url: str | None = Field(None, description="Audio URL (gateway extension).")
    data: str | None = Field(None, description="Inline base64-encoded audio bytes.")
    mime_type: str | None = Field(None, description="MIME type of the audio.")


class ModelOutputStep(BaseModel):
    """A model_output step carrying the result content blocks."""

    model_config = _PERMISSIVE
    type: Literal["model_output"] = Field("model_output", description="A model output step.")
    content: list[dict[str, Any]] = Field(..., description="Typed result blocks.")


class TaskError(BaseModel):
    model_config = _PERMISSIVE
    code: str = Field(..., description="Error code, e.g. failed.")
    message: str = Field(..., description="Error message.")


class ImageTaskResponse(BaseModel):
    """GET /v1/images/{id} -- the Gemini steps/content poll shape."""

    model_config = _PERMISSIVE
    id: str = Field(..., description="Task id.")
    model: str | None = Field(None, description="Model that produced the result.")
    status: TaskStatus = Field(..., description="Task lifecycle status.")
    steps: list[ModelOutputStep] | None = Field(None, description="model_output steps with content blocks.")
    output_image: str | None = Field(None, description="Last inline image block's base64.")
    output_image_url: str | None = Field(None, description="Last image block's URL.")
    error: TaskError | None = Field(None, description="Present only on failure.")
    usage: UsageCost | None = Field(None, description="Cost summary.")
    created_at: str | None = Field(None, description="Creation timestamp (ISO 8601).")
    completed_at: str | None = Field(None, description="Completion timestamp (ISO 8601).")


class VideoContentEnvelope(BaseModel):
    """Seedance poll `content` envelope with the generated video URL."""

    model_config = _PERMISSIVE
    video_url: str | None = Field(None, description="Generated video URL.")
    last_frame_url: str | None = Field(None, description="Last frame URL for chaining.")


class VideoTaskResponse(BaseModel):
    """GET /v1/videos/{id} -- the Seedance poll shape."""

    model_config = _PERMISSIVE
    id: str = Field(..., description="Task id.")
    model: str | None = Field(None, description="Model that produced the result.")
    status: TaskStatus = Field(..., description="Task lifecycle status.")
    content: VideoContentEnvelope | None = Field(None, description="Generated video URLs.")
    error: TaskError | None = Field(None, description="Present only on failure.")
    usage: UsageCost | None = Field(None, description="Cost summary.")


class MusicTaskResponse(BaseModel):
    """GET /v1/music/{id} -- the Lyria steps/content poll shape."""

    model_config = _PERMISSIVE
    id: str = Field(..., description="Task id.")
    model: str | None = Field(None, description="Model that produced the result.")
    status: TaskStatus = Field(..., description="Task lifecycle status.")
    steps: list[ModelOutputStep] | None = Field(None, description="model_output steps with content blocks.")
    output_audio: str | None = Field(None, description="Last inline audio block's base64.")
    output_audio_url: str | None = Field(None, description="Last audio block's URL.")
    output_text: str | None = Field(None, description="Generated lyrics/structure text.")
    error: TaskError | None = Field(None, description="Present only on failure.")
    usage: UsageCost | None = Field(None, description="Cost summary.")


# --------------------------------------------------------------------------- #
# Meta responses
# --------------------------------------------------------------------------- #


class ModelEntry(BaseModel):
    """An entry in the GET /v1/models list."""

    model_config = _PERMISSIVE
    id: str = Field(..., description="Model id or gateway alias.")
    type: str | None = Field(None, description="Backend type (alias entries).")
    underlying: str | None = Field(None, description="Real backend model id (alias entries).")
    provider: str | None = Field(None, description="Backend name (direct entries).")
    modality: str = Field(..., description='One of "image", "video", "music".')


class ModelListResponse(BaseModel):
    model_config = _PERMISSIVE
    object: Literal["list"] = Field("list", description="Always 'list'.")
    data: list[ModelEntry] = Field(..., description="Available models for the caller's key.")


class HealthResponse(BaseModel):
    model_config = _PERMISSIVE
    status: Literal["ok"] = Field("ok", description="Always 'ok' when healthy.")


__all__ = [
    "AudioBlock", "ContentPart", "CreateResponse", "DraftPart",
    "ErrorBody", "ErrorEnvelope", "HealthResponse", "ImageBlock",
    "ImageConfig", "ImagePart", "ImageRequest", "ImageTaskResponse",
    "ImageUrlPart", "ModelEntry", "ModelListResponse", "ModelOutputStep",
    "MusicConfig", "MusicRequest", "MusicResponseFormat", "MusicTaskResponse",
    "ProviderDirective", "TaskError", "TaskStatus", "TextPart",
    "UsageCost", "VideoContentEnvelope", "VideoRequest", "VideoTaskResponse",
    "VideoUrlPart", "AudioUrlPart",
]
