"""Unified video schemas - the canonical internal representation.

The model carries an ordered array of typed text/image/video/audio parts plus a
flat set of generation controls. The public REST layer maps neutral names into
this representation; each adapter maps the applicable subset to its native API.

Video generation uses a task lifecycle: ``pending -> running -> succeeded |
failed``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, RootModel

# --------------------------------------------------------------------------- #
# Content parts
# --------------------------------------------------------------------------- #

ImageRole = Literal["first_frame", "last_frame", "reference_image"]
VideoRole = Literal["reference_video"]
AudioRole = Literal["reference_audio"]


class _Url(BaseModel):
    url: str


class VideoTextPart(BaseModel):
    """The prompt. Required for text-to-video."""

    type: Literal["text"]
    text: str


class VideoImagePart(BaseModel):
    """An input image. ``role`` picks first/last frame or reference image."""

    type: Literal["image_url"]
    image_url: _Url
    role: ImageRole = "first_frame"


class VideoVideoPart(BaseModel):
    """A reference video (Seedance 2.0 multi-modal reference)."""

    type: Literal["video_url"]
    video_url: _Url
    role: VideoRole = "reference_video"


class VideoAudioPart(BaseModel):
    """A reference audio (Seedance 2.0 multi-modal reference)."""

    type: Literal["audio_url"]
    audio_url: _Url
    role: AudioRole = "reference_audio"


class VideoDraftPart(BaseModel):
    """Resume a draft task by id (Seedance draft workflow)."""

    type: Literal["draft_task"]
    draft_task: dict[str, Any]


class VideoContentPart(RootModel[VideoTextPart | VideoImagePart | VideoVideoPart
                                 | VideoAudioPart | VideoDraftPart]):
    """Discriminated union over the ``type`` field of a content part."""


def text_part(text: str) -> VideoTextPart:
    return VideoTextPart(type="text", text=text)


def image_part(url: str, role: ImageRole = "first_frame") -> VideoImagePart:
    return VideoImagePart(type="image_url", image_url=_Url(url=url), role=role)


def video_part(url: str, role: VideoRole = "reference_video") -> VideoVideoPart:
    return VideoVideoPart(type="video_url", video_url=_Url(url=url), role=role)


def audio_part(url: str, role: AudioRole = "reference_audio") -> VideoAudioPart:
    return VideoAudioPart(type="audio_url", audio_url=_Url(url=url), role=role)


def draft_part(draft: dict[str, Any]) -> VideoDraftPart:
    return VideoDraftPart(type="draft_task", draft_task=draft)


# --------------------------------------------------------------------------- #
# Request / response
# --------------------------------------------------------------------------- #


class UnifiedVideoRequest(BaseModel):
    """The canonical internal video request.

    ``content`` carries the typed parts (prompt text, first/last-frame and
    reference images, reference videos/audios, draft resumes). The flat fields
    are the union of neutral generation controls; providers read whichever
    subset they support.
    """

    model: str
    content: list[VideoContentPart] = Field(default_factory=list)
    negative_prompt: str | None = None
    duration: float | None = None
    ratio: str | None = Field(None, description="Aspect ratio, e.g. '16:9'.")
    resolution: str | None = None
    size: str | None = None
    width: int | None = None
    height: int | None = None
    fps: int | None = None
    seed: int | None = None
    generate_audio: bool | None = None
    camera_fixed: bool | None = None
    watermark: bool | None = None
    prompt_extend: bool | None = None
    callback_url: str | None = None
    return_last_frame: bool | None = None
    guidance_scale: float | None = None
    motion_intensity: int | None = None
    frame_count: int | None = None
    output_format: str | None = None
    provider: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    # -- convenience accessors for providers that don't speak content[] natively #

    def prompt(self) -> str | None:
        """Concatenated text parts, or None if there are none."""
        texts = [p.root.text for p in self.content
                 if isinstance(p.root, VideoTextPart)]
        return "\n".join(texts) if texts else None

    def first_image(self) -> str | None:
        """URL of the first image_url part whose role is first_frame (or the
        first image_url part if none is tagged)."""
        fallback = None
        for p in self.content:
            if isinstance(p.root, VideoImagePart):
                if p.root.role == "first_frame":
                    return p.root.image_url.url
                if fallback is None:
                    fallback = p.root.image_url.url
        return fallback

    def last_image(self) -> str | None:
        for p in self.content:
            if isinstance(p.root, VideoImagePart) and p.root.role == "last_frame":
                return p.root.image_url.url
        return None

    def reference_images(self) -> list[str]:
        return [p.root.image_url.url for p in self.content
                if isinstance(p.root, VideoImagePart)
                and p.root.role == "reference_image"]

    def reference_videos(self) -> list[str]:
        return [p.root.video_url.url for p in self.content
                if isinstance(p.root, VideoVideoPart)]

    def reference_audios(self) -> list[str]:
        return [p.root.audio_url.url for p in self.content
                if isinstance(p.root, VideoAudioPart)]

    def draft(self) -> dict[str, Any] | None:
        for p in self.content:
            if isinstance(p.root, VideoDraftPart):
                return p.root.draft_task
        return None


TaskStatus = Literal["pending", "running", "succeeded", "failed", "cancelled", "expired"]


class VideoUsage(BaseModel):
    cost: float | None = None
    video_count: int | None = None
    video_duration: int | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class UnifiedVideoTask(BaseModel):
    task_id: str
    provider: str
    model: str
    status: TaskStatus
    # One or more output video URLs once succeeded.
    video_urls: list[str] = Field(default_factory=list)
    # Optional cover/thumbnail URL.
    cover_url: str | None = None
    error: str | None = None
    usage: VideoUsage | None = None
    # Provider-native raw response, for clients that want the full envelope.
    raw: dict[str, Any] | None = None
    created_at: int | None = None
    completed_at: int | None = None
