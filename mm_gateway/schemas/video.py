"""Unified video schemas — the canonical internal representation.

Video generation is async on every provider, so the unified model centres on a
*task* with a lifecycle: ``pending -> running -> succeeded | failed``. The
gateway may optionally block until completion (sync-style) for clients that
prefer a one-shot call, then fall back to returning a task handle.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class VideoFrameImage(BaseModel):
    url: str | None = None
    b64_json: str | None = None
    frame_type: Literal["first_frame", "last_frame"] | None = None


class UnifiedVideoRequest(BaseModel):
    model: str
    prompt: str | None = None
    negative_prompt: str | None = None
    duration: float | None = None
    aspect_ratio: str | None = None
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
    # For image-to-video / first-frame:
    image: str | None = Field(None, description="URL or data: URI of an input image")
    last_frame_image: str | None = None
    reference_images: list[str] | None = None
    callback_url: str | None = None
    provider: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


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
