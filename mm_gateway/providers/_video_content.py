"""Shared handling of video request content for models with exclusive input modes."""

from __future__ import annotations

from mm_gateway.schemas.video import UnifiedVideoRequest, VideoAudioPart, VideoContentPart, VideoImagePart, VideoVideoPart


def _frame(item: object) -> bool:
    return isinstance(item, VideoImagePart) and item.role in ("first_frame", "last_frame")


def _reference(item: object) -> bool:
    return isinstance(item, (VideoVideoPart, VideoAudioPart)) or (
        isinstance(item, VideoImagePart) and item.role == "reference_image"
    )


def frames_over_references(request: UnifiedVideoRequest) -> tuple[list[VideoContentPart], int]:
    """``request.content`` for a model whose frames and references are exclusive modes, and how many parts were cut.

    A first or last frame keeps the image-to-video mode, so the reference images, videos and audio clips are
    dropped (``exclusive_frames_and_references`` in the limits catalogue); without a frame nothing changes.
    """
    if not any(_frame(p.root) for p in request.content):
        return list(request.content), 0
    kept = [p for p in request.content if not _reference(p.root)]
    return kept, len(request.content) - len(kept)


__all__ = ["frames_over_references"]
