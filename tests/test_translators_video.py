"""Tests for the Seedance-compatible video translator."""

from __future__ import annotations

from mm_gateway.schemas.video import UnifiedVideoTask, VideoUsage
from mm_gateway.translators.video import seedance_compat


def _task(status: str = "succeeded") -> UnifiedVideoTask:
    t = UnifiedVideoTask(
        task_id="t-1", provider="volcengine", model="doubao-seedance-1-0-pro-250528",
        status=status, video_urls=["https://x.test/v.mp4"] if status == "succeeded" else [],
    )
    if status == "succeeded":
        t.usage = VideoUsage(cost=0.02)
    return t


# -- from_seedance --------------------------------------------------------- #


def test_seedance_request_text_and_first_frame():
    unified = seedance_compat.from_seedance({
        "model": "doubao-seedance-1-0-pro-250528",
        "content": [
            {"type": "text", "text": "a cat playing"},
            {"type": "image_url", "image_url": {"url": "https://x.test/first.png"}, "role": "first_frame"},
        ],
        "duration": 5,
        "ratio": "16:9",
    })
    assert unified.prompt() == "a cat playing"
    assert unified.first_image() == "https://x.test/first.png"
    assert unified.duration == 5
    assert unified.ratio == "16:9"


def test_seedance_request_last_frame_role():
    unified = seedance_compat.from_seedance({
        "model": "m", "content": [
            {"type": "image_url", "image_url": {"url": "https://x.test/last.png"}, "role": "last_frame"},
        ],
    })
    assert unified.last_image() == "https://x.test/last.png"


def test_seedance_aspect_ratio_maps_to_ratio():
    unified = seedance_compat.from_seedance({"model": "m", "aspect_ratio": "9:16"})
    assert unified.ratio == "9:16"


# -- to_seedance_create / to_seedance_task --------------------------------- #


def test_seedance_create_response_is_just_id():
    out = seedance_compat.to_seedance_create(_task("pending"))
    assert out == {"id": "t-1"}


def test_seedance_task_response_has_content():
    out = seedance_compat.to_seedance_task(_task("succeeded"))
    assert out["id"] == "t-1"
    assert out["status"] == "succeeded"
    assert out["content"]["video_url"] == "https://x.test/v.mp4"
