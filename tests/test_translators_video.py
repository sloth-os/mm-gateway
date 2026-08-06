"""Tests for the video translators (Seedance <-> unified <-> OpenRouter)."""

from __future__ import annotations

from mm_gateway.schemas.video import UnifiedVideoTask, VideoUsage
from mm_gateway.translators.video import openrouter_compat, seedance_compat


def _task(status: str = "succeeded") -> UnifiedVideoTask:
    t = UnifiedVideoTask(
        task_id="t-1", provider="volcengine", model="doubao-seedance-1-0-pro-250528",
        status=status, video_urls=["https://x.test/v.mp4"] if status == "succeeded" else [],
    )
    if status == "succeeded":
        t.usage = VideoUsage(cost=0.02)
    return t


# -- Seedance compat --------------------------------------------------------- #

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
    assert unified.prompt == "a cat playing"
    assert unified.image == "https://x.test/first.png"
    assert unified.duration == 5
    assert unified.aspect_ratio == "16:9"


def test_seedance_request_last_frame_role():
    unified = seedance_compat.from_seedance({
        "model": "m", "content": [
            {"type": "image_url", "image_url": {"url": "https://x.test/last.png"}, "role": "last_frame"},
        ],
    })
    assert unified.last_frame_image == "https://x.test/last.png"


def test_seedance_create_response_is_just_id():
    out = seedance_compat.to_seedance_create(_task("pending"))
    assert out == {"id": "t-1"}


def test_seedance_task_response_has_content():
    out = seedance_compat.to_seedance_task(_task("succeeded"))
    assert out["id"] == "t-1"
    assert out["status"] == "succeeded"
    assert out["content"]["video_url"] == "https://x.test/v.mp4"


# -- OpenRouter compat ------------------------------------------------------- #

def test_openrouter_request_known_fields():
    unified = openrouter_compat.from_openrouter({"model": "m", "prompt": "x", "duration": 4, "aspect_ratio": "1:1"})
    assert unified.duration == 4
    assert unified.aspect_ratio == "1:1"


def test_openrouter_request_frame_images():
    unified = openrouter_compat.from_openrouter({
        "model": "m", "prompt": "x",
        "frame_images": [
            {"image_url": {"url": "https://x.test/a.png"}, "frame_type": "first_frame"},
            {"image_url": {"url": "https://x.test/b.png"}, "frame_type": "last_frame"},
        ],
    })
    assert unified.image == "https://x.test/a.png"
    assert unified.last_frame_image == "https://x.test/b.png"


def test_openrouter_response_polling_url_and_unsigned():
    out = openrouter_compat.to_openrouter(_task("succeeded"), base_url="https://gw.test")
    assert out["id"] == "t-1"
    assert out["status"] == "succeeded"
    assert out["polling_url"] == "https://gw.test/api/v1/videos/t-1"
    assert out["unsigned_urls"] == ["https://x.test/v.mp4"]
    assert out["usage"]["cost"] == 0.02


def test_openrouter_response_pending_has_no_unsigned_urls():
    out = openrouter_compat.to_openrouter(_task("pending"))
    assert "unsigned_urls" not in out
