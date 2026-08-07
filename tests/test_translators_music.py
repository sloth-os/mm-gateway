"""Tests for the music translator (Gemini Lyria 3 <-> unified)."""

from __future__ import annotations

from mm_gateway.schemas.music import MusicUsage, UnifiedMusicTask
from mm_gateway.translators.music import lyria_compat


def _task(status: str = "succeeded") -> UnifiedMusicTask:
    t = UnifiedMusicTask(
        task_id="m-1", provider="google", model="lyria-3", status=status,
    )
    if status == "succeeded":
        t.audio_b64 = "AAAA"
        t.audio_media_type = "audio/wav"
        t.lyrics = "la la la"
        t.usage = MusicUsage(cost=0.01)
    return t


# -- request: from_lyria ----------------------------------------------------- #


def test_lyria_request_string_input():
    unified = lyria_compat.from_lyria({"model": "lyria-3", "input": "a happy song"})
    assert unified.prompt() == "a happy song"
    assert unified.model == "lyria-3"


def test_lyria_request_parts_input_text():
    unified = lyria_compat.from_lyria({
        "model": "lyria-3",
        "input": [{"type": "text", "text": "verse one"}, {"type": "text", "text": "verse two"}],
    })
    # Multiple text parts concatenate with newlines.
    assert unified.prompt() == "verse one\nverse two"


def test_lyria_request_image_part_stashed_in_extra():
    # Lyria inline image parts ({mime_type, data}) are not a 1:1 fit for the
    # unified content[] model; the translator stashes them in extra["images"] for
    # providers that consume inline reference images.
    unified = lyria_compat.from_lyria({
        "model": "lyria-3",
        "input": [
            {"type": "text", "text": "make it upbeat"},
            {"type": "image", "mime_type": "image/png", "data": "iVBOR..."},
        ],
    })
    assert unified.prompt() == "make it upbeat"
    assert unified.extra["images"] == [{"mime_type": "image/png", "data": "iVBOR..."}]


def test_lyria_request_flat_knobs():
    unified = lyria_compat.from_lyria({
        "model": "lyria-3", "input": "x",
        "duration": 30, "bpm": 120, "seed": 7, "is_instrumental": True,
    })
    assert unified.duration == 30
    assert unified.bpm == 120
    assert unified.seed == 7
    assert unified.is_instrumental is True


def test_lyria_response_format_audio_maps_to_wav():
    unified = lyria_compat.from_lyria({
        "model": "lyria-3", "input": "x", "response_format": {"type": "audio"},
    })
    assert unified.audio_format == "wav"


def test_lyria_unknown_fields_go_to_extra():
    unified = lyria_compat.from_lyria({"model": "lyria-3", "input": "x", "style": "edm"})
    assert unified.extra["style"] == "edm"


def test_lyria_request_missing_model_raises():
    import pytest

    from mm_gateway.core.exceptions import ValidationError
    with pytest.raises(ValidationError):
        lyria_compat.from_lyria({"input": "x"})


# -- response: to_lyria_* ---------------------------------------------------- #


def test_lyria_create_response_is_just_id():
    out = lyria_compat.to_lyria_create(_task("pending"))
    assert out == {"id": "m-1"}


def test_lyria_task_response_has_steps_content_blocks():
    out = lyria_compat.to_lyria_task(_task("succeeded"))
    assert out["id"] == "m-1"
    assert out["status"] == "succeeded"
    # The audio + lyrics ride a model_output step's content array as typed blocks.
    step = out["steps"][0]
    assert step["type"] == "model_output"
    blocks = step["content"]
    audio_block = next(b for b in blocks if b["type"] == "audio")
    assert audio_block["data"] == "AAAA"
    assert audio_block["mime_type"] == "audio/wav"
    text_block = next(b for b in blocks if b["type"] == "text")
    assert text_block["text"] == "la la la"
    # Convenience accessors mirror the SDK's interaction.output_audio / output_text.
    assert out["output_audio"] == "AAAA"
    assert out["output_text"] == "la la la"


def test_lyria_task_response_url_when_no_inline_audio():
    # Async backends (udioapi, mureka) return a URL rather than inline base64;
    # the translator surfaces it as output_audio_url + a url-bearing audio block.
    t = UnifiedMusicTask(
        task_id="m-2", provider="udioapi", model="chirp-v5", status="succeeded",
        audio_urls=["https://x.test/song.mp3"], audio_media_type="audio/mpeg",
    )
    out = lyria_compat.to_lyria_task(t)
    assert out["output_audio_url"] == "https://x.test/song.mp3"
    block = out["steps"][0]["content"][0]
    assert block["type"] == "audio"
    assert block["url"] == "https://x.test/song.mp3"
    assert "output_audio" not in out


def test_lyria_task_response_error_block():
    t = UnifiedMusicTask(
        task_id="m-3", provider="mureka", model="mureka-song-1", status="failed",
        error="moderation blocked",
    )
    out = lyria_compat.to_lyria_task(t)
    assert out["status"] == "failed"
    assert out["error"] == {"code": "failed", "message": "moderation blocked"}


def test_lyria_task_response_usage_cost():
    out = lyria_compat.to_lyria_task(_task("succeeded"))
    assert out["usage"] == {"cost": 0.01}
