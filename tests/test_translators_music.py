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


def test_lyria_request_config_object_merges_knobs():
    # The Gemini Interactions `config` object is the abstraction over all
    # backend music functions: known knobs merge onto the unified flat fields.
    unified = lyria_compat.from_lyria({
        "model": "lyria-3", "input": "a happy song",
        "config": {
            "negative_prompt": "drums", "duration": 30, "bpm": 120,
            "key_scale": "C major", "time_signature": "4/4",
            "vocal_language": "en", "audio_format": "wav",
            "audio_quality": "44100_128", "is_instrumental": False,
            "generate_audio": True, "seed": 7, "guidance_scale": 3.0,
            "n": 2,
        },
    })
    assert unified.negative_prompt == "drums"
    assert unified.duration == 30
    assert unified.bpm == 120
    assert unified.key_scale == "C major"
    assert unified.time_signature == "4/4"
    assert unified.vocal_language == "en"
    assert unified.audio_format == "wav"
    assert unified.audio_quality == "44100_128"
    assert unified.is_instrumental is False
    assert unified.generate_audio is True
    assert unified.seed == 7
    assert unified.guidance_scale == 3.0
    assert unified.n == 2


def test_lyria_request_config_unknown_keys_go_to_extra():
    # Provider-specific knobs (ElevenLabs finetune_id, ACE-Step inference_steps,
    # Lyria lyria_config, ...) pass through to providers via extra.
    unified = lyria_compat.from_lyria({
        "model": "lyria-3", "input": "x",
        "config": {
            "finetune_id": "ft-123", "inference_steps": 50,
            "lyria_config": {"denoising": 0.8},
        },
    })
    assert unified.extra["finetune_id"] == "ft-123"
    assert unified.extra["inference_steps"] == 50
    assert unified.extra["lyria_config"] == {"denoising": 0.8}


def test_lyria_request_config_overrides_flat_knobs():
    # config is the canonical (new) shape; where a knob is set in both places,
    # config wins.
    unified = lyria_compat.from_lyria({
        "model": "lyria-3", "input": "x", "duration": 10,
        "config": {"duration": 45},
    })
    assert unified.duration == 45


def test_lyria_request_config_response_format_envelope():
    # Lyria's response_format envelope inside config maps to audio_format/quality.
    unified = lyria_compat.from_lyria({
        "model": "lyria-3", "input": "x",
        "config": {"response_format": {"type": "audio", "quality": "44100_128"}},
    })
    assert unified.audio_format == "wav"
    assert unified.audio_quality == "44100_128"


def test_lyria_request_config_and_flat_both_accepted():
    # A body mixing flat knobs (legacy) and config (new) merges both.
    unified = lyria_compat.from_lyria({
        "model": "lyria-3", "input": "x", "bpm": 90,
        "config": {"seed": 11},
    })
    assert unified.bpm == 90
    assert unified.seed == 11


def test_lyria_request_provider_directive_is_dropped():
    # ``provider`` is a routing directive dict ({tag}/{backend}) read from the
    # raw body by ``routing_overrides``; it must NOT be assigned to the unified
    # ``provider: str`` field (would fail validation) nor leak into ``extra``.
    unified = lyria_compat.from_lyria(
        {"model": "lyria-3", "input": "x", "provider": {"backend": "mm-a"}}
    )
    assert unified.provider is None
    assert "provider" not in unified.extra


def test_lyria_request_config_provider_directive_is_dropped():
    unified = lyria_compat.from_lyria(
        {"model": "lyria-3", "input": "x", "config": {"provider": {"tag": "t"}}}
    )
    assert unified.provider is None
    assert "provider" not in unified.extra


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
