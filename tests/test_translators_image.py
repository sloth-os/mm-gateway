"""Tests for the Gemini-compatible image translator."""

from __future__ import annotations

from mm_gateway.schemas.image import ImageData, UnifiedImageTask
from mm_gateway.translators.image import gemini_compat


def _task(status: str = "succeeded", *, url: str | None = None,
          b64: str | None = None) -> UnifiedImageTask:
    images = []
    if url or b64:
        images.append(ImageData(url=url, b64_json=b64, revised_prompt="a cat"))
    return UnifiedImageTask(
        task_id="img-1", provider="openai", model="gpt-image-1", status=status,
        images=images, created_at=1, completed_at=2,
    )


# -- from_gemini ----------------------------------------------------------- #


def test_request_string_prompt_becomes_content_text():
    unified = gemini_compat.from_gemini(
        {"model": "gpt-image-1", "input": "a cat", "n": 2, "size": "1024x1024"}
    )
    assert unified.model == "gpt-image-1"
    assert unified.prompt() == "a cat"
    assert unified.n == 2
    assert unified.size == "1024x1024"


def test_request_parts_array_text_and_image_url():
    unified = gemini_compat.from_gemini({
        "model": "gpt-image-1",
        "input": [
            {"type": "text", "text": "edit"},
            {"type": "image", "url": "https://x.test/in.png"},
        ],
    })
    assert unified.prompt() == "edit"
    imgs = unified.input_images()
    assert imgs and imgs[0].url == "https://x.test/in.png"


def test_request_inline_image_data_part():
    unified = gemini_compat.from_gemini({
        "model": "gpt-image-1",
        "input": [
            {"type": "image", "mime_type": "image/png", "data": "AAAA"},
        ],
    })
    imgs = unified.input_images()
    assert imgs and imgs[0].data == "AAAA" and imgs[0].mime_type == "image/png"


def test_request_unknown_fields_go_to_extra():
    unified = gemini_compat.from_gemini(
        {"model": "gpt-image-1", "input": "x", "magic_knob": 7}
    )
    assert unified.extra == {"magic_knob": 7}


def test_request_response_format_string_mapped():
    unified = gemini_compat.from_gemini(
        {"model": "gpt-image-1", "input": "x", "response_format": "b64_json"}
    )
    assert unified.response_format == "b64_json"


def test_request_config_object_merges_known_knobs():
    # The Gemini Interactions `config` object is the abstraction over all
    # backend image functions: known knobs merge onto the unified flat fields.
    unified = gemini_compat.from_gemini({
        "model": "gpt-image-1", "input": "a cat",
        "config": {
            "negative_prompt": "blurry", "n": 3, "size": "1024x1024",
            "seed": 7, "guidance_scale": 7.5, "aspect_ratio": "16:9",
            "output_format": "png", "background": "transparent", "user": "u1",
        },
    })
    assert unified.negative_prompt == "blurry"
    assert unified.n == 3
    assert unified.size == "1024x1024"
    assert unified.seed == 7
    assert unified.guidance_scale == 7.5
    assert unified.aspect_ratio == "16:9"
    assert unified.output_format == "png"
    assert unified.background == "transparent"
    assert unified.user == "u1"


def test_request_config_unknown_keys_go_to_extra():
    # Provider-specific knobs the unified model doesn't name pass through to
    # providers via extra (best-effort policy).
    unified = gemini_compat.from_gemini({
        "model": "gpt-image-1", "input": "a cat",
        "config": {"thinking_mode": True, "motion_bucket_id": 127},
    })
    assert unified.extra["thinking_mode"] is True
    assert unified.extra["motion_bucket_id"] == 127


def test_request_config_overrides_flat_knobs():
    # config is the canonical (new) shape; where a knob is set in both places,
    # config wins.
    unified = gemini_compat.from_gemini({
        "model": "gpt-image-1", "input": "a cat", "n": 1,
        "config": {"n": 4},
    })
    assert unified.n == 4


def test_request_config_response_format_envelope():
    # Imagen/Lyria-style response_format envelope inside config.
    unified = gemini_compat.from_gemini({
        "model": "imagen-4.0-generate-001", "input": "a cat",
        "config": {"response_format": {"type": "b64_json", "quality": "high"}},
    })
    assert unified.response_format == "b64_json"
    assert unified.quality == "high"


def test_request_config_and_flat_both_accepted():
    # A body mixing flat knobs (legacy) and config (new) merges both.
    unified = gemini_compat.from_gemini({
        "model": "gpt-image-1", "input": "a cat", "style": "vivid",
        "config": {"seed": 11},
    })
    assert unified.style == "vivid"
    assert unified.seed == 11


def test_request_provider_directive_is_dropped():
    # ``provider`` is a routing directive dict ({tag}/{backend}) read from the
    # raw body by ``routing_overrides``; it must NOT be assigned to the unified
    # ``provider: str`` field (would fail validation) nor leak into ``extra``.
    unified = gemini_compat.from_gemini(
        {"model": "gpt-image-1", "input": "a cat", "provider": {"backend": "img-b"}}
    )
    assert unified.provider is None
    assert "provider" not in unified.extra


def test_request_config_provider_directive_is_dropped():
    # The routing directive is honoured at the top level only; a provider inside
    # ``config`` has no routing effect and must not break the translation.
    unified = gemini_compat.from_gemini(
        {"model": "gpt-image-1", "input": "a cat", "config": {"provider": {"tag": "t"}}}
    )
    assert unified.provider is None
    assert "provider" not in unified.extra


def test_request_missing_model_raises():
    import pytest

    from mm_gateway.core.exceptions import ValidationError
    with pytest.raises(ValidationError):
        gemini_compat.from_gemini({"input": "a cat"})


# -- to_gemini_create / to_gemini_task ------------------------------------ #


def test_create_returns_only_id():
    out = gemini_compat.to_gemini_create(_task(status="pending"))
    assert out == {"id": "img-1"}


def test_task_url_image_block():
    out = gemini_compat.to_gemini_task(_task(url="https://x.test/a.png"))
    assert out["id"] == "img-1" and out["status"] == "succeeded"
    block = out["steps"][0]["content"][0]
    assert block == {"type": "image", "url": "https://x.test/a.png"}
    assert out["output_image_url"] == "https://x.test/a.png"
    assert "output_image" not in out  # falls back to url when no b64


def test_task_b64_image_block_and_text():
    out = gemini_compat.to_gemini_task(_task(b64="AAAA"))
    blocks = out["steps"][0]["content"]
    assert {"type": "image", "data": "AAAA"} in blocks
    assert {"type": "text", "text": "a cat"} in blocks
    assert out["output_image"] == "AAAA"
    assert "output_image_url" not in out


def test_task_error_envelope():
    task = UnifiedImageTask(
        task_id="img-1", provider="openai", model="m", status="failed",
        error="boom",
    )
    out = gemini_compat.to_gemini_task(task)
    assert out["error"] == {"code": "failed", "message": "boom"}
    assert "steps" not in out  # no content blocks on failure
