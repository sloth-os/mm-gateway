"""Tests for the image translators (OpenAI <-> unified <-> OpenRouter)."""

from __future__ import annotations

import time

from mm_gateway.schemas.image import ImageData, UnifiedImageResponse
from mm_gateway.translators.image import openai_compat, openrouter_compat


def _resp(url: str = "https://x.test/a.png") -> UnifiedImageResponse:
    return UnifiedImageResponse(
        created=int(time.time()), model="gpt-image-1", provider="openai",
        data=[ImageData(url=url, revised_prompt="a cat")],
    )


# -- OpenAI compat ----------------------------------------------------------- #

def test_openai_request_basic_fields():
    unified = openai_compat.from_openai({"model": "gpt-image-1", "prompt": "a cat", "n": 2, "size": "1024x1024"})
    assert unified.model == "gpt-image-1"
    assert unified.prompt == "a cat"
    assert unified.n == 2
    assert unified.size == "1024x1024"


def test_openai_request_unknown_fields_go_to_extra():
    unified = openai_compat.from_openai({"model": "gpt-image-1", "prompt": "x", "magic_knob": 7})
    assert unified.extra == {"magic_knob": 7}


def test_openai_request_image_string_becomes_input_images():
    unified = openai_compat.from_openai({"model": "gpt-image-1", "prompt": "edit", "image": "https://x.test/in.png"})
    assert unified.input_images and unified.input_images[0].url == "https://x.test/in.png"


def test_openai_response_shape():
    out = openai_compat.to_openai(_resp())
    assert out["data"][0]["url"] == "https://x.test/a.png"
    assert out["data"][0]["revised_prompt"] == "a cat"
    assert "created" in out


# -- OpenRouter compat ------------------------------------------------------- #

def test_openrouter_request_known_fields():
    unified = openrouter_compat.from_openrouter({"model": "flux-2", "prompt": "a cat", "aspect_ratio": "16:9"})
    assert unified.aspect_ratio == "16:9"


def test_openrouter_request_provider_field_is_dropped():
    unified = openrouter_compat.from_openrouter({"model": "flux-2", "prompt": "x", "provider": {"only": "flux"}})
    assert unified.provider is None  # routing handled by the route layer


def test_openrouter_request_input_references_data_uri():
    unified = openrouter_compat.from_openrouter({
        "model": "flux-2", "prompt": "x",
        "input_references": [{"image_url": {"url": "data:image/png;base64,AAAA"}}],
    })
    assert unified.input_images and unified.input_images[0].b64_json == "AAAA"


def test_openrouter_response_prefers_b64_then_url():
    out = openrouter_compat.to_openrouter(_resp(url="https://x.test/a.png"))
    assert out["data"][0]["b64_json"] == "https://x.test/a.png"
    assert out["data"][0]["media_type"] == "image/png"


# -- Round trip -------------------------------------------------------------- #

def test_openai_to_openrouter_via_unified():
    """A request that arrives as OpenAI and is emitted as OpenRouter keeps the image."""
    unified = openai_compat.from_openai({"model": "gpt-image-1", "prompt": "a cat"})
    resp = UnifiedImageResponse(
        created=1, model=unified.model, provider="openai",
        data=[ImageData(b64_json="AAAA")],
    )
    out = openrouter_compat.to_openrouter(resp)
    assert out["data"][0]["b64_json"] == "AAAA"
