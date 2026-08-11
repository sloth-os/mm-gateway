"""Tests for canonical pixel-dimension translation at provider boundaries."""

from __future__ import annotations

from types import SimpleNamespace

from mm_gateway.providers._dimensions import (
    aspect_ratio,
    image_resolution,
    pixel_size,
    video_resolution,
)
from mm_gateway.providers.openrouter import _image_body, _video_body
from mm_gateway.schemas.image import UnifiedImageRequest
from mm_gateway.schemas.image import text_part as image_text_part
from mm_gateway.schemas.video import UnifiedVideoRequest
from mm_gateway.schemas.video import text_part as video_text_part


def _dimensions(width: int | None, height: int | None) -> SimpleNamespace:
    return SimpleNamespace(
        width=width,
        height=height,
        size=None,
        resolution=None,
        aspect_ratio=None,
        ratio=None,
    )


def test_dimensions_derive_each_provider_wire_convention() -> None:
    request = _dimensions(1920, 1080)

    assert pixel_size(request) == "1920x1080"
    assert pixel_size(request, "*") == "1920*1080"
    assert aspect_ratio(request) == "16:9"
    assert image_resolution(request) == "2k"
    assert video_resolution(request) == "1080p"


def test_dimensions_require_both_pixel_axes() -> None:
    for request in (_dimensions(None, None), _dimensions(1024, None)):
        assert pixel_size(request) is None
        assert aspect_ratio(request) is None
        assert image_resolution(request) is None
        assert video_resolution(request) is None


def test_image_resolution_uses_one_kilopixel_tier_through_1024() -> None:
    assert image_resolution(_dimensions(1024, 768)) == "1k"
    assert image_resolution(_dimensions(1025, 768)) == "2k"


def test_openrouter_derives_all_image_and_video_dimension_fields() -> None:
    image = _image_body(UnifiedImageRequest(
        model="image-model",
        content=[image_text_part("x")],
        width=1536,
        height=1024,
    ))
    video = _video_body(UnifiedVideoRequest(
        model="video-model",
        content=[video_text_part("x")],
        width=1280,
        height=720,
    ))

    assert image["size"] == "1536x1024"
    assert image["aspect_ratio"] == "3:2"
    assert image["resolution"] == "2k"
    assert video["size"] == "1280x720"
    assert video["aspect_ratio"] == "16:9"
    assert video["resolution"] == "720p"
