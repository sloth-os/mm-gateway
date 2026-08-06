"""Unified image schemas — the canonical internal representation.

Every provider maps its native request/response into these models, and every
front-end format (OpenAI-compatible, OpenRouter-compatible) maps into/out of
them. Keeping one superset model means translators are O(formats + providers)
rather than O(formats × providers).

Design notes
------------
- ``size`` is kept as a free string ("WxH" or "W*H") because providers disagree
  on the separator; ``width``/``height`` are the parsed ints used by providers
  that want numbers. Translators normalise.
- ``response_format`` is ``url | b64_json``; providers that only return one are
  post-processed by the provider adapter (e.g. download a URL to b64).
- ``extra`` is an opaque passthrough dict for provider-specific knobs that the
  unified model doesn't name — the "best effort" escape hatch.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class ImageInput(BaseModel):
    """A reference/input image for edit / image-to-image, by URL or base64."""

    url: str | None = None
    b64_json: str | None = None

    @model_validator(mode="after")
    def _one_of(self) -> "ImageInput":
        if not self.url and not self.b64_json:
            raise ValueError("ImageInput requires either url or b64_json")
        return self


class UnifiedImageRequest(BaseModel):
    model: str = Field(..., description="Provider model id, or a gateway alias.")
    prompt: str
    negative_prompt: str | None = None
    n: int = Field(1, ge=1, le=16)
    size: str | None = Field(None, description='"WxH" or "W*H"')
    width: int | None = None
    height: int | None = None
    aspect_ratio: str | None = None
    resolution: str | None = None
    quality: str | None = None
    style: str | None = None
    seed: int | None = None
    guidance_scale: float | None = None
    num_inference_steps: int | None = None
    strength: float | None = None
    watermark: bool | None = None
    response_format: Literal["url", "b64_json"] | None = None
    output_format: str | None = None
    output_compression: int | None = None
    background: str | None = None
    input_images: list[ImageInput] | None = None
    mask: ImageInput | None = None
    user: str | None = None
    # Routing override; if None the registry routes by model alias or default.
    provider: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class ImageData(BaseModel):
    url: str | None = None
    b64_json: str | None = None
    revised_prompt: str | None = None
    media_type: str | None = None


class ImageUsage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cost: float | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class UnifiedImageResponse(BaseModel):
    created: int
    data: list[ImageData]
    model: str
    provider: str
    usage: ImageUsage | None = None
