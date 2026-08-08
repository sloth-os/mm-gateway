"""Unified image schemas — the canonical internal representation.

Mirrors the music/video schemas: a ``content`` array of typed parts (text /
image) plus a flat set of generation knobs (size, n, seed, ...). Every
front-end shape (Gemini-compatible) is translated into this one model, and
every provider pulls the bits it cares about out of ``content`` and the knobs.

The front-end format is **Gemini-compatible**: a request is ``{model, input, ...}``
where ``input`` is a string or a parts array (``{type:"text",text}`` and
``{type:"image",mime_type,data}``/``{type:"image",url}``), and the response is a
task whose ``steps[].content[]`` blocks carry the generated image (inline base64
or URL). See ``translators/image/gemini_compat.py``.

Image generation is synchronous on most providers (OpenAI DALL·E, Imagen,
Stable Image, xAI, Volcengine Seedream) and asynchronous on a few (DashScope
Wanx, FLUX runapi). The synchronous adapters wrap their result as a synthetic
in-memory task so the gateway's create/poll surface is uniform across all
three modalities (image/video/music).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, RootModel, model_validator


# --------------------------------------------------------------------------- #
# Content parts
# --------------------------------------------------------------------------- #


class ImageTextPart(BaseModel):
    """The prompt. Required for text-to-image."""

    type: Literal["text"]
    text: str


class ImageImagePart(BaseModel):
    """A reference/input image for edit or image-to-image, by URL or base64."""

    type: Literal["image"]
    url: str | None = None
    data: str | None = None
    mime_type: str | None = None

    @model_validator(mode="after")
    def _one_of(self) -> "ImageImagePart":
        if not self.url and not self.data:
            raise ValueError("ImageImagePart requires either url or data")
        return self


class ImageContentPart(RootModel[ImageTextPart | ImageImagePart]):
    """Discriminated union over the ``type`` field of a content part."""


def text_part(text: str) -> ImageTextPart:
    return ImageTextPart(type="text", text=text)


def image_part(url: str | None = None, *, data: str | None = None,
               mime_type: str | None = None) -> ImageImagePart:
    return ImageImagePart(type="image", url=url, data=data, mime_type=mime_type)


# --------------------------------------------------------------------------- #
# Request / response
# --------------------------------------------------------------------------- #


class UnifiedImageRequest(BaseModel):
    """The canonical internal image request.

    ``content`` carries the typed parts (prompt text, input/reference images).
    The flat fields are the generation knobs the union of providers exposes; each
    provider reads the subset it supports and ignores the rest.
    """

    model: str
    content: list[ImageContentPart] = Field(default_factory=list)
    negative_prompt: str | None = None
    n: int | None = Field(None, ge=1, le=16)
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
    # Whether to stream the result back (provider passthrough; most ignore it).
    stream: bool | None = None
    callback_url: str | None = None
    user: str | None = None
    # Routing override; if None the registry routes by model alias or default.
    provider: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    # -- convenience accessors for providers that don't speak content[] natively #

    def prompt(self) -> str | None:
        """Concatenated text parts, or None if there are none."""
        texts = [p.root.text for p in self.content
                 if isinstance(p.root, ImageTextPart)]
        return "\n".join(texts) if texts else None

    def input_images(self) -> list[ImageImagePart]:
        """All image parts in order (URLs or inline base64)."""
        return [p.root for p in self.content
                if isinstance(p.root, ImageImagePart)]

    def first_image_url(self) -> str | None:
        for p in self.content:
            if isinstance(p.root, ImageImagePart) and p.root.url:
                return p.root.url
        return None


TaskStatus = Literal["pending", "running", "succeeded", "failed", "cancelled", "expired"]


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
    """Provider-native response, wrapped by a task. Providers build this then
    the synthetic-task adapters lift it into a ``UnifiedImageTask`` on poll."""

    created: int
    data: list[ImageData]
    model: str
    provider: str
    usage: ImageUsage | None = None


class UnifiedImageTask(BaseModel):
    task_id: str
    provider: str
    model: str
    status: TaskStatus
    # Generated image data once succeeded (one entry per generated image).
    images: list[ImageData] = Field(default_factory=list)
    error: str | None = None
    usage: ImageUsage | None = None
    # Provider-native raw response, for clients that want the full envelope.
    raw: dict[str, Any] | None = None
    created_at: int | None = None
    completed_at: int | None = None
