"""OpenAI-compatible image translator.

Maps the classic ``POST /v1/images/generations`` request/response shape to and
from the unified image schema. Best-effort: unknown fields are preserved in
``extra`` so they survive a round-trip to providers that accept them.
"""

from __future__ import annotations

from typing import Any

from mm_gateway.core.exceptions import ValidationError
from mm_gateway.schemas.image import ImageData, ImageInput, UnifiedImageRequest, UnifiedImageResponse

# Fields with a named home on the unified model; everything else -> extra.
_KNOWN = {
    "model", "prompt", "negative_prompt", "n", "size", "width", "height",
    "aspect_ratio", "resolution", "quality", "style", "seed", "guidance_scale",
    "num_inference_steps", "strength", "response_format", "output_format",
    "output_compression", "background", "user",
}


def from_openai(body: dict[str, Any]) -> UnifiedImageRequest:
    if "model" not in body or "prompt" not in body:
        raise ValidationError("`model` and `prompt` are required for image generation.")
    kwargs: dict[str, Any] = {
        "model": body["model"],
        "prompt": body["prompt"],
    }
    extra: dict[str, Any] = {}
    for k, v in body.items():
        if k in kwargs or k in _KNOWN:
            if k not in kwargs and k not in ("model", "prompt"):
                kwargs[k] = v
        else:
            extra[k] = v
    # OpenAI puts image references under different names depending on model;
    # accept a few common ones as input images.
    for ref_key in ("image", "images", "input_image", "input_images"):
        if ref_key in body:
            imgs = body[ref_key]
            if isinstance(imgs, str):
                kwargs["input_images"] = [ImageInput(url=imgs)]
            elif isinstance(imgs, list) and imgs and isinstance(imgs[0], str):
                kwargs["input_images"] = [ImageInput(url=u) for u in imgs]
            extra.pop(ref_key, None)
            break
    if extra:
        kwargs["extra"] = extra
    return UnifiedImageRequest(**kwargs)


def to_openai(resp: UnifiedImageResponse) -> dict[str, Any]:
    data = []
    for d in resp.data or []:
        item: dict[str, Any] = {}
        if d.url:
            item["url"] = d.url
        if d.b64_json:
            item["b64_json"] = d.b64_json
        if d.revised_prompt:
            item["revised_prompt"] = d.revised_prompt
        data.append(item)
    out: dict[str, Any] = {"created": resp.created, "data": data}
    if resp.usage:
        usage: dict[str, Any] = {}
        for f in ("input_tokens", "output_tokens", "total_tokens"):
            v = getattr(resp.usage, f, None)
            if v is not None:
                usage[f] = v
        if usage:
            out["usage"] = usage
    return out


__all__ = ["from_openai", "to_openai"]
