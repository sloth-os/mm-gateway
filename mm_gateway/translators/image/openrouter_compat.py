"""OpenRouter-compatible image translator.

Maps OpenRouter's unified ``POST /api/v1/images`` shape to/from the unified
image schema. OpenRouter always returns base64 in ``data[].b64_json``, so the
response translator prefers b64 and falls back to url.
"""

from __future__ import annotations

from typing import Any

from mm_gateway.core.exceptions import ValidationError
from mm_gateway.schemas.image import ImageData, ImageInput, UnifiedImageRequest, UnifiedImageResponse

_KNOWN = {
    "model", "prompt", "n", "aspect_ratio", "resolution", "size", "quality",
    "seed", "background", "output_format", "output_compression",
}


def from_openrouter(body: dict[str, Any]) -> UnifiedImageRequest:
    if "model" not in body or "prompt" not in body:
        raise ValidationError("`model` and `prompt` are required for image generation.")
    kwargs: dict[str, Any] = {"model": body["model"], "prompt": body["prompt"]}
    extra: dict[str, Any] = {}
    for k, v in body.items():
        if k in _KNOWN and k not in ("model", "prompt"):
            kwargs[k] = v
        elif k == "input_references":
            imgs: list[ImageInput] = []
            for ref in v or []:
                url = (ref.get("image_url") or {}).get("url") if isinstance(ref, dict) else None
                if url and url.startswith("data:image"):
                    _, _, b64 = url.partition(",")
                    imgs.append(ImageInput(b64_json=b64))
                elif url:
                    imgs.append(ImageInput(url=url))
            if imgs:
                kwargs["input_images"] = imgs
        elif k in ("provider", "stream"):
            # provider prefs / streaming are handled by the route, not the model.
            continue
        else:
            extra[k] = v
    if extra:
        kwargs["extra"] = extra
    return UnifiedImageRequest(**kwargs)


def to_openrouter(resp: UnifiedImageResponse) -> dict[str, Any]:
    data = []
    for d in resp.data or []:
        item = {"b64_json": d.b64_json or d.url, "media_type": d.media_type or "image/png"}
        data.append(item)
    out: dict[str, Any] = {"created": resp.created, "data": data}
    if resp.usage:
        usage = {}
        if resp.usage.cost is not None:
            usage["cost"] = resp.usage.cost
        if usage:
            out["usage"] = usage
    return out


__all__ = ["from_openrouter", "to_openrouter"]
