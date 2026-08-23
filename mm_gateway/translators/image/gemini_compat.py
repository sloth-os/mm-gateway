"""Gemini-compatible image translator.

The Gemini image REST shape (``POST /v1/images``) is the **Interactions**
surface — ``{model, input, config}`` — the same envelope Imagen
(``GenerateImagesConfig``) and Lyria (``generation_config``) speak:

    { "model": "imagen-4.0-generate-001",
      "input": "a string prompt" | [ {type:"text", text}
                                    | {type:"image", url}
                                    | {type:"image", mime_type, data} ... ],
      "config": { "negative_prompt": ..., "n": ..., "size": ...,
                  "aspect_ratio": ..., "quality": ..., "style": ...,
                  "seed": ..., "guidance_scale": ..., "output_format": ...,
                  ...any provider-specific knob } }

``config`` is the abstraction over **all** backend image functions: every knob
every provider reads (OpenAI's ``background``/``output_compression``/``user``,
Stability's ``cfg_scale``/``steps``, Volcengine's ``watermark``, DashScope's
``watermark``/``thinking_mode``, xAI's ``resolution``/``user``, FLUX's
``output_resolution``, OpenRouter's ``output_compression``/``input_references``)
has a home in the unified flat fields or rides through as a provider-specific
``extra`` key. ``config`` is the canonical place to set them; the flat knobs
remain accepted at the top level for backwards compatibility. Where a knob is
set in both places, ``config`` wins (it is the new canonical shape).

The response is a task whose ``steps[].content[]`` blocks carry the generated
image (inline base64 ``{type:"image", data, mime_type}`` or a URL
``{type:"image", url}``). The SDK exposes ``output_image`` (last image block's
base64) / ``output_image_url`` convenience accessors; we surface the same
values as top-level fields for REST clients, mirroring the Lyria music
translator's ``output_audio`` / ``output_audio_url``.

This translator maps that front-end shape to/from the unified
``UnifiedImageRequest`` / ``UnifiedImageTask`` schema, so every backend
provider (OpenAI, Imagen, Stability, xAI, Volcengine, FLUX, DashScope Wanx,
OpenRouter) is reachable through a Gemini-shaped request.
"""

from __future__ import annotations

from typing import Any

from mm_gateway.core.exceptions import ValidationError
from mm_gateway.schemas.image import (
    UnifiedImageRequest,
    UnifiedImageTask,
    image_part,
    text_part,
)

# Top-level fields the unified model owns; everything else -> extra. These are
# the keys the abstraction accepts in both the flat top-level form and inside
# ``config`` — the union of knobs every backend image provider reads.
_KNOWN = {
    "model", "content", "negative_prompt", "n", "size", "width", "height",
    "aspect_ratio", "resolution", "quality", "style", "seed", "guidance_scale",
    "num_inference_steps", "strength", "watermark", "response_format",
    "output_format", "output_compression", "background", "stream",
    "callback_url", "user", "provider",
}

# Flat generation knobs copied from the top-level form (the legacy shape, still
# accepted). The same names are also read from ``config``. ``provider`` is NOT a
# generation knob — it is a routing directive dict ({tag}/{backend}) read from
# the raw body by ``routing_overrides`` — so it stays out of the unified fields.
_FLAT_KNOBS = (
    "negative_prompt", "n", "size", "width", "height", "aspect_ratio",
    "resolution", "quality", "style", "seed", "guidance_scale",
    "num_inference_steps", "strength", "watermark", "output_format",
    "output_compression", "background", "stream", "callback_url", "user",
)


def from_gemini(body: dict[str, Any]) -> UnifiedImageRequest:
    if "model" not in body:
        raise ValidationError("`model` is required for image generation.")
    kwargs: dict[str, Any] = {"model": body["model"]}

    # ``input`` is either a string prompt or a parts array. Gemini image parts
    # are {type:"text", text} and {type:"image", url} or {type:"image",
    # mime_type, data}; the gateway accepts both URL and inline-base64 images.
    parts = []
    inp = body.get("input")
    if isinstance(inp, str):
        if inp:
            parts.append(text_part(inp))
    elif isinstance(inp, list):
        for part in inp:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype == "text":
                if part.get("text"):
                    parts.append(text_part(part.get("text") or ""))
            elif ptype == "image":
                url = part.get("url")
                data = part.get("data")
                if url or data:
                    parts.append(image_part(
                        url=url, data=data,
                        mime_type=part.get("mime_type"),
                    ))
    if parts:
        kwargs["content"] = parts

    # response_format selects url vs b64_json output (gateway extension; Gemini
    # native output is usually inline base64).
    rf = body.get("response_format")
    if isinstance(rf, str):
        kwargs["response_format"] = rf  # type: ignore[assignment]

    # Flat generation knobs (legacy top-level form). These are the fallback;
    # ``config`` below overrides them where both are present.
    for k in _FLAT_KNOBS:
        if (v := body.get(k)) is not None:
            kwargs[k] = v

    # ``config`` is the Gemini Interactions abstraction over all backend image
    # functions. Known keys (the union every provider reads) merge onto the
    # unified flat fields, overriding the legacy top-level form; unknown keys
    # drop into ``extra`` and pass through to providers that accept them
    # (best-effort policy). This lets a single front-end body reach every
    # backend knob without per-provider shapes.
    config = body.get("config")
    if isinstance(config, dict):
        for k, v in config.items():
            if v is None:
                continue
            if k == "response_format" and isinstance(v, dict):
                # Imagen/Lyria-style response_format envelope: {type, quality}.
                # type "url"/"b64_json" selects delivery; quality maps to the
                # unified quality knob.
                rtype = v.get("type")
                if isinstance(rtype, str):
                    kwargs["response_format"] = rtype  # type: ignore[assignment]
                if v.get("quality"):
                    kwargs["quality"] = v["quality"]
                continue
            if k == "provider":
                # Routing directive, not a generation knob (and may be a dict);
                # ``routing_overrides`` reads it from the raw body, not here.
                continue
            if k in _KNOWN:
                kwargs[k] = v
            else:
                kwargs.setdefault("extra", {})[k] = v

    extra: dict[str, Any] = {}
    for k, v in body.items():
        if k not in _KNOWN and k not in ("input", "response_format", "config"):
            extra[k] = v
    if extra:
        kwargs["extra"] = {**kwargs.get("extra", {}), **extra}
    return UnifiedImageRequest(**kwargs)


def to_gemini_create(task: UnifiedImageTask) -> dict[str, Any]:
    """The create endpoint returns only the task id."""
    return {"id": task.task_id}


def to_gemini_task(task: UnifiedImageTask) -> dict[str, Any]:
    """Map the unified task to the Gemini steps/content response shape.

    ``model_output`` steps carry a ``content`` array of typed blocks: image
    blocks ({type:"image", url} or {type:"image", data, mime_type}) and text
    blocks ({type:"text", text} for revised prompts). Convenience top-level
    ``output_image`` (last inline image block's base64) and ``output_image_url``
    (last URL block's url) mirror the SDK accessors.
    """
    out: dict[str, Any] = {
        "id": task.task_id,
        "model": task.model,
        "status": task.status,
    }
    content: list[dict[str, Any]] = []
    for d in task.images or []:
        if d.b64_json:
            block: dict[str, Any] = {"type": "image", "data": d.b64_json}
            if d.media_type:
                block["mime_type"] = d.media_type
            content.append(block)
        elif d.url:
            b: dict[str, Any] = {"type": "image", "url": d.url}
            if d.media_type:
                b["mime_type"] = d.media_type
            content.append(b)
        if d.revised_prompt:
            content.append({"type": "text", "text": d.revised_prompt})
    if content:
        out["steps"] = [{"type": "model_output", "content": content}]

    # Convenience accessors mirroring the Lyria output_audio / output_audio_url.
    last_b64 = next((d.b64_json for d in reversed(task.images or []) if d.b64_json), None)
    last_url = next((d.url for d in reversed(task.images or []) if d.url), None)
    if last_b64:
        out["output_image"] = last_b64
    elif last_url:
        out["output_image_url"] = last_url

    if task.error:
        out["error"] = {"code": "failed", "message": task.error}
    if task.usage and task.usage.cost is not None:
        out["usage"] = {"cost": task.usage.cost}
    if task.created_at is not None:
        out["created_at"] = task.created_at
    if task.completed_at is not None:
        out["completed_at"] = task.completed_at
    return out


__all__ = ["from_gemini", "to_gemini_create", "to_gemini_task"]
