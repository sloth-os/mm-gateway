"""Gemini Lyria 3-compatible music translator.

The Lyria 3 REST shape (``POST /v1beta/interactions``) is the **Interactions**
surface — ``{model, input, config}``, where ``config`` is this gateway's
abstraction over every backend music knob (Google's literal wire shape uses
top-level ``response_format`` / ``generation_config`` instead)::

    { "model": "lyria-3-pro-preview",
      "input": "a string prompt" | [ {type:"text", text} | {type:"image", mime_type, data} ... ],
      "config": { "negative_prompt": ..., "duration": ..., "bpm": ...,
                  "key_scale": ..., "time_signature": ..., "vocal_language": ...,
                  "audio_format": ..., "audio_quality": ..., "is_instrumental": ...,
                  "generate_audio": ..., "seed": ..., "guidance_scale": ...,
                  "n": ..., "response_format": {"type": "audio", "quality": ...},
                  ...any provider-specific knob } }

``config`` is the abstraction over **all** backend music functions: every knob
every provider reads (ElevenLabs' ``finetune_id``/``music_length``/``seed``,
MiniMax's ``lyrics``/``lyrics_optimizer``/``cover_feature_id``, udioapi's
``style``/``title``/``gender``/``style_weight``, Mureka's ``title``/
``audio_config``/``voice_id``, ACE-Step's ``inference_steps``/``batch_size``/
``task_type``/``reference_audio_path``, Lyria's ``number_of_outputs``/
``lyria_config``) has a home in the unified flat fields or rides through as a
provider-specific ``extra`` key. ``config`` is the canonical place to set them;
the flat knobs remain accepted at the top level for backwards compatibility.
Where a knob is set in both places, ``config`` wins. ``response_format`` is
accepted at the top level (the legacy Lyria shape) and inside ``config`` (the
Interactions shape).

The response is a ``steps`` array whose ``model_output`` steps carry a
``content`` array of typed blocks (``{type:"text", text}`` for lyrics/structure
and ``{type:"audio", data}`` for base64-encoded audio). The SDK also exposes
``interaction.output_audio`` / ``interaction.output_text`` convenience
accessors; we surface the same values as top-level fields for REST clients.

This translator maps that front-end shape to/from the unified ``UnifiedMusic``
schema, so every backend provider (ElevenLabs, MiniMax, udioapi, Mureka,
ACE-Step, Lyria itself) is reachable through a Lyria-shaped request.
"""

from __future__ import annotations

from typing import Any

from mm_gateway.core.exceptions import ValidationError
from mm_gateway.schemas.music import (
    UnifiedMusicRequest,
    UnifiedMusicTask,
    audio_part,
    image_part,
    text_part,
)

# Top-level fields the unified model owns; everything else -> extra. These are
# the keys the abstraction accepts in both the flat top-level form and inside
# ``config`` — the union of knobs every backend music provider reads.
_KNOWN = {
    "model", "content", "negative_prompt", "duration", "bpm", "key_scale",
    "key", "scale", "time_signature", "vocal_language", "audio_format",
    "audio_quality", "is_instrumental", "generate_audio", "seed",
    "guidance_scale", "n", "callback_url", "provider",
}

# Flat generation knobs copied from the top-level form (the legacy shape, still
# accepted). The same names are also read from ``config``. ``provider`` is NOT a
# generation knob — it is a routing directive dict ({tag}/{backend}) read from
# the raw body by ``routing_overrides`` — so it stays out of the unified fields.
_FLAT_KNOBS = (
    "negative_prompt", "duration", "bpm", "key_scale", "key", "scale",
    "time_signature", "vocal_language", "audio_format", "audio_quality",
    "is_instrumental", "generate_audio", "seed", "guidance_scale", "n",
    "callback_url",
)

# Lyria `input` text part -> unified; image part (mime_type+data) -> image_url
# is not a 1:1 fit (Lyria carries inline base64 images), so we stash inline
# images in extra["images"] for providers that can consume them (e.g. ACE-Step
# image-to-music fetches its own URL; the data form is a gateway extension).


def _apply_response_format(rf: Any, kwargs: dict[str, Any]) -> None:
    """Map a Lyria ``response_format`` envelope onto audio_format/quality.

    ``{"type": "audio"}`` selects WAV; any other string ``type`` sets
    ``audio_format`` to it; ``quality`` sets ``audio_quality``. Uses
    ``setdefault`` so an explicit flat/config value wins over a later call.
    """
    if not isinstance(rf, dict):
        return
    rtype = rf.get("type")
    if rtype == "audio":
        kwargs.setdefault("audio_format", "wav")
    elif isinstance(rtype, str):
        kwargs.setdefault("audio_format", rtype)
    if rf.get("quality"):
        kwargs.setdefault("audio_quality", rf.get("quality"))


def from_lyria(body: dict[str, Any]) -> UnifiedMusicRequest:
    if "model" not in body:
        raise ValidationError("`model` is required for music generation.")
    kwargs: dict[str, Any] = {"model": body["model"]}

    # ``input`` is either a string prompt or a parts array. Lyria parts are
    # {type:"text", text} and {type:"image", mime_type, data}; the gateway also
    # accepts our native {type:"audio_url",...}/{type:"image_url",...} shapes so
    # the same content[] model as video works.
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
                # Lyria inline image: {mime_type, data}. Stash for providers
                # that consume inline reference images.
                if part.get("data"):
                    kwargs.setdefault("extra", {}).setdefault("images", []).append(
                        {"mime_type": part.get("mime_type") or "image/jpeg",
                         "data": part["data"]}
                    )
            elif ptype == "audio_url":
                url = (part.get("audio_url") or {}).get("url")
                if url:
                    parts.append(audio_part(url, part.get("role") or "reference_audio"))
            elif ptype == "image_url":
                url = (part.get("image_url") or {}).get("url")
                if url:
                    parts.append(image_part(url, part.get("role") or "reference_image"))
    if parts:
        kwargs["content"] = parts

    # Flat generation knobs (legacy top-level form). These are the fallback;
    # ``config`` below overrides them where both are present.
    for k in _FLAT_KNOBS:
        if (v := body.get(k)) is not None:
            kwargs[k] = v

    # response_format {"type": "audio"} selects WAV output (legacy Lyria shape).
    _apply_response_format(body.get("response_format"), kwargs)

    # ``config`` is the Gemini Interactions abstraction over all backend music
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
            if k == "response_format":
                _apply_response_format(v, kwargs)
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
    return UnifiedMusicRequest(**kwargs)


def to_lyria_create(task: UnifiedMusicTask) -> dict[str, Any]:
    """The create endpoint returns only the task id (Lyria interaction id)."""
    return {"id": task.task_id}


def to_lyria_task(task: UnifiedMusicTask) -> dict[str, Any]:
    """Map the unified task to the Lyria steps/content response shape.

    ``model_output`` steps carry a ``content`` array of typed blocks: audio
    blocks ({type:"audio", data, mime_type}) and text blocks ({type:"text",
    text} for lyrics/structure). Convenience top-level ``output_audio`` (last
    audio block's base64) and ``output_text`` mirror the SDK accessors. When a
    provider returned a URL rather than inline bytes (most async backends), the
    URL is exposed as ``output_audio_url`` and the audio block's ``url`` field
    — a gateway extension since Lyria native output is inline base64.
    """
    out: dict[str, Any] = {
        "id": task.task_id,
        "model": task.model,
        "status": task.status,
    }
    content: list[dict[str, Any]] = []
    # Prefer inline base64 (Lyria native); fall back to URL blocks otherwise.
    if task.audio_b64:
        block: dict[str, Any] = {"type": "audio", "data": task.audio_b64}
        if task.audio_media_type:
            block["mime_type"] = task.audio_media_type
        content.append(block)
    for url in task.audio_urls:
        b: dict[str, Any] = {"type": "audio", "url": url}
        if task.audio_media_type:
            b["mime_type"] = task.audio_media_type
        content.append(b)
    if task.lyrics:
        content.append({"type": "text", "text": task.lyrics})
    if content:
        out["steps"] = [{"type": "model_output", "content": content}]

    if task.audio_b64:
        out["output_audio"] = task.audio_b64
    elif task.audio_urls:
        out["output_audio_url"] = task.audio_urls[0]
    if task.lyrics:
        out["output_text"] = task.lyrics
    if task.error:
        out["error"] = {"code": "failed", "message": task.error}
    if task.usage and task.usage.cost is not None:
        out["usage"] = {"cost": task.usage.cost}
    return out


__all__ = ["from_lyria", "to_lyria_create", "to_lyria_task"]
