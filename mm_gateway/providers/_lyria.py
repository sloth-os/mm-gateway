"""Shared Lyria request/response helpers.

Both the AI Studio (google) and Vertex adapters generate music with the
**Lyria 3** Interactions API — ``client.aio.interactions.create()`` (REST
``POST /v1beta/interactions``). The wire shape is identical across the two
surfaces (the only difference is how the ``genai.Client`` is authenticated),
so the body-building and output-extraction logic lives here and is reused by
both providers. AI Studio speaks raw httpx against
``generativelanguage.googleapis.com`` (its legacy base); Vertex goes through
the SDK's ``client.aio.interactions`` against the aiplatform host. Either way
the body is the same and the response is normalized the same way.
"""

from __future__ import annotations

from typing import Any

from mm_gateway.schemas.music import UnifiedMusicRequest


def lyria_body(request: UnifiedMusicRequest) -> dict[str, Any]:
    """Build the Interactions request body for a Lyria 3 call.

    The canonical gateway inputs map onto Lyria ``input`` parts. Wire names
    (``model``, ``input``, ``response_format``, ``generation_config``) stay
    here rather than leaking into the public REST schema.
    """
    parts: list[dict[str, Any]] = []
    if prompt := request.generation_prompt():
        parts.append({"type": "text", "text": prompt})
    if request.lyrics:
        parts.append({"type": "text", "text": f"Lyrics:\n{request.lyrics}"})
    for p in request.content:
        root = p.root
        if hasattr(root, "image_url"):
            url = root.image_url.url
            if url.startswith("data:"):
                header, _, data = url.partition(",")
                parts.append({
                    "type": "image",
                    "mime_type": header[5:].split(";", 1)[0] or "image/png",
                    "data": data,
                })
            else:
                parts.append({
                    "type": "image_url",
                    "image_url": {"url": url},
                    "role": getattr(root, "role", "reference_image"),
                })
        elif hasattr(root, "audio_url"):
            url = root.audio_url.url
            if url.startswith("data:"):
                header, _, data = url.partition(",")
                parts.append({
                    "type": "audio",
                    "mime_type": header[5:].split(";", 1)[0] or "audio/mpeg",
                    "data": data,
                    "role": getattr(root, "role", "reference_audio"),
                })
            else:
                parts.append({
                    "type": "audio_url",
                    "audio_url": {"url": url},
                    "role": getattr(root, "role", "reference_audio"),
                })
    for img in request.extra.get("images", []) or []:
        parts.append({"type": "image", "mime_type": img.get("mime_type", "image/jpeg"),
                      "data": img.get("data")})
    if not parts:
        parts.append({"type": "text", "text": request.lyrics or ""})
    body: dict[str, Any] = {"model": request.model, "input": parts}
    # The Interactions request takes these as top-level fields (not nested in
    # a ``config`` key). ``response_format`` is the AudioResponseFormat
    # envelope: a ``"type": "audio"`` discriminator plus a ``mime_type``
    # drawn from the SDK's output enum (audio/wav, audio/mp3, ...). Omit it
    # entirely when no format is pinned: Lyria's default output is MP3.
    mime = lyria_request_mime(request.audio_format)
    if mime:
        body["response_format"] = {"type": "audio", "mime_type": mime}
    # ``generation_config`` carries the generation knobs the SDK recognises
    # (``seed``). Other provider-specific knobs ride through best-effort
    # inside ``generation_config`` — unknown fields are ignored upstream.
    generation_config: dict[str, Any] = {}
    if request.negative_prompt:
        generation_config["negative_prompt"] = request.negative_prompt
    if request.seed is not None:
        generation_config["seed"] = request.seed
    if request.guidance_scale is not None:
        generation_config["guidance_scale"] = request.guidance_scale
    if request.n is not None:
        generation_config["number_of_outputs"] = request.n
    generation_config.update(request.extra.get("lyria_config") or {})
    if generation_config:
        body["generation_config"] = generation_config
    return body


def lyria_request_mime(audio_format: str | None) -> str | None:
    """SDK-enum value for the Interactions ``response_format.mime_type``.

    The output enum (``AudioResponseFormatMimeType``) is
    ``audio/mp3``/``audio/wav``/``audio/ogg_opus``/``audio/l16``/``audio/alaw``/
    ``audio/mulaw`` — note ``audio/mp3``, NOT ``audio/mpeg`` (that lives in the
    broader *input* audio-parts enum and is not accepted for the output format).
    Returns ``None`` to omit ``response_format`` entirely: Lyria emits MP3 by
    default when no envelope is sent, so a bare MP3 request need not pin it.
    """
    if not audio_format:
        return None
    if audio_format == "mp3":
        return "audio/mp3"
    if audio_format == "wav":
        return "audio/wav"
    if audio_format == "ogg_opus":
        return "audio/ogg_opus"
    return f"audio/{audio_format}"


def lyria_media_type(audio_format: str | None) -> str:
    """Client-facing MIME of the inline audio the Lyria call returns.

    Matches the gateway convention every other music provider uses (minimax /
    elevenlabs / udioapi / mureka all map mp3 -> ``audio/mpeg``, and
    ``rest.py``'s music default is ``audio/mpeg``). The default is MP3 because
    Lyria emits MP3 unless ``response_format`` requests WAV.
    """
    if not audio_format or audio_format == "mp3":
        return "audio/mpeg"
    if audio_format == "wav":
        return "audio/wav"
    if audio_format == "ogg_opus":
        return "audio/ogg"
    return f"audio/{audio_format}"


def extract_lyria_output(data: Any) -> tuple[str | None, str | None]:
    """Pull the inline audio (base64) and any text/lyrics out of a Lyria
    response. Accepts either the raw JSON dict (the AI Studio REST envelope)
    or a pydantic ``Interaction`` model (the Vertex SDK return value), which we
    coerce to a dict before walking it.

    The shape is ``steps[].content[]`` blocks: audio blocks carry
    ``{type:"audio", data, mime_type}``, text blocks ``{type:"text", text}``.
    """
    if hasattr(data, "model_dump"):
        data = data.model_dump(exclude_none=True)
    if not isinstance(data, dict):
        return None, None
    audio_b64: str | None = None
    lyrics: str | None = None
    steps = data.get("steps") or data.get("model_output") or []
    if isinstance(steps, dict):
        steps = [steps]
    for step in steps:
        content = (step or {}).get("content") or (step or {}).get("model_output") or []
        if isinstance(content, dict):
            content = [content]
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "audio" and block.get("data") and not audio_b64:
                audio_b64 = block["data"]
            elif btype == "text" and block.get("text") and not lyrics:
                lyrics = block["text"]
    # Some envelopes surface the audio at the top level instead.
    if not audio_b64:
        out = data.get("output_audio")
        if isinstance(out, dict):
            audio_b64 = out.get("data")
        else:
            audio_b64 = out
    if not audio_b64:
        audio_b64 = data.get("audio")
    if not lyrics:
        lyrics = data.get("output_text") or data.get("text")
    return audio_b64, lyrics
