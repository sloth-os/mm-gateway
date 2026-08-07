"""Unified music schemas — the canonical internal representation.

Mirrors the video schema's shape: a ``content`` array of typed parts (text /
audio_url / image_url) plus a flat set of generation knobs (duration, bpm,
key, time_signature, ...). Every front-end shape (Gemini Lyria 3, ElevenLabs,
MiniMax, udioapi, Mureka, ACE-Step) is translated into this one model, and
every provider pulls the bits it cares about out of ``content`` and the knobs.

The front-end format is **Gemini Lyria 3-compatible**: a request is
``{model, input, ...}`` where ``input`` is a string or a parts array, and the
response is a task whose ``steps[].content[]`` blocks carry the generated
audio (base64) and text (lyrics). See ``translators/music/lyria_compat.py``.

Music generation is async on most providers (MiniMax, udioapi, Mureka,
ACE-Step return a task id to poll) and synchronous on ElevenLabs (it streams
audio bytes from a single POST); the ElevenLabs adapter wraps the stream as a
synthetic in-memory task so the gateway's poll surface is uniform.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, RootModel


# --------------------------------------------------------------------------- #
# Content parts
# --------------------------------------------------------------------------- #

AudioRole = Literal["reference_audio", "continuation_audio"]
ImageRole = Literal["reference_image"]


class _Url(BaseModel):
    url: str


class MusicTextPart(BaseModel):
    """The prompt / lyrics. Required for text-to-music."""

    type: Literal["text"]
    text: str


class MusicAudioPart(BaseModel):
    """A reference or continuation audio clip."""

    type: Literal["audio_url"]
    audio_url: _Url
    role: AudioRole = "reference_audio"


class MusicImagePart(BaseModel):
    """A reference image (e.g. ACE-Step image-to-music)."""

    type: Literal["image_url"]
    image_url: _Url
    role: ImageRole = "reference_image"


class MusicContentPart(RootModel[MusicTextPart | MusicAudioPart | MusicImagePart]):
    """Discriminated union over the ``type`` field of a content part."""


def text_part(text: str) -> MusicTextPart:
    return MusicTextPart(type="text", text=text)


def audio_part(url: str, role: AudioRole = "reference_audio") -> MusicAudioPart:
    return MusicAudioPart(type="audio_url", audio_url=_Url(url=url), role=role)


def image_part(url: str, role: ImageRole = "reference_image") -> MusicImagePart:
    return MusicImagePart(type="image_url", image_url=_Url(url=url), role=role)


# --------------------------------------------------------------------------- #
# Request / response
# --------------------------------------------------------------------------- #


class UnifiedMusicRequest(BaseModel):
    """The canonical internal music request.

    ``content`` carries the typed parts (prompt/lyrics text, reference or
    continuation audio, reference image). The flat fields are the generation
    knobs the union of providers exposes; each provider reads the subset it
    supports and ignores the rest.
    """

    model: str
    content: list[MusicContentPart] = Field(default_factory=list)
    negative_prompt: str | None = None
    # Total length in seconds (providers that take ms convert internally).
    duration: float | None = None
    # Tempo in beats per minute.
    bpm: int | None = None
    # Musical key + scale, e.g. "C major", "A minor".
    key_scale: str | None = Field(None, description="Musical key and scale, e.g. 'C major'.")
    key: str | None = Field(None, description="Musical key, e.g. 'C', 'A'.")
    scale: str | None = Field(None, description="Scale, e.g. 'major', 'minor'.")
    time_signature: str | None = Field(None, description="e.g. '4/4'.")
    # Language/tag for vocals, e.g. 'en', 'zh', 'instrumental'.
    vocal_language: str | None = None
    # Output container/codec, e.g. 'mp3', 'wav', 'ogg'.
    audio_format: str | None = None
    # Sample rate + bitrate hint some providers accept, e.g. '44100_128'.
    audio_quality: str | None = None
    is_instrumental: bool | None = None
    # Whether to synthesise audio (vs. only lyrics/structure).
    generate_audio: bool | None = None
    seed: int | None = None
    guidance_scale: float | None = None
    # Number of variations to request.
    n: int | None = None
    callback_url: str | None = None
    provider: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    # -- convenience accessors for providers that don't speak content[] natively #

    def prompt(self) -> str | None:
        """Concatenated text parts, or None if there are none."""
        texts = [p.root.text for p in self.content
                 if isinstance(p.root, MusicTextPart)]
        return "\n".join(texts) if texts else None

    def reference_audios(self) -> list[str]:
        return [p.root.audio_url.url for p in self.content
                if isinstance(p.root, MusicAudioPart)
                and p.root.role == "reference_audio"]

    def continuation_audio(self) -> str | None:
        for p in self.content:
            if isinstance(p.root, MusicAudioPart) and p.root.role == "continuation_audio":
                return p.root.audio_url.url
        return None

    def reference_images(self) -> list[str]:
        return [p.root.image_url.url for p in self.content
                if isinstance(p.root, MusicImagePart)]


TaskStatus = Literal["pending", "running", "succeeded", "failed", "cancelled", "expired"]


class MusicUsage(BaseModel):
    cost: float | None = None
    duration: int | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class UnifiedMusicTask(BaseModel):
    task_id: str
    provider: str
    model: str
    status: TaskStatus
    # One or more output audio URLs once succeeded.
    audio_urls: list[str] = Field(default_factory=list)
    # Inline base64 audio (provider returned bytes, not a URL).
    audio_b64: str | None = None
    # MIME type of the audio (e.g. 'audio/mpeg', 'audio/wav').
    audio_media_type: str | None = None
    # Generated lyrics / textual structure, if the provider returns them.
    lyrics: str | None = None
    error: str | None = None
    usage: MusicUsage | None = None
    # Provider-native raw response, for clients that want the full envelope.
    raw: dict[str, Any] | None = None
    created_at: int | None = None
    completed_at: int | None = None
