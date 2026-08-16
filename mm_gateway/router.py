"""Auto-router — picks a backend+model when the request omits ``model``.

When a request sets ``model`` to ``auto`` (or omits it), the gateway cannot use
the registry's model-alias resolver. Instead it scores every model the usable
backends serve for the requested modality against the request's *input
profile* — the part modalities and roles present, the prompt length, the
output count, the requested dimensions/duration/fps — using the static limits
catalogue in :mod:`mm_gateway.models.limits`.

A model **fits** when every documented hard constraint is satisfied (e.g. an
image-input request must not route to a text-only image model; a 20 s video
must not route to a 8 s-ceiling model). Undocumented limits are *not*
constraints, so an operator-pinned brand-new model id still routes. Among the
fitting candidates the router prefers the one that satisfies the most optional
controls (native dimensions/duration/role support), then falls back to a
stable order (backend config order, then the provider's model-list order) so
the choice is deterministic and operator-controllable via key defaults/tags.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mm_gateway.models.limits import ModelLimits
from mm_gateway.schemas.image import UnifiedImageRequest
from mm_gateway.schemas.music import UnifiedMusicRequest
from mm_gateway.schemas.video import UnifiedVideoRequest

_TEXT = "text"
_IMAGE = "image"
_LYRICS = "lyrics"
_AUDIO = "audio"
_VIDEO = "video"


@dataclass
class RequestProfile:
    """The subset of a request the router scores against, per modality."""

    modality: str
    input_modalities: set[str] = field(default_factory=set)
    prompt_chars: int = 0
    # Image / video role flags.
    wants_image_input: bool = False
    wants_first_frame: bool = False
    wants_last_frame: bool = False
    wants_reference_video: bool = False
    wants_reference_audio: bool = False
    wants_continuation_audio: bool = False
    wants_reference_image: bool = False
    wants_lyrics: bool = False
    # Numeric controls the router checks against documented ceilings.
    input_image_count: int = 0
    output_count: int | None = None
    width: int | None = None
    height: int | None = None
    size: str | None = None
    aspect_ratio: str | None = None
    resolution: str | None = None
    duration_seconds: float | None = None
    fps: int | None = None
    sample_rate_hz: int | None = None


def _image_profile(request: UnifiedImageRequest) -> RequestProfile:
    prompt = request.prompt() or ""
    image_count = len(request.input_images())
    modalities: set[str] = set()
    if prompt:
        modalities.add(_TEXT)
    if image_count:
        modalities.add(_IMAGE)
    width = request.width
    height = request.height
    if request.size:
        w, h = _parse_size(request.size)
        width = width or w
        height = height or h
    return RequestProfile(
        modality="image",
        input_modalities=modalities,
        prompt_chars=len(prompt),
        wants_image_input=image_count > 0,
        input_image_count=image_count,
        output_count=request.n,
        width=width, height=height,
        aspect_ratio=request.aspect_ratio or None,
    )


def _video_profile(request: UnifiedVideoRequest) -> RequestProfile:
    prompt = request.prompt() or ""
    modalities: set[str] = set()
    if prompt:
        modalities.add(_TEXT)
    first = request.first_image() is not None
    last = request.last_image() is not None
    ref_images = request.reference_images()
    ref_videos = request.reference_videos()
    ref_audios = request.reference_audios()
    if first or last or ref_images:
        modalities.add(_IMAGE)
    if ref_videos:
        modalities.add(_VIDEO)
    if ref_audios:
        modalities.add(_AUDIO)
    width = request.width
    height = request.height
    if request.size:
        w, h = _parse_size(request.size)
        width = width or w
        height = height or h
    return RequestProfile(
        modality="video",
        input_modalities=modalities,
        prompt_chars=len(prompt),
        wants_image_input=bool(first or last or ref_images),
        wants_first_frame=first,
        wants_last_frame=last,
        wants_reference_video=bool(ref_videos),
        wants_reference_audio=bool(ref_audios),
        input_image_count=len([p for p in request.content
                               if getattr(p.root, "type", None) == "image_url"]),
        output_count=None,
        width=width, height=height,
        aspect_ratio=request.ratio or None,
        resolution=request.resolution or None,
        duration_seconds=request.duration,
        fps=request.fps,
    )


def _music_profile(request: UnifiedMusicRequest) -> RequestProfile:
    prompt = request.generation_prompt() or ""
    modalities: set[str] = set()
    if prompt:
        modalities.add(_TEXT)
    ref_audios = request.reference_audios()
    continuation = request.continuation_audio()
    ref_images = request.reference_images()
    # Lyrics is a *role* on the text channel, not a separate input modality:
    # every lyrics-capable model expresses it via ``supports_lyrics`` (checked
    # below), and several such models list ``input_modalities`` without
    # "lyrics" (e.g. music_v1, chirp-*). Adding it to the modality subset here
    # would wrongly veto them for any request that sets the ``lyrics`` field.
    if ref_audios or continuation:
        modalities.add(_AUDIO)
    if ref_images:
        modalities.add(_IMAGE)
    return RequestProfile(
        modality="music",
        input_modalities=modalities,
        prompt_chars=len(prompt),
        wants_lyrics=bool(request.lyrics),
        wants_reference_audio=bool(ref_audios),
        wants_continuation_audio=bool(continuation),
        wants_reference_image=bool(ref_images),
        output_count=request.n,
        duration_seconds=request.duration,
        sample_rate_hz=request.sample_rate_hz,
    )


def profile_for(request: Any) -> RequestProfile:
    """Build the router input profile for a unified image/video/music request."""
    if isinstance(request, UnifiedImageRequest):
        return _image_profile(request)
    if isinstance(request, UnifiedVideoRequest):
        return _video_profile(request)
    if isinstance(request, UnifiedMusicRequest):
        return _music_profile(request)
    raise TypeError(f"unsupported request type for auto-routing: {type(request).__name__}")


def _parse_size(size: str) -> tuple[int | None, int | None]:
    """Parse a ``"WxH"`` / ``"W*H"`` size into ints (None on failure)."""
    cleaned = size.replace("*", "x").lower()
    if "x" not in cleaned:
        return None, None
    left, _, right = cleaned.partition("x")
    try:
        return int(left), int(right)
    except ValueError:
        return None, None


@dataclass
class _Score:
    """A candidate model's routing score."""

    fits: bool
    optional_hits: int
    backend_index: int
    model_index: int


def _fits(profile: RequestProfile, limits: ModelLimits) -> tuple[bool, int]:
    """Return ``(fits, optional_hits)`` for a profile against a model's limits.

    ``fits`` is True only when every documented hard constraint is satisfied.
    ``optional_hits`` counts satisfied optional controls (used to prefer models
    that natively honor the request's dimensions/duration/roles).
    """
    optional_hits = 0
    # Input modalities: every modality the request carries must be accepted.
    # Empty ``input_modalities`` (e.g. the permissive fallback) means "no
    # constraint" — the model accepts any input the modality allows.
    if limits.input_modalities and profile.input_modalities:
        accepted = set(limits.input_modalities)
        if not profile.input_modalities <= accepted:
            return False, 0

    # Prompt length.
    if limits.max_prompt_chars is not None and profile.prompt_chars > limits.max_prompt_chars:
        return False, 0
    if limits.max_prompt_tokens is not None and profile.prompt_chars > limits.max_prompt_tokens * 4:
        # Rough chars->tokens guard (4 chars/token) only when no char ceiling
        # was documented; avoids routing a huge prompt to a token-capped model.
        return False, 0

    # Image-specific role support.
    if profile.wants_image_input:
        if limits.supports_image_to_image is False:
            return False, 0
        if limits.max_input_images is not None and profile.input_image_count > limits.max_input_images:
            return False, 0
        optional_hits += 1

    # Output count.
    if profile.output_count is not None and limits.max_output_count is not None:
        if profile.output_count > limits.max_output_count:
            return False, 0
        optional_hits += 1

    # Duration (video + music).
    if profile.duration_seconds is not None:
        if limits.max_duration_seconds is not None and profile.duration_seconds > limits.max_duration_seconds:
            return False, 0
        if limits.min_duration_seconds is not None and profile.duration_seconds < limits.min_duration_seconds:
            return False, 0
        optional_hits += 1

    # FPS.
    if profile.fps is not None and limits.max_fps is not None:
        if profile.fps > limits.max_fps:
            return False, 0
        optional_hits += 1

    # Output geometry: explicit dimensions.
    longest = _longest_side(profile.width, profile.height, profile.size)
    if longest is not None and limits.max_output_longest_side is not None:
        if longest > limits.max_output_longest_side:
            return False, 0
        optional_hits += 1
    if profile.aspect_ratio and limits.aspect_ratios:
        if profile.aspect_ratio in limits.aspect_ratios:
            optional_hits += 1
        else:
            return False, 0

    # Video role flags.
    if profile.wants_first_frame and limits.supports_first_frame is False:
        return False, 0
    if profile.wants_last_frame and limits.supports_last_frame is False:
        return False, 0
    if profile.wants_reference_video and limits.supports_reference_video is False:
        return False, 0
    if profile.wants_reference_audio and limits.supports_reference_audio is False:
        return False, 0
    for flag in (profile.wants_first_frame, profile.wants_last_frame,
                 profile.wants_reference_video, profile.wants_reference_audio):
        if flag:
            optional_hits += 1

    # Music role flags.
    if profile.wants_lyrics and limits.supports_lyrics is False:
        return False, 0
    if profile.wants_continuation_audio and limits.supports_continuation_audio is False:
        return False, 0
    if profile.wants_reference_image and limits.supports_reference_image is False:
        return False, 0
    if profile.sample_rate_hz is not None and limits.supported_sample_rates:
        if profile.sample_rate_hz not in limits.supported_sample_rates:
            return False, 0
        else:
            optional_hits += 1
    for flag in (profile.wants_lyrics, profile.wants_reference_audio,
                 profile.wants_continuation_audio, profile.wants_reference_image):
        if flag:
            optional_hits += 1

    return True, optional_hits


def _longest_side(width: int | None, height: int | None, size: str | None) -> int | None:
    sides = [s for s in (width, height) if s]
    if size:
        w, h = _parse_size(size)
        if w:
            sides.append(w)
        if h:
            sides.append(h)
    return max(sides) if sides else None


def score(profile: RequestProfile, limits: ModelLimits, *, backend_index: int,
          model_index: int) -> _Score:
    fits, optional_hits = _fits(profile, limits)
    return _Score(fits=fits, optional_hits=optional_hits,
                  backend_index=backend_index, model_index=model_index)


def best(scores: list[_Score]) -> _Score | None:
    """Pick the best score: fit > most optional hits > stable backend/model order."""
    fitting = [s for s in scores if s.fits]
    if not fitting:
        return None
    fitting.sort(key=lambda s: (-s.optional_hits, s.backend_index, s.model_index))
    return fitting[0]


__all__ = ["RequestProfile", "profile_for", "score", "best"]
