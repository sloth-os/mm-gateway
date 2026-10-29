"""Static, provider-neutral model capability + limits catalogue.

Each entry describes one gateway model alias (or a backend-native model id) in
terms the public API exposes: the input part *modalities* it accepts, the
generation *controls* it honors (prompt length, input image count, output
count, output dimensions, video duration, music duration, fps, etc.), and the
roles an input image/video/audio part may carry.

The values come from the providers' official documentation (verified via web
search — see ``source_urls`` on each entry). The auto-router uses this
catalogue to pick a usable backend+model whose limits accommodate a request
that omits ``model`` (or sets it to ``auto``); the ``/v1/models`` limits API
returns the same data so a client can craft a prompt for a specific model.

Only the limits that are documented and useful for routing are modelled here.
Unknown limits stay ``None`` and the router treats ``None`` as "unbounded /
not a routing constraint".

Design note: this is deliberately a *static* table, not a live query against
each provider. The catalogue is small and stable, and a live query would make
routing depend on every backend being reachable at request time. An operator
who pins a brand-new model id via ``*_MODEL`` gets a default permissive entry
(see ``limits_for``) so routing still works; they can add an explicit entry
here once the upstream publishes the limits.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from typing import Any, Literal

# The part ``type`` values the public request envelope accepts, per modality.
Lifecycle = Literal["active", "deprecated", "retired"]


def utc_today() -> date:
    """Today's date in UTC (the lifecycle clock; tests patch it)."""
    from datetime import UTC, datetime

    return datetime.now(UTC).date()


_TEXT = "text"
_IMAGE = "image"
_LYRICS = "lyrics"
_AUDIO = "audio"
_VIDEO = "video"


@dataclass(frozen=True)
class ModelLimits:
    """One model's input/output limits, in the gateway's neutral vocabulary.

    Fields are named after the public REST request fields where possible, so
    the limits API reads naturally to a client. Every field is optional; a
    ``None`` (or empty) value means "not a routing constraint / undocumented".
    """

    modality: str
    # Input part ``type`` values the model accepts (subset of the modality's
    # allowed parts). Used to reject e.g. an image-input request against a
    # text-only image model, or a lyrics request against a prompt-only model.
    input_modalities: tuple[str, ...] = ()
    # Generation controls (map to public ``parameters`` fields).
    max_prompt_chars: int | None = None
    max_prompt_tokens: int | None = None
    max_input_images: int | None = None
    max_output_count: int | None = None
    max_duration_seconds: float | None = None
    min_duration_seconds: float | None = None
    max_fps: int | None = None
    # Output geometry. ``aspect_ratios`` / ``supported_sizes`` are documented
    # enumerations; ``max_output_longest_side`` is a pixel ceiling used when the
    # request gives explicit ``dimensions``. Any may be empty/None.
    aspect_ratios: tuple[str, ...] = ()
    supported_sizes: tuple[str, ...] = ()
    max_output_longest_side: int | None = None
    # Per-role input support flags (image parts carry a ``role``; video/audio
    # parts too). The router checks these when a request carries that role.
    supports_image_to_image: bool | None = None
    supports_first_frame: bool | None = None
    supports_last_frame: bool | None = None
    supports_reference_video: bool | None = None
    supports_reference_audio: bool | None = None
    supports_continuation_audio: bool | None = None
    supports_lyrics: bool | None = None
    supports_reference_image: bool | None = None
    # Audio output geometry (music).
    supported_sample_rates: tuple[int, ...] = ()
    # Video models that render sound with the picture (dialogue, effects,
    # music). ``False`` = documented silent; the router never sends an
    # ``include_audio`` request there (docs/design/auto-mode.md#capabilities-native-audio).
    supports_audio_output: bool | None = None
    # Multi-shot video models render several shots, separated by hard cuts, in one request (the prompt lists the
    # shots); ``max_shots`` is how many. Enhancement models take a ``reference_video`` and return it upscaled to the
    # requested ``dimensions`` and/or interpolated to the requested ``fps``
    # (docs/design/auto-mode.md#capabilities-multi-shot-and-enhancement).
    max_shots: int | None = None
    supports_upscale: bool | None = None
    supports_frame_interpolation: bool | None = None
    # Segmentation models take a ``reference_video`` and a prompt naming the subject and return a matte: a grayscale
    # video of the same length, size and frame rate, white where the subject is
    # (docs/design/auto-mode.md#capabilities-segmentation).
    supports_segmentation: bool | None = None
    # Performance models take a driving ``reference_video`` (a person's performance: expressions, lips, head and body
    # motion, timing) and a ``first_frame`` with the character, and return the character performing it
    # (docs/design/auto-mode.md#capabilities-performance).
    supports_performance: bool | None = None
    # ``True``: a ``first_frame``/``last_frame`` cannot be combined with reference images, videos or audio in one
    # request (Wan 3.0); the adapter keeps the frames and drops the references.
    exclusive_frames_and_references: bool | None = None
    # ``False``: the model has no negative prompt field; ``negative_prompt`` is dropped, so exclusions belong in the
    # prompt text.
    supports_negative_prompt: bool | None = None
    supports_voice_cloning: bool | None = None
    supports_instructions: bool | None = None
    max_voice_samples: int | None = None
    min_speed: float | None = None
    max_speed: float | None = None
    supported_file_formats: tuple[str, ...] = ()
    # Model lifecycle (ISO dates, UTC). From ``retired_on`` the model is never
    # auto-routed and a pinned request fails with ``model_retired`` unless it
    # allows a fallback; ``deprecated_on`` ranks it after active models.
    deprecated_on: str | None = None
    retired_on: str | None = None
    replacement: str | None = None
    notes: str = ""
    source_urls: tuple[str, ...] = ()

    def lifecycle(self, today: date | None = None) -> Lifecycle:
        """``active``, ``deprecated`` or ``retired`` on ``today`` (UTC)."""
        day = today or utc_today()
        if self.retired_on and day >= date.fromisoformat(self.retired_on):
            return "retired"
        if self.deprecated_on and day >= date.fromisoformat(self.deprecated_on):
            return "deprecated"
        return "active"

    def to_public_dict(self, today: date | None = None) -> dict[str, Any]:
        """Return the additive, client-facing limits object.

        Omits ``None``/empty fields so the spec stays small; clients ignore
        unknown members, so adding fields later is safe. ``lifecycle`` is always
        present so clients can tell a deprecated or retired model at a glance.
        """
        out: dict[str, Any] = {"modality": self.modality, "lifecycle": self.lifecycle(today)}
        if self.input_modalities:
            out["input_modalities"] = list(self.input_modalities)
        for key in (
            "max_prompt_chars", "max_prompt_tokens", "max_input_images",
            "max_output_count", "max_duration_seconds", "min_duration_seconds",
            "max_fps", "max_output_longest_side", "max_shots",
            "max_voice_samples", "min_speed", "max_speed",
        ):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        if self.aspect_ratios:
            out["aspect_ratios"] = list(self.aspect_ratios)
        if self.supported_sizes:
            out["supported_sizes"] = list(self.supported_sizes)
        if self.supported_sample_rates:
            out["supported_sample_rates"] = list(self.supported_sample_rates)
        if self.supported_file_formats:
            out["supported_file_formats"] = list(self.supported_file_formats)
        for key in (
            "supports_image_to_image", "supports_first_frame", "supports_last_frame",
            "supports_reference_video", "supports_reference_audio",
            "supports_continuation_audio", "supports_lyrics", "supports_reference_image",
            "supports_audio_output", "supports_upscale", "supports_frame_interpolation",
            "supports_segmentation", "supports_performance", "deprecated_on", "retired_on", "replacement",
            "supports_voice_cloning", "supports_instructions", "exclusive_frames_and_references",
            "supports_negative_prompt",
        ):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        if self.notes:
            out["notes"] = self.notes
        if self.source_urls:
            out["source_urls"] = list(self.source_urls)
        return out


# Sentinel returned when a model id has no explicit entry. It is maximally
# permissive so an operator-pinned brand-new model id still routes (every
# constraint passes); the only hard signal is the modality the caller asked for.
def _permissive(modality: str) -> ModelLimits:
    return ModelLimits(modality=modality, notes="No documented limits; routing assumes the model is permissive.")


_LIMITS: dict[str, ModelLimits] = {}

# --------------------------------------------------------------------------- #
# OpenAI
# --------------------------------------------------------------------------- #
# Lifecycle dates below are OpenAI's deprecations page (announcement -> shutdown).
_OPENAI_DEPRECATIONS = "https://developers.openai.com/api/docs/deprecations"
# gpt-image-1 / -mini: multimodal (text+image edit), sizes incl. auto, n up to 10.
_openai_img = ModelLimits(
    modality="image", input_modalities=(_TEXT, _IMAGE),
    supports_image_to_image=True, max_prompt_chars=32000, max_output_count=10,
    supported_sizes=("1024x1024", "1024x1536", "1536x1024", "auto"),
    notes="Output always base64; supports edits endpoint with masks.",
    source_urls=("https://platform.openai.com/docs/guides/image-generation",
                 "https://platform.openai.com/docs/api-reference/images/create"),
)
_LIMITS["gpt-image-1"] = replace(_openai_img,
    deprecated_on="2026-04-22", retired_on="2026-10-23", replacement="gpt-image-2.5-sunburst",
    notes="Output always base64; supports edits endpoint with masks. Shut down on 2026-10-23.",
    source_urls=(*_openai_img.source_urls, _OPENAI_DEPRECATIONS))
_LIMITS["gpt-image-1-mini"] = replace(_openai_img,
    deprecated_on="2026-06-02", retired_on="2026-12-01", replacement="gpt-image-2.5-flare",
    notes="Smaller/faster variant of gpt-image-1 with the same API surface; shuts down on 2026-12-01.",
    source_urls=(*_openai_img.source_urls, _OPENAI_DEPRECATIONS))
_LIMITS["gpt-image-2"] = ModelLimits(
    modality="image", input_modalities=(_TEXT, _IMAGE),
    supports_image_to_image=True, max_prompt_chars=32000, max_output_count=10,
    supported_sizes=("1024x1024", "1536x1024", "1024x1536", "2048x2048",
                     "2048x1152", "3840x2160", "2160x3840", "auto"),
    max_output_longest_side=3840,
    notes="Snapshot gpt-image-2-2026-04-21; arbitrary WxH (both divisible by 16, ratio 1:3-3:1), "
          "above 2560x1440 experimental; quality up to high.",
    source_urls=("https://developers.openai.com/api/docs/models/gpt-image-2",
                 "https://developers.openai.com/api/docs/guides/image-generation"),
)
# GPT Image 2.5 (2026-09-08 snapshots): Sunburst is the quality model, Flare the fastest.
_gpt_image_25 = ModelLimits(
    modality="image", input_modalities=(_TEXT, _IMAGE),
    supports_image_to_image=True, max_prompt_chars=32000, max_output_count=10,
    supported_sizes=("1024x1024", "1536x1024", "1024x1536", "auto"),
    max_output_longest_side=3840,
    notes="Snapshot -2026-09-08. Arbitrary WxH: sides multiples of 16, ratio 1:3-3:1, edge <= 3840, total "
          "655,360-8,294,400 px (above 2560x1440 experimental). Quality low/medium/high/xhigh/max/auto; "
          "transparent background with png/webp; edits take a mask (< 50 MB).",
    source_urls=("https://developers.openai.com/api/docs/guides/image-generation",
                 "https://developers.openai.com/api/docs/models/gpt-image-2.5-sunburst"),
)
_LIMITS["gpt-image-2.5-sunburst"] = _gpt_image_25
_LIMITS["gpt-image-2.5-flare"] = replace(_gpt_image_25,
    notes="OpenAI's fastest image model; same sizes and quality levels as gpt-image-2.5-sunburst. "
          + _gpt_image_25.notes,
    source_urls=("https://developers.openai.com/api/docs/guides/image-generation",
                 "https://developers.openai.com/api/docs/models/gpt-image-2.5-flare"))
_DALL_E_RETIREMENT = dict(deprecated_on="2025-11-14", retired_on="2026-05-12", replacement="gpt-image-2")
_LIMITS["dall-e-2"] = ModelLimits(
    modality="image", input_modalities=(_TEXT, _IMAGE),
    supports_image_to_image=True, max_prompt_chars=1000, max_output_count=10,
    supported_sizes=("256x256", "512x512", "1024x1024"),
    **_DALL_E_RETIREMENT,
    notes="Supports edits + variations with image/mask input; shut down on 2026-05-12.",
    source_urls=("https://platform.openai.com/docs/api-reference/images/create", _OPENAI_DEPRECATIONS),
)
_LIMITS["dall-e-3"] = ModelLimits(
    modality="image", input_modalities=(_TEXT,),
    supports_image_to_image=False, max_prompt_chars=4000, max_output_count=1,
    supported_sizes=("1024x1024", "1792x1024", "1024x1792"),
    **_DALL_E_RETIREMENT,
    notes="Text-only (no image input/edit); returns revised_prompt; shut down on 2026-05-12.",
    source_urls=("https://platform.openai.com/docs/api-reference/images/create", _OPENAI_DEPRECATIONS),
)
# Sora 2: durations 4/8/12/16/20s; i2v via a single input_reference image.
# OpenAI announced the Sora 2 API shutdown on 2026-03-24; the Videos API and
# every Sora 2 model id were removed on 2026-09-24 (no successor was named).
_SORA_RETIREMENT = dict(
    deprecated_on="2026-03-24", retired_on="2026-09-24",
)
_SORA_SHUTDOWN_SOURCES = (_OPENAI_DEPRECATIONS,)
_LIMITS["sora-2"] = ModelLimits(
    modality="video", input_modalities=(_TEXT, _IMAGE),
    supports_first_frame=True, supports_last_frame=False,
    supports_reference_video=False, supports_reference_audio=False,
    max_input_images=1, min_duration_seconds=4, max_duration_seconds=20,
    supported_sizes=("720x1280", "1280x720"),
    supports_audio_output=True, **_SORA_RETIREMENT,
    notes="Durations 4/8/12/16/20s (default 4); 720p only; synchronized audio. Removed from the API on 2026-09-24.",
    source_urls=("https://platform.openai.com/docs/guides/video-generation",
                 "https://developers.openai.com/cookbook/examples/sora/sora2_prompting_guide",
                 *_SORA_SHUTDOWN_SOURCES),
)
_LIMITS["sora-2-pro"] = ModelLimits(
    modality="video", input_modalities=(_TEXT, _IMAGE),
    supports_first_frame=True, supports_last_frame=False,
    supports_reference_video=False, supports_reference_audio=False,
    max_input_images=1, min_duration_seconds=4, max_duration_seconds=20,
    supported_sizes=("720x1280", "1280x720", "1024x1792", "1792x1024",
                    "1080x1920", "1920x1080"),
    supports_audio_output=True, **_SORA_RETIREMENT,
    notes="Adds 1080p and 1024x1792/1792x1024 exports. Removed from the API on 2026-09-24.",
    source_urls=("https://platform.openai.com/docs/guides/video-generation",
                 "https://developers.openai.com/cookbook/examples/sora/sora2_prompting_guide",
                 *_SORA_SHUTDOWN_SOURCES),
)

# --------------------------------------------------------------------------- #
# Google (Imagen / Gemini-Flash-Image / Veo / Lyria)
# --------------------------------------------------------------------------- #
# The Gemini API and Vertex AI (Gemini Enterprise Agent Platform) serve the same ids; an id's lifecycle is the
# Gemini API's deprecations table unless only Vertex lists it (its model-versions table).
_GEMINI_DEPRECATIONS = "https://ai.google.dev/gemini-api/docs/deprecations"
_VERTEX_LIFECYCLE = "https://docs.cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions"
_imagen = ModelLimits(
    modality="image", input_modalities=(_TEXT,),
    supports_image_to_image=False, max_prompt_tokens=480, max_output_count=4,
    aspect_ratios=("1:1", "3:4", "4:3", "9:16", "16:9"),
    notes="Text-only (English prompts); sampleCount 1-4.",
    source_urls=("https://ai.google.dev/gemini-api/docs/imagen",
                 "https://docs.cloud.google.com/vertex-ai/generative-ai/docs/image/generate-images"),
)
_LIMITS["imagen-4.0-generate-001"] = replace(_imagen,
    deprecated_on="2026-06-15", retired_on="2026-08-17", replacement="gemini-nano-banana-2.1",
    notes="Text-only; sampleCount 1-4. Shut down on the Gemini API on 2026-08-17 (Vertex: 2026-06-30).",
    source_urls=(_GEMINI_DEPRECATIONS, _VERTEX_LIFECYCLE))
_LIMITS["imagen-3.0-generate-001"] = replace(_imagen,
    retired_on="2026-06-30", replacement="gemini-2.5-flash-image",
    notes="Text-only; aspect ratios 1:1/3:4/4:3/9:16/16:9; 1K only. Vertex retired it on 2026-06-30.",
    source_urls=(_VERTEX_LIFECYCLE,))
_LIMITS["gemini-2.5-flash-image"] = ModelLimits(
    modality="image", input_modalities=(_TEXT, _IMAGE),
    supports_image_to_image=True, max_prompt_tokens=65536,
    max_input_images=3, max_output_count=10, max_output_longest_side=1536,
    aspect_ratios=("1:1", "3:2", "2:3", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"),
    retired_on="2027-03-15", replacement="gemini-3.1-flash-lite-image",
    notes="a.k.a. nano-banana; up to 3 input images, up to 10 output images. Shuts down on 2027-03-15.",
    source_urls=("https://ai.google.dev/gemini-api/docs/models/gemini-2.5-flash-image",
                 "https://ai.google.dev/gemini-api/docs/image-generation", _GEMINI_DEPRECATIONS),
)
# Gemini 3 image models ("Nano Banana"): up to 14 reference images per request; 131,072 input tokens.
_GEMINI_IMAGE_RATIOS = ("1:1", "3:2", "2:3", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9")
_GEMINI_IMAGE_EXTENDED_RATIOS = (*_GEMINI_IMAGE_RATIOS, "1:4", "4:1", "1:8", "8:1")
_gemini3_image = ModelLimits(
    modality="image", input_modalities=(_TEXT, _IMAGE),
    supports_image_to_image=True, max_prompt_tokens=131072, max_input_images=14,
    aspect_ratios=_GEMINI_IMAGE_EXTENDED_RATIOS, supported_sizes=("1K", "2K", "4K"),
    notes="Nano Banana 2.1: up to 14 reference images (10 objects, 4 characters); 1K (default)/2K/4K, no 512px; "
          "1:4/4:1/1:8/8:1 ratios.",
    source_urls=("https://ai.google.dev/gemini-api/docs/models/gemini-nano-banana-2.1",
                 "https://ai.google.dev/gemini-api/docs/image-generation"),
)
_LIMITS["gemini-nano-banana-2.1"] = _gemini3_image
_LIMITS["gemini-3.1-flash-image"] = replace(_gemini3_image,
    supported_sizes=("0.5K", "1K", "2K", "4K"), deprecated_on="2026-10-06", replacement="gemini-nano-banana-2.1",
    notes="Nano Banana 2: up to 14 reference images (10 objects, 4 characters); 0.5K/1K (default)/2K/4K. "
          "Deprecated on 2026-10-06 in favour of gemini-nano-banana-2.1 (no shutdown date yet).",
    source_urls=("https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-image",
                 "https://ai.google.dev/gemini-api/docs/changelog", _GEMINI_DEPRECATIONS))
_LIMITS["gemini-3.1-flash-lite-image"] = replace(_gemini3_image,
    aspect_ratios=_GEMINI_IMAGE_RATIOS, supported_sizes=("1K",),
    notes="Nano Banana 2 Lite: low latency and cost; up to 14 reference images, 1K only; not optimized for many "
          "references or multi-turn editing.",
    source_urls=("https://ai.google.dev/gemini-api/docs/image-generation", _GEMINI_DEPRECATIONS))
_LIMITS["gemini-3-pro-image"] = replace(_gemini3_image,
    aspect_ratios=_GEMINI_IMAGE_RATIOS,
    notes="Nano Banana Pro: up to 14 reference images (6 objects, 5 characters, 3 styles); 1K (default)/2K/4K.",
    source_urls=("https://ai.google.dev/gemini-api/docs/image-generation", _GEMINI_DEPRECATIONS))
# Veo 2.0 / 3.0: the Gemini API announced the shutdown on 2026-06-15; both surfaces stopped serving them on
# 2026-06-30 (Vertex's upgrade: veo-3.1-generate-001).
_VEO_RETIREMENT = dict(deprecated_on="2026-06-15", retired_on="2026-06-30", replacement="veo-3.1-generate-001")
_LIMITS["veo-2.0-generate-001"] = ModelLimits(
    modality="video", input_modalities=(_TEXT, _IMAGE),
    supports_first_frame=True, supports_last_frame=True,
    supports_reference_video=False, max_input_images=1, max_output_count=2,
    min_duration_seconds=5, max_duration_seconds=8,
    aspect_ratios=("16:9", "9:16"), supports_audio_output=False, **_VEO_RETIREMENT,
    notes="Silent (no audio); durations 5/6/8s; 720p only. Shut down on 2026-06-30.",
    source_urls=("https://ai.google.dev/gemini-api/docs/changelog", _GEMINI_DEPRECATIONS, _VERTEX_LIFECYCLE),
)
_LIMITS["veo-3.0-generate-001"] = ModelLimits(
    modality="video", input_modalities=(_TEXT, _IMAGE),
    supports_first_frame=True, supports_last_frame=True,
    supports_reference_video=False, max_input_images=1, max_output_count=1,
    max_prompt_tokens=1024, min_duration_seconds=4, max_duration_seconds=8,
    aspect_ratios=("16:9", "9:16"), supports_audio_output=True, **_VEO_RETIREMENT,
    notes="Always generates audio; durations 4/6/8s; 720p/1080p; 1 video per request. Shut down on 2026-06-30.",
    source_urls=("https://ai.google.dev/gemini-api/docs/changelog", _GEMINI_DEPRECATIONS, _VERTEX_LIFECYCLE),
)
# The Veo 3.1 previews left the Gemini API on 2026-10-22 (replacement: gemini-omni-1.1-flash).
_veo31_preview = ModelLimits(
    modality="video", input_modalities=(_TEXT, _IMAGE, _VIDEO),
    supports_first_frame=True, supports_last_frame=True,
    supports_reference_video=True, max_input_images=3, max_output_count=1,
    max_prompt_tokens=1024, min_duration_seconds=4, max_duration_seconds=8,
    aspect_ratios=("16:9", "9:16"), supports_audio_output=True,
    retired_on="2026-10-22", replacement="gemini-omni-1.1-flash",
    notes="Audio on; supports video extension + up to 3 reference images; 1 video per request. "
          "Shut down on 2026-10-22.",
    source_urls=("https://ai.google.dev/gemini-api/docs/video", _GEMINI_DEPRECATIONS),
)
_LIMITS["veo-3.1-generate-preview"] = _veo31_preview
_LIMITS["veo-3.1-fast-generate-preview"] = replace(_veo31_preview,
    notes="Faster Veo 3.1 preview. Shut down on 2026-10-22.")
_LIMITS["veo-3.1-lite-generate-preview"] = replace(_veo31_preview,
    input_modalities=(_TEXT, _IMAGE), supports_reference_video=False, max_input_images=1,
    notes="Lowest-cost Veo 3.1 preview (720p/1080p). Shut down on 2026-10-22.")
# Veo 3.1 GA (Vertex AI only): 4/6/8 s, 24 fps, up to 4 videos per prompt; retirement "2026-11-17 or later".
_veo31_ga = ModelLimits(
    modality="video", input_modalities=(_TEXT, _IMAGE),
    supports_first_frame=True, supports_last_frame=True, supports_reference_image=True,
    supports_reference_video=False, max_input_images=3, max_output_count=4,
    min_duration_seconds=4, max_duration_seconds=8, max_fps=24,
    aspect_ratios=("16:9", "9:16"), supports_audio_output=True,
    notes="Vertex GA (us-central1): 4/6/8 s at 24 fps; 720p/1080p/4K; up to 4 videos per prompt; input images "
          "<= 20 MB; reference images only with 8 s. Retirement 2026-11-17 or later.",
    source_urls=("https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/veo/3-1-generate",
                 _VERTEX_LIFECYCLE),
)
_LIMITS["veo-3.1-generate-001"] = _veo31_ga
_LIMITS["veo-3.1-fast-generate-001"] = replace(_veo31_ga,
    notes="Vertex GA fast tier: 4/6/8 s at 24 fps; 720p/1080p; up to 4 videos per prompt; reference images "
          "only with 8 s. Retirement 2026-11-17 or later.")
_LIMITS["veo-3.1-lite-generate-001"] = replace(_veo31_ga,
    supports_last_frame=None, supports_reference_image=False, max_input_images=1,
    notes="Vertex preview (2026-04-02): 4/6/8 s at 24 fps; 720p/1080p; no reference images.")
# Gemini Omni Flash: video through the Interactions API (POST /v1beta/interactions), not generate_videos.
_omni_video = ModelLimits(
    modality="video", input_modalities=(_TEXT, _IMAGE, _VIDEO),
    supports_first_frame=True, supports_last_frame=True, supports_reference_image=True,
    supports_reference_video=True, supports_reference_audio=False, supports_audio_output=True,
    supports_negative_prompt=False, max_input_images=10, max_output_count=1,
    min_duration_seconds=3, max_duration_seconds=10, max_fps=24,
    aspect_ratios=("16:9", "9:16"),
    notes="Interactions API; 3-10 s at 24 fps with sound described in the prompt; 360p/720p (default)/1080p/4k; "
          "first+last frame (two images), reference images, up to 3 reference videos of <= 3 s; videos to edit "
          "or extend <= 10 s. No audio input, negative prompt or system instructions.",
    source_urls=("https://ai.google.dev/gemini-api/docs/omni",
                 "https://ai.google.dev/gemini-api/docs/models/gemini-omni-flash"),
)
_LIMITS["gemini-omni-1.1-flash"] = _omni_video
_LIMITS["gemini-omni-1.1-flash-preview"] = replace(_omni_video,
    max_prompt_tokens=131072,
    notes="Vertex preview (global, 2026-08-27): 131,072 input tokens, 10 images and 3 videos per prompt. "
          + _omni_video.notes,
    source_urls=("https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/omni-1-1-flash",
                 "https://ai.google.dev/gemini-api/docs/omni"))
# Lyria (Interactions API): text plus images in, MP3 (or WAV) and the lyrics out. Single turn.
_lyria = ModelLimits(
    modality="music", input_modalities=(_TEXT, _IMAGE),
    max_prompt_tokens=131072, max_input_images=10, max_output_count=1,
    supports_lyrics=True, supports_reference_audio=False, supports_continuation_audio=False,
    supported_sample_rates=(44100,),
    notes="Full-length songs whose length follows the prompt; up to 10 images; MP3 by default or WAV, 44.1 kHz "
          "stereo; lyrics in the prompt. Single-turn (no reference/continuation audio).",
    source_urls=("https://ai.google.dev/gemini-api/docs/music-generation",
                 "https://ai.google.dev/gemini-api/docs/models/lyria-3.5"),
)
_LIMITS["lyria-3.5"] = _lyria
_LIMITS["lyria-3-pro-preview"] = replace(_lyria,
    max_duration_seconds=184, supported_sample_rates=(), replacement="lyria-3.5",
    notes="Previous generation (preview, 2026-03-25): full songs up to 184 s; MP3. Single-turn.",
    source_urls=("https://ai.google.dev/gemini-api/docs/models/lyria-3-pro-preview",
                 "https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/lyria/lyria-3"))
_LIMITS["lyria-3-clip-preview"] = replace(_lyria,
    min_duration_seconds=30, max_duration_seconds=30, supported_sample_rates=(),
    notes="Preview (2026-03-25): 30-second clips; MP3. Single-turn.",
    source_urls=("https://ai.google.dev/gemini-api/docs/models/lyria-3-clip-preview",
                 "https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/lyria/lyria-3"))

# --------------------------------------------------------------------------- #
# xAI (Grok Imagine)
# --------------------------------------------------------------------------- #
_xai_img = ModelLimits(
    modality="image", input_modalities=(_TEXT,),
    supports_image_to_image=False, max_output_count=10,
    aspect_ratios=("1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "2:1",
                   "1:2", "19.5:9", "9:19.5", "20:9", "9:20", "auto"),
    notes="Resolution 1K or 2K (~2048px); output as url or base64.",
    source_urls=("https://docs.x.ai/developers/model-capabilities/images/generation",
                 "https://docs.x.ai/developers/models/grok-imagine-image"),
)
_LIMITS["grok-imagine-image"] = _xai_img
_LIMITS["grok-imagine-image-2.0"] = replace(_xai_img,
    aspect_ratios=(*_xai_img.aspect_ratios[:-1], "21:9", "5:2", "auto"),
    notes="Recommended xAI image model: resolution 1k (default)/2k; quality low/medium/auto (extra); n 1-10; "
          "the edits endpoint takes up to five source images (this adapter calls generations only).",
    source_urls=("https://docs.x.ai/developers/model-capabilities/images/generation",
                 "https://docs.x.ai/developers/models/grok-imagine-image-2.0"))
# grok-imagine-image-pro was retired on 2026-05-15 (its slug now redirects); grok-imagine-image-quality's 60-day
# notice began on 2026-09-02 and it retires on 2026-11-02, after which the slug is served by 2.0 at low quality.
_LIMITS["grok-imagine-image-pro"] = replace(_xai_img,
    retired_on="2026-05-15", replacement="grok-imagine-image-2.0",
    notes="Retired on 2026-05-15; the slug is an alias of grok-imagine-image-quality.",
    source_urls=("https://docs.x.ai/developers/migration/may-15-retirement",
                 "https://docs.x.ai/developers/models/grok-imagine-image-pro"))
_LIMITS["grok-imagine-image-quality"] = replace(_xai_img,
    deprecated_on="2026-09-02", retired_on="2026-11-02", replacement="grok-imagine-image-2.0",
    notes="Quality tier; resolution 1K/2K; output as url or base64. Retires on 2026-11-02.",
    source_urls=("https://docs.x.ai/developers/migration/imagine-image-quality-nov-2",
                 "https://docs.x.ai/developers/models/grok-imagine-image-pro"))
_XAI_VIDEO_RATIOS = ("1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "21:9", "5:2")
_LIMITS["grok-imagine-video"] = ModelLimits(
    modality="video", input_modalities=(_TEXT, _IMAGE, _VIDEO),
    supports_first_frame=True, supports_last_frame=None, supports_reference_image=True,
    supports_reference_video=None, max_input_images=7, min_duration_seconds=1, max_duration_seconds=15,
    aspect_ratios=_XAI_VIDEO_RATIOS, supports_audio_output=True,
    notes="Resolution 480p (default)/720p; i2v via image param; up to 7 reference images (reference-to-video "
          "<= 10 s, not combined with image); edits keep the input's length (<= 8.7 s). Sound by default.",
    source_urls=("https://docs.x.ai/developers/model-capabilities/video/generation",
                 "https://docs.x.ai/developers/model-capabilities/video/reference-to-video"),
)
_xai_video_15 = ModelLimits(
    modality="video", input_modalities=(_TEXT, _IMAGE),
    supports_first_frame=True, supports_reference_image=True, supports_reference_audio=False,
    max_input_images=14, min_duration_seconds=1, max_duration_seconds=15,
    aspect_ratios=_XAI_VIDEO_RATIOS, supports_audio_output=True,
    notes="Native 1080p text/image-to-video (reference-to-video up to 720p); up to 14 reference images and 4 "
          "keyframes; lip-synced speech, up to 3 preset voices via reference_audios (extra).",
    source_urls=("https://docs.x.ai/developers/model-capabilities/video/generation",
                 "https://docs.x.ai/developers/model-capabilities/video/reference-to-video",
                 "https://docs.x.ai/developers/models/grok-imagine-video-1.5"),
)
_LIMITS["grok-imagine-video-1.5"] = _xai_video_15
_LIMITS["grok-imagine-video-1.5-preview"] = replace(_xai_video_15,
    notes="Alias of grok-imagine-video-1.5. " + _xai_video_15.notes)
_LIMITS["grok-imagine-video-1.5-lite"] = replace(_xai_video_15,
    supports_reference_image=False, max_input_images=1,
    notes="Text/image-to-video; 1080p by upscaling 720p; lip-synced speech; no reference-to-video.",
    source_urls=("https://docs.x.ai/developers/model-capabilities/video/generation",
                 "https://docs.x.ai/developers/models/grok-imagine-video-1.5-lite"))

# --------------------------------------------------------------------------- #
# Stability (SD3.5 / SDXL / Stable-Image / Stable-Video)
# --------------------------------------------------------------------------- #
_sd35 = ModelLimits(
    modality="image", input_modalities=(_TEXT, _IMAGE),
    supports_image_to_image=True, max_prompt_chars=10000,
    aspect_ratios=("16:9", "1:1", "21:9", "2:3", "3:2", "4:5", "5:4", "9:16", "9:21"),
    notes="Supports negative_prompt (max 10000 chars); 1MP output (default 1024x1024).",
    source_urls=("https://platform.stability.ai/docs/api-reference",
                 "https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-diffusion-3-5-large.html"),
)
_LIMITS["sd3.5-large"] = _sd35
_LIMITS["sd3.5-medium"] = replace(_sd35,
    notes="Same endpoint/params as sd3.5-large; 1MP output.")
_LIMITS["sd3.5-large-turbo"] = replace(_sd35,
    notes="Distilled SD3.5 Large (4 steps) on the same endpoint; 1MP output.",
    source_urls=("https://docs.stability.ai/api-reference/generate/stable-diffusion-35",))
_LIMITS["sd3.5-flash"] = replace(_sd35,
    notes="Fastest SD3.5 model on the same endpoint (image-to-image works best at strength 0.94-0.97).",
    source_urls=("https://docs.stability.ai/api-reference/generate/stable-diffusion-35",))
_LIMITS["sdxl"] = ModelLimits(
    modality="image", input_modalities=(_TEXT, _IMAGE),
    supports_image_to_image=True, max_prompt_chars=10000,
    supported_sizes=("1024x1024", "1152x896", "896x1152", "1216x832",
                    "1344x768", "768x1344", "1536x640", "640x1536"),
    notes="Legacy v1 API; image-to-image via strength 0-1; cfg_scale 0-35.",
    source_urls=("https://platform.stability.ai/docs/api-reference",
                 "https://platform.stability.ai/pricing"),
)
_LIMITS["stable-image-core"] = ModelLimits(
    modality="image", input_modalities=(_TEXT,),
    supports_image_to_image=False, max_prompt_chars=10000,
    aspect_ratios=("16:9", "1:1", "21:9", "2:3", "3:2", "4:5", "5:4", "9:16", "9:21"),
    notes="Supports negative_prompt (max 10000 chars); output 1MP.",
    source_urls=("https://platform.stability.ai/docs/api-reference",
                 "https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-diffusion-stable-image-core-text-image-request-response.html"),
)
_LIMITS["stable-image-ultra"] = ModelLimits(
    modality="image", input_modalities=(_TEXT,),
    supports_image_to_image=False, max_prompt_chars=10000,
    aspect_ratios=("16:9", "1:1", "21:9", "2:3", "3:2", "4:5", "5:4", "9:16", "9:21"),
    notes="Powered by SD3.5 Large; output 1MP (1024x1024). The upstream also takes an image for image-to-image; "
          "this adapter sends text only.",
    source_urls=("https://docs.stability.ai/api-reference/generate/stable-image-ultra",
                 "https://platform.stability.ai/pricing"),
)
# Stability stopped serving the Stable Video Diffusion API on 2025-07-24 (no API replacement).
_svd = ModelLimits(
    modality="video", input_modalities=(_IMAGE,),
    supports_first_frame=True, max_fps=25, supports_audio_output=False,
    deprecated_on="2025-07-24", retired_on="2025-07-24",
    notes="Image-to-video only (init_image required, no text-to-video); silent; motion_bucket_id 1-255. "
          "The API was discontinued on 2025-07-24.",
    source_urls=("https://docs.stability.ai/release-notes",
                 "https://docs.api.nvidia.com/nim/reference/stabilityai-stable-video-diffusion"),
)
_LIMITS["stable-video-1-1"] = replace(_svd, notes="i2v only; native 1024x576; ~25 frames (~2s). " + _svd.notes)
_LIMITS["stable-video-1-0"] = replace(_svd, notes="i2v only; native 576x1024; superseded by 1.1. " + _svd.notes)
_LIMITS["stable-video-diffusion"] = _svd

# --------------------------------------------------------------------------- #
# FLUX-2 (runapi) — text-to-image vs remix (i2i)
# --------------------------------------------------------------------------- #
_flux_t2i = ModelLimits(
    modality="image", input_modalities=(_TEXT,),
    supports_image_to_image=False,
    notes="Width/height min 64 each, output up to 4MP; output_format jpeg/png/webp.",
    source_urls=("https://docs.bfl.ml/flux_2/flux2_overview",
                 "https://docs.bfl.ml/api-reference"),
)
_LIMITS["flux-2-flex-text-to-image"] = _flux_t2i
_LIMITS["flux-2-max-text-to-image"] = _flux_t2i
_LIMITS["flux-2-pro-text-to-image"] = _flux_t2i
_flux_remix = ModelLimits(
    modality="image", input_modalities=(_TEXT, _IMAGE),
    supports_image_to_image=True, max_input_images=8,
    notes="Remix via input_image_1..8; output up to 4MP.",
    source_urls=("https://docs.bfl.ml/flux_2/flux2_overview",
                 "https://docs.bfl.ml/api-reference"),
)
_LIMITS["flux-2-flex-remix-image"] = _flux_remix
_LIMITS["flux-2-max-remix-image"] = _flux_remix
_LIMITS["flux-2-pro-remix-image"] = _flux_remix

# --------------------------------------------------------------------------- #
# Volcengine Ark (Seedream image / Seedance video)
# --------------------------------------------------------------------------- #
# Lifecycle: Ark's model-deprecation notice (docs/82379/1350667). deprecated_on is the batch's notification
# date, retired_on its EOS (service offline) date; generation models are shut down, not migrated.
_ARK_MODELS = "https://docs.volcengine.com/docs/82379/1330310"
_ARK_DEPRECATIONS = "https://docs.volcengine.com/docs/82379/1350667"
_ARK_IMAGE_API = "https://docs.volcengine.com/docs/82379/1541523"
_ARK_VIDEO_API = "https://docs.volcengine.com/docs/82379/1520757"
_LIMITS["doubao-seedream-3-0-t2i-250415"] = ModelLimits(
    modality="image", input_modalities=(_TEXT,),
    supports_image_to_image=False,
    supported_sizes=("512x512", "1024x1024", "2048x2048"),
    deprecated_on="2026-04-03", retired_on="2026-05-11", replacement="doubao-seedream-5-0-pro-260628",
    notes="T2I only; size explicit W*H, sides in [512, 2048]; default 1024x1024. Offline since 2026-05-11.",
    source_urls=("https://docs.volcengine.com/docs/6492/2172373", _ARK_DEPRECATIONS),
)
_LIMITS["doubao-seedream-4-0-250828"] = ModelLimits(
    modality="image", input_modalities=(_TEXT, _IMAGE),
    supports_image_to_image=True, max_input_images=14, max_output_longest_side=4096,
    supported_sizes=("1K", "2K", "4K", "1024x1024", "1280x720", "2048x2048", "4096x4096"),
    deprecated_on="2026-09-22", retired_on="2026-11-24",
    notes="T2I + single/multi-ref image edit (up to 14 reference images; references + group outputs <= 15); "
          "1K/2K/4K or WxH with 921,600-16,777,216 px and ratio 1/16-16. Goes offline on 2026-11-24 (Ark names "
          "doubao-seedream-4-0-20260415 as the migration target).",
    source_urls=(_ARK_IMAGE_API, _ARK_DEPRECATIONS),
)
# Seedream 5.0 pro / flash: one image per request (no group generation), up to 10 reference images, layer split.
_seedream5 = ModelLimits(
    modality="image", input_modalities=(_TEXT, _IMAGE),
    supports_image_to_image=True, max_input_images=10, max_output_count=1,
    supported_sizes=("1K", "1.5K", "2K"),
    notes="T2I, single-image and multi-reference (up to 10) editing; one image per request; size 1K/1.5K/2K "
          "(default 2K) or WxH with 921,600-4,624,220 px and ratio 1/16-16; png/jpeg; layer_decomposition and "
          "transparent backgrounds via extra.",
    source_urls=(_ARK_IMAGE_API, _ARK_MODELS),
)
_LIMITS["doubao-seedream-5-0-pro-260628"] = _seedream5
_LIMITS["doubao-seedream-5-0-flash-260915"] = replace(_seedream5,
    notes="Faster Seedream 5.0 tier with pro's capabilities. " + _seedream5.notes)
_ARK_VIDEO_RATIOS = ("16:9", "4:3", "1:1", "3:4", "9:16", "21:9", "adaptive")
# Seedance 1.0 renders silent video (generate_audio is a Seedance 2.x parameter).
_seedance1 = ModelLimits(
    modality="video", input_modalities=(_TEXT, _IMAGE),
    supports_first_frame=True, supports_last_frame=True,
    supports_reference_video=False, supports_audio_output=False,
    min_duration_seconds=2, max_duration_seconds=12,
    aspect_ratios=_ARK_VIDEO_RATIOS,
    notes="Silent; 24 fps, mp4; duration 2-12s, default 5s; 480p/720p/1080p, default 1080p.",
    source_urls=(_ARK_VIDEO_API, _ARK_MODELS),
)
_LIMITS["doubao-seedance-1-0-pro-250528"] = replace(_seedance1,
    deprecated_on="2026-09-22", retired_on="2026-11-24", replacement="doubao-seedance-2-0-mini-260615",
    notes=_seedance1.notes + " Goes offline on 2026-11-24.",
    source_urls=(_ARK_VIDEO_API, _ARK_DEPRECATIONS))
_LIMITS["doubao-seedance-1-0-lite-i2v-250428"] = replace(_seedance1,
    max_input_images=4, deprecated_on="2026-04-03", retired_on="2026-05-11",
    replacement="doubao-seedance-2-0-fast-260128",
    notes="I2V only (first frame, first+last frame, reference images up to 4); 1080p unsupported in "
          "reference-image mode. Offline since 2026-05-11.",
    source_urls=(_ARK_DEPRECATIONS,))
# Seedance 2.x: one omni model per tier; first frame, first+last frame and multimodal references are three
# mutually exclusive modes, so the adapter keeps the frames and drops the references.
_seedance2 = ModelLimits(
    modality="video", input_modalities=(_TEXT, _IMAGE, _VIDEO, _AUDIO),
    supports_first_frame=True, supports_last_frame=True, supports_reference_image=True,
    supports_reference_video=True, supports_reference_audio=True, supports_audio_output=True,
    exclusive_frames_and_references=True, max_input_images=9,
    min_duration_seconds=4, max_duration_seconds=15, max_fps=24,
    aspect_ratios=_ARK_VIDEO_RATIOS, supported_sizes=("480p", "720p", "1080p", "4k"),
    notes="Omni (T2V/I2V/r2v/edit/extend) with generated audio; references: 9 images, 3 videos and 3 audio clips "
          "(2-15 s each, 15 s in total; audio needs an image or video). 4-15 s or -1 (model chooses); 4k is 10-bit.",
    source_urls=(_ARK_VIDEO_API, "https://docs.volcengine.com/docs/82379/2291680", _ARK_MODELS),
)
_LIMITS["doubao-seedance-2-0-260128"] = _seedance2
_LIMITS["doubao-seedance-2-0-fast-260128"] = replace(_seedance2,
    supported_sizes=("480p", "720p"), notes="Faster Seedance 2.0 tier; 480p/720p. " + _seedance2.notes)
_LIMITS["doubao-seedance-2-0-mini-260615"] = replace(_seedance2,
    supported_sizes=("480p", "720p"), notes="Lowest-cost Seedance 2.0 tier; 480p/720p. " + _seedance2.notes)
_LIMITS["doubao-seedance-2-5-260628"] = replace(_seedance2,
    max_input_images=30, max_duration_seconds=30, supported_sizes=("480p", "720p", "1080p"),
    notes="Omni with generated audio; 4-30 s or -1; references: 30 images, 10 videos and 10 audio clips (30 s in "
          "total; audio alone works); 1080p is 10-bit; mp4 or mov; draft (preview) mode. Edit, extend and "
          "first-frame tasks keep the input's ratio (ratio adaptive).",
    source_urls=(_ARK_VIDEO_API, "https://docs.volcengine.com/docs/82379/2607688", _ARK_MODELS))

# --------------------------------------------------------------------------- #
# Alibaba DashScope (Wanx / Qwen-Image / Wan2.x image + Wan2.1 video)
# --------------------------------------------------------------------------- #
_wanx21 = ModelLimits(
    modality="image", input_modalities=(_TEXT,),
    supports_image_to_image=False, max_prompt_chars=500, max_output_count=4,
    supported_sizes=("512x512", "1024x1024", "1440x1440"), max_output_longest_side=1440,
    notes="T2I only; size W*H, each side [512,1440], default 1024x1024.",
    source_urls=("https://help.aliyun.com/zh/model-studio/text-to-image-v2-api-reference",
                 "https://help.aliyun.com/en/model-studio/image-model"),
)
_LIMITS["wanx2.1-t2i-turbo"] = _wanx21
_LIMITS["wanx2.1-t2i-plus"] = replace(_wanx21, notes="T2I only (Wan 2.1 Pro); each side [512,1440].")
_LIMITS["wan2.2-t2i-flash"] = replace(_wanx21, notes="T2I only (Wan 2.2 Flash); each side [512,1440].")
_LIMITS["wan2.2-t2i-plus"] = replace(_wanx21, notes="T2I only (Wan 2.2 Plus); each side [512,1440].")
# Wan 2.5+ T2I: total pixels 1280*1280-1440*1440 at ratio 1:4-4:1, so the long side reaches 2880 (e.g. 768*2700).
_LIMITS["wan2.5-t2i-preview"] = replace(_wanx21,
    max_prompt_chars=2000, supported_sizes=("1280x1280", "1440x1440"), max_output_longest_side=2880,
    notes="T2I only; size W*H with total pixels 1280*1280-1440*1440 and ratio 1:4-4:1; default 1280*1280.")
_LIMITS["wan2.6-t2i"] = replace(_wanx21,
    max_prompt_chars=2100, max_output_longest_side=2880,
    supported_sizes=("1280x1280", "1104x1472", "1472x1104", "960x1696", "1696x960"),
    notes="T2I on the messages-shaped generation surface; size W*H with total pixels 1280*1280-1440*1440 and "
          "ratio 1:4-4:1; default 1280*1280; n 1-4.")
_qwen_image = ModelLimits(
    modality="image", input_modalities=(_TEXT, _IMAGE),
    supports_image_to_image=True, max_output_count=6, max_output_longest_side=2688,
    supported_sizes=("512x512", "1024x1024", "2048x2048", "2688x1536",
                    "1536x2688", "2368x1728", "1728x2368"),
    notes="T2I + image editing; messages-shaped multimodal input; default 2048x2048.",
    source_urls=("https://help.aliyun.com/en/model-studio/qwen-image-api",
                 "https://help.aliyun.com/en/model-studio/image-model"),
)
_LIMITS["qwen-image-2.0-pro"] = _qwen_image
_LIMITS["qwen-image-2.0"] = replace(_qwen_image, notes="Faster Qwen-Image 2.0 tier. " + _qwen_image.notes)
_LIMITS["qwen-image-3.0-pro"] = replace(_qwen_image,
    max_prompt_tokens=4500,
    notes="Flagship Qwen-Image: dense layouts, fine print, multilingual typefaces; up to 4.5k input tokens. "
          + _qwen_image.notes)
_LIMITS["qwen-image-3.0"] = replace(_qwen_image, notes="Faster Qwen-Image 3.0 tier. " + _qwen_image.notes)
_LIMITS["wan2.7-image-pro"] = ModelLimits(
    modality="image", input_modalities=(_TEXT, _IMAGE),
    supports_image_to_image=True, max_prompt_chars=5000, max_output_count=4,
    max_input_images=9, max_output_longest_side=4096,
    notes="Messages-shaped T2I + editing; up to 9 input images; 4096x4096 text-to-image, 2048x2048 editing.",
    source_urls=("https://help.aliyun.com/en/model-studio/image-model",
                 "https://help.aliyun.com/en/model-studio/wan-image-edit"),
)
_LIMITS["wan2.7-image"] = ModelLimits(
    modality="image", input_modalities=(_TEXT, _IMAGE),
    supports_image_to_image=True, max_prompt_chars=5000, max_output_count=4,
    max_input_images=9, max_output_longest_side=2048,
    notes="Messages-shaped T2I + editing; up to 9 input images; image sides [240, 8000].",
    source_urls=("https://help.aliyun.com/en/model-studio/wan-image-edit",
                 "https://help.aliyun.com/en/model-studio/image-model"),
)
_LIMITS["wan2.6-image"] = ModelLimits(
    modality="image", input_modalities=(_TEXT, _IMAGE),
    supports_image_to_image=True, max_prompt_chars=2000, max_output_count=4,
    max_output_longest_side=2048,
    notes="Messages-shaped T2I + editing; accepts 1-4 input images; total pixels up to 2048x2048 (1K/2K tiers); sides [240, 8000].",
    source_urls=("https://help.aliyun.com/en/model-studio/wan-image-edit",
                 "https://help.aliyun.com/en/model-studio/image-model"),
)
# Text-to-video, legacy protocol (Wan 2.6 and earlier): an exact size W*H from the model's list.
_WAN_T2V_LEGACY = "https://www.alibabacloud.com/help/en/model-studio/legacy-wan-text-to-video-api-reference"
_wanx21_t2v = ModelLimits(
    modality="video", input_modalities=(_TEXT,),
    max_prompt_chars=800, max_duration_seconds=5, supports_audio_output=False,
    notes="T2V, silent; 5s fixed, 30 fps, MP4 (H.264).",
    source_urls=("https://help.aliyun.com/en/model-studio/video-generate-edit-model", _WAN_T2V_LEGACY),
)
_LIMITS["wanx2.1-t2v-turbo"] = replace(_wanx21_t2v, supported_sizes=("480P", "720P"))
_LIMITS["wanx2.1-t2v-plus"] = replace(_wanx21_t2v, supported_sizes=("720P",))
_LIMITS["wan2.2-t2v-plus"] = replace(_wanx21_t2v,
    supported_sizes=("832x480", "480x832", "624x624", "1920x1080", "1080x1920", "1440x1440", "1632x1248",
                     "1248x1632"),
    notes="T2V, silent; 5s fixed; 480P or 1080P sizes (default 1920*1080); 30 fps, MP4.")
_LIMITS["wan2.5-t2v-preview"] = replace(_wanx21_t2v,
    max_prompt_chars=1500, min_duration_seconds=5, max_duration_seconds=10, supports_audio_output=True,
    supported_sizes=("832x480", "480x832", "624x624", "1280x720", "720x1280", "960x960", "1088x832",
                     "832x1088", "1920x1080", "1080x1920", "1440x1440", "1632x1248", "1248x1632"),
    notes="T2V with generated audio (or input.audio_url); 5 or 10 s; 480P/720P/1080P sizes; 30 fps, MP4.")
_LIMITS["wan2.6-t2v"] = replace(_wanx21_t2v,
    max_prompt_chars=1500, min_duration_seconds=2, max_duration_seconds=15, supports_audio_output=True,
    supported_sizes=("1280x720", "720x1280", "960x960", "1088x832", "832x1088", "1920x1080", "1080x1920",
                     "1440x1440", "1632x1248", "1248x1632"),
    notes="T2V with generated audio (or input.audio_url); 2-15 s; 720P/1080P sizes (default 1920*1080); "
          "multi-shot via shot_type=multi with prompt_extend; 30 fps, MP4.")
_LIMITS["wan2.7-t2v"] = ModelLimits(
    modality="video", input_modalities=(_TEXT,),
    max_prompt_chars=5000, min_duration_seconds=2, max_duration_seconds=15, supports_audio_output=True,
    supported_sizes=("720P", "1080P"), aspect_ratios=("16:9", "9:16", "1:1", "4:3", "3:4"),
    notes="T2V with generated audio (or input.audio_url, 2-30 s); resolution 720P/1080P (default) and ratio "
          "instead of size; 2-15 s; multi-shot from the prompt; 30 fps, MP4.",
    source_urls=("https://help.aliyun.com/en/model-studio/text-to-video-api-reference",
                 "https://help.aliyun.com/en/model-studio/video-generate-edit-model"),
)
# Image-to-video: the first frame as img_url and a resolution tier (no size or ratio: the output follows the image).
_WAN_I2V = "https://help.aliyun.com/en/model-studio/image-to-video-api-reference"
_wanx21_i2v = ModelLimits(
    modality="video", input_modalities=(_TEXT, _IMAGE),
    supports_first_frame=True, supports_last_frame=False, max_input_images=1,
    max_prompt_chars=800, min_duration_seconds=3, max_duration_seconds=5,
    supports_audio_output=False,
    notes="I2V (first-frame image via img_url), silent; 3-5 s; resolution 480P/720P (default 720P); 30 fps, MP4.",
    source_urls=("https://help.aliyun.com/en/model-studio/video-generate-edit-model", _WAN_I2V),
)
_LIMITS["wanx2.1-i2v-turbo"] = replace(_wanx21_i2v, supported_sizes=("480P", "720P"))
_LIMITS["wanx2.1-i2v-plus"] = replace(_wanx21_i2v, supported_sizes=("720P",),
    min_duration_seconds=5,
    notes="I2V (first-frame image via img_url), silent; duration fixed at 5s (not the turbo's 3-5s); 30 fps, MP4.",
)
_LIMITS["wan2.2-i2v-flash"] = replace(_wanx21_i2v, supported_sizes=("480P", "720P", "1080P"),
    min_duration_seconds=5,
    notes="I2V, silent; 5 s; resolution 480P/720P (default)/1080P; 30 fps, MP4.")
_LIMITS["wan2.2-i2v-plus"] = replace(_wanx21_i2v, supported_sizes=("480P", "1080P"),
    min_duration_seconds=5,
    notes="I2V, silent; 5 s; resolution 480P/1080P (default); 30 fps, MP4.")
_LIMITS["wan2.5-i2v-preview"] = replace(_wanx21_i2v, supported_sizes=("480P", "720P", "1080P"),
    max_prompt_chars=1500, min_duration_seconds=5, max_duration_seconds=10, supports_audio_output=True,
    notes="I2V with generated audio (or input.audio_url); 5 or 10 s; resolution 480P/720P/1080P (default).")
_LIMITS["wan2.6-i2v"] = replace(_wanx21_i2v, supported_sizes=("720P", "1080P"),
    max_prompt_chars=1500, min_duration_seconds=2, max_duration_seconds=15, supports_audio_output=True,
    notes="I2V with generated audio (or input.audio_url); 2-15 s; resolution 720P/1080P (default); "
          "multi-shot via shot_type.")
_LIMITS["wan2.6-i2v-flash"] = replace(_LIMITS["wan2.6-i2v"],
    notes="Faster Wan 2.6 I2V; parameters.audio=false renders silent video. 2-15 s; resolution 720P/1080P.")

_wan3_video = ModelLimits(
    modality="video", input_modalities=(_TEXT, _IMAGE, _VIDEO, _AUDIO),
    max_prompt_chars=20000, max_input_images=10,
    min_duration_seconds=2, max_duration_seconds=30, max_fps=30,
    supported_sizes=("480P", "720P", "1080P"),
    aspect_ratios=("adaptive", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"),
    supports_first_frame=True, supports_last_frame=True, supports_reference_image=True,
    supports_reference_video=True, supports_reference_audio=True, supports_audio_output=True,
    exclusive_frames_and_references=True, supports_negative_prompt=False,
    # The prompt sets the shots: "Shot N [start-end s]: ..." segments of 2-5 s each; the guide's examples run up to
    # eight shots in 25-30 s.
    max_shots=8,
    notes=(
        "Native media body (input.media: first_frame | last_frame | up to 10 reference_image, 5 reference_video and "
        "5 reference_audio, the reference clips at most 15 s together). A first or last frame cannot be combined with "
        "references. Duration 2-30 s (-1: the model chooses); 30 fps MP4 with sound unless audio=false. No negative "
        "prompt: write exclusions in the prompt. prompt_extend (on by default) rewrites the prompt."
    ),
    source_urls=(
        "https://www.alibabacloud.com/help/en/model-studio/wan3-video-generation-api-reference",
        "https://www.alibabacloud.com/help/en/model-studio/wan3-video-generation-prompt-guide",
    ),
)
_LIMITS["wan3.0-video"] = _wan3_video
_LIMITS["wan3.0-video-prime"] = replace(
    _wan3_video, notes="High-speed Wan 3.0 with the standard model's capabilities. " + _wan3_video.notes,
)
# wan2.7-i2v (native input.media): first_frame, first+last frame, optional driving_audio for lip-sync and timing.
_LIMITS["wan2.7-i2v"] = ModelLimits(
    modality="video", input_modalities=(_TEXT, _IMAGE, _AUDIO),
    max_prompt_chars=5000, max_input_images=2, min_duration_seconds=2, max_duration_seconds=15,
    supported_sizes=("720P", "1080P"),
    supports_first_frame=True, supports_last_frame=True, supports_reference_image=False,
    supports_reference_video=False, supports_reference_audio=True, supports_audio_output=True,
    notes="First frame or first+last frame, plus an optional driving audio (wav/mp3, 2-30 s) that drives lip-sync "
          "and timing (otherwise the model adds music or effects); resolution 720P/1080P (default), the ratio "
          "follows the frame; 2-15 s; negative prompt up to 500 chars; MP4.",
    source_urls=("https://www.alibabacloud.com/help/en/model-studio/image-to-video-general-api-reference",
                 "https://help.aliyun.com/en/model-studio/video-generate-edit-model"),
)
# HappyHorse 1.1 (native input.media): one model per mode; 3-15 s at 24 fps with sound; watermark on by default.
_HAPPYHORSE = "https://www.alibabacloud.com/help/en/model-studio/happyhorse-api-reference/"
_happyhorse = ModelLimits(
    modality="video", input_modalities=(_TEXT,),
    max_prompt_chars=5000, min_duration_seconds=3, max_duration_seconds=15, max_fps=24,
    supported_sizes=("480P", "720P", "1080P"),
    aspect_ratios=("16:9", "9:16", "1:1", "4:3", "3:4", "4:5", "5:4", "9:21", "21:9"),
    supports_audio_output=True,
    notes="Text-to-video; 480P/720P/1080P (default); 3-15 s; prompts up to 5000 characters (2500 Chinese); "
          "the 'Happy Horse' watermark is on unless watermark=false.",
    source_urls=("https://www.alibabacloud.com/help/en/model-studio/happyhorse-text-to-video-api-reference",
                 _HAPPYHORSE),
)
_LIMITS["happyhorse-1.1-t2v"] = _happyhorse
_LIMITS["happyhorse-1.1-i2v"] = replace(_happyhorse,
    input_modalities=(_TEXT, _IMAGE), supports_first_frame=True, supports_last_frame=False,
    supports_reference_image=False, max_input_images=1, aspect_ratios=(),
    notes="Image-to-video from exactly one first frame (ratio 1:2.5-2.5:1; the output keeps it); "
          "480P/720P/1080P; 3-15 s; watermark on by default.",
    source_urls=("https://www.alibabacloud.com/help/en/model-studio/happyhorse-image-to-video-api-reference",
                 _HAPPYHORSE))
_LIMITS["happyhorse-1.1-r2v"] = replace(_happyhorse,
    input_modalities=(_TEXT, _IMAGE), supports_reference_image=True, supports_first_frame=False,
    supports_last_frame=False, max_input_images=9,
    notes="Reference-to-video from 1-9 reference images cited in the prompt as [Image N] (shorter edge >= 400 px); "
          "480P/720P/1080P; 3-15 s; watermark on by default.",
    source_urls=("https://www.alibabacloud.com/help/en/model-studio/happyhorse-reference-to-video-api-reference",
                 _HAPPYHORSE))

# --------------------------------------------------------------------------- #
# MiniMax (music + H3 video)
# --------------------------------------------------------------------------- #
# The paid Music Generation API is closed to new MiniMax accounts from 2026-08-20; existing paying users keep it.
_MINIMAX_MUSIC_NOTE = " Not available to new MiniMax accounts from 2026-08-20."
_LIMITS["music-3.0"] = ModelLimits(
    modality="music", input_modalities=(_TEXT, _LYRICS),
    max_prompt_chars=2000, max_duration_seconds=300, supports_lyrics=True,
    max_output_count=1, supported_sample_rates=(16000, 24000, 32000, 44100),
    notes="Lyrics 1-3500 chars when non-instrumental; 300s is the model ceiling." + _MINIMAX_MUSIC_NOTE,
    source_urls=("https://platform.minimax.io/docs/api-reference/music-generation",),
)
_LIMITS["music-2.6"] = ModelLimits(
    modality="music", input_modalities=(_TEXT, _LYRICS),
    max_prompt_chars=2000, supports_lyrics=True, max_output_count=1,
    supported_sample_rates=(16000, 24000, 32000, 44100),
    notes="Constraints identical to music-3.0; no duration field exposed." + _MINIMAX_MUSIC_NOTE,
    source_urls=("https://platform.minimax.io/docs/api-reference/music-generation",),
)
_LIMITS["music-cover"] = ModelLimits(
    modality="music", input_modalities=(_TEXT, _LYRICS, _AUDIO),
    max_prompt_chars=300, max_duration_seconds=360, supports_lyrics=True,
    supports_reference_audio=True, max_output_count=1,
    supported_sample_rates=(16000, 24000, 32000, 44100),
    notes="Audio-to-song cover (audio_url/base64 6s-6min <=50MB or cover_feature_id); prompt 10-300 chars, "
          "lyrics 10-1000 (transcribed when omitted)." + _MINIMAX_MUSIC_NOTE,
    source_urls=("https://platform.minimax.io/docs/api-reference/music-generation",
                 "https://platform.minimax.io/docs/api-reference/music-cover-preprocess"),
)
# H3 family (POST /v2/video_generation): image-to-video and reference-to-video are mutually exclusive, so the
# adapter keeps the frames and drops the references.
_minimax_h3 = ModelLimits(
    modality="video", input_modalities=(_TEXT, _IMAGE, _VIDEO, _AUDIO),
    max_prompt_chars=7000, min_duration_seconds=4, max_duration_seconds=15,
    supports_first_frame=True, supports_last_frame=True, supports_reference_image=True,
    supports_reference_video=True, supports_reference_audio=True, exclusive_frames_and_references=True,
    max_input_images=9, max_output_count=1,
    aspect_ratios=("adaptive", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"), supported_sizes=("768P", "2K"),
    notes="Omni video model; content[] roles pick t2v/i2v/r2v; resolution 768P|2K. References: 9 images, 3 videos "
          "and 3 audio clips (2-15 s each, 15 s in total). Text-to-video needs a concrete ratio.",
    source_urls=("https://platform.minimax.io/docs/api-reference/video-generation-v2-create",),
)
_LIMITS["MiniMax-H3"] = _minimax_h3
_LIMITS["MiniMax-H3-Highspeed"] = replace(_minimax_h3, supported_sizes=("768P",),
    notes="Faster H3; 768P only; 4-15 s. References: 9 images, 3 videos and 3 audio clips (15 s in total).")
_LIMITS["MiniMax-H3-Lite"] = replace(_minimax_h3, supported_sizes=("480P", "768P"),
    notes="Lower-cost H3; 480P/768P (default 768P); 4-15 s; reference-to-video needs a concrete ratio.")
_LIMITS["MiniMax-H3-Max"] = replace(_minimax_h3, supported_sizes=("480P", "768P"), min_duration_seconds=5,
    notes="Speed-optimized H3; 480P/768P (default 768P); 5-15 s; extra.prompt_expansion_mode "
          "disabled|balanced|quality.")

# --------------------------------------------------------------------------- #
# ElevenLabs (music)
# --------------------------------------------------------------------------- #
_LIMITS["music_v1"] = ModelLimits(
    modality="music", input_modalities=(_TEXT,),
    max_prompt_chars=4100, min_duration_seconds=3, max_duration_seconds=600,
    supports_lyrics=True, max_output_count=1,
    supported_sample_rates=(8000, 16000, 22050, 24000, 32000, 44100, 48000),
    notes="API default during v1->v2 transition; lyrics via prompt or composition_plan.",
    source_urls=("https://elevenlabs.io/docs/api-reference/music/compose",),
)
_LIMITS["music_v2"] = ModelLimits(
    modality="music", input_modalities=(_TEXT, _AUDIO),
    max_prompt_chars=4100, min_duration_seconds=3, max_duration_seconds=600,
    supports_lyrics=True, supports_reference_audio=True, max_output_count=1,
    supported_sample_rates=(8000, 16000, 22050, 24000, 32000, 44100, 48000),
    notes="Default in Eleven Music UI; audio reference via composition_plan conditioning_ref.",
    source_urls=("https://elevenlabs.io/docs/api-reference/music/compose",
                 "https://elevenlabs.io/docs/eleven-creative/products/music"),
)
_LIMITS["music_v2_5"] = replace(_LIMITS["music_v2"],
    notes="Most advanced Eleven Music model: better quality and prompt adherence than music_v2; inpainting, audio "
          "reference (composition_plan conditioning_ref) and composition plans (chunks of 3-120 s).",
    source_urls=("https://elevenlabs.io/docs/api-reference/music/compose",
                 "https://elevenlabs.io/docs/overview/models"))

# --------------------------------------------------------------------------- #
# UdioAPI (chirp)
# --------------------------------------------------------------------------- #
# Every model takes a 5000-character Custom Mode prompt (Inspiration Mode: 3000), style <= 1000 and title <= 80.
# The current models are chirp-v6/-mini/-wild; legacy ids are still accepted but remapped: chirp-v5 and
# chirp-v5-5 run on chirp-v6, the older ones on chirp-v6-mini.
_chirp = ModelLimits(
    modality="music", input_modalities=(_TEXT,),
    max_prompt_chars=5000, supports_lyrics=True, max_output_count=2,
    notes="style <=1000 chars, title <=80; duration param ignored.",
    source_urls=("https://udioapi.pro/docs/v2-generate",),
)
_LIMITS["chirp-v6"] = replace(_chirp, notes="Current flagship; style <=1000 chars, title <=80; duration ignored.")
_LIMITS["chirp-v6-mini"] = replace(_chirp,
    notes="Lower-cost current model (the default when model is omitted); style <=1000, title <=80.")
_LIMITS["chirp-v6-wild"] = replace(_chirp,
    notes="Current chirp-v6 variant (billed like chirp-v6); style <=1000, title <=80.")
_LIMITS["chirp-v4-0"] = replace(_chirp, replacement="chirp-v6-mini",
    notes="Legacy id, served by chirp-v6-mini; style <=1000, title <=80.")
_LIMITS["chirp-v4-5"] = replace(_chirp, replacement="chirp-v6-mini",
    notes="Legacy id, served by chirp-v6-mini; style <=1000, title <=80.")
_LIMITS["chirp-v4-5-plus"] = replace(_chirp, replacement="chirp-v6-mini",
    notes="Legacy id, served by chirp-v6-mini; style <=1000, title <=80.")
_LIMITS["chirp-v5"] = replace(_chirp, replacement="chirp-v6",
    notes="Legacy id, served by chirp-v6; style <=1000, title <=80; duration ignored.")
_LIMITS["chirp-v5-5"] = replace(_chirp, replacement="chirp-v6",
    min_duration_seconds=10, max_duration_seconds=360,
    notes="Legacy id, served by chirp-v6; the only id accepting duration (10-360s, default 20, Custom Mode only).",
    source_urls=("https://udioapi.pro/docs/v2-generate", "https://udioapi.pro/docs/changelog"))

# --------------------------------------------------------------------------- #
# Mureka (song)
# --------------------------------------------------------------------------- #
# POST /v1/song/generate takes model auto | mureka-7.6 | mureka-o2 | mureka-8 | mureka-9 | mureka-9.5; lyrics
# <= 5000 characters, prompt <= 1024, n 1-3 (default 2).
_mureka_song = ModelLimits(
    modality="music", input_modalities=(_TEXT, _LYRICS, _AUDIO),
    max_prompt_chars=1024, supports_lyrics=True, supports_reference_audio=True,
    supports_continuation_audio=True, max_output_count=3,
    notes="Lyrics <= 5000 chars; continuation via separate /song/extend; reference via reference_id/vocal_id/melody_id.",
    source_urls=("https://platform.mureka.ai/docs/api/operations/post-v1-song-generate.html",
                 "https://platform.mureka.ai/docs/en/changelog.html"),
)
_LIMITS["mureka-9.5"] = replace(_mureka_song, notes="Latest Mureka model (2026-08-28). " + _mureka_song.notes)
_LIMITS["mureka-9"] = replace(_mureka_song, notes="Mureka 9 (2026-04-09). " + _mureka_song.notes)
_LIMITS["mureka-8"] = replace(_mureka_song, notes="Mureka 8 (2026-03-02). " + _mureka_song.notes)
_LIMITS["mureka-7.6"] = replace(_mureka_song, notes="Mureka 7.6 (2025-12-09). " + _mureka_song.notes)
_LIMITS["mureka-o2"] = replace(_mureka_song,
    notes="Reasoning music model (2025-12-09); no vocal_id or melody_id. Lyrics <= 5000 chars; reference via "
          "reference_id.")

# --------------------------------------------------------------------------- #
# ACE-Step (self-hosted / acemusic.ai cloud)
# --------------------------------------------------------------------------- #
# The official ace-step model cards do not publish hard numeric limits; the
# adapter accepts prompt + structured lyrics + a source/continuation audio
# (src_audio_path, task_type=cover) and an audio_duration knob. Modelled here
# from the adapter contract + the official acestep.sh defaults.
_acestep = ModelLimits(
    modality="music", input_modalities=(_TEXT, _LYRICS, _AUDIO),
    supports_lyrics=True, supports_reference_audio=True,
    supports_continuation_audio=True, max_output_count=4,
    notes="src_audio_path carries reference/continuation audio (task_type=cover); audio_duration controls length.",
    source_urls=("https://raw.githubusercontent.com/ace-step/ace-step-skills/main/skills/acestep/scripts/acestep.sh",
                 "https://huggingface.co/ACE-Step/acestep-v15-xl-base"),
)
# Checkpoints: acestep-v15-{turbo,sft,base} (2B DiT) and acestep-v15-xl-{turbo,sft,base} (4B DiT, ACE-Step 1.5 XL).
for _model in ("acestep-v15-turbo", "acestep-v15-sft", "acestep-v15-base", "acestep-v15-turbo-shift3",
               "acestep-v15-xl-turbo", "acestep-v15-xl-sft", "acestep-v15-xl-base", "ace-step-1.5"):
    _LIMITS[_model] = _acestep


# Speech constraints checked against official APIs on 2026-10-03.
_LIMITS["azure-tts"] = ModelLimits(
    modality="audio", input_modalities=(_TEXT,), supports_voice_cloning=False,
    supports_instructions=False, min_speed=0.5, max_speed=2,
    supported_file_formats=("mp3", "wav", "pcm", "opus"),
    supported_sample_rates=(8000, 16000, 22050, 24000, 44100, 48000),
    source_urls=("https://learn.microsoft.com/en-us/azure/ai-services/speech-service/speech-services-quotas-and-limits",
                 "https://learn.microsoft.com/en-us/azure/ai-services/speech-service/speech-synthesis-markup-voice",
                 "https://learn.microsoft.com/en-us/python/api/azure-cognitiveservices-speech/azure.cognitiveservices.speech.speechsynthesisoutputformat"),
    notes="Official Speech SDK; 64 KiB escaped synthesis message, up to 10 minutes of audio. "
          "MP3 and Ogg Opus have format-specific sample rates; language requires a locale and a compatible multilingual voice.",
)
_openai_speech = ModelLimits(
    modality="audio", input_modalities=(_TEXT,), max_prompt_chars=4096,
    supports_voice_cloning=True, max_voice_samples=1, supports_instructions=True,
    min_speed=0.25, max_speed=4,
    supported_file_formats=("mp3", "opus", "aac", "flac", "wav", "pcm"),
    source_urls=("https://developers.openai.com/api/reference/resources/audio/subresources/speech/methods/create",
                 "https://developers.openai.com/api/docs/guides/custom-voices"),
    notes="Cloning requires eligible account access, operator opt-in and a consent recording. PCM is 24 kHz mono.",
)
# OpenAI deprecated every /v1/audio/speech model on 2026-10-01 (shutdown 2027-01-06). Its suggested replacement,
# gpt-realtime-2.1-mini, runs on the Realtime API and is not served by the speech endpoint.
_openai_speech = replace(_openai_speech,
    deprecated_on="2026-10-01", retired_on="2027-01-06",
    notes=_openai_speech.notes + " Shuts down on 2027-01-06; OpenAI's replacement (gpt-realtime-2.1-mini) is "
          "Realtime-API only.",
    source_urls=(*_openai_speech.source_urls, _OPENAI_DEPRECATIONS,
                 "https://developers.openai.com/api/docs/guides/text-to-speech"))
for _model in ("gpt-4o-mini-tts", "gpt-4o-mini-tts-2025-12-15"):
    _LIMITS[_model] = _openai_speech
for _model in ("tts-1", "tts-1-hd"):
    _LIMITS[_model] = replace(_openai_speech, supports_instructions=False, supports_voice_cloning=False)
_eleven_speech = ModelLimits(
    modality="audio", input_modalities=(_TEXT,), supports_voice_cloning=True,
    max_voice_samples=10, supports_instructions=False, min_speed=0.7, max_speed=1.2,
    supported_file_formats=("mp3", "pcm", "wav", "opus"),
    supported_sample_rates=(8000, 16000, 22050, 24000, 32000, 44100, 48000),
    source_urls=("https://elevenlabs.io/docs/api-reference/text-to-speech/convert",
                 "https://elevenlabs.io/docs/api-reference/voices/ivc/create"),
    notes="Output encoding combinations and account plan restrictions apply; WAV wraps PCM as mono S16LE.",
)
for _model, _chars in (("eleven_v3", 5000), ("eleven_multilingual_v2", 10000),
                       ("eleven_flash_v2_5", 40000)):
    _LIMITS[_model] = replace(_eleven_speech, max_prompt_chars=_chars)
# ElevenLabs lists eleven_turbo_v2_5 as deprecated in favour of eleven_flash_v2_5 without a date; 2026-10-29 is
# the day that listing was checked.
_LIMITS["eleven_turbo_v2_5"] = replace(_eleven_speech, max_prompt_chars=40000,
    deprecated_on="2026-10-29", replacement="eleven_flash_v2_5",
    notes="Deprecated: functionally equivalent to eleven_flash_v2_5 with higher latency. " + _eleven_speech.notes,
    source_urls=(*_eleven_speech.source_urls, "https://elevenlabs.io/docs/overview/models"))
# Eleven v4 (2026-09-28) has only Stability and Similarity settings: no speed control and no SSML.
_eleven_v4 = replace(_eleven_speech, min_speed=1.0, max_speed=1.0,
    notes="No speed or style setting (speed stays 1.0) and no SSML; 90+ languages; audio tags. "
          + _eleven_speech.notes,
    source_urls=(*_eleven_speech.source_urls, "https://elevenlabs.io/docs/overview/models",
                 "https://elevenlabs.io/docs/overview/capabilities/text-to-speech/eleven-v4"))
_LIMITS["eleven_v4"] = replace(_eleven_v4, max_prompt_chars=10000)
_LIMITS["eleven_v4_turbo"] = replace(_eleven_v4,
    notes="Real-time Eleven v4 (~100 ms median inference). " + _eleven_v4.notes)
_minimax_speech = ModelLimits(
    modality="audio", input_modalities=(_TEXT,), max_prompt_chars=9999,
    supports_voice_cloning=True, max_voice_samples=1, supports_instructions=False,
    min_speed=0.5, max_speed=2,
    supported_file_formats=("mp3", "pcm", "flac", "wav", "opus"),
    supported_sample_rates=(8000, 16000, 22050, 24000, 32000, 44100),
    source_urls=("https://platform.minimax.io/docs/api-reference/speech-t2a-http",
                 "https://platform.minimax.io/docs/api-reference/voice-cloning-clone"),
    notes="Cloning accepts one MP3/M4A/WAV sample of 10–300 seconds, up to 20 MB; unused clones expire after seven days.",
)
for _model in ("speech-2.8-hd", "speech-2.8-turbo", "speech-2.6-hd", "speech-2.6-turbo",
               "speech-02-hd", "speech-02-turbo", "speech-01-hd", "speech-01-turbo"):
    _LIMITS[_model] = _minimax_speech


def limits_for(model: str, modality: str) -> ModelLimits:
    """Return the limits catalogue entry for ``model``.

    Falls back to a maximally-permissive entry when the model has no explicit
    entry (e.g. an operator-pinned brand-new id), so routing never rejects a
    configured model purely for lacking documented limits. The modality is
    forced to the caller's requested modality in that fallback.
    """
    entry = _LIMITS.get(model)
    if entry is None:
        return _permissive(modality)
    if entry.modality != modality:
        # An entry exists but for a different modality (e.g. an omni model id
        # listed under one modality). Return a permissive modality-correct view
        # rather than routing against the wrong modality's limits.
        return _permissive(modality)
    return entry


def builtin_limits() -> dict[str, ModelLimits]:
    """The built-in catalogue (a copy; :class:`mm_gateway.models.catalog.Catalog` merges overrides)."""
    return dict(_LIMITS)


__all__ = ["Lifecycle", "ModelLimits", "builtin_limits", "limits_for", "utc_today"]
