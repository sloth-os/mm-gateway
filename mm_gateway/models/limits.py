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
from typing import Any

# The part ``type`` values the public request envelope accepts, per modality.
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
    notes: str = ""
    source_urls: tuple[str, ...] = ()

    def to_public_dict(self) -> dict[str, Any]:
        """Return the additive, client-facing limits object.

        Omits ``None``/empty fields so the spec stays small; clients ignore
        unknown members, so adding fields later is safe.
        """
        out: dict[str, Any] = {"modality": self.modality}
        if self.input_modalities:
            out["input_modalities"] = list(self.input_modalities)
        for key in (
            "max_prompt_chars", "max_prompt_tokens", "max_input_images",
            "max_output_count", "max_duration_seconds", "min_duration_seconds",
            "max_fps", "max_output_longest_side",
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
        for key in (
            "supports_image_to_image", "supports_first_frame", "supports_last_frame",
            "supports_reference_video", "supports_reference_audio",
            "supports_continuation_audio", "supports_lyrics", "supports_reference_image",
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
# gpt-image-1 / -mini: multimodal (text+image edit), sizes incl. auto, n up to 10.
_openai_img = ModelLimits(
    modality="image", input_modalities=(_TEXT, _IMAGE),
    supports_image_to_image=True, max_prompt_chars=32000, max_output_count=10,
    supported_sizes=("1024x1024", "1024x1536", "1536x1024", "auto"),
    notes="Output always base64; supports edits endpoint with masks.",
    source_urls=("https://platform.openai.com/docs/guides/image-generation",
                 "https://platform.openai.com/docs/api-reference/images/create"),
)
_LIMITS["gpt-image-1"] = _openai_img
_LIMITS["gpt-image-1-mini"] = replace(_openai_img,
    notes="Smaller/faster variant of gpt-image-1 with the same API surface.")
_LIMITS["gpt-image-2"] = ModelLimits(
    modality="image", input_modalities=(_TEXT, _IMAGE),
    supports_image_to_image=True, max_prompt_chars=32000, max_output_count=10,
    supported_sizes=("1024x1024", "1536x1024", "1024x1536", "2048x2048",
                     "2048x1152", "3840x2160", "2160x3840", "auto"),
    max_output_longest_side=3840,
    notes="Newest model; arbitrary WxH (both divisible by 16, ratio 1:3-3:1).",
    source_urls=("https://platform.openai.com/docs/guides/image-generation",
                 "https://platform.openai.com/docs/api-reference/images/create"),
)
_LIMITS["dall-e-2"] = ModelLimits(
    modality="image", input_modalities=(_TEXT, _IMAGE),
    supports_image_to_image=True, max_prompt_chars=1000, max_output_count=10,
    supported_sizes=("256x256", "512x512", "1024x1024"),
    notes="Supports edits + variations with image/mask input; deprecated.",
    source_urls=("https://platform.openai.com/docs/api-reference/images/create",
                 "https://platform.openai.com/docs/models/dall-e-2"),
)
_LIMITS["dall-e-3"] = ModelLimits(
    modality="image", input_modalities=(_TEXT,),
    supports_image_to_image=False, max_prompt_chars=4000, max_output_count=1,
    supported_sizes=("1024x1024", "1792x1024", "1024x1792"),
    notes="Text-only (no image input/edit); returns revised_prompt; deprecated.",
    source_urls=("https://platform.openai.com/docs/api-reference/images/create",
                 "https://platform.openai.com/docs/models/dall-e-3"),
)
# Sora 2: durations 4/8/12/16/20s; i2v via a single input_reference image.
_LIMITS["sora-2"] = ModelLimits(
    modality="video", input_modalities=(_TEXT, _IMAGE),
    supports_first_frame=True, supports_last_frame=False,
    supports_reference_video=False, supports_reference_audio=False,
    max_input_images=1, min_duration_seconds=4, max_duration_seconds=20,
    supported_sizes=("720x1280", "1280x720"),
    notes="Durations 4/8/12/16/20s (default 4); 720p only.",
    source_urls=("https://platform.openai.com/docs/guides/video-generation",
                 "https://developers.openai.com/cookbook/examples/sora/sora2_prompting_guide"),
)
_LIMITS["sora-2-pro"] = ModelLimits(
    modality="video", input_modalities=(_TEXT, _IMAGE),
    supports_first_frame=True, supports_last_frame=False,
    supports_reference_video=False, supports_reference_audio=False,
    max_input_images=1, min_duration_seconds=4, max_duration_seconds=20,
    supported_sizes=("720x1280", "1280x720", "1024x1792", "1792x1024",
                    "1080x1920", "1920x1080"),
    notes="Adds 1080p and 1024x1792/1792x1024 exports.",
    source_urls=("https://platform.openai.com/docs/guides/video-generation",
                 "https://developers.openai.com/cookbook/examples/sora/sora2_prompting_guide"),
)

# --------------------------------------------------------------------------- #
# Google (Imagen / Gemini-Flash-Image / Veo / Lyria)
# --------------------------------------------------------------------------- #
_imagen = ModelLimits(
    modality="image", input_modalities=(_TEXT,),
    supports_image_to_image=False, max_prompt_tokens=480, max_output_count=4,
    aspect_ratios=("1:1", "3:4", "4:3", "9:16", "16:9"),
    notes="Text-only (English prompts); sampleCount 1-4.",
    source_urls=("https://ai.google.dev/gemini-api/docs/imagen",
                 "https://docs.cloud.google.com/vertex-ai/generative-ai/docs/image/generate-images"),
)
_LIMITS["imagen-4.0-generate-001"] = _imagen
_LIMITS["imagen-3.0-generate-001"] = replace(_imagen,
    notes="Text-only; aspect ratios 1:1/3:4/4:3/9:16/16:9; 1K only.")
_LIMITS["gemini-2.5-flash-image"] = ModelLimits(
    modality="image", input_modalities=(_TEXT, _IMAGE),
    supports_image_to_image=True, max_prompt_tokens=65536,
    max_input_images=3, max_output_count=10, max_output_longest_side=1536,
    aspect_ratios=("1:1", "3:2", "2:3", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"),
    notes="a.k.a. nano-banana; up to 3 input images, up to 10 output images.",
    source_urls=("https://ai.google.dev/gemini-api/docs/models/gemini-2.5-flash-image",
                 "https://ai.google.dev/gemini-api/docs/image-generation"),
)
_LIMITS["veo-2.0-generate-001"] = ModelLimits(
    modality="video", input_modalities=(_TEXT, _IMAGE),
    supports_first_frame=True, supports_last_frame=True,
    supports_reference_video=False, max_input_images=1, max_output_count=2,
    min_duration_seconds=5, max_duration_seconds=8,
    aspect_ratios=("16:9", "9:16"),
    notes="Silent (no audio); durations 5/6/8s; 720p only.",
    source_urls=("https://ai.google.dev/gemini-api/docs/models/veo-2.0-generate-001",
                 "https://ai.google.dev/gemini-api/docs/veo"),
)
_LIMITS["veo-3.0-generate-001"] = ModelLimits(
    modality="video", input_modalities=(_TEXT, _IMAGE),
    supports_first_frame=True, supports_last_frame=True,
    supports_reference_video=False, max_input_images=1, max_output_count=1,
    max_prompt_tokens=1024, min_duration_seconds=4, max_duration_seconds=8,
    aspect_ratios=("16:9", "9:16"),
    notes="Always generates audio; durations 4/6/8s; 720p/1080p; 1 video per request.",
    source_urls=("https://ai.google.dev/gemini-api/docs/veo",
                 "https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/veo/3-0-generate"),
)
_LIMITS["veo-3.1-generate-preview"] = ModelLimits(
    modality="video", input_modalities=(_TEXT, _IMAGE, _VIDEO),
    supports_first_frame=True, supports_last_frame=True,
    supports_reference_video=True, max_input_images=3, max_output_count=1,
    max_prompt_tokens=1024, min_duration_seconds=4, max_duration_seconds=8,
    aspect_ratios=("16:9", "9:16"),
    notes="Audio on; supports video extension + up to 3 reference images; 1 video per request.",
    source_urls=("https://ai.google.dev/gemini-api/docs/veo",
                 "https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/veo/3-1-generate"),
)
_LIMITS["lyria-3"] = ModelLimits(
    modality="music", input_modalities=(_TEXT, _IMAGE),
    supports_image_to_image=None, max_prompt_tokens=131072, max_input_images=10,
    max_output_count=1, min_duration_seconds=30, max_duration_seconds=180,
    supports_lyrics=True, supports_reference_audio=False,
    supports_continuation_audio=False,
    notes="Single-turn (no reference/continuation audio); structured lyrics supported.",
    source_urls=("https://ai.google.dev/gemini-api/docs/music-generation",
                 "https://ai.google.dev/gemini-api/docs/models/lyria-3-pro-preview"),
)

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
_LIMITS["grok-imagine-image-pro"] = replace(_xai_img,
    notes="Pro tier; resolution 1K/2K; text-to-image takes no image input.")
_LIMITS["grok-imagine-image-quality"] = replace(_xai_img,
    notes="Quality tier; resolution 1K/2K; output as url or base64.")
_LIMITS["grok-imagine-video"] = ModelLimits(
    modality="video", input_modalities=(_TEXT, _IMAGE, _VIDEO),
    supports_first_frame=True, supports_last_frame=None,
    supports_reference_video=None, min_duration_seconds=1, max_duration_seconds=15,
    aspect_ratios=("1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"),
    notes="Resolution 480p/720p only (no 1080p); i2v via image param.",
    source_urls=("https://docs.x.ai/developers/model-capabilities/video/generation",
                 "https://docs.x.ai/developers/models/grok-imagine-video"),
)
_LIMITS["grok-imagine-video-1.5-preview"] = ModelLimits(
    modality="video", input_modalities=(_TEXT, _IMAGE, _AUDIO),
    supports_first_frame=True, min_duration_seconds=1, max_duration_seconds=15,
    supports_reference_audio=True,
    aspect_ratios=("1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"),
    notes="480p/720p/1080p; up to 7 reference images and 3 preset voices.",
    source_urls=("https://docs.x.ai/developers/model-capabilities/video/generation",
                 "https://docs.x.ai/developers/models/grok-imagine-video-1.5-preview"),
)

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
    notes="Powered by SD3.5 Large; output 1MP (1024x1024).",
    source_urls=("https://platform.stability.ai/docs/api-reference",
                 "https://platform.stability.ai/pricing"),
)
_svd = ModelLimits(
    modality="video", input_modalities=(_IMAGE,),
    supports_first_frame=True, max_fps=25,
    notes="Image-to-video only (init_image required, no text-to-video); motion_bucket_id 1-255.",
    source_urls=("https://platform.stability.ai/docs/api-reference",
                 "https://docs.api.nvidia.com/nim/reference/stabilityai-stable-video-diffusion"),
)
_LIMITS["stable-video-1-1"] = replace(_svd, notes="i2v only; native 1024x576; ~25 frames (~2s).")
_LIMITS["stable-video-1-0"] = replace(_svd, notes="i2v only; native 576x1024; superseded by 1.1.")
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
_LIMITS["doubao-seedream-3-0-t2i-250415"] = ModelLimits(
    modality="image", input_modalities=(_TEXT,),
    supports_image_to_image=False,
    supported_sizes=("512x512", "1024x1024", "2048x2048"),
    notes="T2I only; size explicit W*H, sides in [512, 2048]; default 1024x1024.",
    source_urls=("https://docs.volcengine.com/docs/6492/2172373",
                 "https://www.volcengine.com/docs/82379/1330310"),
)
_LIMITS["doubao-seedream-4-0-t2i-250828"] = ModelLimits(
    modality="image", input_modalities=(_TEXT, _IMAGE),
    supports_image_to_image=True, max_input_images=14, max_output_longest_side=4096,
    supported_sizes=("1024x1024", "1280x720", "2048x2048", "4096x4096"),
    notes="T2I + single/multi-ref image edit (up to 14 reference images; ref+output <= 15); sides [1280, 4096].",
    source_urls=("https://docs.volcengine.com/docs/6492/2172373",
                 "https://developer.volcengine.com/articles/7553203404664176650"),
)
_seedance1 = ModelLimits(
    modality="video", input_modalities=(_TEXT, _IMAGE),
    supports_first_frame=True, supports_last_frame=True,
    supports_reference_video=False,
    min_duration_seconds=2, max_duration_seconds=12,
    aspect_ratios=("16:9", "4:3", "1:1", "3:4", "9:16", "21:9", "adaptive"),
    notes="24 fps, mp4; duration 2-12s, default 5s; default resolution 1080p.",
    source_urls=("https://www.volcengine.com/docs/6492/2165104",
                 "https://www.volcengine.com/docs/82379/1330310"),
)
_LIMITS["doubao-seedance-1-0-pro-250528"] = _seedance1
_LIMITS["doubao-seedance-1-0-pro-i2v-250528"] = replace(_seedance1,
    notes="I2V-only pro variant; shares Seedance 1.0 specs.")
_LIMITS["doubao-seedance-1-0-lite-i2v-250428"] = replace(_seedance1,
    max_input_images=4,
    notes="I2V only (first frame, first+last frame, reference images up to 4); 1080p unsupported in reference-image mode.")
_LIMITS["doubao-seedance-2-0-260128"] = ModelLimits(
    modality="video", input_modalities=(_TEXT, _IMAGE, _VIDEO, _AUDIO),
    supports_first_frame=True, supports_last_frame=True,
    supports_reference_video=True, supports_reference_audio=True,
    min_duration_seconds=4, max_duration_seconds=15,
    aspect_ratios=("16:9", "4:3", "1:1", "3:4", "9:16", "21:9", "adaptive"),
    notes="Omni (T2V/I2V/r2v/edit/extend); up to 3 reference videos + 3 reference audios; 4k is 10-bit.",
    source_urls=("https://docs.volcengine.com/docs/6492/2595411",
                 "https://www.volcengine.com/docs/82379/1330310"),
)

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
_LIMITS["wanx2.1-t2i-flash"] = replace(_wanx21, notes="Wan 2.1 Flash variant; shares the wanx2.1 family limits.")
_LIMITS["qwen-image-2.0-pro"] = ModelLimits(
    modality="image", input_modalities=(_TEXT, _IMAGE),
    supports_image_to_image=True, max_output_count=6, max_output_longest_side=2688,
    supported_sizes=("512x512", "1024x1024", "2048x2048", "2688x1536",
                    "1536x2688", "2368x1728", "1728x2368"),
    notes="T2I + image editing; messages-shaped multimodal input; default 2048x2048.",
    source_urls=("https://help.aliyun.com/en/model-studio/qwen-image-api",
                 "https://help.aliyun.com/en/model-studio/image-model"),
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
_wanx21_t2v = ModelLimits(
    modality="video", input_modalities=(_TEXT,),
    max_duration_seconds=5,
    notes="T2V, silent; 5s fixed, 30 fps, MP4 (H.264).",
    source_urls=("https://help.aliyun.com/en/model-studio/video-generate-edit-model",
                 "https://help.aliyun.com/en/model-studio/text-to-video-guide"),
)
_LIMITS["wanx2.1-t2v-turbo"] = replace(_wanx21_t2v, supported_sizes=("480P", "720P"))
_LIMITS["wanx2.1-t2v-plus"] = replace(_wanx21_t2v, supported_sizes=("720P",))
_wanx21_i2v = ModelLimits(
    modality="video", input_modalities=(_TEXT, _IMAGE),
    supports_first_frame=True, min_duration_seconds=3, max_duration_seconds=5,
    notes="I2V (first-frame image via img_url), silent; 30 fps, MP4.",
    source_urls=("https://help.aliyun.com/en/model-studio/video-generate-edit-model",
                 "https://help.aliyun.com/en/model-studio/text-to-video-guide"),
)
_LIMITS["wanx2.1-i2v-turbo"] = replace(_wanx21_i2v, supported_sizes=("480P", "720P"))
_LIMITS["wanx2.1-i2v-plus"] = replace(_wanx21_i2v, supported_sizes=("720P",),
    min_duration_seconds=5,
    notes="I2V (first-frame image via img_url), silent; duration fixed at 5s (not the turbo's 3-5s); 30 fps, MP4.",
)

# --------------------------------------------------------------------------- #
# MiniMax (music + H3 video)
# --------------------------------------------------------------------------- #
_LIMITS["music-3.0"] = ModelLimits(
    modality="music", input_modalities=(_TEXT, _LYRICS),
    max_prompt_chars=2000, max_duration_seconds=300, supports_lyrics=True,
    max_output_count=1, supported_sample_rates=(16000, 24000, 32000, 44100),
    notes="Lyrics 1-3500 chars when non-instrumental; 300s is the model ceiling.",
    source_urls=("https://platform.minimax.io/docs/api-reference/music-generation",),
)
_LIMITS["music-2.6"] = ModelLimits(
    modality="music", input_modalities=(_TEXT, _LYRICS),
    max_prompt_chars=2000, supports_lyrics=True, max_output_count=1,
    supported_sample_rates=(16000, 24000, 32000, 44100),
    notes="Constraints identical to music-3.0; no duration field exposed.",
    source_urls=("https://platform.minimax.io/docs/api-reference/music-generation",),
)
_LIMITS["music-cover"] = ModelLimits(
    modality="music", input_modalities=(_TEXT, _LYRICS, _AUDIO),
    max_prompt_chars=300, max_duration_seconds=360, supports_lyrics=True,
    supports_reference_audio=True, max_output_count=1,
    supported_sample_rates=(16000, 24000, 32000, 44100),
    notes="Audio-to-song cover (audio_url/base64 6s-6min <=50MB or cover_feature_id).",
    source_urls=("https://platform.minimax.io/docs/api-reference/music-generation",
                 "https://platform.minimax.io/docs/api-reference/music-cover-preprocess"),
)
_LIMITS["MiniMax-H3"] = ModelLimits(
    modality="video", input_modalities=(_TEXT, _IMAGE, _VIDEO, _AUDIO),
    max_prompt_chars=7000, min_duration_seconds=4, max_duration_seconds=15,
    supports_reference_audio=True, supports_reference_image=True,
    max_output_count=1, supports_first_frame=True,
    notes="Omni video model; content[] roles pick t2v/i2v/r2v; resolution 768P|2K.",
    source_urls=("https://platform.minimax.io/docs/api-reference/video-generation-v2-create",),
)

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

# --------------------------------------------------------------------------- #
# UdioAPI (chirp)
# --------------------------------------------------------------------------- #
_chirp_v4 = ModelLimits(
    modality="music", input_modalities=(_TEXT,),
    max_prompt_chars=3000, supports_lyrics=True, max_output_count=2,
    notes="style <=200 chars, title <=80; duration param ignored.",
    source_urls=("https://udioapi.pro/docs/v2-generate",),
)
_chirp_v45 = ModelLimits(
    modality="music", input_modalities=(_TEXT,),
    max_prompt_chars=5000, supports_lyrics=True, max_output_count=2,
    notes="style <=1000 chars, title <=80; duration param ignored.",
    source_urls=("https://udioapi.pro/docs/v2-generate",),
)
_LIMITS["chirp-v4-0"] = _chirp_v4
_LIMITS["chirp-v4-5"] = _chirp_v45
_LIMITS["chirp-v4-5-plus"] = replace(_chirp_v45, notes="Same tier limits as chirp-v4-5; duration ignored.")
_LIMITS["chirp-v5"] = replace(_chirp_v45, notes="style <=1000, title <=80; duration ignored (v5-5-only feature).")
_LIMITS["chirp-v5-5"] = ModelLimits(
    modality="music", input_modalities=(_TEXT,),
    max_prompt_chars=5000, min_duration_seconds=10, max_duration_seconds=360,
    supports_lyrics=True, max_output_count=2,
    notes="Only model accepting duration (10-360s, default 20, Custom Mode only).",
    source_urls=("https://udioapi.pro/docs/v2-generate", "https://udioapi.pro/docs/changelog"),
)

# --------------------------------------------------------------------------- #
# Mureka (song)
# --------------------------------------------------------------------------- #
_mureka_song = ModelLimits(
    modality="music", input_modalities=(_TEXT, _LYRICS, _AUDIO),
    max_prompt_chars=1024, supports_lyrics=True, supports_reference_audio=True,
    supports_continuation_audio=True, max_output_count=3,
    notes="Continuation via separate /song/extend; reference via reference_id/vocal_id/melody_id.",
    source_urls=("https://platform.mureka.ai/docs/en/changelog.html",
                 "https://docs.cloudsway.net/maasapi/api-reference/music/mureka/"),
)
_LIMITS["mureka-song-1"] = _mureka_song
_LIMITS["mureka-song-1.5"] = replace(_mureka_song,
    notes="Gateway-only alias; same upstream endpoint as mureka-song-1.")

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
    source_urls=("https://raw.githubusercontent.com/ace-step/ace-step-skills/main/skills/acestep/scripts/acestep.sh",),
)
_LIMITS["acestep-v15-turbo"] = _acestep
_LIMITS["acestep-v15-xl-turbo"] = _acestep
_LIMITS["acestep-v15-base"] = _acestep
_LIMITS["acestep-v15-turbo-shift3"] = _acestep
_LIMITS["ace-step-1.5"] = _acestep


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


__all__ = ["ModelLimits", "limits_for"]
