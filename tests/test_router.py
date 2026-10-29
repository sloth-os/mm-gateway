"""Tests for the model-limits catalogue and the auto-router."""

from __future__ import annotations

import pytest

from mm_gateway.config import BackendConfig, KeyConfig, Settings
from mm_gateway.core.exceptions import GatewayError
from mm_gateway.models.limits import limits_for
from mm_gateway.registry import Registry
from mm_gateway.router import profile_for, score
from mm_gateway.schemas.image import UnifiedImageRequest
from mm_gateway.schemas.image import text_part as img_text
from mm_gateway.schemas.image import image_part as img_image
from mm_gateway.schemas.video import UnifiedVideoRequest
from mm_gateway.schemas.video import text_part as vid_text
from mm_gateway.schemas.video import image_part as vid_image
from mm_gateway.schemas.music import UnifiedMusicRequest
from mm_gateway.schemas.music import text_part as mus_text


# --------------------------------------------------------------------------- #
# Catalogue lookups
# --------------------------------------------------------------------------- #

def test_known_models_resolve_to_documented_limits():
    gpt = limits_for("gpt-image-1", "image")
    assert gpt.modality == "image"
    assert "text" in gpt.input_modalities and "image" in gpt.input_modalities
    assert gpt.supports_image_to_image is True
    assert gpt.max_output_count == 10


def test_dall_e_3_is_text_only():
    d3 = limits_for("dall-e-3", "image")
    assert d3.input_modalities == ("text",)
    assert d3.supports_image_to_image is False


def test_unknown_model_falls_back_to_permissive_entry():
    permissive = limits_for("brand-new-id", "image")
    assert permissive.modality == "image"
    assert permissive.input_modalities == ()  # no documented constraint
    assert "permissive" in permissive.notes


def test_limits_public_dict_omits_empty_fields():
    payload = limits_for("gpt-image-1", "image").to_public_dict()
    # Documented fields present; per-modality fields the model lacks absent.
    assert payload["modality"] == "image"
    assert "max_prompt_chars" in payload
    # No video duration on an image model.
    assert "max_duration_seconds" not in payload
    assert "supports_first_frame" not in payload


def test_doc_grounded_limits_pinned_against_official_docs():
    # Values below were doc-grounded against each model's official docs; these
    # assertions pin them so a regression (e.g. re-copying a sibling's value)
    # turns the suite red rather than silently mis-routing.
    veo3 = limits_for("veo-3.0-generate-001", "video")
    assert veo3.max_output_count == 1  # Veo 3: 1 video per request (not 4).
    veo31 = limits_for("veo-3.1-generate-preview", "video")
    assert veo31.max_output_count == 1  # Veo 3.1: 1 video; 3 ref images guide it.
    assert veo31.max_input_images == 3
    seedream4 = limits_for("doubao-seedream-4-0-250828", "image")
    assert seedream4.max_input_images == 14  # Seedream 4.0: up to 14 ref images.
    qwen = limits_for("qwen-image-2.0-pro", "image")
    assert qwen.max_output_longest_side == 2688  # supported_sizes lists 2688x1536.
    assert "2688x1536" in qwen.supported_sizes
    wan26 = limits_for("wan2.6-image", "image")
    assert wan26.max_output_longest_side == 2048  # 2K tier (~2048x2048), not 1440.
    i2v_plus = limits_for("wanx2.1-i2v-plus", "video")
    assert i2v_plus.min_duration_seconds == 5  # i2v-plus: fixed 5s (turbo is 3-5).
    assert i2v_plus.max_duration_seconds == 5
    # Wan 3.0: 2-30 s, 10 reference images, frames exclusive with references, no negative prompt.
    for model in ("wan3.0-video", "wan3.0-video-prime"):
        wan3 = limits_for(model, "video")
        assert (wan3.min_duration_seconds, wan3.max_duration_seconds) == (2, 30)
        assert wan3.max_input_images == 10
        assert wan3.max_prompt_chars == 20000
        assert wan3.exclusive_frames_and_references is True
        assert wan3.supports_negative_prompt is False
        assert wan3.supports_audio_output is True
        public = wan3.to_public_dict()
        assert public["exclusive_frames_and_references"] is True
        assert public["supports_negative_prompt"] is False


def test_lifecycles_pinned_against_official_deprecation_notices():
    # (model, modality, deprecated_on, retired_on, replacement) from each provider's deprecation notice.
    cases = [
        ("gpt-image-1", "image", "2026-04-22", "2026-10-23", "gpt-image-2.5-sunburst"),
        ("gpt-image-1-mini", "image", "2026-06-02", "2026-12-01", "gpt-image-2.5-flare"),
        ("dall-e-3", "image", "2025-11-14", "2026-05-12", "gpt-image-2"),
        ("gpt-4o-mini-tts", "audio", "2026-10-01", "2027-01-06", None),
        ("tts-1-hd", "audio", "2026-10-01", "2027-01-06", None),
        ("imagen-4.0-generate-001", "image", "2026-06-15", "2026-08-17", "gemini-nano-banana-2.1"),
        ("gemini-2.5-flash-image", "image", None, "2027-03-15", "gemini-3.1-flash-lite-image"),
        ("gemini-3.1-flash-image", "image", "2026-10-06", None, "gemini-nano-banana-2.1"),
        ("veo-2.0-generate-001", "video", "2026-06-15", "2026-06-30", "veo-3.1-generate-001"),
        ("veo-3.1-generate-preview", "video", None, "2026-10-22", "gemini-omni-1.1-flash"),
        ("grok-imagine-image-pro", "image", None, "2026-05-15", "grok-imagine-image-2.0"),
        ("grok-imagine-image-quality", "image", "2026-09-02", "2026-11-02", "grok-imagine-image-2.0"),
        ("doubao-seedream-3-0-t2i-250415", "image", "2026-04-03", "2026-05-11", "doubao-seedream-5-0-pro-260628"),
        ("doubao-seedream-4-0-250828", "image", "2026-09-22", "2026-11-24", None),
        ("doubao-seedance-1-0-pro-250528", "video", "2026-09-22", "2026-11-24", "doubao-seedance-2-0-mini-260615"),
        ("doubao-seedance-1-0-lite-i2v-250428", "video", "2026-04-03", "2026-05-11",
         "doubao-seedance-2-0-fast-260128"),
        ("stable-video-diffusion", "video", "2025-07-24", "2025-07-24", None),
        ("eleven_turbo_v2_5", "audio", "2026-10-29", None, "eleven_flash_v2_5"),
    ]
    for model, modality, deprecated_on, retired_on, replacement in cases:
        entry = limits_for(model, modality)
        assert (entry.deprecated_on, entry.retired_on, entry.replacement) == (
            deprecated_on, retired_on, replacement), model


def test_new_models_pinned_against_official_docs():
    sunburst = limits_for("gpt-image-2.5-sunburst", "image")
    assert sunburst.max_output_longest_side == 3840 and sunburst.max_prompt_chars == 32000
    assert sunburst.lifecycle() == "active"
    omni = limits_for("gemini-omni-1.1-flash", "video")
    assert (omni.min_duration_seconds, omni.max_duration_seconds) == (3, 10)
    assert omni.aspect_ratios == ("16:9", "9:16")
    assert omni.supports_reference_audio is False and omni.supports_negative_prompt is False
    assert limits_for("gemini-nano-banana-2.1", "image").max_input_images == 14
    assert "1:8" in limits_for("gemini-nano-banana-2.1", "image").aspect_ratios
    assert limits_for("gemini-3.1-flash-lite-image", "image").supported_sizes == ("1K",)
    assert limits_for("veo-3.1-generate-001", "video").max_output_count == 4
    assert limits_for("lyria-3.5", "music").max_input_images == 10
    assert limits_for("lyria-3-clip-preview", "music").max_duration_seconds == 30
    assert limits_for("lyria-3-pro-preview", "music").max_duration_seconds == 184
    # "lyria-3" was never a Lyria model id: it now gets the permissive fallback.
    assert "permissive" in limits_for("lyria-3", "music").notes
    assert limits_for("grok-imagine-video-1.5", "video").max_input_images == 14
    assert limits_for("grok-imagine-video", "video").max_input_images == 7
    assert "5:2" in limits_for("grok-imagine-image-2.0", "image").aspect_ratios
    seedance25 = limits_for("doubao-seedance-2-5-260628", "video")
    assert (seedance25.min_duration_seconds, seedance25.max_duration_seconds) == (4, 30)
    assert seedance25.max_input_images == 30 and seedance25.exclusive_frames_and_references is True
    assert limits_for("doubao-seedance-2-0-mini-260615", "video").max_input_images == 9
    assert limits_for("doubao-seedance-1-0-pro-250528", "video").supports_audio_output is False
    seedream5 = limits_for("doubao-seedream-5-0-pro-260628", "image")
    assert seedream5.max_input_images == 10 and seedream5.max_output_count == 1
    assert limits_for("MiniMax-H3", "video").exclusive_frames_and_references is True
    assert limits_for("MiniMax-H3-Max", "video").min_duration_seconds == 5
    assert limits_for("sd3.5-flash", "image").max_prompt_chars == 10000
    eleven_v4 = limits_for("eleven_v4", "audio")
    assert (eleven_v4.min_speed, eleven_v4.max_speed) == (1.0, 1.0) and eleven_v4.max_prompt_chars == 10000
    assert limits_for("music_v2_5", "music").max_duration_seconds == 600
    assert limits_for("chirp-v4-0", "music").max_prompt_chars == 5000
    assert limits_for("chirp-v5", "music").replacement == "chirp-v6"
    assert limits_for("mureka-9.5", "music").max_output_count == 3
    assert limits_for("wan2.6-t2i", "image").max_output_longest_side == 2880
    i2v = limits_for("wan2.6-i2v", "video")
    assert (i2v.min_duration_seconds, i2v.max_duration_seconds) == (2, 15) and i2v.supports_last_frame is False
    assert limits_for("wan2.2-t2v-plus", "video").supports_audio_output is False
    assert limits_for("happyhorse-1.1-r2v", "video").max_input_images == 9
    assert (limits_for("happyhorse-1.1-t2v", "video").min_duration_seconds,
            limits_for("happyhorse-1.1-t2v", "video").max_duration_seconds) == (3, 15)
    assert limits_for("happyhorse-1.1-i2v", "video").aspect_ratios == ()
    assert limits_for("wan2.7-i2v", "video").supports_reference_audio is True


# --------------------------------------------------------------------------- #
# Router scoring (against real catalogue entries)
# --------------------------------------------------------------------------- #

def _img_request(*parts, size=None, aspect_ratio=None, n=None, model="auto"):
    return UnifiedImageRequest(
        model=model, content=list(parts),
        size=size, aspect_ratio=aspect_ratio, n=n,
    )


def test_text_only_image_request_fits_text_only_and_multimodal_models():
    req = _img_request(img_text("a prompt"))
    profile = profile_for(req)
    # dall-e-3 is text-only -> fits.
    fits_d3, _ = score(profile, limits_for("dall-e-3", "image"),
                       backend_index=0, model_index=0).fits, None
    assert fits_d3
    # gpt-image-1 accepts text -> fits too.
    s = score(profile, limits_for("gpt-image-1", "image"),
              backend_index=0, model_index=0)
    assert s.fits


def test_image_input_request_does_not_fit_text_only_model():
    req = _img_request(img_text("edit"), img_image(data="AAAA", mime_type="image/png"))
    profile = profile_for(req)
    # dall-e-3 rejects image input; gpt-image-1 accepts it.
    assert score(profile, limits_for("dall-e-3", "image"),
                 backend_index=0, model_index=0).fits is False
    assert score(profile, limits_for("gpt-image-1", "image"),
                 backend_index=0, model_index=0).fits is True


def test_oversize_prompt_is_rejected():
    long = "x" * 5000
    req = _img_request(img_text(long))
    profile = profile_for(req)
    # dall-e-3 caps at 4000 chars.
    assert score(profile, limits_for("dall-e-3", "image"),
                 backend_index=0, model_index=0).fits is False
    # gpt-image-1 caps at 32000 -> fits.
    assert score(profile, limits_for("gpt-image-1", "image"),
                 backend_index=0, model_index=0).fits is True


def test_oversize_output_longest_side_is_rejected():
    big = _img_request(img_text("big"), size="5000x5000")
    profile = profile_for(big)
    # gpt-image-2 caps the longest side at 3840 -> rejects 5000.
    assert score(profile, limits_for("gpt-image-2", "image"),
                 backend_index=0, model_index=0).fits is False
    # Within the ceiling it fits.
    ok = _img_request(img_text("ok"), size="2048x2048")
    assert score(profile_for(ok), limits_for("gpt-image-2", "image"),
                 backend_index=0, model_index=0).fits is True


def test_video_duration_within_and_outside_ceiling():
    req5 = UnifiedVideoRequest(model="auto", content=[vid_text("5s")], duration=5)
    req20 = UnifiedVideoRequest(model="auto", content=[vid_text("20s")], duration=20)
    p5, p20 = profile_for(req5), profile_for(req20)
    # Veo 2.0 caps at 8s.
    assert score(p5, limits_for("veo-2.0-generate-001", "video"),
                 backend_index=0, model_index=0).fits is True
    assert score(p20, limits_for("veo-2.0-generate-001", "video"),
                 backend_index=0, model_index=0).fits is False
    # Sora 2 caps at 20s -> 20s fits.
    assert score(p20, limits_for("sora-2", "video"),
                 backend_index=0, model_index=0).fits is True


def test_video_reference_video_only_fits_seedance_2():
    from mm_gateway.schemas.video import video_part
    req = UnifiedVideoRequest(
        model="auto",
        content=[vid_text("r2v"), video_part("https://x/v.mp4")],
        duration=8,
    )
    profile = profile_for(req)
    # Seedance 1.0 has no reference-video support -> rejects; 2.0 accepts.
    assert score(profile, limits_for("doubao-seedance-1-0-pro-250528", "video"),
                 backend_index=0, model_index=0).fits is False
    assert score(profile, limits_for("doubao-seedance-2-0-260128", "video"),
                 backend_index=0, model_index=0).fits is True


def test_music_lyrics_request_rejected_by_prompt_only_model():
    req = UnifiedMusicRequest(model="auto", content=[mus_text("a song")], lyrics="[verse]\nla")
    profile = profile_for(req)
    # All current music backends accept lyrics, so use a synthetic text-only
    # limits entry to prove the flag is enforced.
    from mm_gateway.models.limits import ModelLimits
    text_only = ModelLimits(modality="music", input_modalities=("text",),
                           supports_lyrics=False)
    assert score(profile, text_only, backend_index=0, model_index=0).fits is False


# --------------------------------------------------------------------------- #
# _fits branch coverage (each guards a documented constraint the router
# hard-fails on; a regression in any branch would otherwise ship green).
# --------------------------------------------------------------------------- #

def test_prompt_tokens_only_guard_rejects_over_long_prompt():
    # imagen-4.0 caps prompt length in TOKENS (480) with no char ceiling, so
    # only the *4 chars/token heuristic (router.py) applies. 480*4 = 1920.
    imagen = limits_for("imagen-4.0-generate-001", "image")
    assert imagen.max_prompt_chars is None and imagen.max_prompt_tokens == 480
    over = _img_request(img_text("x" * 5000))
    assert score(profile_for(over), imagen, backend_index=0, model_index=0).fits is False
    # Just under the token-derived ceiling fits.
    under = _img_request(img_text("x" * 1900))
    assert score(profile_for(under), imagen, backend_index=0, model_index=0).fits is True


def test_video_sub_minimum_duration_is_rejected():
    # veo-2.0 floor is 5s; only the max branch was previously exercised.
    veo2 = limits_for("veo-2.0-generate-001", "video")
    assert veo2.min_duration_seconds == 5
    too_short = UnifiedVideoRequest(model="auto", content=[vid_text("1s")], duration=1)
    assert score(profile_for(too_short), veo2,
                 backend_index=0, model_index=0).fits is False
    at_floor = UnifiedVideoRequest(model="auto", content=[vid_text("5s")], duration=5)
    assert score(profile_for(at_floor), veo2,
                 backend_index=0, model_index=0).fits is True


def test_aspect_ratio_hit_and_hard_fail():
    imagen = limits_for("imagen-4.0-generate-001", "image")
    # imagen accepts 16:9 -> fits and earns an optional hit.
    hit = _img_request(img_text("wide"), aspect_ratio="16:9")
    s_hit = score(profile_for(hit), imagen, backend_index=0, model_index=0)
    assert s_hit.fits
    # 21:9 is not in imagen's set -> hard fail (the else branch).
    miss = _img_request(img_text("ultrawide"), aspect_ratio="21:9")
    assert score(profile_for(miss), imagen,
                 backend_index=0, model_index=0).fits is False


def test_resolve_auto_picks_the_model_whose_aspect_ratio_fits():
    # Backend A serves imagen-4.0 (no 21:9); backend B serves sd3.5 (has 21:9).
    a = BackendConfig(name="aaa", type="google", api_key="k")
    b = BackendConfig(name="bbb", type="stability", api_key="k")
    reg = _registry([
        _StubProvider(a, image=["imagen-4.0-generate-001"]),
        _StubProvider(b, image=["sd3.5-large"]),
    ])
    key = KeyConfig(id="t", key="")
    req = UnifiedImageRequest(
        model="auto", content=[img_text("ultrawide panorama")], aspect_ratio="21:9",
    )
    _provider, real_model, backend = reg.resolve_auto(req, key, modality="image")
    assert backend == "bbb"
    assert real_model == "sd3.5-large"


# --------------------------------------------------------------------------- #
# Registry.auto routing end-to-end (no network)
# --------------------------------------------------------------------------- #

class _StubProvider:
    """A minimal provider that advertises a configurable model list."""

    def __init__(self, cfg, *, image=(), video=(), music=()):
        self.backend = cfg
        self.name = cfg.name
        self.image_models = list(image)
        self.video_models = list(video)
        self.music_models = list(music)

    supports_image = True
    supports_video = True
    supports_music = True


def _registry(providers, keys=None):
    settings = Settings(
        backends=[p.backend for p in providers],
        keys=keys or [KeyConfig(id="t", key="")],
    )
    reg = Registry(settings)
    for p in providers:
        reg._backends[p.name] = p
        reg._configs[p.name] = settings.backend(p.name)
    return reg


def test_resolve_auto_picks_a_fitting_model():
    cfg = BackendConfig(name="oai", type="openai", api_key="k")
    prov = _StubProvider(cfg, image=["gpt-image-2"])
    reg = _registry([prov])
    key = KeyConfig(id="t", key="")

    req = UnifiedImageRequest(model="auto", content=[img_text("hello")])
    provider, real_model, backend = reg.resolve_auto(req, key, modality="image")
    assert backend == "oai"
    assert real_model == "gpt-image-2"


def test_resolve_auto_skips_text_only_model_for_image_input():
    cfg = BackendConfig(name="oai", type="openai", api_key="k")
    # Two models: stable-image-core (text-only) and gpt-image-2 (multimodal).
    prov = _StubProvider(cfg, image=["stable-image-core", "gpt-image-2"])
    reg = _registry([prov])
    key = KeyConfig(id="t", key="")

    req = UnifiedImageRequest(
        model="auto", content=[img_text("edit"), img_image(data="AAAA", mime_type="image/png")]
    )
    _provider, real_model, _backend = reg.resolve_auto(req, key, modality="image")
    assert real_model == "gpt-image-2"


def test_resolve_auto_raises_when_no_model_fits():
    cfg = BackendConfig(name="oai", type="openai", api_key="k")
    # Only a text-only model, but the request carries an image.
    prov = _StubProvider(cfg, image=["stable-image-core"])
    reg = _registry([prov])
    key = KeyConfig(id="t", key="")

    req = UnifiedImageRequest(
        model="auto", content=[img_text("edit"), img_image(data="AAAA", mime_type="image/png")]
    )
    # Auto-route failure is a 422 validation_error (the request's input is
    # incompatible with every configured model), not a 404 model_not_found:
    # no specific model id was requested.
    with pytest.raises(GatewayError) as exc_info:
        reg.resolve_auto(req, key, modality="image")
    assert exc_info.value.status_code == 422
    assert exc_info.value.code == "validation_error"


def test_resolve_auto_music_with_lyrics_routes_to_lyrics_capable_model():
    # Fix A lock-in: lyrics is a role, not an input modality. A music request
    # that sets `lyrics` must still route to a lyrics-capable backend whose
    # catalogue entry lists input_modalities WITHOUT "lyrics" (music_v1 here).
    cfg = BackendConfig(name="eleven", type="elevenlabs", api_key="k")
    prov = _StubProvider(cfg, music=["music_v1"])
    reg = _registry([prov])
    key = KeyConfig(id="t", key="")

    req = UnifiedMusicRequest(model="auto", content=[mus_text("a happy song")],
                              lyrics="[verse]\nla la")
    _provider, real_model, backend = reg.resolve_auto(req, key, modality="music")
    assert backend == "eleven"
    assert real_model == "music_v1"


def test_resolve_auto_prefers_key_default_backend():
    a = BackendConfig(name="aaa", type="openai", api_key="k", tags=["quality"])
    b = BackendConfig(name="bbb", type="google", api_key="k", tags=["fast"])
    prov_a = _StubProvider(a, image=["gpt-image-2"])
    prov_b = _StubProvider(b, image=["gemini-nano-banana-2.1"])
    key = KeyConfig(id="t", key="", default_image_backend="bbb")
    reg = _registry([prov_a, prov_b], keys=[key])

    req = UnifiedImageRequest(model="auto", content=[img_text("hi")])
    _provider, _real_model, backend = reg.resolve_auto(req, key, modality="image")
    assert backend == "bbb"


def test_resolve_dispatches_on_auto_to_auto_router():
    cfg = BackendConfig(name="oai", type="openai", api_key="k")
    prov = _StubProvider(cfg, image=["gpt-image-2"])
    reg = _registry([prov])
    key = KeyConfig(id="t", key="")

    req = UnifiedImageRequest(model="auto", content=[img_text("hi")])
    provider, real_model, backend = reg.resolve(req.model, key, modality="image", request=req)
    assert backend == "oai"
    assert real_model == "gpt-image-2"


def test_resolve_auto_without_request_raises_validation_error():
    cfg = BackendConfig(name="oai", type="openai", api_key="k")
    prov = _StubProvider(cfg, image=["gpt-image-2"])
    reg = _registry([prov])
    key = KeyConfig(id="t", key="")
    with pytest.raises(Exception):  # ValidationError
        reg.resolve(None, key, modality="image")
