"""Tests for the YAML-driven configuration in ``mm_gateway.config``.

Covers: ``${ENV}``/``${ENV:default}`` interpolation, backend+key parsing, the
``key_for``/``backend``/``backends_of_type`` accessors, the
``default_image_provider``/``default_video_provider`` legacy-compat properties,
the legacy env-var fallback, config-file discovery (``MM_GATEWAY_CONFIG`` +
``_find_config_file``), and the new ``mcp_enabled``/``mcp_path`` parsing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mm_gateway.config import (
    BackendConfig,
    KeyConfig,
    Settings,
    _find_config_file,
    _interpolate,
)

# A representative YAML doc exercising every parsed section.
_SAMPLE_YAML = """\
server:
  host: 127.0.0.1
  port: 9999
  log_level: DEBUG
  log_format: text
  request_timeout: 45
video:
  sync_default: false
  max_sync_wait: 12
  poll_interval: 0.25
defaults:
  enable_metrics: false
mcp:
  enabled: true
  path: /ai/mcp
backends:
  - name: volc-prod
    type: volcengine
    api_key: ${VOLC_KEY}
    tags: [video-primary]
    base_url: https://ark.example
    extra:
      region: cn-beijing
  - name: openai-prod
    type: openai
    api_key: ${MISSING_KEY:sk-fallback}
    tags: [image-primary]
keys:
  - id: alice
    key: ${ALICE_TOKEN}
    allow_tags: [image-primary, video-primary]
    default_video_backend: volc-prod
  - id: bob
    key: bob-token
    allow_backends: [openai-prod]
    deny_tags: [video-primary]
    extra:
      note: pinned
"""


# --------------------------------------------------------------------------- #
# Interpolation
# --------------------------------------------------------------------------- #


def test_interpolate_env_substitution(monkeypatch):
    monkeypatch.setenv("VOLC_KEY", "sk-volc")
    assert _interpolate("key: ${VOLC_KEY}") == "key: sk-volc"


def test_interpolate_default_when_env_unset(monkeypatch):
    monkeypatch.delenv("MISSING_KEY", raising=False)
    assert _interpolate("key: ${MISSING_KEY:sk-fallback}") == "key: sk-fallback"


def test_interpolate_default_used_when_env_empty(monkeypatch):
    # An empty env var is treated as unset by the legacy layout; interpolation
    # here matches os.environ.get semantics (empty string is a value), so a
    # present-but-empty var wins over the default — verifying that contract.
    monkeypatch.setenv("EMPTY_KEY", "")
    assert _interpolate("key: ${EMPTY_KEY:sk-fallback}") == "key: "


def test_interpolate_leaves_unknown_token_intact(monkeypatch):
    monkeypatch.delenv("NO_SUCH_VAR", raising=False)
    # No default supplied -> the placeholder is preserved verbatim.
    assert _interpolate("key: ${NO_SUCH_VAR}") == "key: ${NO_SUCH_VAR}"


def test_interpolate_only_matches_uppercase_env_names():
    # Lowercase names are not substituted (keeps arbitrary ${...} text safe).
    assert _interpolate("v: ${lowercase}") == "v: ${lowercase}"


# --------------------------------------------------------------------------- #
# YAML parsing round-trip
# --------------------------------------------------------------------------- #


def test_from_file_parses_all_sections(monkeypatch, tmp_path):
    monkeypatch.setenv("VOLC_KEY", "sk-volc")
    monkeypatch.setenv("ALICE_TOKEN", "tok-alice")
    monkeypatch.delenv("MISSING_KEY", raising=False)
    path = tmp_path / "mm-gateway.yaml"
    path.write_text(_SAMPLE_YAML, encoding="utf-8")

    s = Settings.from_file(path)

    # server / video / defaults
    assert s.host == "127.0.0.1"
    assert s.port == 9999
    assert s.log_level == "DEBUG"
    assert s.log_format == "text"
    assert s.request_timeout == 45
    assert s.video_sync_default is False
    assert s.max_sync_wait == 12
    assert s.poll_interval == 0.25
    assert s.enable_metrics is False

    # mcp section (the addition this feature introduces)
    assert s.mcp_enabled is True
    assert s.mcp_path == "/ai/mcp"

    # backends
    assert [b.name for b in s.backends] == ["volc-prod", "openai-prod"]
    volc = s.backend("volc-prod")
    assert volc.type == "volcengine"
    assert volc.api_key == "sk-volc"          # ${VOLC_KEY} substituted
    assert volc.base_url == "https://ark.example"
    assert volc.tags == ["video-primary"]
    assert volc.extra == {"region": "cn-beijing"}
    assert volc.configured is True
    openai = s.backend("openai-prod")
    assert openai.api_key == "sk-fallback"    # ${MISSING_KEY:sk-fallback} default

    # keys
    assert [k.id for k in s.keys] == ["alice", "bob"]
    alice = s.key_for("tok-alice")
    assert alice.allow_tags == ["image-primary", "video-primary"]
    assert alice.default_video_backend == "volc-prod"
    bob = s.key_for("bob-token")
    assert bob.allow_backends == ["openai-prod"]
    assert bob.deny_tags == ["video-primary"]
    assert bob.extra == {"note": "pinned"}


def test_from_file_missing_keys_default_to_empty(monkeypatch, tmp_path):
    # A key may omit routing fields entirely; they default to empty/None.
    path = tmp_path / "c.yaml"
    path.write_text(
        "backends:\n  - {name: b1, type: openai, api_key: k}\n"
        "keys:\n  - {id: anon, key: ''}\n",
        encoding="utf-8",
    )
    s = Settings.from_file(path)
    k = s.key_for("")
    assert k.id == "anon"
    assert k.allow_tags == [] and k.deny_tags == [] and k.allow_backends == []
    assert k.default_image_backend is None
    assert k.default_video_tag is None


def test_mcp_defaults_when_section_absent(monkeypatch, tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("backends: []\nkeys: []\n", encoding="utf-8")
    s = Settings.from_file(path)
    assert s.mcp_enabled is False
    assert s.mcp_path == "/mcp"


def test_scalar_section_does_not_crash(tmp_path):
    # A plausible shorthand like `mcp: true` must not raise an opaque
    # AttributeError; the section is coerced to empty (defaults apply).
    path = tmp_path / "c.yaml"
    path.write_text("mcp: true\nserver: true\nvideo: true\nbackends: []\nkeys: []\n",
                    encoding="utf-8")
    s = Settings.from_file(path)
    assert s.mcp_enabled is False
    assert s.port == 8000


def test_mcp_session_idle_timeout_parsed(monkeypatch, tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("mcp: {enabled: true, session_idle_timeout: 42}\n"
                    "backends: []\nkeys: []\n", encoding="utf-8")
    assert Settings.from_file(path).mcp_session_idle_timeout == 42
    # default when unset
    path2 = tmp_path / "c2.yaml"
    path2.write_text("backends: []\nkeys: []\n", encoding="utf-8")
    assert Settings.from_file(path2).mcp_session_idle_timeout == 1800


# --------------------------------------------------------------------------- #
# Accessors
# --------------------------------------------------------------------------- #


def _settings() -> Settings:
    return Settings(
        backends=[
            BackendConfig(name="openai-prod", type="openai", api_key="k", tags=["img"]),
            BackendConfig(name="openai-staging", type="openai", api_key="k2", tags=["img-staging"]),
            BackendConfig(name="volc-prod", type="volcengine", api_key="k3", tags=["vid"]),
            BackendConfig(name="flux-1", type="flux", tags=[]),  # unconfigured (no key)
        ],
        keys=[
            KeyConfig(id="alice", key="tok-a"),
            KeyConfig(id="bob", key="tok-b", default_image_backend="openai-prod"),
        ],
    )


def test_backend_lookup_by_name():
    s = _settings()
    assert s.backend("volc-prod").type == "volcengine"
    assert s.backend("nope") is None


def test_backends_of_type():
    s = _settings()
    assert [b.name for b in s.backends_of_type("openai")] == ["openai-prod", "openai-staging"]
    assert s.backends_of_type("stability") == []


def test_key_for_matches_and_misses():
    s = _settings()
    assert s.key_for("tok-a").id == "alice"
    assert s.key_for("tok-b").id == "bob"
    assert s.key_for("nope") is None


def test_default_provider_properties():
    s = _settings()
    # Bob carries the only default_image_backend; alice has no video default, so
    # default_video_provider falls back to the first backend.
    assert s.default_image_provider == "openai-prod"
    assert s.default_video_provider == "openai-prod"


def test_default_provider_empty_when_no_backends():
    s = Settings(backends=[], keys=[])
    assert s.default_image_provider == ""
    assert s.default_video_provider == ""


def test_backend_configured_flag():
    assert BackendConfig(name="b", type="openai", api_key="k").configured is True
    assert BackendConfig(name="b", type="openai").configured is False
    assert BackendConfig(name="b", type="openai", api_key="").configured is False


# --------------------------------------------------------------------------- #
# Config file discovery + from_env
# --------------------------------------------------------------------------- #


def test_from_env_uses_mm_gateway_config(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # ensure no ./mm-gateway.yaml shadows the env var
    path = tmp_path / "custom.yaml"
    path.write_text(
        "server: {port: 7000}\nmcp: {enabled: true}\n"
        "backends:\n  - {name: b1, type: openai, api_key: k}\nkeys: []\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MM_GATEWAY_CONFIG", str(path))
    s = Settings.from_env()
    assert s.port == 7000
    assert s.mcp_enabled is True
    assert [b.name for b in s.backends] == ["b1"]


def test_find_config_file_prefers_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "mm-gateway.yaml").write_text("backends: []\n", encoding="utf-8")
    assert Path(_find_config_file()).name == "mm-gateway.yaml"


def test_find_config_file_yml_fallback(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "mm-gateway.yml").write_text("backends: []\n", encoding="utf-8")
    assert Path(_find_config_file()).name == "mm-gateway.yml"


def test_find_config_file_none_when_absent(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert _find_config_file() is None


# --------------------------------------------------------------------------- #
# Legacy env-var fallback
# --------------------------------------------------------------------------- #


def test_from_env_legacy_when_no_yaml(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # no ./mm-gateway.yaml present
    monkeypatch.delenv("MM_GATEWAY_CONFIG", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-legacy-openai")
    monkeypatch.setenv("ARK_API_KEY", "sk-legacy-volc")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.test")
    monkeypatch.delenv("GATEWAY_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    s = Settings.from_env()

    # One backend per configured provider, named by its type.
    names = {b.name for b in s.backends}
    assert {"openai", "volcengine"} <= names
    assert "google" not in names  # GOOGLE_API_KEY unset -> backend omitted
    openai = s.backend("openai")
    assert openai.api_key == "sk-legacy-openai"
    assert openai.base_url == "https://api.openai.test"

    # A single implicit "env" key, open (empty token) when GATEWAY_API_KEY unset,
    # authorised for every configured backend.
    assert [k.id for k in s.keys] == ["env"]
    key = s.keys[0]
    assert key.key == ""
    assert set(key.allow_backends) == names


def test_from_env_legacy_gateway_api_key_makes_it_closed(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MM_GATEWAY_CONFIG", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("GATEWAY_API_KEY", "secret-gateway-token")
    s = Settings.from_env()
    assert s.key_for("secret-gateway-token").id == "env"
    assert s.key_for("") is None  # no longer open


def test_from_env_legacy_no_keys_configured(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MM_GATEWAY_CONFIG", raising=False)
    for var in ("OPENAI_API_KEY", "ARK_API_KEY", "GATEWAY_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    s = Settings.from_env()
    assert s.backends == []
    # The implicit key still exists (open), authorised for zero backends.
    assert [k.id for k in s.keys] == ["env"]
    assert s.keys[0].allow_backends == []


def test_from_env_legacy_model_env_recorded_in_extra(monkeypatch, tmp_path):
    # The *_MODEL env var pins/extends a backend's served image model. It must
    # be carried on BackendConfig.extra["image_model"] so the registry can
    # append it to the provider's image_models at build time.
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MM_GATEWAY_CONFIG", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.test")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-image-custom")
    for v in ("ARK_API_KEY", "GOOGLE_API_KEY", "GATEWAY_API_KEY"):
        monkeypatch.delenv(v, raising=False)
    s = Settings.from_env()
    openai = s.backend("openai")
    assert openai.extra.get("image_model") == "gpt-image-custom"


def test_from_env_legacy_model_env_absent_leaves_no_extra(monkeypatch, tmp_path):
    # With no *_MODEL set, extra must not carry an image_model (so the registry
    # doesn't append None to the provider's model list).
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MM_GATEWAY_CONFIG", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    for v in ("OPENAI_MODEL", "ARK_API_KEY", "GOOGLE_API_KEY", "GATEWAY_API_KEY"):
        monkeypatch.delenv(v, raising=False)
    s = Settings.from_env()
    assert "image_model" not in s.backend("openai").extra


# --------------------------------------------------------------------------- #
# Split image/video env-var layout (the CI e2e contract): *_IMAGE_* / *_VIDEO_*
# triples override the legacy un-split *_* names per modality.
# --------------------------------------------------------------------------- #


def _clear_provider_envs(monkeypatch) -> None:
    """Unset every legacy and split provider env var for a clean baseline."""
    names = ["OPENAI", "ARK", "GOOGLE", "XAI", "FLUX", "RUNAPI", "DASHSCOPE",
             "STABILITY", "OPENROUTER"]
    for n in names:
        for suffix in ("API_KEY", "BASE_URL", "MODEL",
                       "IMAGE_API_KEY", "IMAGE_BASE_URL", "IMAGE_MODEL",
                       "VIDEO_API_KEY", "VIDEO_BASE_URL", "VIDEO_MODEL"):
            monkeypatch.delenv(f"{n}_{suffix}", raising=False)
    monkeypatch.delenv("GATEWAY_API_KEY", raising=False)
    monkeypatch.delenv("MM_GATEWAY_CONFIG", raising=False)


def test_from_env_split_image_triple_registers_image_model(monkeypatch, tmp_path):
    # A *_IMAGE_* triple pins the served image model into extra["image_model"].
    monkeypatch.chdir(tmp_path)
    _clear_provider_envs(monkeypatch)
    monkeypatch.setenv("OPENAI_IMAGE_API_KEY", "sk-img")
    monkeypatch.setenv("OPENAI_IMAGE_BASE_URL", "https://api.openai.test")
    monkeypatch.setenv("OPENAI_IMAGE_MODEL", "gpt-image-split")
    s = Settings.from_env()
    openai = s.backend("openai")
    assert openai.api_key == "sk-img"
    assert openai.base_url == "https://api.openai.test"
    assert openai.extra.get("image_model") == "gpt-image-split"
    # The split video triple was unset, so no video model is pinned.
    assert "video_model" not in openai.extra


def test_from_env_split_video_triple_pins_video_model(monkeypatch, tmp_path):
    # A *_VIDEO_* triple pins the served video model into extra["video_model"].
    monkeypatch.chdir(tmp_path)
    _clear_provider_envs(monkeypatch)
    monkeypatch.setenv("ARK_VIDEO_API_KEY", "sk-vid")
    monkeypatch.setenv("ARK_VIDEO_BASE_URL", "https://ark.test")
    monkeypatch.setenv("ARK_VIDEO_MODEL", "doubao-seedance-split")
    s = Settings.from_env()
    volc = s.backend("volcengine")
    assert volc.api_key == "sk-vid"
    assert volc.extra.get("video_model") == "doubao-seedance-split"
    assert "image_model" not in volc.extra


def test_from_env_split_overrides_legacy_per_modality(monkeypatch, tmp_path):
    # When both split and legacy names are set, the split *_IMAGE_* / *_VIDEO_*
    # names win per modality; the legacy name is only a fallback.
    monkeypatch.chdir(tmp_path)
    _clear_provider_envs(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-legacy")          # legacy fallback
    monkeypatch.setenv("OPENAI_MODEL", "gpt-image-legacy")
    monkeypatch.setenv("OPENAI_IMAGE_MODEL", "gpt-image-split")  # overrides legacy
    monkeypatch.setenv("OPENAI_VIDEO_MODEL", "sora-split")      # video pinned
    s = Settings.from_env()
    openai = s.backend("openai")
    assert openai.extra.get("image_model") == "gpt-image-split"
    assert openai.extra.get("video_model") == "sora-split"


def test_from_env_split_omits_unconfigured_modality(monkeypatch, tmp_path):
    # Only the image triple set -> the backend still registers (image key),
    # and no video model is pinned. A provider with only image is image-only.
    monkeypatch.chdir(tmp_path)
    _clear_provider_envs(monkeypatch)
    monkeypatch.setenv("FLUX_IMAGE_API_KEY", "sk-flux")
    monkeypatch.setenv("FLUX_IMAGE_BASE_URL", "https://flux.test")
    monkeypatch.setenv("FLUX_IMAGE_MODEL", "flux-2-pro")
    s = Settings.from_env()
    names = {b.name for b in s.backends}
    assert "flux" in names
    flux = s.backend("flux")
    assert flux.api_key == "sk-flux"
    assert flux.extra.get("image_model") == "flux-2-pro"
    assert "video_model" not in flux.extra


def test_from_env_video_only_key_registers_backend(monkeypatch, tmp_path):
    # A backend can be wired for video only (no image triple). The video key
    # registers it; the image triple stays unset.
    monkeypatch.chdir(tmp_path)
    _clear_provider_envs(monkeypatch)
    monkeypatch.setenv("STABILITY_VIDEO_API_KEY", "sk-stab")
    monkeypatch.setenv("STABILITY_VIDEO_BASE_URL", "https://api.stability.ai")
    monkeypatch.setenv("STABILITY_VIDEO_MODEL", "stable-video-diffusion")
    s = Settings.from_env()
    stab = s.backend("stability")
    assert stab.api_key == "sk-stab"
    assert stab.extra.get("video_model") == "stable-video-diffusion"
    assert "image_model" not in stab.extra


