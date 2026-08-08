"""Tests for the Google provider (Imagen image + Veo video + Lyria music).

The provider builds two ``genai.Client`` instances (image + video) honoring the
sync/async URL split, plus a music REST base. We capture constructor args by
monkeypatching ``genai.Client`` — no network.
"""

from __future__ import annotations

from typing import Any

import pytest

from mm_gateway.config import BackendConfig
from mm_gateway.core.exceptions import ProviderNotConfiguredError
from mm_gateway.providers import google as google_mod
from mm_gateway.providers.google import GoogleProvider, _GLM_BASE


def _backend(base_url: str | None = None, *, extra: dict[str, Any] | None = None,
             api_key: str = "g-key") -> BackendConfig:
    kw: dict[str, Any] = {"name": "google", "type": "google", "api_key": api_key}
    if base_url is not None:
        kw["base_url"] = base_url
    if extra:
        kw["extra"] = extra
    return BackendConfig(**kw)


def _http_options_base(kwargs: dict[str, Any]) -> str | None:
    """Pull the base_url out of an HttpOptions instance (or None)."""
    opts = kwargs.get("http_options")
    if opts is None:
        return None
    return getattr(opts, "base_url", None)


# -- construction -------------------------------------------------------- #


def test_requires_api_key() -> None:
    with pytest.raises(ProviderNotConfiguredError):
        GoogleProvider(BackendConfig(name="google", type="google"))


def test_split_sync_async_base_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Image (Imagen/generate_content) lands on ``base_url`` (sync), video (Veo)
    on ``extra["video_base_url"]`` (async, recorded by ``config.py`` when it
    differs)."""
    captured: list[dict[str, Any]] = []

    class CapturingClient:
        def __init__(self, **kwargs: Any) -> None:
            captured.append(kwargs)

    monkeypatch.setattr(google_mod, "genai", type("G", (), {"Client": CapturingClient}))
    p = GoogleProvider(_backend(
        "https://image.test", extra={"video_base_url": "https://video.test"},
    ))
    assert p._client is not p._client_video
    assert [_http_options_base(c) for c in captured] == ["https://image.test", "https://video.test"]
    assert all(c["api_key"] == "g-key" for c in captured)


def test_single_base_url_when_not_split(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a separate video base, both clients share the image base."""
    captured: list[dict[str, Any]] = []

    class CapturingClient:
        def __init__(self, **kwargs: Any) -> None:
            captured.append(kwargs)

    monkeypatch.setattr(google_mod, "genai", type("G", (), {"Client": CapturingClient}))
    p = GoogleProvider(_backend("https://google.test"))
    assert p._client is not p._client_video
    assert [_http_options_base(c) for c in captured] == ["https://google.test", "https://google.test"]


def test_default_base_url_omits_http_options(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no base pinned, the SDK default host applies — i.e. no
    ``http_options`` override is passed."""
    captured: list[dict[str, Any]] = []

    class CapturingClient:
        def __init__(self, **kwargs: Any) -> None:
            captured.append(kwargs)

    monkeypatch.setattr(google_mod, "genai", type("G", (), {"Client": CapturingClient}))
    GoogleProvider(_backend())
    assert all("http_options" not in c for c in captured)


def test_music_base_prefers_music_base_url() -> None:
    """Lyria's REST base follows ``music_base_url`` > ``base_url`` > SDK default."""
    p = GoogleProvider(_backend(
        "https://image.test",
        extra={"video_base_url": "https://video.test",
               "music_base_url": "https://music.test"},
    ))
    assert p._music_base == "https://music.test"
    # Falls back to the image base when no music base is pinned.
    p2 = GoogleProvider(_backend("https://image.test"))
    assert p2._music_base == "https://image.test"
    # Falls back to the SDK default host when nothing is pinned.
    p3 = GoogleProvider(_backend())
    assert p3._music_base == _GLM_BASE
