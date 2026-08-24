"""Tests for the Vertex AI provider (Imagen image + Veo video + Lyria music, ADC-only).

Vertex is the Gemini Enterprise Agent Platform surface: it serves the same
Imagen/Veo/Lyria models as the AI Studio (google) adapter and reuses
``GenaiImageVideoMixin`` + the shared ``_lyria`` helpers verbatim — only
``genai.Client`` construction differs. The client is built with
``vertexai=True`` and ADC ``credentials=`` (never an ``api_key``). We capture
constructor args by monkeypatching ``genai.Client`` and the three
``google.auth`` loaders — no network, no real key.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from mm_gateway.config import BackendConfig
from mm_gateway.core.exceptions import ProviderNotConfiguredError
from mm_gateway.providers import vertex as vertex_mod
from mm_gateway.providers.vertex import VertexProvider
from mm_gateway.schemas.music import UnifiedMusicRequest, text_part


def _sa_json(project_id: str = "proj-123") -> str:
    """A minimally well-formed SA key dict (the private key is never used —
    construction stops at the ``genai.Client`` the test monkeypatches)."""
    return json.dumps({
        "type": "service_account",
        "project_id": project_id,
        "private_key": "-----BEGIN PRIVATE KEY-----\nfake\n-----END PRIVATE KEY-----\n",
        "private_key_id": "k1",
        "client_email": f"sa@{project_id}.iam.gserviceaccount.com",
        "client_id": "c1",
        "token_uri": "https://oauth2.googleapis.com/token",
    })


def _backend(*, credentials_json: str | None = None, credentials_file: str | None = None,
             project: str | None = None, location: str | None = "us-central1",
             base_url: str | None = None,
             extra: dict[str, Any] | None = None) -> BackendConfig:
    ex: dict[str, Any] = {}
    if credentials_json is not None:
        ex["credentials_json"] = credentials_json
    if credentials_file is not None:
        ex["credentials_file"] = credentials_file
    if project is not None:
        ex["project"] = project
    if location is not None:
        ex["location"] = location
    if extra:
        ex.update(extra)
    # api_key is deliberately None — Vertex is ADC-only.
    return BackendConfig(name="vertex", type="vertex", api_key=None,
                         base_url=base_url, extra=ex)


def _http_options_base(kwargs: dict[str, Any]) -> str | None:
    """Pull the base_url out of an HttpOptions instance (or None)."""
    opts = kwargs.get("http_options")
    if opts is None:
        return None
    return getattr(opts, "base_url", None)


def _stub_auth(monkeypatch: pytest.MonkeyPatch, *, project_id: str = "proj-123") -> list[dict[str, Any]]:
    """Replace the three google.auth loaders with stubs that return a sentinel.

    Returns the list each call appended its (scopes,) arg list into, so a test
    can assert the cloud-platform scope was requested.
    """
    calls: list[dict[str, Any]] = []
    _creds = object()  # sentinel — its identity proves it reached the client

    def _from_dict(info, scopes=None):
        calls.append({"via": "dict", "scopes": scopes})
        return _creds, project_id

    def _from_file(path, scopes=None):
        calls.append({"via": "file", "scopes": scopes, "path": path})
        return _creds, project_id

    def _default(scopes=None):
        calls.append({"via": "default", "scopes": scopes})
        return _creds, project_id

    monkeypatch.setattr(vertex_mod.google.auth, "load_credentials_from_dict", _from_dict)
    monkeypatch.setattr(vertex_mod.google.auth, "load_credentials_from_file", _from_file)
    monkeypatch.setattr(vertex_mod.google.auth, "default", _default)
    return calls


def _capture_client(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Monkeypatch genai.Client to capture its kwargs; returns the capture list."""
    captured: list[dict[str, Any]] = []

    class CapturingClient:
        def __init__(self, **kwargs: Any) -> None:
            captured.append(kwargs)

    monkeypatch.setattr(vertex_mod, "genai", type("G", (), {"Client": CapturingClient}))
    return captured


# -- construction -------------------------------------------------------- #


def test_location_defaults_to_global(monkeypatch: pytest.MonkeyPatch) -> None:
    """A region is optional — unset, the client defaults to the ``global``
    endpoint (the one Lyria 3 requires on Vertex)."""
    _stub_auth(monkeypatch)
    captured = _capture_client(monkeypatch)

    VertexProvider(_backend(credentials_json=_sa_json(), project="proj-x", location=None))
    # No VERTEX_LOCATION pinned → every client (image, video, music) lands on
    # the global endpoint rather than erroring.
    assert all(c["location"] == "global" for c in captured)
    assert len(captured) == 3  # image + video + music clients


def test_requires_credentials_when_no_key_no_adc(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no SA key and ambient ADC unavailable, construction fails."""
    from google.auth.exceptions import DefaultCredentialsError

    monkeypatch.setattr(vertex_mod.google.auth, "default",
                        lambda scopes=None: (_ for _ in ()).throw(DefaultCredentialsError("no adc")))
    with pytest.raises(ProviderNotConfiguredError):
        VertexProvider(_backend(location="us-central1"))


def test_credentials_json_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """A raw SA-JSON in extra["credentials_json"] is loaded via load_credentials_from_dict."""
    calls = _stub_auth(monkeypatch)
    captured = _capture_client(monkeypatch)

    VertexProvider(_backend(credentials_json=_sa_json(), project="proj-x", location="us-central1"))

    # Credentials resolve ONCE (in __init__ via the static resolver); the
    # three clients reuse that single credential, so the loader is called once.
    assert [c["via"] for c in calls] == ["dict"]
    # The cloud-platform scope was requested.
    assert calls[0]["scopes"] == ["https://www.googleapis.com/auth/cloud-platform"]
    # Three clients built (image + video + music), all in vertexai mode, no api_key.
    assert len(captured) == 3
    assert all(c["vertexai"] is True for c in captured)
    assert all("api_key" not in c for c in captured)
    # project + location threaded through.
    assert all(c["project"] == "proj-x" for c in captured)
    assert all(c["location"] == "us-central1" for c in captured)
    # The sentinel credentials object reached the client.
    assert all(c["credentials"] is not None for c in captured)


def test_credentials_file_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """A key path in extra["credentials_file"] is loaded via load_credentials_from_file."""
    calls = _stub_auth(monkeypatch)
    captured = _capture_client(monkeypatch)

    VertexProvider(_backend(credentials_file="/etc/vertex/sa.json", project="proj-x"))

    assert [c["via"] for c in calls] == ["file"]
    assert calls[0]["path"] == "/etc/vertex/sa.json"
    assert all(c["vertexai"] is True for c in captured)
    assert all("api_key" not in c for c in captured)


def test_ambient_adc_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """With neither key set, ambient ADC (google.auth.default) is used."""
    calls = _stub_auth(monkeypatch)
    captured = _capture_client(monkeypatch)

    VertexProvider(_backend(project="proj-x", location="europe-west1"))

    assert [c["via"] for c in calls] == ["default"]
    assert all(c["location"] == "europe-west1" for c in captured)


def test_project_falls_back_to_sa_key_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """No explicit VERTEX_PROJECT → the SA key's own project_id is used."""
    _stub_auth(monkeypatch, project_id="sa-project-999")
    captured = _capture_client(monkeypatch)

    # No `project` in extra.
    VertexProvider(_backend(credentials_json=_sa_json(project_id="sa-project-999")))

    assert all(c["project"] == "sa-project-999" for c in captured)


def test_two_distinct_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    """Image, video and music get separate client instances (an operator can pin them apart)."""
    _stub_auth(monkeypatch)
    _capture_client(monkeypatch)

    p = VertexProvider(_backend(credentials_json=_sa_json(), project="proj-x"))
    assert p._client is not p._client_video
    assert p._client is not p._client_music
    assert p._client_video is not p._client_music


def test_no_base_url_means_none_host_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no operator base pin, http_options.base_url is None (the SDK derives the
    regional host from the location). http_options is still passed so the logging
    httpx client is injected regardless."""
    _stub_auth(monkeypatch)
    captured = _capture_client(monkeypatch)

    VertexProvider(_backend(credentials_json=_sa_json(), project="proj-x"))
    assert all("http_options" in c for c in captured)
    assert all(_http_options_base(c) is None for c in captured)


def test_image_video_music_base_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    """An operator-pinned image base lands on ``backend.base_url`` (the image
    client), a differing video pin on ``extra["video_base_url"]`` (the video
    client), and a differing music pin on ``extra["music_base_url"]`` (the
    music client). Mirrors the google adapter's split."""
    _stub_auth(monkeypatch)
    captured = _capture_client(monkeypatch)

    p = VertexProvider(_backend(
        credentials_json=_sa_json(), project="proj-x",
        # Image base via base_url, video + music bases via the extras the
        # provider reads.
        location="us-central1",
        extra={"video_base_url": "https://video.test",
               "music_base_url": "https://music.test"},
    ))
    # image_base = backend.base_url (None here); video_base + music_base = the pins.
    bases = [_http_options_base(c) for c in captured]
    assert bases == [None, "https://video.test", "https://music.test"]
    assert p._client is not p._client_video
    assert p._client_video is not p._client_music


def test_music_client_falls_back_to_image_base(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no music-specific base pinned, the music client shares the image base
    (like google's ``_music_base`` = image base)."""
    _stub_auth(monkeypatch)
    captured = _capture_client(monkeypatch)

    VertexProvider(_backend(
        credentials_json=_sa_json(), project="proj-x", location="us-central1",
        base_url="https://image.test",
    ))
    bases = [_http_options_base(c) for c in captured]
    # image + video + music all fall back to backend.base_url when no per-modality
    # pins are set.
    assert bases == ["https://image.test", "https://image.test", "https://image.test"]


# -- modality surface --------------------------------------------------- #


def test_supports_image_video_and_music(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lyria is served on Vertex via the Interactions API — image+video+music."""
    _stub_auth(monkeypatch)
    _capture_client(monkeypatch)

    p = VertexProvider(_backend(credentials_json=_sa_json(), project="proj-x"))
    assert p.supports_image is True
    assert p.supports_video is True
    assert p.supports_music is True
    assert p.image_models == [
        "imagen-4.0-generate-001",
        "imagen-3.0-generate-001",
        "gemini-2.5-flash-image",
    ]
    assert p.video_models == [
        "veo-2.0-generate-001",
        "veo-3.0-generate-001",
        "veo-3.1-generate-preview",
    ]
    assert p.music_models == ["lyria-3"]


# -- Lyria music (Interactions) ---------------------------------------- #


def _stub_interactions(monkeypatch: pytest.MonkeyPatch, audio_b64: str = "UklGRiQAAABXQVZFZmV",
                       lyrics: str | None = "la la la") -> tuple[list[dict[str, Any]], Any]:
    """Replace ``VertexProvider``'s ``_client_music`` with one whose
    ``aio.interactions.create`` returns a dict-shaped Interaction.

    Returns ``(calls, client)`` where ``calls`` is the list of kwargs each
    ``interactions.create`` call received, so a test can assert the body shape
    the adapter built (the same shape ``lyria_body`` produces for the google
    adapter).
    """
    calls: list[dict[str, Any]] = []

    class _Interaction:
        def model_dump(self, exclude_none: bool = False):
            return {
                "steps": [{"content": [
                    {"type": "audio", "data": audio_b64},
                    *([{"type": "text", "text": lyrics}] if lyrics else []),
                ]}],
            }

    class _MusicClient:
        class aio:
            class interactions:
                @staticmethod
                async def create(**kwargs):
                    calls.append(kwargs)
                    return _Interaction()

    return calls, _MusicClient()


def test_vertex_music_poll_calls_interactions_create(monkeypatch: pytest.MonkeyPatch) -> None:
    """The first poll performs the blocking ``interactions.create`` call and
    extracts the inline audio — the same wire shape the google adapter uses,
    only authenticated with the ADC bearer the SDK injects (not x-goog-api-key)."""
    _stub_auth(monkeypatch)
    _capture_client(monkeypatch)
    p = VertexProvider(_backend(credentials_json=_sa_json(), project="proj-x"))

    calls, music_client = _stub_interactions(monkeypatch)
    p._client_music = music_client

    create = asyncio.run(p.create_music_task(
        UnifiedMusicRequest(model="lyria-3", content=[text_part("a happy song")])
    ))
    task = asyncio.run(p.get_music_task(create.task_id))

    assert len(calls) == 1
    body = calls[0]
    # Same Interactions body the google adapter builds (lyria_body).
    assert body["model"] == "lyria-3"
    assert body["input"] == [{"type": "text", "text": "a happy song"}]
    assert task.status == "succeeded"
    assert task.audio_b64 == "UklGRiQAAABXQVZFZmV"
    assert task.audio_media_type == "audio/mpeg"  # default MP3


def test_vertex_music_fails_when_no_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    """No audio block in the Interaction output → the task fails (TaskFailedError)."""
    from mm_gateway.core.exceptions import TaskFailedError

    _stub_auth(monkeypatch)
    _capture_client(monkeypatch)
    p = VertexProvider(_backend(credentials_json=_sa_json(), project="proj-x"))

    _, music_client = _stub_interactions(monkeypatch, audio_b64="", lyrics=None)
    p._client_music = music_client

    create = asyncio.run(p.create_music_task(
        UnifiedMusicRequest(model="lyria-3", content=[text_part("x")])
    ))
    with pytest.raises(TaskFailedError):
        asyncio.run(p.get_music_task(create.task_id))
