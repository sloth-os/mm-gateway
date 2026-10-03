"""Live speech smoke selection and output validation without live credentials."""

import json

import httpx
import pytest

from tests.e2e import smoke


@pytest.fixture(autouse=True)
def clear_provider_environment(monkeypatch):
    names = {name for row in smoke.PROVIDERS for name in row[1:] if name}
    names.update(provider.upper() + "_AUDIO_" + suffix
                 for provider in smoke.AUDIO_PROVIDERS for suffix in ("API_KEY", "BASE_URL", "MODEL"))
    names.update({"E2E_BACKEND", "E2E_AUDIO_MODEL", "E2E_IMAGE_MODEL", "E2E_VIDEO_MODEL", "E2E_MUSIC_MODEL"})
    for name in names:
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("provider", smoke.AUDIO_PROVIDERS)
def test_speech_requires_complete_configuration_and_supports_pinned_backend(monkeypatch, provider):
    prefix = provider.upper() + "_AUDIO_"
    monkeypatch.setenv(prefix + "API_KEY", "test-key")
    monkeypatch.setenv(prefix + "BASE_URL", "https://speech.example")
    assert smoke.candidates() == []
    monkeypatch.setenv(prefix + "MODEL", "speech-model")
    assert smoke.candidates() == [(provider, "audio", "speech-model")]
    monkeypatch.setenv("E2E_BACKEND", provider)
    monkeypatch.setenv("E2E_AUDIO_MODEL", "override-model")
    assert smoke.candidates() == [(provider, "audio", "override-model")]
    monkeypatch.setenv("E2E_BACKEND", "xai")
    with pytest.raises(ValueError, match="no image/video/music/audio"):
        smoke.candidates()


@pytest.mark.parametrize("uri", ["data:audio/mpeg;base64,YXVkaW8=", "data:audio/mpeg;base64,"])
def test_speech_smoke_polls_audio_and_validates_nonempty_bytes(uri):
    paths = []

    def handle(request):
        paths.append(request.url.path)
        if request.method == "POST":
            assert json.loads(request.content)["parameters"]["file_format"] == "mp3"
            return httpx.Response(202, headers={"location": "/v1/audio/aud_test"}, json={"id": "aud_test"})
        return httpx.Response(200, json={"status": "succeeded", "outputs": [{"mime_type": "audio/mpeg", "uri": uri}]})

    with httpx.Client(base_url="http://gateway", transport=httpx.MockTransport(handle)) as client:
        if uri.endswith(","):
            with pytest.raises(RuntimeError, match="empty audio"):
                smoke.generate_audio(client, "speech-model")
        else:
            assert smoke.generate_audio(client, "speech-model") == "audio/mpeg, 5 bytes"
    assert paths == ["/v1/audio", "/v1/audio/aud_test"]
