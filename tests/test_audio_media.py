"""Voice-sample download limits and network boundaries."""

from __future__ import annotations

import asyncio
import socket

import httpx
import pytest

from mm_gateway.core.exceptions import ValidationError
from mm_gateway.providers import _speech_media as media


@pytest.mark.parametrize("address", ["127.0.0.1", "10.1.2.3", "169.254.169.254", "::1", "fd00::1", "224.0.0.1"])
async def test_voice_sample_rejects_local_and_private_addresses(monkeypatch, address):
    async def resolve(*args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 443))]
    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", resolve)
    with pytest.raises(ValidationError, match="public addresses"):
        await media._public_url(httpx.URL("https://samples.example/audio.wav"))


async def test_voice_sample_download_pins_ip_and_keeps_tls_name_without_provider_auth(monkeypatch):
    async def resolve(*args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443))]
    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", resolve)
    captured = []
    def handle(request):
        captured.append(request)
        return httpx.Response(200, content=b"wave", headers={"content-type": "audio/wav"})
    client_class = httpx.AsyncClient
    monkeypatch.setattr(media.httpx, "AsyncClient", lambda **kwargs: client_class(
        transport=httpx.MockTransport(handle), **kwargs))
    name, blob, mime = await media.load_voice_sample("https://samples.example/audio.wav?signature=secret")
    assert (name, blob, mime) == ("sample.wav", b"wave", "audio/wav")
    request = captured[0]
    assert request.url.host == "8.8.8.8"
    assert request.url.params["signature"] == "secret"
    assert request.headers["host"] == "samples.example" and request.extensions["sni_hostname"] == "samples.example"
    assert "authorization" not in request.headers and "xi-api-key" not in request.headers


async def test_redirect_to_private_network_is_rejected(monkeypatch):
    async def resolve(host, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8" if host == "public.example" else "127.0.0.1", 443))]
    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", resolve)
    calls = []
    def handle(request):
        calls.append(request)
        return httpx.Response(302, headers={"location": "http://localhost/private.wav"})
    client_class = httpx.AsyncClient
    monkeypatch.setattr(media.httpx, "AsyncClient", lambda **kwargs: client_class(
        transport=httpx.MockTransport(handle), **kwargs))
    with pytest.raises(ValidationError, match="public addresses"):
        await media.load_voice_sample("https://public.example/audio.wav")
    assert len(calls) == 1


async def test_voice_sample_transport_size_limit(monkeypatch):
    monkeypatch.setattr(media, "MAX_SAMPLE_BYTES", 3)
    with pytest.raises(ValidationError):
        await media.load_voice_sample("data:audio/wav;base64,d2F2ZQ==")
