"""Bounded, credential-free downloads for voice sample uploads."""

from __future__ import annotations

import asyncio
import base64
import ipaddress
import socket

import httpx

from mm_gateway.core.exceptions import ValidationError
from mm_gateway.providers._http import proxy_kwargs

MAX_SAMPLE_BYTES = 20 * 1024 * 1024
_EXTENSIONS = {"audio/wav": "wav", "audio/x-wav": "wav", "audio/mpeg": "mp3",
               "audio/mp3": "mp3", "audio/mp4": "m4a", "audio/x-m4a": "m4a",
               "audio/ogg": "ogg", "audio/aac": "aac", "audio/flac": "flac", "audio/webm": "webm"}


async def _public_url(url: httpx.URL) -> httpx.URL:
    if url.scheme not in {"http", "https"} or not url.host or url.username or url.password:
        raise ValidationError("Voice samples require an HTTP(S) URL.")
    try:
        addresses = await asyncio.get_running_loop().getaddrinfo(
            url.host, url.port or (443 if url.scheme == "https" else 80), type=socket.SOCK_STREAM,
        )
    except OSError as exc:
        raise ValidationError("Voice sample host could not be resolved.") from exc
    ips = [entry[4][0] for entry in addresses]
    if not ips or any(not ipaddress.ip_address(ip).is_global or ipaddress.ip_address(ip).is_multicast for ip in ips):
        raise ValidationError("Voice sample URLs must resolve to public addresses.")
    # Pin the validated IP for the connection, avoiding a second DNS lookup.
    return url.copy_with(host=ips[0])


async def load_voice_sample(uri: str, *, proxy_url: str | None = None) -> tuple[str, bytes, str]:
    if uri.lower().startswith("data:"):
        header, _, data = uri.partition(",")
        mime = header[5:].split(";", 1)[0].lower()
        blob = base64.b64decode(data, validate=True)
    else:
        url = httpx.URL(uri)
        async with httpx.AsyncClient(timeout=30, **proxy_kwargs(proxy_url)) as client:
            for redirect in range(4):
                pinned = await _public_url(url)
                # Keep TLS SNI/certificate verification and HTTP Host on the
                # original hostname. No provider headers are sent to samples.
                async with client.stream("GET", pinned, headers={"Host": url.netloc.decode()},
                                         extensions={"sni_hostname": url.host}) as response:
                    if response.is_redirect:
                        if redirect == 3 or "location" not in response.headers:
                            raise ValidationError("Voice sample redirect limit exceeded.")
                        url = url.join(response.headers["location"])
                        continue
                    response.raise_for_status()
                    mime = response.headers.get("content-type", "").split(";", 1)[0].lower()
                    blob = bytearray()
                    async for chunk in response.aiter_bytes():
                        blob.extend(chunk)
                        if len(blob) > MAX_SAMPLE_BYTES:
                            raise ValidationError("Voice sample exceeds 20 MiB.")
                    blob = bytes(blob)
                    break
    if mime not in _EXTENSIONS or not blob or len(blob) > MAX_SAMPLE_BYTES:
        raise ValidationError("Voice sample must contain supported audio of up to 20 MiB.")
    return f"sample.{_EXTENSIONS[mime]}", blob, mime
