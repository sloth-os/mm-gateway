#!/usr/bin/env python3
"""Minimal client for the provider-neutral mm-gateway REST API.

Start ``mm-gateway``, configure at least one backend, then run::

    pip install httpx
    GATEWAY_API_KEY=... python examples/client.py
"""

from __future__ import annotations

import os
import time
from typing import Any
from uuid import uuid4

import httpx

BASE = os.environ.get("MM_GATEWAY", "http://localhost:8000")
TOKEN = os.environ.get("GATEWAY_API_KEY", "")
TERMINAL = {"succeeded", "failed", "cancelled", "expired"}


def poll(client: httpx.Client, created: httpx.Response) -> dict[str, Any]:
    """Follow the task resource advertised by a successful create response."""
    created.raise_for_status()
    location = created.headers["location"]
    etag: str | None = None
    for _ in range(150):
        headers = {"if-none-match": etag} if etag else None
        response = client.get(location, headers=headers)
        if response.status_code == 304:
            time.sleep(float(response.headers.get("retry-after", "2")))
            continue
        response.raise_for_status()
        etag = response.headers.get("etag")
        task = response.json()
        print(f"  {task['object']} {task['id']}: {task['status']}")
        if task["status"] in TERMINAL:
            return task
        time.sleep(float(response.headers.get("retry-after", "2")))
    raise TimeoutError(f"task did not complete: {location}")


def generate_image(client: httpx.Client) -> None:
    task = poll(
        client,
        client.post(
            "/v1/images",
            headers={
                "idempotency-key": f"example-image-{uuid4().hex}",
            },
            json={
                "model": "gateway-image-pro",
                "input": "a cat in a spacesuit, cinematic",
                "parameters": {
                    "size": "1024x1024",
                    "output_count": 1,
                    "delivery": "url",
                },
            },
        ),
    )
    print("image outputs:", task.get("outputs", []))


def generate_video(client: httpx.Client) -> None:
    task = poll(
        client,
        client.post(
            "/v1/videos",
            headers={
                "idempotency-key": f"example-video-{uuid4().hex}",
            },
            json={
                "model": "gateway-video-pro",
                "input": [
                    {"type": "text", "text": "a cat playing with a ball of yarn"},
                ],
                "parameters": {
                    "duration_seconds": 5,
                    "aspect_ratio": "16:9",
                    "include_audio": True,
                },
            },
        ),
    )
    print("video outputs:", task.get("outputs", []))


def generate_music(client: httpx.Client) -> None:
    task = poll(
        client,
        client.post(
            "/v1/music",
            headers={
                "idempotency-key": f"example-music-{uuid4().hex}",
            },
            json={
                "model": "gateway-music-lyria",
                "input": [
                    {"type": "text", "text": "upbeat cinematic pop"},
                    {"type": "lyrics", "text": "[Verse]\nUnder the city lights"},
                ],
                "parameters": {
                    "duration_seconds": 30,
                    "file_format": "wav",
                    "instrumental": False,
                },
            },
        ),
    )
    print("music outputs:", task.get("outputs", []))


def main() -> None:
    headers = {"authorization": f"Bearer {TOKEN}"} if TOKEN else {}
    with httpx.Client(base_url=BASE, headers=headers, timeout=300) as client:
        generate_image(client)
        generate_video(client)
        generate_music(client)


if __name__ == "__main__":
    main()
