#!/usr/bin/env python3
"""Example mm-gateway client.

Calls both image and video endpoints in the OpenAI and OpenRouter shapes.
Requires the gateway to be running (``mm-gateway``) and at least one provider
configured. Adjust the model ids to match what your gateway exposes (see
``GET /v1/models``).

    pip install httpx
    python examples/client.py
"""

from __future__ import annotations

import os
import time

import httpx

BASE = os.environ.get("MM_GATEWAY", "http://localhost:8000")


def openai_image(c: httpx.Client) -> None:
    r = c.post("/v1/images/generations", json={
        "model": "gateway-image-pro",
        "prompt": "a cat in a spacesuit, cinematic",
        "size": "1024x1024",
    })
    r.raise_for_status()
    data = r.json()["data"][0]
    print("image (openai):", data.get("url") or data.get("b64_json", "")[:40] + "…")


def openrouter_image(c: httpx.Client) -> None:
    r = c.post("/api/v1/images", json={
        "model": "gateway-image-flux",
        "prompt": "a neon city, synthwave",
        "provider": {"only": "flux"},
    })
    r.raise_for_status()
    print("image (openrouter):", r.json()["data"][0])


def seedance_video_async(c: httpx.Client) -> None:
    r = c.post("/v1/videos", json={
        "model": "gateway-video-pro",
        "content": [{"type": "text", "text": "a cat playing with a ball of yarn"}],
    }, headers={"prefer": "respond-async"})
    r.raise_for_status()
    task_id = r.json()["id"]
    print("video (seedance) created:", task_id)
    for _ in range(60):
        p = c.get(f"/v1/videos/{task_id}")
        p.raise_for_status()
        body = p.json()
        status = body.get("status")
        print("  poll:", status)
        if status in ("succeeded", "failed", "cancelled", "expired"):
            print("  result:", body.get("content") or body)
            return
        time.sleep(2)


def openrouter_video(c: httpx.Client) -> None:
    r = c.post("/api/v1/videos", json={
        "model": "gateway-video-seedance-2",
        "prompt": "a drone shot over a misty forest",
    }, headers={"prefer": "respond-async"})
    r.raise_for_status()
    handle = r.json()
    print("video (openrouter) created:", handle["id"], "poll:", handle["polling_url"])


def main() -> None:
    with httpx.Client(base_url=BASE, timeout=300) as c:
        openai_image(c)
        openrouter_image(c)
        seedance_video_async(c)
        openrouter_video(c)


if __name__ == "__main__":
    main()
