#!/usr/bin/env python3
"""Example mm-gateway client.

Calls the image and video endpoints in their canonical front-end shapes: the
Gemini image shape (``POST /v1/images`` with ``{model, input}`` then poll
``GET /v1/images/{id}`` for an image block) and the Seedance video shape
(``POST /v1/videos`` then poll ``GET /v1/videos/{id}``). Requires the gateway to
be running (``mm-gateway``) and at least one provider configured. Adjust the
model ids to match what your gateway exposes (see ``GET /v1/models``).

    pip install httpx
    python examples/client.py
"""

from __future__ import annotations

import os
import time

import httpx

BASE = os.environ.get("MM_GATEWAY", "http://localhost:8000")

_TERMINAL = ("succeeded", "failed", "cancelled", "expired")


def generate_image(c: httpx.Client) -> None:
    # Gemini shape: POST /v1/images with {model, input} returns a task id; the
    # sync frontend (?wait=true) blocks until completion, then poll
    # GET /v1/images/{id} for the steps[].content[] image block (url or base64).
    r = c.post("/v1/images", json={
        "model": "gateway-image-pro",
        "input": "a cat in a spacesuit, cinematic",
    }, params={"wait": "true"})
    r.raise_for_status()
    task_id = r.json()["id"]
    print("image created:", task_id)
    for _ in range(60):
        p = c.get(f"/v1/images/{task_id}")
        p.raise_for_status()
        body = p.json()
        status = body.get("status")
        print("  poll:", status)
        if status in _TERMINAL:
            blocks = ((body.get("steps") or [{}])[0]).get("content") or []
            img = next((b for b in blocks if b.get("type") == "image"), {})
            print("  image:", img.get("url") or (img.get("data", "")[:40] + "…"))
            return
        time.sleep(2)


def generate_video(c: httpx.Client) -> None:
    # Seedance shape: POST /v1/videos with a content array returns a task id;
    # poll GET /v1/videos/{id} for the content.video_url.
    r = c.post("/v1/videos", json={
        "model": "gateway-video-pro",
        "content": [{"type": "text", "text": "a cat playing with a ball of yarn"}],
    }, headers={"prefer": "respond-async"})
    r.raise_for_status()
    task_id = r.json()["id"]
    print("video created:", task_id)
    for _ in range(60):
        p = c.get(f"/v1/videos/{task_id}")
        p.raise_for_status()
        body = p.json()
        status = body.get("status")
        print("  poll:", status)
        if status in _TERMINAL:
            print("  result:", body.get("content") or body)
            return
        time.sleep(2)


def main() -> None:
    with httpx.Client(base_url=BASE, timeout=300) as c:
        generate_image(c)
        generate_video(c)


if __name__ == "__main__":
    main()
