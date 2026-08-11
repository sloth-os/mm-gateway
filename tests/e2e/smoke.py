#!/usr/bin/env python3
"""Real-provider end-to-end smoke test for a running mm-gateway.

Unlike ``tests/test_routes_e2e.py`` (which drives the full FastAPI stack against
a fake in-memory provider), this script talks to a **live** gateway and a **real**
upstream provider. It is intended to run in CI against the published Docker
image, with provider credentials supplied via environment variables using the
gateway's legacy env-var layout split by modality: an image triple
(``*_IMAGE_API_KEY`` + ``*_IMAGE_BASE_URL`` + ``*_IMAGE_MODEL``), a video triple
(``*_VIDEO_*``), and a music triple (``*_MUSIC_*``) per provider, plus an
implicit front-end key of ``GATEWAY_API_KEY``.

When to run
-----------
This script exercises a backend for a modality only when that modality is
**fully configured**: all three of its ``*_IMAGE_*`` (or ``*_VIDEO_*`` or
``*_MUSIC_*``) env vars are set. The ``*_BASE_URL`` proves the operator pointed
at a real endpoint, and ``*_MODEL`` pins the exact upstream model id to call
(rather than relying on a hard-coded alias). When *no* modality of any backend
has all three set the script **exits 0** (skips), so the workflow is green
before secrets are configured and only spends real provider calls once a
modality is fully wired.

Behaviour
---------
* Collects every (backend, modality) whose ``*_IMAGE_*`` / ``*_VIDEO_*`` /
  ``*_MUSIC_*`` triple is fully set. Falls back to the per-backend default
  gateway alias when the user pins a backend via ``E2E_BACKEND`` but leaves the
  model env unset.
* Confirms the chosen model is actually served by ``GET /v1/models`` — so a
  provider whose SDK failed to import in the image is skipped rather than
  failing the run.
* Runs one asynchronous task through each configured modality API, polling the
  normalized task resource until real image, video, or audio output is present.
  Music uses a typed lyrics input with verse/chorus sections.

Configuration (env)
-------------------
  MM_GATEWAY          gateway base URL (default http://127.0.0.1:8000)
  GATEWAY_API_KEY     bearer token to send (the gateway's front-end key)
  E2E_BACKEND         pin one backend type instead of collecting all configured
  E2E_IMAGE_MODEL     pin an image model id instead of the backend's *_IMAGE_MODEL
  E2E_VIDEO_MODEL     pin a video model id instead of the backend's *_VIDEO_MODEL
  E2E_MUSIC_MODEL     pin a music model id instead of the backend's *_MUSIC_MODEL
  E2E_PROMPT          override the prompt (default a deterministic string)
  E2E_LYRICS          override the music lyrics (default structured example lyrics)
  E2E_TIMEOUT         per-request timeout seconds (default 120)
  E2E_MUSIC_TIMEOUT   music poll budget seconds (default 300); music is slower
                      than image/video and gets its own budget so it doesn't
                      require raising E2E_TIMEOUT for the faster modalities.

Exit codes: 0 ok / skipped / some candidates failed but at least one succeeded
(the gateway is healthy; per-backend upstream failures are logged as warnings),
1 every configured candidate failed (the gateway itself is broken), 2 misconfigured.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import httpx

# One row per provider type. Each carries its image triple env names + default
# gateway alias, its video triple env names + alias, and its music triple env
# names + alias. A (backend, modality) is "fully configured" when all three of
# that modality's env vars are set; only those are exercised. Order is the
# collection priority. Providers that don't serve a modality carry empty env
# names for it (so it can never be "fully configured"). OpenRouter has no
# first-class image/video alias, so its *_MODEL (a real upstream slug) is used
# directly — the registry accepts any model id when the backend's catalogue is
# dynamic. The five music-only providers (elevenlabs, minimax, udioapi, mureka,
# acestep) carry only a music triple.
PROVIDERS: list[tuple[str, str, str, str, str, str, str, str, str, str, str, str, str]] = [
    # type,
    #   img_key, img_url, img_model, img_alias,
    #   vid_key, vid_url, vid_model, vid_alias,
    #   mus_key, mus_url, mus_model, mus_alias,
    ("openai", "OPENAI_IMAGE_API_KEY", "OPENAI_IMAGE_BASE_URL", "OPENAI_IMAGE_MODEL", "gateway-image-pro",
     "OPENAI_VIDEO_API_KEY", "OPENAI_VIDEO_BASE_URL", "OPENAI_VIDEO_MODEL", "gateway-video-sora",
     "", "", "", ""),
    ("volcengine", "ARK_IMAGE_API_KEY", "ARK_IMAGE_BASE_URL", "ARK_IMAGE_MODEL", "gateway-image-seedream",
     "ARK_VIDEO_API_KEY", "ARK_VIDEO_BASE_URL", "ARK_VIDEO_MODEL", "gateway-video-pro",
     "", "", "", ""),
    ("google", "GOOGLE_IMAGE_API_KEY", "GOOGLE_IMAGE_BASE_URL", "GOOGLE_IMAGE_MODEL", "gateway-image-imagen",
     "GOOGLE_VIDEO_API_KEY", "GOOGLE_VIDEO_BASE_URL", "GOOGLE_VIDEO_MODEL", "gateway-video-veo",
     "GOOGLE_MUSIC_API_KEY", "GOOGLE_MUSIC_BASE_URL", "GOOGLE_MUSIC_MODEL", "gateway-music-lyria"),
    ("xai", "XAI_IMAGE_API_KEY", "XAI_IMAGE_BASE_URL", "XAI_IMAGE_MODEL", "gateway-image-grok",
     "XAI_VIDEO_API_KEY", "XAI_VIDEO_BASE_URL", "XAI_VIDEO_MODEL", "gateway-video-grok",
     "", "", "", ""),
    ("flux", "FLUX_IMAGE_API_KEY", "FLUX_IMAGE_BASE_URL", "FLUX_IMAGE_MODEL", "gateway-image-flux",
     "FLUX_VIDEO_API_KEY", "FLUX_VIDEO_BASE_URL", "FLUX_VIDEO_MODEL", "",
     "", "", "", ""),
    ("dashscope", "DASHSCOPE_IMAGE_API_KEY", "DASHSCOPE_IMAGE_BASE_URL", "DASHSCOPE_IMAGE_MODEL", "gateway-image-wanx",
     "DASHSCOPE_VIDEO_API_KEY", "DASHSCOPE_VIDEO_BASE_URL", "DASHSCOPE_VIDEO_MODEL", "gateway-video-wan",
     "", "", "", ""),
    ("stability", "STABILITY_IMAGE_API_KEY", "STABILITY_IMAGE_BASE_URL", "STABILITY_IMAGE_MODEL", "gateway-image-sd",
     "STABILITY_VIDEO_API_KEY", "STABILITY_VIDEO_BASE_URL", "STABILITY_VIDEO_MODEL", "gateway-video-svd",
     "", "", "", ""),
    ("openrouter", "OPENROUTER_IMAGE_API_KEY", "OPENROUTER_IMAGE_BASE_URL", "OPENROUTER_IMAGE_MODEL", "",
     "OPENROUTER_VIDEO_API_KEY", "OPENROUTER_VIDEO_BASE_URL", "OPENROUTER_VIDEO_MODEL", "",
     "", "", "", ""),
    ("elevenlabs", "", "", "", "",
     "", "", "", "",
     "ELEVENLABS_MUSIC_API_KEY", "ELEVENLABS_MUSIC_BASE_URL", "ELEVENLABS_MUSIC_MODEL", "gateway-music-elevenlabs"),
    ("minimax", "", "", "", "",
     "", "", "", "",
     "MINIMAX_MUSIC_API_KEY", "MINIMAX_MUSIC_BASE_URL", "MINIMAX_MUSIC_MODEL", "gateway-music-minimax"),
    ("udioapi", "", "", "", "",
     "", "", "", "",
     "UDIOAPI_MUSIC_API_KEY", "UDIOAPI_MUSIC_BASE_URL", "UDIOAPI_MUSIC_MODEL", "gateway-music-udio"),
    ("mureka", "", "", "", "",
     "", "", "", "",
     "MUREKA_MUSIC_API_KEY", "MUREKA_MUSIC_BASE_URL", "MUREKA_MUSIC_MODEL", "gateway-music-mureka"),
    ("acestep", "", "", "", "",
     "", "", "", "",
     "ACESTEP_MUSIC_API_KEY", "ACESTEP_MUSIC_BASE_URL", "ACESTEP_MUSIC_MODEL", "gateway-music-acestep"),
]

BASE = os.environ.get("MM_GATEWAY", "http://127.0.0.1:8000").rstrip("/")
TOKEN = os.environ.get("GATEWAY_API_KEY", "")
TIMEOUT = float(os.environ.get("E2E_TIMEOUT", "120"))
MUSIC_TIMEOUT = float(os.environ.get("E2E_MUSIC_TIMEOUT", "300"))
PROMPT = os.environ.get("E2E_PROMPT", "a cat in a spacesuit, cinematic, 4k")

# Structured example lyrics (verse/chorus sections) used for every music
# candidate. Music providers take structured lyrics as input rather than a flat
# prompt, so this mirrors the real usage the gateway is built for. Overridable
# via E2E_LYRICS. The public request carries these as a typed lyrics part.
_DEFAULT_LYRICS = """[Intro]

[Verse 1]
Walking through the empty streets
Thinking of your gentle touch
Summer nights and softer dreams

[Chorus]
We rise together
Into the light
This is our moment tonight

[Verse 2]
Stars are falling from the sky
Your hand fits perfectly in mine

[Bridge]
If tomorrow never comes
At least we had this

[Chorus]
We rise together
Into the light
This is our moment tonight

[Outro]
"""
LYRICS = os.environ.get("E2E_LYRICS", _DEFAULT_LYRICS).strip()


def log(msg: str) -> None:
    print(f"[e2e] {msg}", flush=True)


def auth_headers() -> dict[str, str]:
    return {"authorization": f"Bearer {TOKEN}"} if TOKEN else {}


def wait_for_health(client: httpx.Client) -> None:
    deadline = time.time() + 60
    last = ""
    while time.time() < deadline:
        try:
            r = client.get("/health", timeout=5)
            if r.status_code == 200:
                return
            last = f"status {r.status_code}"
        except Exception as exc:  # noqa: BLE001
            last = str(exc)
        time.sleep(1)
    raise RuntimeError(f"gateway did not become healthy within 60s (last: {last})")


def fully_configured(envs: dict[str, str]) -> bool:
    """True iff every required env var has a non-empty value."""
    return all(os.environ.get(v) for v in envs.values())


def candidates() -> list[tuple[str, str, str]]:
    """(backend_type, modality, model_id) candidates that are fully configured.

    A (backend, modality) qualifies only when its ``*_IMAGE_*`` / ``*_VIDEO_*`` /
    ``*_MUSIC_*`` triple is fully set — "three envs of one type provider set" —
    so the e2e targets a real, operator-pinned model rather than a default alias.
    ``E2E_BACKEND`` pins a single backend (raising if no modality is fully
    configured); ``E2E_IMAGE_MODEL`` / ``E2E_VIDEO_MODEL`` / ``E2E_MUSIC_MODEL``
    override the model id for the selected image / video / music candidates.
    """
    pinned_backend = os.environ.get("E2E_BACKEND", "").strip().lower()
    pinned_image_model = os.environ.get("E2E_IMAGE_MODEL", "").strip()
    pinned_video_model = os.environ.get("E2E_VIDEO_MODEL", "").strip()
    pinned_music_model = os.environ.get("E2E_MUSIC_MODEL", "").strip()

    rows: list[tuple[str, str, str]] = []

    def add(backend: str, modality: str, model_env: str, default_alias: str,
            pinned: str) -> None:
        model = pinned or os.environ[model_env] or default_alias
        if model:
            rows.append((backend, modality, model))

    if pinned_backend:
        spec = next((p for p in PROVIDERS if p[0] == pinned_backend), None)
        if spec is None:
            raise ValueError(f"E2E_BACKEND={pinned_backend!r} is not a known backend")
        backend, ik, iu, im, ia, vk, vu, vm, va, mk, mu, mm, ma = spec
        img_set = fully_configured({"api_key": ik, "base_url": iu, "model": im})
        vid_set = fully_configured({"api_key": vk, "base_url": vu, "model": vm})
        mus_set = fully_configured({"api_key": mk, "base_url": mu, "model": mm})
        if not (img_set or vid_set or mus_set):
            # Only list env names that exist for this backend (music-only
            # providers have empty image/video env names — skip those).
            missing = [v for v in (ik, iu, im, vk, vu, vm, mk, mu, mm)
                       if v and not os.environ.get(v)]
            raise ValueError(
                f"E2E_BACKEND={pinned_backend!r} pinned but no image/video/music "
                f"triple is fully configured (missing: {', '.join(missing)})"
            )
        if img_set:
            add(backend, "image", im, ia, pinned_image_model)
        if vid_set:
            add(backend, "video", vm, va, pinned_video_model)
        if mus_set:
            add(backend, "music", mm, ma, pinned_music_model)
        return rows

    for backend, ik, iu, im, ia, vk, vu, vm, va, mk, mu, mm, ma in PROVIDERS:
        if fully_configured({"api_key": ik, "base_url": iu, "model": im}):
            add(backend, "image", im, ia, pinned_image_model)
        if fully_configured({"api_key": vk, "base_url": vu, "model": vm}):
            add(backend, "video", vm, va, pinned_video_model)
        if fully_configured({"api_key": mk, "base_url": mu, "model": mm}):
            add(backend, "music", mm, ma, pinned_music_model)
    return rows


def choose_models(client: httpx.Client, cands: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    """Keep the candidates whose model is actually served by the gateway."""
    # The registry only registers backends whose SDK imported cleanly; a broken
    # one is absent from /v1/models, so we skip it rather than failing.
    try:
        models = client.get("/v1/models", headers=auth_headers(), timeout=30)
    except Exception as exc:
        raise RuntimeError(f"GET /v1/models failed: {exc}") from exc
    if models.status_code != 200:
        raise RuntimeError(f"GET /v1/models -> {models.status_code}: {models.text[:200]}")
    served = {m["id"] for m in models.json().get("data", [])}

    kept: list[tuple[str, str, str]] = []
    for backend, modality, model in cands:
        if model in served:
            kept.append((backend, modality, model))
        else:
            log(f"backend {backend!r} {modality} model {model!r} not in /v1/models — skipping it")
    if not kept:
        raise RuntimeError(
            "configured provider(s) are set but none of their models are served; "
            "check the gateway logs for backend_init_failed"
        )
    return kept


def generate_image(client: httpx.Client, model: str) -> dict[str, Any]:
    """Create an image task and poll its normalized resource."""
    create = client.post(
        "/v1/images",
        headers={**auth_headers(), "content-type": "application/json"},
        json={"model": model, "input": PROMPT},
        timeout=TIMEOUT,
    )
    if create.status_code != 202:
        raise RuntimeError(f"POST /v1/images -> {create.status_code}: {create.text[:500]}")
    task_id = create.json().get("id")
    if not task_id:
        raise RuntimeError(f"image create returned no task id: {str(create.json())[:500]}")
    resource_url = create.headers.get("location") or f"/v1/images/{task_id}"

    deadline = time.time() + TIMEOUT
    last: dict[str, Any] = {}
    while time.time() < deadline:
        r = client.get(resource_url, headers=auth_headers(), timeout=TIMEOUT)
        if r.status_code != 200:
            raise RuntimeError(f"GET /v1/images/{task_id} -> {r.status_code}: {r.text[:500]}")
        last = r.json()
        status = last.get("status")
        if status == "succeeded":
            img = (last.get("outputs") or [{}])[0]
            url = img.get("url")
            b64 = img.get("data")
            if not (url or b64):
                raise RuntimeError(f"image succeeded but has no output: {str(last)[:500]}")
            return {"url": url, "data": b64}
        if status in ("failed", "cancelled", "expired"):
            raise RuntimeError(f"image task {status}: {str(last)[:500]}")
        time.sleep(2)
    raise RuntimeError(f"image task did not complete within {TIMEOUT}s (last: {str(last)[:500]})")


def generate_video(client: httpx.Client, model: str) -> str:
    """Create a video task and poll until it has a video URL."""
    create = client.post(
        "/v1/videos",
        headers={**auth_headers(), "content-type": "application/json"},
        json={"model": model, "input": [{"type": "text", "text": PROMPT}]},
        timeout=TIMEOUT,
    )
    if create.status_code != 202:
        raise RuntimeError(f"POST /v1/videos -> {create.status_code}: {create.text[:500]}")
    task_id = create.json().get("id")
    if not task_id:
        raise RuntimeError(f"video create returned no task id: {str(create.json())[:500]}")
    resource_url = create.headers.get("location") or f"/v1/videos/{task_id}"

    deadline = time.time() + TIMEOUT
    last: dict[str, Any] = {}
    while time.time() < deadline:
        r = client.get(resource_url, headers=auth_headers(), timeout=TIMEOUT)
        if r.status_code != 200:
            raise RuntimeError(f"GET /v1/videos/{task_id} -> {r.status_code}: {r.text[:500]}")
        last = r.json()
        status = last.get("status")
        if status == "succeeded":
            url = ((last.get("outputs") or [{}])[0]).get("url")
            if not url:
                raise RuntimeError(f"video succeeded but no video_url: {str(last)[:500]}")
            return url
        if status in ("failed", "cancelled", "expired"):
            raise RuntimeError(f"video task {status}: {str(last)[:500]}")
        time.sleep(2)
    raise RuntimeError(f"video task did not complete within {TIMEOUT}s (last: {str(last)[:500]})")


def generate_music(client: httpx.Client, model: str) -> str:
    """Create a music task and poll until it has an audio result."""
    create = client.post(
        "/v1/music",
        headers={**auth_headers(), "content-type": "application/json"},
        json={
            "model": model,
            "input": [
                {"type": "text", "text": "Compose a complete song."},
                {"type": "lyrics", "text": LYRICS},
            ],
        },
        timeout=TIMEOUT,
    )
    if create.status_code != 202:
        raise RuntimeError(f"POST /v1/music -> {create.status_code}: {create.text[:500]}")
    task_id = create.json().get("id")
    if not task_id:
        raise RuntimeError(f"music create returned no task id: {str(create.json())[:500]}")
    resource_url = create.headers.get("location") or f"/v1/music/{task_id}"

    deadline = time.time() + MUSIC_TIMEOUT
    last: dict[str, Any] = {}
    while time.time() < deadline:
        r = client.get(resource_url, headers=auth_headers(), timeout=TIMEOUT)
        if r.status_code != 200:
            raise RuntimeError(f"GET /v1/music/{task_id} -> {r.status_code}: {r.text[:500]}")
        last = r.json()
        status = last.get("status")
        if status == "succeeded":
            audio = (last.get("outputs") or [{}])[0]
            url = audio.get("url")
            b64 = audio.get("data")
            if not (url or b64):
                raise RuntimeError(f"music succeeded but has no output: {str(last)[:500]}")
            return url or "<b64_audio>"
        if status in ("failed", "cancelled", "expired"):
            raise RuntimeError(f"music task {status}: {str(last)[:500]}")
        time.sleep(3)
    raise RuntimeError(f"music task did not complete within {MUSIC_TIMEOUT}s (last: {str(last)[:500]})")


def run_one(candidate: tuple[str, str, str]) -> tuple[str, str, str, str]:
    """Exercise a single (backend, modality, model) in its own client/thread.

    Each worker holds its own ``httpx.Client`` — the client is not safe to
    share across concurrent requests, so parallel candidates must not reuse one.
    Returns ``(backend, modality, model, result)`` where result is either an
    ok-summary string or an error message prefixed with ``FAILED: ``.
    """
    backend, modality, model = candidate
    try:
        with httpx.Client(base_url=BASE, timeout=TIMEOUT) as client:
            if modality == "image":
                out = generate_image(client, model)
                return backend, modality, model, f"image ok: {out.get('url') or '<b64_json>'}"
            if modality == "video":
                url = generate_video(client, model)
                return backend, modality, model, f"video ok: {url}"
            url = generate_music(client, model)
            return backend, modality, model, f"music ok: {url}"
    except Exception as exc:  # noqa: BLE001
        return backend, modality, model, f"FAILED: {exc}"


def main() -> int:
    log(f"target gateway: {BASE}")
    cands = candidates()
    if not cands:
        log(
            "no provider modality is fully configured (needs *_IMAGE_* / "
            "*_VIDEO_* / *_MUSIC_* triple) and E2E_BACKEND is unset — "
            "skipping real-provider e2e"
        )
        return 0
    log(f"candidate (backend, modality): {[(c[0], c[1]) for c in cands]}")
    with httpx.Client(base_url=BASE, timeout=TIMEOUT) as client:
        wait_for_health(client)
        log("gateway healthy")
        chosen = choose_models(client, cands)
    # Run every configured candidate concurrently so a slow/failing one (e.g.
    # an upstream 500) no longer blocks the rest; each backend gets the full
    # E2E_TIMEOUT budget instead of sharing the wall-clock of a serial loop.
    failures: list[str] = []
    ok = 0
    nworkers = min(len(chosen) or 1, 8)
    with ThreadPoolExecutor(max_workers=nworkers) as pool:
        futures = {pool.submit(run_one, c): c for c in chosen}
        for fut in as_completed(futures):
            backend, modality, model, result = fut.result()
            log(f"[{backend}/{modality}] {model} -> {result}")
            if result.startswith("FAILED:"):
                failures.append(f"{backend}/{modality}/{model}: {result}")
            else:
                ok += 1
    # The e2e's job is to prove the *gateway* works end-to-end, not to assert
    # third-party upstream SLAs. A backend whose own provider returns 500/timeout
    # (or is pointed at an incompatible endpoint) is a per-backend issue, not a
    # gateway regression — the /v1/models check above already catches
    # backend_init_failed / SDK-import failures. So: green as long as at least
    # one candidate succeeded (the front-end + routing path is exercised); only
    # fail when *every* configured candidate failed, which signals the gateway
    # itself is broken. Per-candidate failures are still logged as warnings.
    if ok and failures:
        log(f"WARN: {len(failures)} of {len(chosen)} candidate(s) failed but {ok} succeeded — gateway is healthy:")
        for f in failures:
            log(f"  - {f}")
        return 0
    if failures:
        log(f"FAILED: all {len(failures)} of {len(chosen)} candidate(s) failed:")
        for f in failures:
            log(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        log(f"FAILED: {exc}")
        raise SystemExit(1)
