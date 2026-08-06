#!/usr/bin/env python3
"""Real-provider end-to-end smoke test for a running mm-gateway.

Unlike ``tests/test_routes_e2e.py`` (which drives the full FastAPI stack against
a fake in-memory provider), this script talks to a **live** gateway and a **real**
upstream provider. It is intended to run in CI against the published Docker
image, with provider credentials supplied via environment variables using the
gateway's legacy env-var layout (one backend per ``*_API_KEY`` plus optional
``*_BASE_URL`` and ``*_MODEL`` overrides, and an implicit front-end key of
``GATEWAY_API_KEY``).

When to run
-----------
This script runs the e2e against a backend only when the backend is **fully
configured**: all three of its ``*_API_KEY``, ``*_BASE_URL``, and ``*_MODEL``
environment variables are set. The ``*_BASE_URL`` proves the operator pointed at
a real endpoint, and ``*_MODEL`` pins the exact upstream model id to call (rather
than relying on a hard-coded alias). When *no* backend has all three set the
script **exits 0** (skips), so the workflow is green before secrets are
configured and only spends real provider calls once a backend is fully wired.

Behaviour
---------
* Collects every backend whose ``*_API_KEY`` + ``*_BASE_URL`` + ``*_MODEL`` are
  all set. Falls back to the per-backend default gateway alias when the user
  pins a backend via ``E2E_BACKEND`` but leaves ``E2E_IMAGE_MODEL`` unset.
* Confirms the chosen model is actually served by ``GET /v1/models`` — so a
  provider whose SDK failed to import in the image is skipped rather than
  failing the run.
* Runs one image generation through the OpenAI-shape front-end and asserts real
  image data (a ``url`` or ``b64_json``) comes back.

Configuration (env)
-------------------
  MM_GATEWAY          gateway base URL (default http://127.0.0.1:8000)
  GATEWAY_API_KEY     bearer token to send (the gateway's front-end key)
  E2E_BACKEND         pin one backend type instead of collecting all configured
  E2E_IMAGE_MODEL     pin a model id instead of using the backend's *_MODEL
  E2E_PROMPT          override the prompt (default a deterministic string)
  E2E_TIMEOUT         per-request timeout seconds (default 120)

Exit codes: 0 ok / skipped, 1 a configured e2e failed, 2 misconfigured.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

import httpx

# Backend type -> (api_key env, base_url env, model env, default gateway alias).
# Order is the collection priority. A backend is "fully configured" when all
# three env vars are set; only those are exercised. OpenRouter is included but
# has no first-class image alias, so its *_MODEL (a real upstream slug) is used
# directly — the registry accepts any model id when the backend's catalogue is
# dynamic.
PROVIDERS: list[tuple[str, str, str, str, str]] = [
    ("openai", "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL", "gateway-image-pro"),
    ("volcengine", "ARK_API_KEY", "ARK_BASE_URL", "ARK_MODEL", "gateway-image-seedream"),
    ("google", "GOOGLE_API_KEY", "GOOGLE_BASE_URL", "GOOGLE_MODEL", "gateway-image-imagen"),
    ("xai", "XAI_API_KEY", "XAI_BASE_URL", "XAI_MODEL", "gateway-image-grok"),
    ("flux", "RUNAPI_API_KEY", "FLUX_BASE_URL", "FLUX_MODEL", "gateway-image-flux"),
    ("dashscope", "DASHSCOPE_API_KEY", "DASHSCOPE_BASE_URL", "DASHSCOPE_MODEL", "gateway-image-wanx"),
    ("stability", "STABILITY_API_KEY", "STABILITY_BASE_URL", "STABILITY_MODEL", "gateway-image-sd"),
    ("openrouter", "OPENROUTER_API_KEY", "OPENROUTER_BASE_URL", "OPENROUTER_MODEL", ""),
]

BASE = os.environ.get("MM_GATEWAY", "http://127.0.0.1:8000").rstrip("/")
TOKEN = os.environ.get("GATEWAY_API_KEY", "")
TIMEOUT = float(os.environ.get("E2E_TIMEOUT", "120"))
PROMPT = os.environ.get("E2E_PROMPT", "a cat in a spacesuit, cinematic, 4k")


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


def candidates() -> list[tuple[str, str]]:
    """(backend_type, model_id) candidates that are fully configured.

    A backend qualifies only when its *_API_KEY, *_BASE_URL, and *_MODEL are all
    set — "three envs of one type provider set" — so the e2e targets a real,
    operator-pinned model rather than a default alias. ``E2E_BACKEND`` pins a
    single backend (raising if it isn't fully configured); ``E2E_IMAGE_MODEL``
    overrides the model id for whichever backends are selected.
    """
    pinned_backend = os.environ.get("E2E_BACKEND", "").strip().lower()
    pinned_model = os.environ.get("E2E_IMAGE_MODEL", "").strip()

    if pinned_backend:
        spec = next((p for p in PROVIDERS if p[0] == pinned_backend), None)
        if spec is None:
            raise ValueError(f"E2E_BACKEND={pinned_backend!r} is not a known backend")
        _, key_env, url_env, model_env, default_alias = spec
        envs = {"api_key": key_env, "base_url": url_env, "model": model_env}
        if not fully_configured(envs):
            missing = [v for v in envs.values() if not os.environ.get(v)]
            raise ValueError(
                f"E2E_BACKEND={pinned_backend!r} pinned but not fully configured "
                f"(missing: {', '.join(missing)})"
            )
        return [(pinned_backend, pinned_model or os.environ[model_env] or default_alias)]

    found: list[tuple[str, str]] = []
    for backend, key_env, url_env, model_env, default_alias in PROVIDERS:
        envs = {"api_key": key_env, "base_url": url_env, "model": model_env}
        if not fully_configured(envs):
            continue
        model = pinned_model or os.environ[model_env] or default_alias
        if not model:
            continue
        found.append((backend, model))
    return found


def choose_model(client: httpx.Client, cands: list[tuple[str, str]]) -> tuple[str, str]:
    """Pick the first candidate whose model is actually served by the gateway."""
    # The registry only registers backends whose SDK imported cleanly; a broken
    # one is absent from /v1/models, so we skip it rather than failing.
    try:
        models = client.get("/v1/models", headers=auth_headers(), timeout=30)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"GET /v1/models failed: {exc}") from exc
    if models.status_code != 200:
        raise RuntimeError(f"GET /v1/models -> {models.status_code}: {models.text[:200]}")
    served = {m["id"] for m in models.json().get("data", [])}

    for backend, model in cands:
        if model in served:
            return backend, model
        log(f"backend {backend!r} model {model!r} not in /v1/models — skipping it")

    raise RuntimeError(
        "configured provider(s) are set but none of their models are served; "
        "check the gateway logs for backend_init_failed"
    )


def generate_image(client: httpx.Client, model: str) -> dict[str, Any]:
    r = client.post(
        "/v1/images/generations",
        headers={**auth_headers(), "content-type": "application/json"},
        json={"model": model, "prompt": PROMPT, "n": 1, "size": "1024x1024"},
        timeout=TIMEOUT,
    )
    if r.status_code != 200:
        raise RuntimeError(
            f"POST /v1/images/generations -> {r.status_code}: {r.text[:500]}"
        )
    body = r.json()
    data = body.get("data") or []
    if not data or not (data[0].get("url") or data[0].get("b64_json")):
        raise RuntimeError(f"response had no image data: {str(body)[:500]}")
    return data[0]


def main() -> int:
    log(f"target gateway: {BASE}")
    cands = candidates()
    if not cands:
        log(
            "no provider is fully configured (needs *_API_KEY + *_BASE_URL + "
            "*_MODEL) and E2E_BACKEND is unset — skipping real-provider e2e"
        )
        return 0
    log(f"candidate backends: {[c[0] for c in cands]}")
    with httpx.Client(base_url=BASE, timeout=TIMEOUT) as client:
        wait_for_health(client)
        log("gateway healthy")
        backend, model = choose_model(client, cands)
        log(f"using backend={backend} model={model}")
        out = generate_image(client, model)
        url = out.get("url")
        log(f"image ok: {url or '<b64_json>'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        log(f"FAILED: {exc}")
        raise SystemExit(1)
