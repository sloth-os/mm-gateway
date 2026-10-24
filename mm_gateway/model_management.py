"""Operator model suggestions and authenticated upstream catalogue discovery."""

from __future__ import annotations

import asyncio
import importlib
from typing import Any
from urllib.parse import urlsplit

import httpx

from mm_gateway.config import BackendConfig, _valid_model_id
from mm_gateway.core.base import AudioProvider, ImageProvider, MusicProvider, VideoProvider
from mm_gateway.core.exceptions import GatewayError
from mm_gateway.providers._http import proxy_kwargs

MODALITIES = {"image": ImageProvider, "video": VideoProvider,
              "music": MusicProvider, "audio": AudioProvider}


def builtin_models(provider_type: str) -> dict[str, list[str]]:
    # Reading class metadata does not initialize SDK clients or require credentials.
    from mm_gateway.registry import _PROVIDER_CLASSES

    class_name = _PROVIDER_CLASSES.get(provider_type)
    if not class_name:
        raise GatewayError("Unknown backend type.", code="invalid_config", status_code=422)
    provider = getattr(importlib.import_module(f"mm_gateway.providers.{provider_type}"), class_name)
    return {modality: list(getattr(provider, f"{modality}_models", []))
            for modality, capability in MODALITIES.items() if issubclass(provider, capability)}


# Defaults mirror the adapters. Other compatible services can use a base_url
# and, when needed, extra.models_path (relative to that base URL).
_BASES = {
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "google": "https://generativelanguage.googleapis.com",
    "xai": "https://api.x.ai/v1",
    "volcengine": "https://ark.cn-beijing.volces.com/api/v3",
    "elevenlabs": "https://api.elevenlabs.io",
}


def _listing_request(backend: BackendConfig, outbound_proxy: str | None) -> tuple[str, dict, str | None]:
    account = next((account for account in backend.accounts() if account[1]), None)
    if account is None:
        raise GatewayError("Enter a provider API key before listing upstream models.",
                           code="model_discovery_unavailable", status_code=422)
    _, api_key, account_base, extra = account
    base = account_base or backend.base_url or _BASES.get(backend.type)
    if not base:
        raise GatewayError("This provider has no default model-list API. Use built-in models, enter a model manually, "
                           "or set a compatible Base URL and provider option models_path.",
                           code="model_discovery_unavailable", status_code=422)
    parsed = urlsplit(base)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.query or parsed.fragment:
        raise GatewayError("Model discovery requires an HTTP(S) Base URL without query or fragment.",
                           code="invalid_config", status_code=422)
    path = extra.get("models_path")
    if path is None:
        path = "models"
        if backend.type == "google" and not parsed.path.rstrip("/").endswith(("/v1beta", "/v1")):
            path = "v1beta/models"
        elif backend.type in {"elevenlabs", "xai"} and not parsed.path.rstrip("/").endswith("/v1"):
            path = "v1/models"
    if not isinstance(path, str) or not path or urlsplit(path).scheme or path.startswith("//"):
        raise GatewayError("Provider option models_path must be a relative model-list path.",
                           code="invalid_config", status_code=422)
    header = {"google": "x-goog-api-key", "elevenlabs": "xi-api-key"}.get(backend.type, "Authorization")
    headers = {header: api_key if header != "Authorization" else f"Bearer {api_key}"}
    custom_headers = extra.get("models_headers", {})
    if not isinstance(custom_headers, dict) or any(not isinstance(value, str) for value in custom_headers.values()):
        raise GatewayError("Provider option models_headers must be a header object.", code="invalid_config", status_code=422)
    headers.update(custom_headers)
    return base.rstrip("/") + "/" + path.lstrip("/"), headers, extra.get("outbound_proxy") or outbound_proxy


async def discover_models(backend: BackendConfig, outbound_proxy: str | None = None) -> list[dict]:
    builtins = builtin_models(backend.type)
    try:
        url, headers, proxy = _listing_request(backend, outbound_proxy)
        request_proxy = proxy_kwargs(proxy)
    except GatewayError as exc:
        if exc.code == "provider_error":
            raise GatewayError("Could not configure model discovery's outbound proxy. Check the proxy settings and dependencies.",
                               code="model_discovery_unavailable", status_code=422) from exc
        raise
    except (ValueError, TypeError) as exc:
        raise GatewayError("Check the backend Base URL and model-list options.", code="invalid_config", status_code=422) from exc
    discovered: dict[str, dict] = {}
    params: dict[str, str] = {}
    pages: set[str] = set()
    try:
        async with asyncio.timeout(12), httpx.AsyncClient(timeout=10, **request_proxy) as client:
            for _ in range(100):
                response = await client.get(httpx.URL(url).copy_merge_params(params), headers=headers)
                if response.status_code >= 400:
                    raise GatewayError(
                        f"The upstream models API returned HTTP {response.status_code}. Check the Base URL, credentials "
                        "and models_path, or use built-in models / manual entry.",
                        code="model_discovery_failed", status_code=502)
                payload = response.json()
                entries: Any = payload if isinstance(payload, list) else payload.get("data", payload.get("models"))
                if not isinstance(entries, list):
                    raise ValueError("Expected a model list.")
                for entry in entries:
                    model_id = entry if isinstance(entry, str) else entry.get("id") or entry.get("name") or entry.get("model_id")
                    if isinstance(model_id, str) and backend.type == "google":
                        model_id = model_id.removeprefix("models/")
                    if not _valid_model_id(model_id):
                        raise ValueError("Invalid model id.")
                    modalities = [modality for modality, models in builtins.items() if model_id in models]
                    discovered[model_id] = {"id": model_id, "modalities": modalities}
                if isinstance(payload, list):
                    return list(discovered.values())
                if payload.get("nextPageToken"):
                    params = {"pageToken": str(payload["nextPageToken"])}
                elif payload.get("next_page_token"):
                    params = {"page_token": str(payload["next_page_token"])}
                elif payload.get("has_more") and entries:
                    params = {"after": str(entries[-1]["id"])}
                else:
                    return list(discovered.values())
                marker = str(params)
                if marker in pages:
                    raise ValueError("Repeated pagination token.")
                pages.add(marker)
    except (httpx.TimeoutException, TimeoutError) as exc:
        raise GatewayError("The upstream models API timed out. You can still use built-in models or manual entry.",
                           code="model_discovery_failed", status_code=504) from exc
    except (httpx.HTTPError, ValueError, TypeError, AttributeError, KeyError) as exc:
        # Never return upstream bodies, credential-bearing URLs or transport exception text.
        raise GatewayError("Could not read the upstream model list. Check the models API path and response format, "
                           "or use built-in models / manual entry.",
                           code="model_discovery_failed", status_code=502) from exc
    raise GatewayError("The upstream model list exceeded the pagination limit. Use manual entry for additional models.",
                       code="model_discovery_failed", status_code=502)
