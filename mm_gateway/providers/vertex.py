"""Google Vertex AI provider — the Gemini Enterprise Agent Platform surface.

Vertex AI exposes the **same** generative models as the AI Studio (Gemini
Developer) surface — Imagen (image) and Veo (video) — and is reached through the
**same** ``google-genai`` SDK, just with the client built in Vertex mode. The
SDK's own docstring calls Vertex the "Gemini Enterprise Agent Platform". So this
adapter reuses the AI Studio adapter's Imagen/Veo request/response logic
(:class:`~mm_gateway.providers._genai_media.GenaiImageVideoMixin`) verbatim; only
the ``genai.Client`` construction differs.

Authentication is **Application Default Credentials via a service-account JSON
key** (no API key). Two ways to supply the key, resolved in this order:

* ``backend.extra["credentials_json"]`` — the **raw JSON content** of an SA key.
  Used by CI, where a GitHub secret holds the file contents verbatim. Loaded
  with ``google.auth.load_credentials_from_dict`` (no temp file written).
* ``backend.extra["credentials_file"]`` — a **filesystem path** to an SA key
  JSON. Used by YAML config deployments. Loaded with
  ``google.auth.load_credentials_from_file``.
* Otherwise the SDK's own ADC resolution (``google.auth.default``) is used, so a
  deployment that has run ``gcloud auth application-default login`` (or runs on
  GCE/GKE/Workload Identity, or sets ``GOOGLE_APPLICATION_CREDENTIALS``) works
  with no explicit key — the key file is discovered from the environment.

In every case the resulting ``Credentials`` are passed straight to
``genai.Client(credentials=...)``. The **project** comes from the explicit
``VERTEX_PROJECT`` (or the SA key's own ``project_id`` when one is loaded);
``VERTEX_LOCATION`` pins the region (e.g. ``us-central1``), which selects the
``https://{location}-aiplatform.googleapis.com`` endpoint. A region is required —
ADC yields credentials but not a region, so the operator must set one.

**Music (Lyria) is not available on Vertex** — the SDK raises
``NotImplementedError`` for live-music in Vertex mode, and the Interactions
REST surface the AI Studio adapter uses lives only on
``generativelanguage.googleapis.com``. So this backend implements
``ImageProvider`` + ``VideoProvider`` only.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

import google.auth
import httpx
from google import genai
from google.auth.exceptions import DefaultCredentialsError
from google.genai import types

from mm_gateway.core.base import ImageProvider, VideoProvider
from mm_gateway.core.exceptions import ProviderNotConfiguredError
from mm_gateway.observability.httplog import backend_event_hooks
from mm_gateway.observability.logging import get_logger
from mm_gateway.providers._genai_media import GenaiImageVideoMixin
from mm_gateway.providers._sync_image import SyncImageTaskMixin

log = get_logger("provider.vertex")

# Scopes Vertex requests when it resolves ADC itself; matched by SA keys too.
_VERTEX_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


class VertexProvider(SyncImageTaskMixin, GenaiImageVideoMixin, ImageProvider, VideoProvider):
    name = "vertex"
    # Vertex serves the same model ids as AI Studio (Imagen/Veo). Lyria is
    # absent on Vertex, so there is no music list here.
    image_models: ClassVar[list[str]] = [
        "imagen-4.0-generate-001",
        "imagen-3.0-generate-001",
        "gemini-2.5-flash-image",
    ]
    video_models: ClassVar[list[str]] = [
        "veo-2.0-generate-001",
        "veo-3.0-generate-001",
        "veo-3.1-generate-preview",
    ]

    def __init__(self, backend):
        super().__init__(backend)
        credentials, project, location = self._resolve_credentials(backend)
        if credentials is None:
            raise ProviderNotConfiguredError(
                "vertex",
                "Vertex requires a service-account JSON key (credentials_json or "
                "credentials_file) or application default credentials.",
            )
        if not location:
            raise ProviderNotConfiguredError(
                "vertex",
                "Vertex requires a region (VERTEX_LOCATION) — ADC yields credentials "
                "but not a region.",
            )
        image_base = backend.base_url or None
        video_base = backend.extra.get("video_base_url") or image_base
        self._client = self._build_vertex_client(credentials, project, location, image_base)
        self._client_video = self._build_vertex_client(credentials, project, location, video_base)

    @staticmethod
    def _resolve_credentials(backend) -> tuple[Any, str | None, str | None]:
        """Resolve ADC credentials, project, and location from the backend.

        Returns ``(credentials, project, location)``. ``credentials`` is None
        when no SA key and no ambient ADC are available. ``project`` is the
        explicit VERTEX_PROJECT or the SA key's project_id; ``location`` is
        VERTEX_LOCATION.
        """
        project = backend.extra.get("project") or None
        location = backend.extra.get("location") or None
        cred_json = backend.extra.get("credentials_json")
        cred_file = backend.extra.get("credentials_file")
        try:
            if cred_json:
                creds, pid = google.auth.load_credentials_from_dict(
                    json.loads(cred_json) if isinstance(cred_json, str) else cred_json,
                    scopes=_VERTEX_SCOPES,
                )
            elif cred_file:
                creds, pid = google.auth.load_credentials_from_file(cred_file, scopes=_VERTEX_SCOPES)
            else:
                # Ambient ADC: GOOGLE_APPLICATION_CREDENTIALS env or the
                # metadata server (GCE/GKE/Workload Identity).
                creds, pid = google.auth.default(scopes=_VERTEX_SCOPES)
        except (DefaultCredentialsError, ValueError, OSError) as exc:
            log.info("vertex_adc_unavailable", error=str(exc))
            return None, project, location
        return creds, project or pid, location

    @staticmethod
    def _build_vertex_client(credentials, project, location, base_url):
        kwargs: dict[str, Any] = {
            "vertexai": True,
            "credentials": credentials,
            "project": project,
            "location": location,
        }
        # Inject an httpx client whose event hooks log the backend request/
        # response (curl format + masked sensitive headers), matching the AI
        # Studio adapter. base_url overrides the endpoint when an operator pins
        # a regional host via VERTEX_*_BASE_URL.
        http_kwargs: dict[str, Any] = {
            "httpxAsyncClient": httpx.AsyncClient(event_hooks=backend_event_hooks()),
        }
        if base_url:
            kwargs["http_options"] = types.HttpOptions(base_url=base_url, **http_kwargs)
        else:
            kwargs["http_options"] = types.HttpOptions(**http_kwargs)
        return genai.Client(**kwargs)
