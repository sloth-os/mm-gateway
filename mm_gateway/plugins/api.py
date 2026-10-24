"""Small, provider-independent HTTP hook API."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from starlette.datastructures import MutableHeaders

from mm_gateway.config import BackendConfig


@dataclass(frozen=True)
class HookContext:
    plugin_id: str
    request_id: str
    key_id: str
    # Private to this plugin and this request; shared by its response hooks.
    state: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BackendContext:
    backend: BackendConfig
    plugin_id: str = ""
    request_id: str = ""
    key_id: str = ""

    @property
    def account_id(self) -> str:
        return self.backend.extra.get("__account_id", "default")


@dataclass
class BackendRequest:
    method: str
    url: str
    headers: MutableHeaders
    # Available for materialized HTTP bodies; never consumes upload streams.
    body: bytes | None = None


@dataclass(frozen=True)
class RouteCandidate:
    id: str
    backend: str
    account_id: str
    model: str
    estimated_cost: float | None
    tags: tuple[str, ...]


@dataclass(frozen=True)
class RouteContext:
    modality: str
    key_id: str
    request: dict[str, Any]
    candidates: tuple[RouteCandidate, ...]


@dataclass
class PluginRequest:
    method: str
    path: str
    headers: MutableHeaders
    query_string: bytes = b""
    body: bytes = b""

    def json(self) -> Any:
        return json.loads(self.body)

    def set_json(self, value: Any) -> None:
        self.body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode(
            "utf-8"
        )
        self.headers["content-type"] = "application/json"
        if "content-encoding" in self.headers:
            del self.headers["content-encoding"]


@dataclass
class PluginResponse:
    status_code: int = 200
    headers: MutableHeaders = field(default_factory=MutableHeaders)
    # None means the body is streamed, non-JSON, or encoded. Use the chunk hook
    # for streaming bytes; only manifests with response_body="json" buffer JSON.
    body: bytes | None = None
    streaming: bool = False

    def json(self) -> Any:
        if self.body is None:
            raise ValueError("This response body is not buffered.")
        return json.loads(self.body)

    def set_json(self, value: Any) -> None:
        if self.streaming:
            raise ValueError("Use on_response_chunk to modify a streaming body.")
        self.body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode(
            "utf-8"
        )
        self.headers["content-type"] = "application/json"

    @classmethod
    def from_json(cls, value: Any, status_code: int = 200) -> PluginResponse:
        response = cls(status_code=status_code)
        response.set_json(value)
        return response
