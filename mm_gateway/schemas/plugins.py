"""Plugin package, installation and management contracts."""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PluginId = str
HookName = Literal[
    "on_request",
    "on_response",
    "on_response_chunk",
    "prepare_backend",
    "on_backend_request",
    "route",
]
ID_PATTERN = r"^[a-z][a-z0-9_-]{0,63}$"


class PluginModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PluginManifest(PluginModel):
    id: str = Field(pattern=ID_PATTERN)
    name: str = Field(min_length=1, max_length=120)
    version: str = Field(min_length=1, max_length=80)
    api_version: Literal["1"] = "1"
    description: str = Field(default="", max_length=2000)
    author: str = Field(default="", max_length=120)
    homepage: str | None = Field(default=None, max_length=500)
    hooks: list[HookName] = Field(default_factory=list)
    providers: list[str] = Field(
        default_factory=list, description="New provider types exported by this plugin."
    )
    backend_types: list[str] = Field(
        default_factory=list,
        description="Backend types for automatic authentication hooks; * matches all.",
    )
    # Prefixes always match against the gateway path, without ROOT_PATH.
    paths: list[str] = Field(default_factory=lambda: ["/v1/", "/proxy/"])
    methods: list[str] = Field(default_factory=list)
    response_body: Literal["none", "json"] = "none"
    default_config: dict[str, Any] = Field(default_factory=dict)

    @field_validator("paths")
    @classmethod
    def valid_paths(cls, values):
        if not values or any(not p.startswith(("/v1/", "/proxy/")) for p in values):
            raise ValueError("paths must be non-empty prefixes under /v1/ or /proxy/")
        return values

    @field_validator("methods")
    @classmethod
    def valid_methods(cls, values):
        if any(
            v not in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}
            for v in values
        ):
            raise ValueError("methods must be uppercase HTTP methods")
        return values

    @model_validator(mode="after")
    def valid_hooks(self):
        if not self.hooks and not self.providers:
            raise ValueError("A plugin must declare hooks or provider types")
        if len(self.hooks) != len(set(self.hooks)):
            raise ValueError("hooks must be unique")
        if len(self.providers) != len(set(self.providers)) or any(
            not re.fullmatch(ID_PATTERN, p) for p in self.providers
        ):
            raise ValueError("Provider types must be unique lowercase identifiers")
        if self.response_body == "json" and "on_response" not in self.hooks:
            raise ValueError("response_body=json requires on_response")
        return self


class PluginOptions(PluginModel):
    enabled: bool = True
    order: int = Field(default=0, ge=-10000, le=10000)
    config: dict[str, Any] = Field(default_factory=dict)
    timeout_ms: int = Field(default=1000, ge=10, le=30000)
    failure_mode: Literal["closed", "open"] = "closed"


class GitHubPluginInstall(PluginModel):
    repository: str = Field(description="HTTPS GitHub repository URL.")
    ref: str = Field(
        default="HEAD", max_length=200, description="Branch, tag or commit to fetch."
    )
    subdirectory: str = Field(default="", max_length=300)
    options: PluginOptions = Field(default_factory=PluginOptions)

    @field_validator("repository")
    @classmethod
    def valid_repository(cls, value):
        if not re.fullmatch(
            r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?/?", value
        ):
            raise ValueError(
                "Use an HTTPS GitHub repository URL without credentials or query parameters"
            )
        return value.rstrip("/")

    @field_validator("ref")
    @classmethod
    def valid_ref(cls, value):
        if (
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", value)
            or ".." in value
            or "//" in value
            or value.endswith(("/", "."))
        ):
            raise ValueError("Use a branch, tag or commit id")
        return value

    @field_validator("subdirectory")
    @classmethod
    def valid_subdirectory(cls, value):
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or "\\" in value:
            raise ValueError("subdirectory must stay inside the repository")
        return "" if str(path) == "." else str(path)


class PluginSource(PluginModel):
    kind: Literal["builtin", "local", "github"]
    path: str = ""
    repository: str | None = None
    ref: str | None = None
    commit: str | None = None
    subdirectory: str = ""


class PluginRecord(PluginModel):
    options: PluginOptions
    source: PluginSource


class PluginEntry(PluginModel):
    id: str
    manifest: PluginManifest | None = None
    source: PluginSource
    installed: bool = False
    options: PluginOptions | None = None
    active: bool = False
    generation: str | None = None
    loaded_at: float | None = None
    error: str | None = None
    calls: int = 0
    failures: int = 0


class PluginDiscoveryError(PluginModel):
    path: str
    error: str


class PluginStore(PluginModel):
    revision: str
    persistent: bool
    watching: bool
    directory: str
    data: list[PluginEntry]
    errors: list[PluginDiscoveryError]
