"""Administrative contracts, kept separate from the provider-neutral media API."""

from __future__ import annotations

from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator

from mm_gateway.schemas.api import UsageResponse

REDACTED = "[redacted]"
Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")]
Optimize = Literal["balanced", "cost", "latency"]
Fallback = Literal["none", "same_model", "any"]
Modality = Literal["image", "video", "music", "audio", "voice"]
TaskStatus = Literal["pending", "running", "succeeded", "failed", "cancelled", "expired"]
Amount = Annotated[float, Field(ge=0)]


class ManagementModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class BackendCredential(ManagementModel):
    id: Identifier
    api_key: str | None = None
    base_url: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class ManagedBackend(ManagementModel):
    name: Identifier
    type: str = Field(min_length=1)
    enabled: bool = True
    api_key: str | None = None
    base_url: str | None = None
    tags: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)
    credentials: list[BackendCredential] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_accounts(self):
        ids = [entry.id for entry in self.credentials]
        if len(ids) != len(set(ids)):
            raise ValueError("Backend credential ids must be unique.")
        return self


class ManagedBudget(ManagementModel):
    limit_usd: Amount | None = None
    period: Literal["day", "month", "total"] = "month"
    scopes_limit_usd: Amount | None = None


class ManagedKey(ManagementModel):
    id: Identifier
    key: str
    enabled: bool = True
    allow_tags: list[str] = Field(default_factory=list)
    deny_tags: list[str] = Field(default_factory=list)
    allow_backends: list[str] = Field(default_factory=list)
    default_image_tag: str | None = None
    default_video_tag: str | None = None
    default_music_tag: str | None = None
    default_audio_tag: str | None = None
    default_image_backend: str | None = None
    default_video_backend: str | None = None
    default_music_backend: str | None = None
    default_audio_backend: str | None = None
    budget: ManagedBudget | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class ProxyAccount(ManagementModel):
    id: Identifier
    headers: dict[str, str] = Field(default_factory=dict)


class ManagedProxy(ManagementModel):
    domain: str | None = None
    base_url: str
    enabled: bool = True
    tags: list[str] = Field(default_factory=list)
    headers: dict[str, str] = Field(default_factory=dict)
    accounts: list[ProxyAccount] = Field(default_factory=list)
    timeout: float = Field(default=120, gt=0)
    outbound_proxy: str | None = None

    @field_validator("base_url")
    @classmethod
    def valid_url(cls, value):
        if value != REDACTED:
            parsed = urlsplit(value)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.fragment:
                raise ValueError("Proxy base_url must be an HTTP(S) URL without a fragment.")
        return value

    @model_validator(mode="after")
    def unique_accounts(self):
        ids = [entry.id for entry in self.accounts]
        if len(ids) != len(set(ids)):
            raise ValueError("Proxy account ids must be unique.")
        if self.domain and self.base_url != REDACTED and self.domain.lower() != urlsplit(self.base_url).hostname:
            raise ValueError("Proxy domain must match the base_url hostname.")
        return self


class ManagedRoutingProfile(ManagementModel):
    tags: list[str] = Field(default_factory=list)
    optimize: Optimize | None = None
    max_cost_usd: float | None = Field(default=None, gt=0)
    fallback: Fallback | None = None


class ManagementConfig(ManagementModel):
    backends: list[ManagedBackend] = Field(default_factory=list)
    keys: list[ManagedKey] = Field(default_factory=list)
    proxies: list[ManagedProxy] = Field(default_factory=list)
    routing_default_optimize: Optimize = "balanced"
    routing_profiles: dict[str, ManagedRoutingProfile] = Field(default_factory=dict)
    catalog_models: dict[str, Any] = Field(default_factory=dict)
    budget_allow_unpriced: bool = False
    outbound_proxy: str | None = None

    @model_validator(mode="after")
    def unique_identities(self):
        for label, values in (
            ("backend names", [b.name for b in self.backends]),
            ("key ids", [k.id for k in self.keys]),
            ("proxy domains", [p.domain or urlsplit(p.base_url).hostname for p in self.proxies]),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"Duplicate {label}.")
        tokens = [key.key for key in self.keys if key.key and key.key != REDACTED]
        if len(tokens) != len(set(tokens)):
            raise ValueError("API key tokens must be unique.")
        return self


class ManagementConfigResponse(ManagementModel):
    revision: str
    persistent: bool
    config: ManagementConfig


class BackendRuntime(ManagementModel):
    name: str
    type: str
    enabled: bool
    configured: bool
    active: bool
    tags: list[str]
    accounts: list[str]
    models: dict[str, list[str]]


class ProxyRuntime(ManagementModel):
    domain: str
    enabled: bool
    active: bool
    accounts: list[str]


class ManagementStatus(ManagementModel):
    status: Literal["ok"] = "ok"
    version: str
    uptime_seconds: float
    revision: str
    persistent: bool
    metrics_enabled: bool
    backend_types: list[str]
    backends: list[BackendRuntime]
    proxies: list[ProxyRuntime]
    keys_count: int
    enabled_keys_count: int
    tasks_by_status: dict[str, int]


class CounterSample(ManagementModel):
    name: str
    labels: dict[str, str]
    value: float


class HistogramSample(ManagementModel):
    name: str
    labels: dict[str, str]
    count: int
    sum: float
    min: float
    max: float
    mean: float


class SelectionHealth(ManagementModel):
    backend: str
    account: str
    model: str | None
    modality: str
    success_rate: float | None
    latency_s: float | None
    rate_limited: bool
    cooldown_remaining_s: float
    attempts: int


class ManagementMetrics(ManagementModel):
    collected_at: str
    enabled: bool
    counters: list[CounterSample]
    histograms: list[HistogramSample]
    selection: list[SelectionHealth]


class ManagedTask(ManagementModel):
    id: str
    owner_key_id: str
    backend: str
    model: str
    modality: Modality
    status: TaskStatus
    created_at: int
    completed_at: int | None = None


class ManagementTaskList(ManagementModel):
    object: Literal["list"] = "list"
    data: list[ManagedTask]
    total: int
    offset: int
    limit: int


class ManagedUsage(ManagementModel):
    key_id: str
    enabled: bool
    usage: UsageResponse


class ManagementUsageList(ManagementModel):
    object: Literal["list"] = "list"
    data: list[ManagedUsage]
