"""The effective model catalogue: built-in limits and prices plus operator overrides.

``catalog.models.<id>`` in the configuration can retire or deprecate a model,
name its replacement, declare native audio, adjust a few routing limits, and
set its price — without waiting for a gateway release
(docs/design/auto-mode.md#configuration). The registry owns one ``Catalog``;
the router, the estimate endpoint and the ledger all read through it, so they
agree on every model's limits, lifecycle and price.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from datetime import date
from typing import Any

from mm_gateway.core.exceptions import ConfigError
from mm_gateway.models.limits import Lifecycle, ModelLimits, builtin_limits, limits_for, utc_today
from mm_gateway.models.pricing import ModelPrice, builtin_prices

# ModelLimits fields an operator may override (routing-relevant, documented values only).
_OVERRIDABLE = {
    "deprecated_on", "retired_on", "replacement", "supports_audio_output",
    "supports_reference_audio", "supports_reference_image", "supports_reference_video",
    "supports_first_frame", "supports_last_frame", "min_duration_seconds",
    "max_duration_seconds", "max_input_images", "notes", "max_shots", "max_fps",
    "supports_upscale", "supports_frame_interpolation", "supports_segmentation",
}
_DATES = {"deprecated_on", "retired_on"}


@dataclasses.dataclass(frozen=True)
class ModelOverride:
    modality: str | None
    fields: dict[str, Any]
    price: ModelPrice | None


def parse_overrides(raw: dict[str, Any] | None) -> dict[str, ModelOverride]:
    """Validate ``catalog.models`` from the configuration (raises ConfigError)."""
    out: dict[str, ModelOverride] = {}
    for model, spec in (raw or {}).items():
        if not isinstance(spec, dict):
            raise ConfigError(f"catalog.models.{model} must be a mapping")
        spec = dict(spec)
        modality = spec.pop("modality", None)
        if modality is not None and modality not in ("image", "video", "music"):
            raise ConfigError(f"catalog.models.{model}.modality must be image, video or music")
        price_raw = spec.pop("price", None)
        unknown = set(spec) - _OVERRIDABLE
        if unknown:
            raise ConfigError(
                f"catalog.models.{model}: unknown field(s) {', '.join(sorted(unknown))}"
            )
        for name in _DATES & set(spec):
            try:
                spec[name] = date.fromisoformat(str(spec[name])).isoformat()
            except ValueError as exc:
                raise ConfigError(f"catalog.models.{model}.{name} must be an ISO date") from exc
        try:
            price = ModelPrice.from_mapping(price_raw) if price_raw is not None else None
        except (ValueError, KeyError, TypeError) as exc:
            raise ConfigError(f"catalog.models.{model}.price: {exc}") from exc
        out[str(model)] = ModelOverride(modality=modality, fields=spec, price=price)
    return out


class Catalog:
    """Limits, lifecycle and prices of every model the gateway can route to."""

    def __init__(
        self,
        overrides: dict[str, ModelOverride] | None = None,
        *,
        today: Callable[[], date] = utc_today,
    ) -> None:
        self._overrides = dict(overrides or {})
        self._limits = builtin_limits()
        self._prices = builtin_prices()
        self._today = today

    def today(self) -> date:
        return self._today()

    def limits_for(self, model: str, modality: str) -> ModelLimits:
        """Built-in limits (or the permissive default) with the operator's overrides applied."""
        base = limits_for(model, modality)
        override = self._overrides.get(model)
        if override is None or (override.modality and override.modality != modality):
            return base
        return dataclasses.replace(base, **override.fields) if override.fields else base

    def price_for(self, model: str) -> ModelPrice | None:
        override = self._overrides.get(model)
        if override is not None and override.price is not None:
            return override.price
        return self._prices.get(model)

    def lifecycle(self, model: str, modality: str) -> Lifecycle:
        return self.limits_for(model, modality).lifecycle(self.today())

    def public_limits(self, model: str, modality: str) -> dict[str, Any]:
        """The client-facing limits object, including the model's price when known."""
        out = self.limits_for(model, modality).to_public_dict(self.today())
        price = self.price_for(model)
        if price is not None and price.priced:
            out["price"] = price.to_public_dict()
        return out


__all__ = ["Catalog", "ModelOverride", "parse_overrides"]
