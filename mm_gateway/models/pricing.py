"""Static, provider-neutral price catalogue (USD) used by auto mode's cost control.

Prices sit next to the limits catalogue (:mod:`mm_gateway.models.limits`) and
follow the same rules: only values published on a provider's official pricing
page are built in, each with ``source_urls`` and the date it was checked
(``as_of``). Operators set or override any price in configuration
(``catalog.models.<id>.price``) — a contract price, a reseller's price, or the
cost of a self-hosted model — so routing by cost and budgets never depend on
this table being complete. A model without a price is *unpriced*: its estimate
is ``None`` (docs/design/auto-mode.md#prices-and-estimates).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mm_gateway.models.limits import ModelLimits

# Defaults used when a request leaves the quantity open.
DEFAULT_VIDEO_SECONDS = 5.0
DEFAULT_MUSIC_SECONDS = 30.0


@dataclass(frozen=True)
class PriceTier:
    """A per-second price that applies up to an output size (longest side, px)."""

    max_longest_side: int
    per_second: float


@dataclass(frozen=True)
class ModelPrice:
    """One model's list price. Every field is optional; ``None`` = not charged that way."""

    per_second: float | None = None
    per_second_tiers: tuple[PriceTier, ...] = ()
    per_image: float | None = None
    per_request: float | None = None
    currency: str = "USD"
    as_of: str = ""
    source_urls: tuple[str, ...] = field(default=())

    @property
    def priced(self) -> bool:
        return any(v is not None for v in (self.per_second, self.per_image, self.per_request)) \
            or bool(self.per_second_tiers)

    def rate_for(self, longest_side: int | None) -> float | None:
        """The per-second rate for an output whose longest side is ``longest_side``."""
        if self.per_second_tiers:
            tiers = sorted(self.per_second_tiers, key=lambda t: t.max_longest_side)
            if longest_side is None:
                return tiers[0].per_second
            for tier in tiers:
                if longest_side <= tier.max_longest_side:
                    return tier.per_second
            return tiers[-1].per_second
        return self.per_second

    def to_public_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"currency": self.currency}
        for key in ("per_second", "per_image", "per_request"):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        if self.per_second_tiers:
            out["per_second_tiers"] = [
                {"max_longest_side": t.max_longest_side, "per_second": t.per_second}
                for t in self.per_second_tiers
            ]
        if self.as_of:
            out["as_of"] = self.as_of
        return out

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "ModelPrice":
        """Build a price from a configuration mapping (validated, USD only)."""
        allowed = {"per_second", "per_second_tiers", "per_image", "per_request", "currency", "as_of"}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"unknown price field(s): {', '.join(sorted(unknown))}")
        currency = str(raw.get("currency") or "USD").upper()
        if currency != "USD":
            raise ValueError("prices must be in USD")

        def amount(name: str) -> float | None:
            value = raw.get(name)
            if value is None:
                return None
            number = float(value)
            if number < 0:
                raise ValueError(f"{name} must not be negative")
            return number

        tiers = []
        for tier in raw.get("per_second_tiers") or []:
            side = int(tier["max_longest_side"])
            rate = float(tier["per_second"])
            if side <= 0 or rate < 0:
                raise ValueError("per_second_tiers need a positive max_longest_side and a non-negative per_second")
            tiers.append(PriceTier(max_longest_side=side, per_second=rate))
        return cls(
            per_second=amount("per_second"),
            per_second_tiers=tuple(tiers),
            per_image=amount("per_image"),
            per_request=amount("per_request"),
            as_of=str(raw.get("as_of") or ""),
        )


_GEMINI_PRICING = ("https://ai.google.dev/gemini-api/docs/pricing",)

_PRICES: dict[str, ModelPrice] = {
    # Veo 3.1 (Gemini API): per second of generated video, audio included.
    "veo-3.1-generate-preview": ModelPrice(
        per_second_tiers=(PriceTier(1920, 0.40), PriceTier(3840, 0.60)),
        as_of="2026-10-01", source_urls=_GEMINI_PRICING,
    ),
    "veo-3.1-fast-generate-preview": ModelPrice(
        per_second_tiers=(PriceTier(1280, 0.10), PriceTier(1920, 0.12), PriceTier(3840, 0.30)),
        as_of="2026-10-01", source_urls=_GEMINI_PRICING,
    ),
    "veo-3.1-lite-generate-preview": ModelPrice(
        per_second_tiers=(PriceTier(1280, 0.05), PriceTier(1920, 0.08)),
        as_of="2026-10-01", source_urls=_GEMINI_PRICING,
    ),
    # Gemini 2.5 Flash Image (standard tier): per output image.
    "gemini-2.5-flash-image": ModelPrice(
        per_image=0.039, as_of="2026-10-01", source_urls=_GEMINI_PRICING,
    ),
}


def builtin_prices() -> dict[str, ModelPrice]:
    """The built-in price table (a copy; the catalog merges configured overrides)."""
    return dict(_PRICES)


def estimate_cost(
    price: ModelPrice | None,
    *,
    modality: str,
    limits: ModelLimits,
    duration_seconds: float | None = None,
    output_count: int | None = None,
    longest_side: int | None = None,
) -> float | None:
    """USD estimate of one task, or ``None`` when the model is unpriced.

    image: outputs × ``per_image``; video: seconds × the size tier's rate (the
    model's minimum duration, else 5 s, when the request leaves it open); music:
    seconds × ``per_second`` (30 s by default). ``per_request`` is added to all.
    """
    if price is None or not price.priced:
        return None
    total = price.per_request or 0.0
    if modality == "image":
        if price.per_image is not None:
            total += price.per_image * max(1, output_count or 1)
    elif modality == "video":
        seconds = duration_seconds or limits.min_duration_seconds or DEFAULT_VIDEO_SECONDS
        rate = price.rate_for(longest_side)
        if rate is not None:
            total += rate * seconds * max(1, output_count or 1)
    else:
        seconds = duration_seconds or DEFAULT_MUSIC_SECONDS
        if price.per_second is not None:
            total += price.per_second * seconds * max(1, output_count or 1)
    return round(total, 6)


__all__ = [
    "DEFAULT_MUSIC_SECONDS",
    "DEFAULT_VIDEO_SECONDS",
    "ModelPrice",
    "PriceTier",
    "builtin_prices",
    "estimate_cost",
]
