"""Translate canonical pixel dimensions to provider wire conventions."""

from __future__ import annotations

from math import gcd
from typing import Protocol


class DimensionedRequest(Protocol):
    width: int | None
    height: int | None
    size: str | None
    resolution: str | None


def pixel_size(request: DimensionedRequest, separator: str = "x") -> str | None:
    if request.size:
        return request.size.replace("*", separator).replace("x", separator)
    if request.width is None or request.height is None:
        return None
    return f"{request.width}{separator}{request.height}"


def aspect_ratio(request: DimensionedRequest) -> str | None:
    explicit = getattr(request, "aspect_ratio", None) or getattr(request, "ratio", None)
    if explicit:
        return explicit
    if request.width is None or request.height is None:
        return None
    divisor = gcd(request.width, request.height)
    return f"{request.width // divisor}:{request.height // divisor}"


def image_resolution(request: DimensionedRequest) -> str | None:
    if request.resolution:
        return request.resolution
    if request.width is None or request.height is None:
        return None
    return "1k" if max(request.width, request.height) <= 1024 else "2k"


def video_resolution(request: DimensionedRequest) -> str | None:
    if request.resolution:
        return request.resolution
    if request.width is None or request.height is None:
        return None
    return f"{min(request.width, request.height)}p"
