"""Abstract provider interfaces.

The gateway talks to providers only through these protocols, so adding a new
provider never touches the HTTP layer or the translators. Two capabilities are
modelled separately — ``ImageProvider`` and ``VideoProvider`` — because not
every provider supports both (e.g. FLUX is image-only). A provider class may
implement either or both.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from mm_gateway.schemas.image import UnifiedImageRequest, UnifiedImageResponse
from mm_gateway.schemas.video import UnifiedVideoRequest, UnifiedVideoTask


class Provider(ABC):
    """Base class for all providers.

    Subclasses declare ``name`` and the model ids they support, and implement
    whichever capability protocols apply. Construction receives the resolved
    ``BackendConfig``; a provider should raise ``ProviderNotConfiguredError``
    in ``__init__`` if it cannot operate with the given credentials, so the
    registry can skip it.
    """

    name: ClassVar[str] = ""
    # Human-readable list of model ids this provider accepts, for /models and docs.
    image_models: ClassVar[list[str]] = []
    video_models: ClassVar[list[str]] = []

    def __init__(self, backend: Any):
        self.backend = backend

    @property
    def supports_image(self) -> bool:
        return isinstance(self, ImageProvider)

    @property
    def supports_video(self) -> bool:
        return isinstance(self, VideoProvider)


class ImageProvider(Provider):
    @abstractmethod
    async def generate_image(self, request: UnifiedImageRequest) -> UnifiedImageResponse:
        """Generate one or more images. Must populate ``provider`` on the response."""
        ...


class VideoProvider(Provider):
    @abstractmethod
    async def create_video_task(self, request: UnifiedVideoRequest) -> UnifiedVideoTask:
        """Submit a video generation task; return a handle the gateway can poll."""
        ...

    @abstractmethod
    async def get_video_task(self, task_id: str) -> UnifiedVideoTask:
        """Poll a previously submitted task by its provider-local id."""
        ...
