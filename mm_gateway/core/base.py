"""Abstract provider interfaces.

The gateway talks to providers only through these protocols, so adding a new
provider never touches the HTTP layer or the translators. Capabilities are
modelled separately — ``ImageProvider``, ``VideoProvider`` and
``MusicProvider`` — because not every provider supports all of them (e.g.
FLUX is image-only). A provider class may implement any subset.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from mm_gateway.schemas.image import UnifiedImageRequest, UnifiedImageTask
from mm_gateway.schemas.music import UnifiedMusicRequest, UnifiedMusicTask
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
    music_models: ClassVar[list[str]] = []

    def __init__(self, backend: Any):
        self.backend = backend

    @property
    def supports_image(self) -> bool:
        return isinstance(self, ImageProvider)

    @property
    def supports_video(self) -> bool:
        return isinstance(self, VideoProvider)

    @property
    def supports_music(self) -> bool:
        return isinstance(self, MusicProvider)


class ImageProvider(Provider):
    @abstractmethod
    async def create_image_task(self, request: UnifiedImageRequest) -> UnifiedImageTask:
        """Submit an image generation task; return a handle the gateway can poll.

        For synchronous providers (OpenAI, Imagen, Stability, xAI, Volcengine,
        FLUX) the adapter mints a synthetic in-memory task id and runs the
        blocking call on the first poll, so the create/poll surface is uniform.
        """
        ...

    @abstractmethod
    async def get_image_task(self, task_id: str) -> UnifiedImageTask:
        """Poll a previously submitted task by its provider-local id."""
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


class MusicProvider(Provider):
    @abstractmethod
    async def create_music_task(self, request: UnifiedMusicRequest) -> UnifiedMusicTask:
        """Submit a music generation task; return a handle the gateway can poll."""
        ...

    @abstractmethod
    async def get_music_task(self, task_id: str) -> UnifiedMusicTask:
        """Poll a previously submitted task by its provider-local id."""
        ...
