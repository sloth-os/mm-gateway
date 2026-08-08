"""Shared test fixtures.

A fake image+video provider is wired into a Registry so the full HTTP path can
be exercised without any network calls or real SDK credentials.
"""

from __future__ import annotations

import time
from typing import Any

import pytest
from fastapi.testclient import TestClient

from mm_gateway.config import BackendConfig, KeyConfig, Settings
from mm_gateway.core.base import ImageProvider, MusicProvider, VideoProvider
from mm_gateway.schemas.image import (
    ImageData,
    UnifiedImageRequest,
    UnifiedImageResponse,
    UnifiedImageTask,
)
from mm_gateway.schemas.music import MusicUsage, UnifiedMusicRequest, UnifiedMusicTask
from mm_gateway.schemas.video import UnifiedVideoRequest, UnifiedVideoTask, VideoUsage
from mm_gateway.server.app import create_app


class FakeProvider(ImageProvider, VideoProvider, MusicProvider):
    """In-memory provider that records calls and returns deterministic output."""

    image_models = ["fake-image-1"]
    video_models = ["fake-video-1"]
    music_models = ["fake-music-1"]

    def __init__(self, backend: Any):
        super().__init__(backend)
        self.image_calls: list[UnifiedImageRequest] = []
        self.video_calls: list[UnifiedVideoRequest] = []
        self.music_calls: list[UnifiedMusicRequest] = []
        # task_id -> status (transitions pending -> running -> succeeded on poll)
        self._tasks: dict[str, str] = {}
        self._polls: dict[str, int] = {}

    @property
    def name(self) -> str:  # type: ignore[override]
        # ``Provider.name`` is a class attribute ("fake" on this class); instances
        # built for differently-named backends (fake-img, fake-vid, ...) must
        # report the backend name they were constructed with so the gateway's
        # routing/store layers can look the owning backend up by it.
        return self.backend.name

    async def create_image_task(self, request: UnifiedImageRequest) -> UnifiedImageTask:
        self.image_calls.append(request)
        task_id = f"img-{len(self.image_calls)}"
        self._tasks[task_id] = "pending"
        self._polls[task_id] = 0
        return UnifiedImageTask(
            task_id=task_id, provider=self.name, model=request.model, status="pending",
        )

    async def get_image_task(self, task_id: str) -> UnifiedImageTask:
        if task_id not in self._tasks:
            from mm_gateway.core.exceptions import ProviderRequestError
            raise ProviderRequestError(
                f"image task {task_id} not found", provider=self.name, status_code=404,
            )
        state = self._tasks.get(task_id, "pending")
        n = self._polls.get(task_id, 0) + 1
        self._polls[task_id] = n
        if state == "pending":
            state = "running"
        elif state == "running":
            state = "succeeded"
        self._tasks[task_id] = state
        task = UnifiedImageTask(
            task_id=task_id, provider=self.name, model="fake-image-1", status=state,
        )
        if state == "succeeded":
            req = self.image_calls[0] if self.image_calls else None
            task.images = [ImageData(
                url="https://example.test/out.png",
                revised_prompt=req.prompt() if req else None,
            )]
        return task

    async def create_video_task(self, request: UnifiedVideoRequest) -> UnifiedVideoTask:
        self.video_calls.append(request)
        task_id = f"task-{len(self.video_calls)}"
        self._tasks[task_id] = "pending"
        self._polls[task_id] = 0
        return UnifiedVideoTask(
            task_id=task_id, provider=self.name, model=request.model, status="pending",
        )

    async def get_video_task(self, task_id: str) -> UnifiedVideoTask:
        state = self._tasks.get(task_id, "pending")
        n = self._polls.get(task_id, 0) + 1
        self._polls[task_id] = n
        # pending -> running -> succeeded across two polls.
        if state == "pending":
            state = "running"
        elif state == "running":
            state = "succeeded"
        self._tasks[task_id] = state
        task = UnifiedVideoTask(
            task_id=task_id, provider=self.name, model="fake-video-1", status=state,
        )
        if state == "succeeded":
            task.video_urls = ["https://example.test/out.mp4"]
            task.usage = VideoUsage(cost=0.01, video_count=1)
        return task

    async def create_music_task(self, request: UnifiedMusicRequest) -> UnifiedMusicTask:
        self.music_calls.append(request)
        task_id = f"music-{len(self.music_calls)}"
        self._tasks[task_id] = "pending"
        self._polls[task_id] = 0
        return UnifiedMusicTask(
            task_id=task_id, provider=self.name, model=request.model, status="pending",
        )

    async def get_music_task(self, task_id: str) -> UnifiedMusicTask:
        if task_id not in self._tasks:
            # Unknown id -> not found, mirroring real providers that reject ids
            # they never minted.
            from mm_gateway.core.exceptions import ProviderRequestError
            raise ProviderRequestError(
                f"task {task_id} not found", provider=self.name, status_code=404,
            )
        state = self._tasks.get(task_id, "pending")
        n = self._polls.get(task_id, 0) + 1
        self._polls[task_id] = n
        if state == "pending":
            state = "running"
        elif state == "running":
            state = "succeeded"
        self._tasks[task_id] = state
        task = UnifiedMusicTask(
            task_id=task_id, provider=self.name, model="fake-music-1", status=state,
        )
        if state == "succeeded":
            task.audio_b64 = "AAAA"  # tiny base64 placeholder
            task.audio_media_type = "audio/wav"
            task.lyrics = "la la la"
            task.usage = MusicUsage(cost=0.01, duration=8)
        return task


@pytest.fixture
def fake_provider() -> FakeProvider:
    return FakeProvider(BackendConfig(name="fake", type="fake", api_key="test"))


@pytest.fixture
def settings() -> Settings:
    # An open key (no allow_tags/allow_backends) may use every backend, and an
    # empty token means no Authorization header is required.
    key = KeyConfig(id="test", key="")
    return Settings(
        backends=[BackendConfig(name="fake", type="fake", api_key="test")],
        keys=[key],
        video_sync_default=True,
        max_sync_wait=5.0,
        poll_interval=0.01,
    )


@pytest.fixture
def app(settings, fake_provider):
    app = create_app(settings)
    # Inject the fake provider directly, bypassing the registry's constructor
    # logic (which would try to import a real provider module).
    app.state.registry._backends["fake"] = fake_provider
    app.state.registry._configs["fake"] = settings.backends[0]
    return app


@pytest.fixture
def client(app) -> TestClient:
    return TestClient(app)
