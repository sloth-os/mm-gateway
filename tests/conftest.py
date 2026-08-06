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
from mm_gateway.core.base import ImageProvider, VideoProvider
from mm_gateway.schemas.image import ImageData, UnifiedImageRequest, UnifiedImageResponse
from mm_gateway.schemas.video import UnifiedVideoRequest, UnifiedVideoTask, VideoUsage
from mm_gateway.server.app import create_app


class FakeProvider(ImageProvider, VideoProvider):
    """In-memory provider that records calls and returns deterministic output."""

    image_models = ["fake-image-1"]
    video_models = ["fake-video-1"]

    def __init__(self, backend: Any):
        super().__init__(backend)
        self.image_calls: list[UnifiedImageRequest] = []
        self.video_calls: list[UnifiedVideoRequest] = []
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

    async def generate_image(self, request: UnifiedImageRequest) -> UnifiedImageResponse:
        self.image_calls.append(request)
        return UnifiedImageResponse(
            created=int(time.time()),
            model=request.model,
            provider=self.name,
            data=[ImageData(url="https://example.test/out.png", revised_prompt=request.prompt)],
        )

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
