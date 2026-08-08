"""Synthetic create/poll task surface for synchronous image providers.

Most image backends (OpenAI, Imagen, Seedream, xAI, OpenRouter, Stable Image)
are synchronous: a single blocking call returns the finished images. The
gateway's image surface is task-based (create → poll) like video and music, so
this mixin wraps a provider's synchronous ``_generate_image`` as a synthetic
in-memory task: create mints a gateway-local id and records the request; the
first poll runs the blocking call and the task moves ``pending → running →
succeeded | failed``. Subsequent polls return the cached terminal task.

This mirrors the ElevenLabs music adapter and the Stability SVD video adapter.
The store is per-instance (one provider instance per backend in the registry),
so there is no cross-backend collision; it is process-local, like every other
in-memory task store in the gateway.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from mm_gateway.core.exceptions import (
    GatewayError,
    ProviderRequestError,
    TaskFailedError,
)
from mm_gateway.schemas.image import UnifiedImageRequest, UnifiedImageResponse, UnifiedImageTask


class SyncImageTaskMixin:
    """Mixin: implement ``create_image_task``/``get_image_task`` over a
    synchronous ``_generate_image``.

    Subclasses must call ``super().__init__(backend)`` (so this ``__init__``
    runs) and implement ``async def _generate_image(request) -> UnifiedImageResponse``.
    """

    def __init__(self, backend: Any) -> None:
        super().__init__(backend)  # type: ignore[misc]
        self._image_tasks: dict[str, dict[str, Any]] = {}

    async def create_image_task(self, request: UnifiedImageRequest) -> UnifiedImageTask:
        task_id = f"img-{uuid.uuid4().hex}"
        created_at = int(time.time())
        self._image_tasks[task_id] = {
            "model": request.model,
            "request": request,
            "status": "pending",
            "created_at": created_at,
        }
        return UnifiedImageTask(
            task_id=task_id, provider=self.name, model=request.model,  # type: ignore[attr-defined]
            status="pending", created_at=created_at,
        )

    async def get_image_task(self, task_id: str) -> UnifiedImageTask:
        rec = self._image_tasks.get(task_id)
        if rec is None:
            raise ProviderRequestError(
                f"image task {task_id} not found", provider=self.name,  # type: ignore[attr-defined]
                status_code=404,
            )
        if rec["status"] in ("succeeded", "failed"):
            return UnifiedImageTask(
                task_id=task_id, provider=self.name, model=rec["model"],  # type: ignore[attr-defined]
                status=rec["status"], images=rec.get("images", []),
                error=rec.get("error"), usage=rec.get("usage"),
                created_at=rec["created_at"], completed_at=rec.get("completed_at"),
            )
        # Run the blocking generation now.
        rec["status"] = "running"
        request: UnifiedImageRequest = rec["request"]
        try:
            resp: UnifiedImageResponse = await self._generate_image(request)  # type: ignore[attr-defined]
        except TaskFailedError as exc:
            rec["status"] = "failed"; rec["error"] = exc.message
            raise
        except GatewayError as exc:
            rec["status"] = "failed"; rec["error"] = exc.message
            raise
        except Exception as exc:  # noqa: BLE001
            rec["status"] = "failed"; rec["error"] = str(exc)
            raise ProviderRequestError(
                f"{self.name} image failed: {exc}",  # type: ignore[attr-defined]
                provider=self.name,  # type: ignore[attr-defined]
            ) from exc
        if not resp.data:
            rec["status"] = "failed"; rec["error"] = "provider returned no images"
            raise TaskFailedError(
                f"{self.name} image returned no images",  # type: ignore[attr-defined]
                provider=self.name,  # type: ignore[attr-defined]
            )
        rec["status"] = "succeeded"
        rec["completed_at"] = int(time.time())
        rec["images"] = resp.data
        rec["usage"] = resp.usage
        return UnifiedImageTask(
            task_id=task_id, provider=self.name, model=rec["model"],  # type: ignore[attr-defined]
            status="succeeded", images=resp.data, usage=resp.usage,
            created_at=rec["created_at"], completed_at=rec["completed_at"],
        )
