import time
from typing import ClassVar

from mm_gateway.core.base import ImageProvider
from mm_gateway.core.exceptions import ProviderNotConfiguredError
from mm_gateway.providers._http import make_client, request_json
from mm_gateway.providers._sync_image import SyncImageTaskMixin
from mm_gateway.schemas.image import ImageData, UnifiedImageResponse


class ExampleImageProvider(SyncImageTaskMixin, ImageProvider):
    image_models: ClassVar[list[str]] = ["example-image-v1"]

    def __init__(self, backend):
        super().__init__(backend)
        if not backend.base_url:
            raise ProviderNotConfiguredError(
                backend.name, "Set the example image service's Base URL."
            )
        self.name = backend.name
        self.client = make_client(
            backend.base_url,
            timeout=60,
            headers={"authorization": "Bearer " + backend.api_key}
            if backend.api_key
            else {},
            proxy_url=backend.extra.get("outbound_proxy"),
        )

    async def _generate_image(self, request):
        # This is an illustrative service contract; replace it with your vendor's.
        payload = await request_json(
            self.client,
            "POST",
            "generate",
            provider=self.name,
            json={
                "model": request.model,
                "prompt": request.prompt(),
                "count": request.n,
            },
        )
        return UnifiedImageResponse(
            created=int(time.time()),
            model=request.model,
            provider=self.name,
            data=[ImageData(url=uri) for uri in payload["images"]],
        )

    async def aclose(self):
        await self.client.aclose()


class Plugin:
    providers: ClassVar[dict[str, type[ImageProvider]]] = {
        "example-image": ExampleImageProvider
    }


def create_plugin(config):
    return Plugin()
