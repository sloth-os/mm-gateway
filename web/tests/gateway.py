"""Local browser-test gateway. Uses in-memory providers; never calls a vendor."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import uvicorn
import tests.conftest as fakes
from mm_gateway.config import BackendConfig, KeyConfig, Settings
from mm_gateway.registry import _PROVIDER_CLASSES
from mm_gateway.server.app import create_app

_PROVIDER_CLASSES["fake"] = "FakeProvider"
sys.modules["mm_gateway.providers.fake"] = fakes

settings = Settings(
    root_path=os.environ.get("GATEWAY_TEST_ROOT_PATH", ""),
    management_api_key="management-test",
    keys=[KeyConfig(id="browser", key="browser-generation")],
    backends=[BackendConfig(name="test-provider", type="fake", api_key="test-provider-credential", tags=["test"])],
    poll_interval=0.01,
    log_level="WARNING",
)
app = create_app(settings)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("GATEWAY_TEST_PORT", "8765")),
                root_path=settings.root_path, log_level="warning")
