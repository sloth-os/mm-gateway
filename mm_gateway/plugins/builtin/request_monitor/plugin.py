import time

from mm_gateway.observability.logging import get_logger

log = get_logger("plugin.request-monitor")


class RequestMonitor:
    async def on_request(self, request, context):
        context.state.update(
            started=time.monotonic(), method=request.method, path=request.path
        )

    async def on_response(self, response, context):
        log.info(
            "plugin_request_observed",
            plugin_id=context.plugin_id,
            request_id=context.request_id,
            key_id=context.key_id,
            method=context.state["method"],
            path=context.state["path"],
            status=response.status_code,
            elapsed_seconds=time.monotonic() - context.state["started"],
        )


def create_plugin(config):
    return RequestMonitor()
