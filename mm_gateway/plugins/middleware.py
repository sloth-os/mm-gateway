"""HTTP hooks with bounded JSON buffering and a streaming ASGI send path."""

from __future__ import annotations

import asyncio
import re
import time
from copy import deepcopy

from starlette.datastructures import MutableHeaders
from starlette.requests import Request
from starlette.responses import JSONResponse

from mm_gateway.core.exceptions import GatewayError
from mm_gateway.observability.logging import get_logger
from mm_gateway.observability.metrics import STORE as METRICS
from mm_gateway.plugins.api import HookContext, PluginRequest, PluginResponse
from mm_gateway.server.auth import _extract_token, resolve_key

log = get_logger("plugins")
FAILED = object()


class PipelineError(Exception):
    def __init__(
        self,
        detail="A gateway plugin could not process this request.",
        *,
        code="plugin_error",
        status=500,
    ):
        self.detail, self.code, self.status = detail, code, status


def _validate_headers(headers):
    if not isinstance(headers, MutableHeaders):
        raise TypeError("headers must be MutableHeaders")
    for name, value in headers.raw:
        if not re.fullmatch(rb"[!#$%&'*+.^_`|~0-9A-Za-z-]+", name):
            raise ValueError("Invalid header name")
        if any(c in value for c in (b"\r", b"\n", b"\0")):
            raise ValueError("Invalid header value")


def _validate_response(response, *, streaming):
    if type(response.status_code) is not int or not 200 <= response.status_code <= 599:
        raise ValueError("Invalid HTTP response status")
    _validate_headers(response.headers)
    if response.streaming != streaming or (streaming and response.body is not None):
        raise ValueError("Cannot replace a streaming body in on_response")
    if response.body is not None and not isinstance(response.body, bytes):
        raise TypeError("response.body must be bytes")


def _invalidate_representation(headers):
    for name in (
        "content-length",
        "etag",
        "content-md5",
        "digest",
        "content-digest",
        "last-modified",
    ):
        if name in headers:
            del headers[name]


async def _invoke(plugin, hook, args, validate):
    plugin.calls += 1
    started = time.monotonic()
    outcome = "success"
    try:
        async with asyncio.timeout(plugin.options.timeout_ms / 1000):
            result = await getattr(plugin.instance, hook)(*args)
        validate(result)
        return result
    except Exception as exc:  # noqa: BLE001 - contributor exceptions follow the configured failure policy
        outcome = "error"
        plugin.failures += 1
        # Exception messages can contain inspected bodies or plugin credentials.
        log.warning(
            "plugin_hook_failed",
            plugin_id=plugin.manifest.id,
            hook=hook,
            error_type=type(exc).__name__,
        )
        if plugin.options.failure_mode == "closed":
            raise PipelineError() from None
        return FAILED
    finally:
        METRICS.inc_counter(
            "gateway_plugin_hooks_total",
            plugin=plugin.manifest.id,
            hook=hook,
            outcome=outcome,
        )
        METRICS.observe(
            "gateway_plugin_hook_duration_seconds",
            time.monotonic() - started,
            plugin=plugin.manifest.id,
            hook=hook,
        )


class PluginMiddleware:
    def __init__(self, app, manager):
        self.app, self.manager = app, manager

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        path = scope["path"]
        prefix = scope.get("root_path", "").rstrip("/")
        if prefix and path.startswith(prefix + "/"):
            path = path[len(prefix) :]
        # The control plane, console, health and metrics never execute plugins.
        if (
            not path.startswith(("/v1/", "/proxy/"))
            or path == "/v1/management"
            or path.startswith("/v1/management/")
        ):
            return await self.app(scope, receive, send)
        async with self.manager.lease(path, scope["method"]) as plugins:
            if not plugins:
                return await self.app(scope, receive, send)
            started = False

            async def tracked_send(message):
                nonlocal started
                if message["type"] == "http.response.start":
                    started = True
                await send(message)

            try:
                await self._process(scope, receive, tracked_send, plugins, path)
            except (PipelineError, GatewayError) as exc:
                if started:
                    # HTTP headers already reached the client: abort the stream.
                    raise RuntimeError("Plugin stream processing failed.") from None
                status = (
                    exc.status if isinstance(exc, PipelineError) else exc.status_code
                )
                code = exc.code if isinstance(exc, PipelineError) else exc.public_code
                detail = (
                    exc.detail if isinstance(exc, PipelineError) else exc.public_message
                )
                response = JSONResponse(
                    {
                        "type": f"urn:mm-gateway:problem:{code}",
                        "title": code.replace("_", " ").title(),
                        "status": status,
                        "code": code,
                        "detail": detail,
                        "instance": scope["path"],
                        "request_id": scope.get("state", {}).get("request_id"),
                    },
                    status_code=status,
                    media_type="application/problem+json",
                )
                await response(scope, receive, tracked_send)

    async def _process(self, scope, receive, send, plugins, path):
        original_request = Request(scope, receive)
        # Resolve the original caller before any code runs. The route dependency
        # uses this same identity even if a plugin edits Authorization headers.
        key = resolve_key(scope["app"].state.settings, _extract_token(original_request))
        scope.setdefault("state", {})["plugin_api_key"] = key
        limit = self.manager.settings.plugins_max_body_bytes
        body = bytearray()
        needs_body = any("on_request" in p.manifest.hooks for p in plugins)
        if needs_body:
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                body.extend(message.get("body", b""))
                if len(body) > limit:
                    raise PipelineError(
                        "Request exceeds the plugin body limit.",
                        code="plugin_body_too_large",
                        status=413,
                    )
                if not message.get("more_body", False):
                    break
        request = PluginRequest(
            scope["method"],
            path,
            MutableHeaders(raw=list(scope["headers"])),
            scope.get("query_string", b""),
            bytes(body),
        )
        contexts = {
            p.manifest.id: HookContext(
                p.manifest.id, scope["state"].get("request_id", ""), key.id
            )
            for p in plugins
        }
        entered = []
        short_circuit = None
        for plugin in plugins:
            entered.append(plugin)
            if "on_request" not in plugin.manifest.hooks:
                continue
            draft = deepcopy(request)

            def validate(result, draft=draft):
                _validate_headers(draft.headers)
                if draft.path != path or draft.method != scope["method"]:
                    raise ValueError("Request method and path are read-only")
                if not isinstance(draft.body, bytes) or not isinstance(
                    draft.query_string, bytes
                ):
                    raise TypeError("Request body and query_string must be bytes")
                if len(draft.body) > limit:
                    raise ValueError("Modified request exceeds the plugin body limit")
                if result is not None:
                    if not isinstance(result, PluginResponse):
                        raise TypeError("on_request must return None or PluginResponse")
                    _validate_response(result, streaming=False)
                    if result.body is not None and len(result.body) > limit:
                        raise ValueError(
                            "Plugin response exceeds the plugin body limit"
                        )

            result = await _invoke(
                plugin, "on_request", (draft, contexts[plugin.manifest.id]), validate
            )
            if result is FAILED:
                continue
            request = draft
            if result is not None:
                short_circuit = result
                break
        plugins = tuple(reversed(entered))
        scope = {
            **scope,
            "headers": list(request.headers.raw),
            "query_string": request.query_string,
        }
        if needs_body:
            headers = MutableHeaders(scope=scope)
            if request.body != bytes(body):
                _invalidate_representation(headers)
            headers["content-length"] = str(len(request.body))
            replayed = False
            original_receive = receive

            async def replay_receive():
                nonlocal replayed
                if not replayed:
                    replayed = True
                    return {
                        "type": "http.request",
                        "body": request.body,
                        "more_body": False,
                    }
                return await original_receive()

            receive = replay_receive

        chunk_plugins = [p for p in plugins if "on_response_chunk" in p.manifest.hooks]

        async def response_hooks(response):
            for plugin in plugins:
                if "on_response" not in plugin.manifest.hooks:
                    continue
                draft = deepcopy(response)

                def validate(result, draft=draft, streaming=response.streaming):
                    if result is not None:
                        raise TypeError("on_response must return None")
                    _validate_response(draft, streaming=streaming)
                    if draft.body is not None and len(draft.body) > limit:
                        raise ValueError(
                            "Modified response exceeds the plugin body limit"
                        )

                result = await _invoke(
                    plugin,
                    "on_response",
                    (draft, contexts[plugin.manifest.id]),
                    validate,
                )
                if result is not FAILED:
                    response = draft
            return response

        async def chunk_hooks(chunk, more_body):
            for plugin in chunk_plugins:

                def validate(result):
                    if result is not None and not isinstance(result, bytes):
                        raise TypeError("on_response_chunk must return bytes or None")
                    if isinstance(result, bytes) and len(result) > limit:
                        raise ValueError("Modified chunk exceeds the plugin body limit")

                result = await _invoke(
                    plugin,
                    "on_response_chunk",
                    (chunk, contexts[plugin.manifest.id], more_body),
                    validate,
                )
                if result is not FAILED and result is not None:
                    chunk = result
            return chunk

        head = None
        response = None
        buffer = bytearray()
        buffering = False

        def body_allowed():
            return scope["method"] != "HEAD" and response.status_code not in {
                204,
                205,
                304,
            }

        async def process_send(message):
            nonlocal head, response, buffering
            if message["type"] == "http.response.start":
                head = dict(message)
                headers = MutableHeaders(raw=list(message["headers"]))
                content_type = (
                    headers.get("content-type", "").split(";", 1)[0].strip().lower()
                )
                is_json = content_type == "application/json" or content_type.endswith(
                    "+json"
                )
                buffering = (
                    scope["method"] != "HEAD"
                    and message["status"] not in {204, 205, 304}
                    and is_json
                    and headers.get("content-encoding", "identity") == "identity"
                    and any(p.manifest.response_body == "json" for p in plugins)
                )
                response = PluginResponse(
                    message["status"], headers, streaming=not buffering
                )
                if not buffering:
                    response = await response_hooks(response)
                    if chunk_plugins:
                        _invalidate_representation(response.headers)
                    if (
                        response.status_code in {204, 205, 304}
                        and "content-length" in response.headers
                    ):
                        del response.headers["content-length"]
                    head.update(
                        status=response.status_code, headers=list(response.headers.raw)
                    )
                    await send(head)
            elif message["type"] == "http.response.body":
                chunk = message.get("body", b"")
                more_body = message.get("more_body", False)
                if buffering:
                    buffer.extend(chunk)
                    if len(buffer) > limit:
                        raise PipelineError(
                            "Response exceeds the plugin body limit.",
                            code="plugin_body_too_large",
                            status=502,
                        )
                    if more_body:
                        return
                    original = bytes(buffer)
                    response.body = original
                    response = await response_hooks(response)
                    chunk = await chunk_hooks(response.body or b"", False)
                    if not body_allowed():
                        chunk = b""
                    if chunk != original or chunk_plugins:
                        _invalidate_representation(response.headers)
                    if response.status_code in {204, 205, 304}:
                        if "content-length" in response.headers:
                            del response.headers["content-length"]
                    else:
                        response.headers["content-length"] = str(len(chunk))
                    head.update(
                        status=response.status_code, headers=list(response.headers.raw)
                    )
                    await send(head)
                else:
                    chunk = await chunk_hooks(chunk, more_body)
                    if not body_allowed():
                        chunk = b""
                await send({**message, "body": chunk})
            else:
                await send(message)

        if short_circuit:
            await process_send(
                {
                    "type": "http.response.start",
                    "status": short_circuit.status_code,
                    "headers": list(short_circuit.headers.raw),
                }
            )
            await process_send(
                {
                    "type": "http.response.body",
                    "body": short_circuit.body or b"",
                    "more_body": False,
                }
            )
        else:
            await self.app(scope, receive, process_send)
