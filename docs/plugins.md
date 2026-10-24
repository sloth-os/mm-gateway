# Plugins

The Plugin Store installs and hot reloads contributor code for HTTP traffic,
backend authentication, new provider adapters and custom routing policies.
Open `/admin/` and choose **Plugin Store**. Plugins are disabled until installed;
discovering a package does not import its Python code.

## Install and operate

The store lists bundled packages, packages under `PLUGINS_DIR` (default
`./plugins`), and installed GitHub packages. Use **Install** to activate a package,
configure it with a JSON object, and choose its execution order, hook timeout and
failure policy. **Disable**, **Enable**, **Reload**, and **Uninstall** apply live.
Uninstalling a local package preserves the contributor's source files.

**Install from GitHub** takes an HTTPS repository URL such as
`https://github.com/your-org/your-plugins`, a branch/tag/commit (default `HEAD`),
and an optional directory containing `plugin.json` and `plugin.py`. Use the root
directory when one repository contains one plugin; use a subdirectory for a
collection. The gateway fetches the public repository with Git, records its exact
commit, and copies the selected package to its managed installation directory.
**Update** fetches that saved ref again and activates the new package after
validation. Pin a commit to keep a reproducible version. A package must retain
its plugin ID when updated.

Plugins run trusted Python with the gateway process's permissions. Install code
you trust and provide any additional Python dependencies in the gateway's
environment/image. Installing a repository does not run its package managers,
install dependencies, initialize submodules or run Git hooks. The runtime Docker
image includes Git. The initial installer supports public GitHub repositories.

The bundled packages are:

| ID | Configuration | Purpose |
| --- | --- | --- |
| `request-monitor` | `{}` | Log request/key identity, response status and elapsed time |
| `response-headers` | `{"headers":{"x-example":"value"}}` | Add response headers |
| `backend-bearer` | `{"token":"your-backend-token"}` | Supply SDK and outgoing HTTP credentials |
| `route-preference` | `{"preferred_backends":["primary","secondary"]}` | Prefer named admissible backends |

Plugin settings are persisted in `PLUGINS_STATE_PATH`, or `plugins.json` beside
`MANAGEMENT_CONFIG_PATH`. GitHub source lives in `installed-plugins` beside that
state file. Docker Compose keeps both in its existing data volume. If neither
state path is configured, install settings last for the process lifetime and
downloaded source uses `.mm-gateway/installed-plugins`. Configuration fields
recognized as credentials are encrypted with the gateway's secret store and
redacted from management responses. Preserve `[redacted]` values when editing.
Keep secrets in installed configuration rather than source or manifest defaults.

## Develop locally

```bash
pip install -e '.[dev]'
mm-gateway-plugin new my-plugin
mm-gateway-plugin check plugins/my-plugin
export MANAGEMENT_API_KEY=your-admin-token
export MANAGEMENT_CONFIG_PATH=./data/management.json
mm-gateway
```

Install **My Plugin** in the store, then edit its source. The watcher hashes
package source and resources, including relative helper modules, and reloads
changes without restarting the server. Packages load from fresh snapshots and
module namespaces. Existing HTTP requests/streams and provider tasks retain
their owning generation. Syntax errors, invalid hooks or provider initialization
failures leave the last working generation active and show an error in the store.
Newly discovered packages require an explicit install.

`mm-gateway-plugin check PATH` imports trusted code, constructs the plugin with
its defaults, validates its declared async hooks and provider classes, and closes
it. It does not test real provider credentials or upstream behavior.

Each directory contains a manifest:

```json
{
  "id": "my-plugin",
  "name": "My Plugin",
  "version": "0.1.0",
  "api_version": "1",
  "description": "Monitor and transform generation traffic",
  "author": "Your name",
  "hooks": ["on_request", "on_response"],
  "paths": ["/v1/", "/proxy/"],
  "methods": [],
  "response_body": "json",
  "default_config": {"label": "my-plugin"}
}
```

IDs start with a lowercase letter and use up to 64 lowercase letters, digits,
hyphens or underscores. API version `1` is supported. `hooks` must list the
methods implemented by the plugin; a provider-only package can omit hooks and
declare `providers` instead. A package must contain `plugin.py`, export
`create_plugin(config)`, and return an object with the declared methods. The
factory may be synchronous or asynchronous. Callbacks must be asynchronous.
Installed configuration overrides manifest defaults at the top level. An optional
async `aclose()` releases plugin resources after its generation is no longer in use.
Plugin packages cannot contain symlinks and are limited to 512 files / 16 MiB.
Use relative imports for helpers, e.g. `from .helpers import transform`.

## HTTP hooks

```python
from mm_gateway.plugins import HookContext, PluginRequest, PluginResponse

class Plugin:
    def __init__(self, config):
        self.label = config["label"]

    async def on_request(self, request: PluginRequest, context: HookContext):
        context.state["path"] = request.path
        if request.method == "POST" and request.path == "/v1/images":
            payload = request.json()
            payload["input"][0]["text"] += " — cinematic lighting"
            request.set_json(payload)

    async def on_response(self, response: PluginResponse, context: HookContext):
        response.headers["x-plugin"] = self.label
        if response.body is not None:
            payload = response.json()
            payload["plugin_label"] = self.label
            response.set_json(payload)

def create_plugin(config):
    return Plugin(config)
```

`on_request(request, context)` can edit `headers` (Starlette `MutableHeaders`),
`query_string` (bytes), and `body` (bytes). Method and path are read-only.
`json()` / `set_json(value)` handle JSON encoding. Return `None` to continue or
`PluginResponse.from_json(payload, status_code=403)` to short circuit the handler.
The gateway still validates modified generation input before contacting providers.
The original API-key identity is resolved before hooks and remains authoritative.

`on_response(response, context)` can edit status and headers. Set
`response_body: "json"` to buffer uncompressed JSON bodies and modify their bytes
or use `json()` / `set_json()`. With `response_body: "none"` (default), or for
non-JSON/encoded/event-stream responses, `body` is `None` and `streaming` is true;
the hook sees headers without buffering the body. `HEAD`, 204, 205 and 304 bodies
are never buffered. Changed representations have their length/validators repaired.

To monitor or transform streams, declare `on_response_chunk`:

```python
async def on_response_chunk(self, chunk: bytes, context: HookContext, more_body: bool):
    # Return bytes to replace this chunk, or None to observe it unchanged.
    return chunk
```

Chunks are raw transport bytes, including any encoding; UTF-8 characters and
SSE events can span chunk boundaries. Keep parsing state in `context.state` if
needed. The last call has `more_body=False`. Streams flow incrementally; JSON
bodies already buffered for `on_response` are delivered as one final chunk.
Chunk hooks remove the upstream content length and validators before streaming.

`HookContext` contains `plugin_id`, `request_id`, `key_id` and a private per-request
`state` dictionary shared by that plugin's HTTP hooks. Request hooks run in
ascending `(order, id)` order; response/chunk hooks run in reverse order. A short
circuit unwinds only plugins already entered. Path/method filters apply to HTTP
hooks and use paths without the reverse proxy's `ROOT_PATH` prefix.
Management endpoints, the admin console, health, metrics, MCP transport and
WebSocket frames are outside these HTTP hooks. Provider and routing extensions
are shared by generation through REST and MCP.

## Backend authentication

Set a backend's provider options to `{"auth_plugin":"backend-bearer"}` and
configure the plugin's `token`. A separate static provider API key is optional.
For HTTP pass-through proxies, set the proxy's `auth_plugin` field; this uses
`on_backend_request` on each account attempt. The proxy auth hook covers HTTP,
including streamed responses, and does not change WebSocket handshake auth.

Contributors can implement two independent hooks:

```python
from dataclasses import replace

async def prepare_backend(self, backend, context):
    token = await self.obtain_token()
    return replace(backend, api_key=token)

async def on_backend_request(self, request, context):
    token = await self.obtain_or_refresh_token()
    request.headers["authorization"] = "Bearer " + token
```

Declare them in `hooks`. `prepare_backend` receives a per-account `BackendConfig`
and `BackendContext` before provider construction. Return a new config with
derived `api_key`, `base_url`, or `extra`, or mutate `extra` and return `None`.
Backend identity, models, tags and routing permissions stay fixed. This supports
SDKs that consume credentials at construction, including transports other than
HTTPX. Derived credentials are runtime-only and never replace saved backend settings.

`on_backend_request` runs on HTTPX requests from gateway provider clients, model
discovery, and HTTP proxies. It receives `BackendRequest` (method, URL, mutable
headers, and materialized body bytes when available) plus `BackendContext`
(per-account backend, plugin ID, request ID and account ID). It can change URL
and headers, allowing token refresh and request signing immediately before send.
It never consumes streaming uploads. New adapters should use
`mm_gateway.providers._http.make_client` or `backend_event_hooks()` on their HTTPX
clients to participate. Use `asyncio.to_thread` for synchronous SDK calls with
wire auth hooks. Other SDK transports use prepared credentials or their adapter's
native authentication extension.

An explicit `auth_plugin` selects that plugin. Alternatively a manifest's
`backend_types: ["openai", "example-image"]` (or `["*"]`) applies auth hooks
automatically to those types. Type `proxy` selects HTTP proxies. Disabling an
explicitly selected auth plugin makes that backend unavailable; requests do not
fall back to anonymous authentication. Prepare credentials again by reloading
the plugin; per-request refresh belongs in `on_backend_request`.

## New backend providers

Declare `"providers": ["example-image"]` in the manifest and return an instance
with a `providers` mapping from each declared type to a concrete gateway provider
class. Existing built-in types cannot be replaced.

```python
class Plugin:
    providers = {"example-image": ExampleImageProvider}

def create_plugin(config):
    return Plugin()
```

Subclass the relevant interfaces in `mm_gateway.core.base`: `ImageProvider`,
`VideoProvider`, `MusicProvider`, `AudioProvider`, or `VoiceCloneProvider`.
Declare the modality's model list and implement its async create/get task
methods. Constructors receive the resolved per-account `BackendConfig`.
Adapters may raise `ProviderNotConfiguredError` for missing configuration and
provide `aclose()` to release clients. The gateway owns polling, task caching,
cost settlement and account retries. Use unique task IDs across generations.

The [complete image service example](../examples/plugins/example_provider/plugin.py)
adapts `POST /generate` with `{model,prompt,count}` to an illustrative upstream
that returns `{"images":["https://..."]}`. To try it, copy its package into
`PLUGINS_DIR`, install **Example HTTP image provider**, then add a backend of type
`example-image` with your service's Base URL. The type and its model suggestions
appear in the backend editor. It supports the normal `/v1/images`, model listing,
estimates, API-key permissions and task lifecycle without editing gateway code.

## Custom routing policies

Declare `"hooks": ["route"]` and implement:

```python
async def route(self, routing, context):
    # Example: prefer an allowed backend for this request, retaining fallbacks.
    ordered = sorted(routing.candidates, key=lambda c: c.backend != "preferred")
    return [candidate.id for candidate in ordered]
```

`RouteContext` supplies modality, key ID, the translated request dictionary,
and an immutable tuple of candidates. Each candidate has an ID, backend,
account ID, model, estimated cost and tags. Return `None` to keep the ordering
or an ordered, unique subset of the provided candidate IDs to rank/filter them.
Return `[]` to reject all candidates. Filters appear as `plugin_policy` in
estimates. Unknown or duplicate IDs fail the hook.

Both create and estimate call the policy after standard permissions, modality,
model limits, lifecycle, price and budget checks. Plugins cannot introduce
inadmissible backends/models or override an explicit model pin. The executor
still checks/reserves the live budget before each attempt and retries remaining
admissible candidates. Policies should be deterministic for a given request and
candidate set when estimate/create agreement matters.

## Configuration, errors and management API

```yaml
plugins:
  directory: ./plugins
  state_path: ./data/plugins.json
  watch: true
  poll_interval: 1
  max_body_bytes: 8388608
```

The environment equivalents are `PLUGINS_DIR`, `PLUGINS_STATE_PATH`,
`PLUGINS_WATCH`, `PLUGINS_POLL_INTERVAL`, and `PLUGINS_MAX_BODY_BYTES`.
Hot reload and live management use one gateway process, as do existing live
configuration and task snapshots.

Hooks default to a 1000 ms timeout and `failure_mode: "closed"`. A failed hook
returns a generic error without leaking plugin exception values. If a stream has
already sent its headers, failure aborts it. With `failure_mode: "open"`, failed
hook mutations are discarded and the previous request/response/order continues.
The body limit bounds HTTP request buffering, JSON response buffering and
transformed chunks; it does not buffer SSE or non-JSON response bodies.

All store APIs require the management bearer token. GET returns an `ETag` and
`revision`; writes require the plugin store's current `If-Match`. Missing/stale
revisions return 428/412. Gateway configuration uses its own ETag.

| Method | Path | Action |
| --- | --- | --- |
| GET | `/v1/management/plugins` | Catalog, installs, errors, active generations and hook counts |
| PUT | `/v1/management/plugins/{id}` | Install/configure/enable/disable a catalog package |
| POST | `/v1/management/plugins/install` | Install `{repository, ref, subdirectory, options}` from GitHub |
| POST | `/v1/management/plugins/{id}/reload` | Reload the installed source from disk |
| POST | `/v1/management/plugins/{id}/update` | Fetch the saved GitHub ref and atomically update |
| DELETE | `/v1/management/plugins/{id}` | Uninstall; remove managed GitHub source |

Options are `{enabled, order, config, timeout_ms, failure_mode}`. Every successful
write returns the current store snapshot and a new ETag. The store surfaces
load errors and counts hook calls/failures. Prometheus and management metrics
include `gateway_plugin_hooks_total` (plugin, hook, outcome) and
`gateway_plugin_hook_duration_seconds` (plugin, hook).
