# Management API and console

The management surface controls the running gateway's backends, generation keys,
pass-through proxies, routing profiles, catalogue overrides and outbound proxy.
It also exposes task snapshots, per-key usage, counters, duration summaries and
provider/account selection health. It is separate from the provider-neutral media API.

## Enable management

Management endpoints are always included in OpenAPI so generated clients have a
stable contract. Requests require `Authorization: Bearer <management-token>`.
Set `MANAGEMENT_API_KEY`, or add this section to the base YAML configuration:

```yaml
management:
  api_key: ${MANAGEMENT_API_KEY}
  config_path: ${MANAGEMENT_CONFIG_PATH:}
```

An unset admin token disables management access. Ordinary generation keys and
empty/open generation keys do not authorize management. `/health` and the
existing `/metrics` Prometheus endpoint keep their existing public access rules.
`defaults.enable_metrics: false` disables metrics exposition, including the
structured management metrics view.

Docker builds include the console at `/admin/`. For a source checkout, run
`npm ci` and `npm run build` in `web/` before starting the gateway. This writes
the static console to `mm_gateway/server/static/admin/`; Python wheels built
afterward include those files. All console API traffic uses the same origin.

## Caddy and subpath deployments

The same frontend build works at `/admin/` and at, for example,
`/gateway/admin/`. Assets use relative URLs; the SDK and API documentation link
derive the gateway prefix from the console URL at runtime. No frontend rebuild
or build-time path setting is needed when the prefix changes.

When Caddy strips a prefix with [`handle_path`](https://caddyserver.com/docs/caddyfile/directives/handle_path),
set the gateway's `ROOT_PATH` to that prefix so redirects and the API docs use
the public URL. For example, start the gateway with:

```bash
ROOT_PATH=/gateway MANAGEMENT_API_KEY=your-admin-token mm-gateway
```

Or configure it in YAML:

```yaml
server:
  root_path: /gateway
```

Use this Caddyfile and open `https://example.com/gateway/admin/`:

```caddyfile
example.com {
    redir /gateway /gateway/admin/ 308

    handle_path /gateway/* {
        reverse_proxy 127.0.0.1:8000
    }
}
```

Replace `/gateway` in both settings with your prefix; nested prefixes such as
`/tools/gateway` work too. Proxy the entire gateway under that prefix so the
console, `/v1`, `/metrics`, `/docs` and `/openapi.json` share the same base URL.
`ROOT_PATH` defaults to empty for root deployments. If you launch Uvicorn
directly instead of `mm-gateway`, also pass `--root-path /gateway` so its ASGI
request paths include the prefix. See [FastAPI's proxy documentation](https://fastapi.tiangolo.com/advanced/behind-a-proxy/).

## Read and change configuration

`GET /v1/management/config` returns:

```json
{
  "revision": "opaque-revision",
  "persistent": true,
  "config": {
    "backends": [],
    "keys": [],
    "proxies": [],
    "routing_default_optimize": "balanced",
    "routing_profiles": {},
    "catalog_models": {},
    "budget_allow_unpriced": false,
    "outbound_proxy": null
  }
}
```

The response includes `ETag: "opaque-revision"` and `Cache-Control: no-store`.
API tokens, credential material, all proxy header values and URLs containing
authentication are replaced with `[redacted]`. Management request bodies are
excluded from HTTP body logs, and validation errors omit secret-bearing inputs.

To change the complete configuration, send only the `config` object to
`PUT /v1/management/config`, with the current `ETag` in `If-Match`. A full
replacement removes resources omitted from the supplied configuration. The
console exposes this operation for routing settings and uses the individual
resource endpoints for backend/key/proxy edits:

| Endpoint | Mutation |
| --- | --- |
| `PUT /v1/management/backends/{name}` | Create or replace a named backend |
| `DELETE /v1/management/backends/{name}` | Remove the backend |
| `PUT /v1/management/keys/{key_id}` | Create or replace a generation key |
| `DELETE /v1/management/keys/{key_id}` | Revoke and remove the generation key |
| `PUT /v1/management/proxies/{domain}` | Create or replace the upstream-domain proxy |
| `DELETE /v1/management/proxies/{domain}` | Remove the proxy |

These operations also require `If-Match` and return the complete updated,
redacted configuration with a new revision. An absent revision returns `428`;
a stale revision returns `412`. Reload and review the current configuration
before resubmitting an edit. The API does not accept `If-Match: *`.

Preserve an existing secret by sending `[redacted]` in the same field on the
same resource/account identity. Send a new value to rotate it or `null` to
clear an optional credential. Named credential and proxy account pools preserve
secrets when reordered. A placeholder on a new resource is rejected. In the
console's secret input fields, leaving an existing secret blank preserves it;
the Full JSON editor supports explicit clearing and every configuration field.

Backends, keys and proxies accept `enabled: false`. Disabling a backend removes
it from new routing; disabling a key revokes its authentication immediately.
Backend initialization and catalogue validation complete before configuration is
published. Invalid changes or a failed persistent write leave the previous
runtime configuration and revision intact. Unchanged backends retain their SDK
clients. Background monitors for already accepted tasks keep their owning
provider client, and administrative observation never initiates provider polling.

## Persistence and deployment

`MANAGEMENT_CONFIG_PATH` (or YAML `management.config_path`) selects a JSON
overlay. Successful writes use a temporary file, `fsync` and atomic replacement;
the file has mode `0600` and contains the actual credentials. On startup the
overlay replaces the managed portion of the base environment/YAML settings.
The admin token and server settings continue to come from the base configuration.
To return to the base configuration, stop the gateway and remove the overlay.

If no path is configured, changes are process-local and disappear on restart.
The console shows this persistence mode. A Docker deployment can mount a volume
and set, for example, `MANAGEMENT_CONFIG_PATH=/app/data/management.json`.
The image creates `/app/data` with ownership for its runtime user, so a named
volume mounted there inherits a writable directory on first use.

Use one gateway worker for live management. Multiple workers or replicas have
independent registries, metrics, task stores, ledgers and configuration revisions;
the overlay is not a shared or broadcast control plane. Task and usage history
remains process-local with the default stores, even when configuration is persisted.

## Observability endpoints

- `GET /v1/management/status`: process uptime, configured backend types,
  active/configured/enabled state, account ids, models, proxies, key counts and
  task totals by lifecycle state.
- `GET /v1/management/metrics`: timestamped counters and duration summaries
  (`count`, `sum`, `min`, `max`, `mean`), plus live account selection health.
  Counter and duration storage holds aggregates rather than one entry per
  observation. Selection success and attempts are time-decayed values;
  cooldowns and EWMA latency match the auto-router's selection store.
- `GET /v1/management/tasks`: cached task summaries with public task id, owner,
  backend, model, modality, status and timestamps. Prompts and output media are
  omitted. Filter by `modality`, `status`, `key_id` and `backend`; paginate with
  `offset` and `limit` (1–200, default 50), newest first.
- `GET /v1/management/usage`: the cost ledger's current budget period, spend,
  reservations, scope and model breakdown for every configured key. Optionally
  filter with `key_id`.

HTTP counters use route templates as labels rather than individual task ids.
Provider-call duration statistics and async task statistics keep their existing
metric names. The console has explicit empty states until real activity occurs.
Its default refresh interval is ten seconds and pauses while the tab is hidden.
Overview health badges show active cooldowns, unobserved accounts, and degraded
health when the observed success rate is below 90%; the metrics view shows the
underlying rates and latencies.

## TypeScript SDK and frontend development

The console depends on `@sloth-os/mm-gateway-ts` directly from a pinned commit
of `sloth-os/mm-gateway-ts`. The current upstream SDK predates the management
endpoints. `web/src/sdk/management.ts` extends its `MetaApi` and shares the
upstream `Configuration`, Axios instance and bearer credential. It adds the
management operation methods, with contract types generated from
`docs/openapi.json` in `web/src/sdk/schema.d.ts`. Public Prometheus retrieval
uses the upstream SDK's `getMetrics` method. There is no separate fetch client.

The existing `openapi.yml` workflow will generate the management classes into
the upstream TypeScript, Python and Go SDKs when the updated contract is
published. Until then, the local management bindings allow the frontend to
build against the current upstream commit.

To develop against a running local gateway:

```bash
# In the repository root:
MANAGEMENT_API_KEY=your-admin-token mm-gateway

# In another terminal:
cd web
npm ci
npm run dev
```

Open `http://127.0.0.1:5173/admin/`. Vite forwards API requests to port 8000;
set `GATEWAY_URL` on the Vite process to use a different local upstream. The
admin token is held in memory only, cleared on disconnect, and must be entered
again after a reload.

After changing the API contract:

```bash
python scripts/generate_openapi.py
cd web
npm run sdk:generate
npm run build
npx playwright install chromium
npm test
npm run test:production
```

The browser suite starts a local gateway with in-memory providers, exercises
real authenticated SDK traffic and mutations, and verifies desktop/mobile
rendering without vendor calls. Production tests repeat the suite through a
prefix-stripping proxy and check assets, redirects and API docs. The portable
test proxy needs only Node; set `CADDY_TEST_BINARY=/path/to/caddy` to run those
tests through real Caddy using `web/tests/Caddyfile`.
It defaults to `../.venv/bin/python`; set
`GATEWAY_TEST_PYTHON=python` if the test dependencies are installed elsewhere.
CI runs Python tests, checks generated schema freshness, builds the console,
and runs the browser suite against the compiled console before building the Docker image.
