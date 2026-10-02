# Auto mode: routing, fallbacks and cost control

Auto mode is how the gateway chooses *which model on which backend and account* serves a request. A request
enters auto mode when it omits `model` (or sends `"auto"`), and a pinned request can opt into the same
machinery for fallbacks with `routing.fallback`. Auto mode has three jobs:

1. **Route**: pick a model whose documented limits and capabilities fit the request, that is still in
   service, and that matches the caller's policy (cost, latency or the operator's default ranking).
2. **Fall back**: when the chosen candidate is retired, rate limited, out of capacity, timing out or failing
   upstream, try the next candidate instead of surfacing the failure.
3. **Control cost**: estimate every candidate's price before calling it, enforce per-request cost caps and
   per-key or per-scope budgets, record what each task cost, and report spend.

Models come and go (OpenAI removed every Sora 2 model from its API on 24 September 2026), so clients should
not have to hard-code model ids or track provider pricing. They express *what* they need (inputs,
duration, native audio, a cost ceiling, a budget) and auto mode decides *where* it runs.

## Pipeline

Every create request runs the same declarative pipeline. Each stage either filters candidates (recording
why) or orders them; nothing in a later stage can re-admit a filtered candidate.

| # | Stage | Input → output | Filter reasons |
|---|---|---|---|
| 1 | **Candidates** | the key's usable backends × the models each serves for the modality (auto), or every backend and account serving the pinned model (pinned with fallback) | – |
| 2 | **Policy** | `routing` directive + the operator's profile → effective policy `{tags, optimize, max_cost_usd, fallback, budget}` | `tag` (backend lacks the policy's tag) |
| 3 | **Fit** | the limits catalogue's hard constraints against the request profile (modalities, roles, prompt length, counts, duration, size, fps, **native audio**) | `limits` |
| 4 | **Lifecycle** | drop models whose `retired_on` date has passed; flag deprecated ones | `retired` |
| 5 | **Price** | estimate each remaining candidate's cost from the price catalogue | `max_cost` (estimate above `max_cost_usd`), `unpriced` (a cap or budget applies but the model has no price) |
| 6 | **Budget** | check the key budget and the scope budget can absorb the estimate | `budget` |
| 7 | **Rank** | order the admissible candidates by the policy's `optimize` mode, then live health | – |
| 8 | **Attempt** | try candidates best-first; retryable failures move to the next one; the accepted task reserves its estimate | – |
| 9 | **Settle** | when the task reaches a terminal state, record its cost (provider-reported, else the estimate) and release the reservation | – |

If stage 3–6 leave nothing, the request fails **before any provider is contacted**, with the most
specific reason (see [Errors](#errors)).

## Routing directive

`routing` is optional on every create (and estimate) request. Every member is optional.

```json
"routing": {
  "profile": "quality",
  "optimize": "cost",
  "max_cost_usd": 1.5,
  "fallback": "any",
  "budget": {"scope": "rideo:prj_123", "limit_usd": 50}
}
```

| Member | Values | Meaning |
|---|---|---|
| `profile` | string | A server-defined policy from `routing.profiles` in the configuration. A name that is not a configured profile is a **backend tag** (the original behaviour): only backends carrying it are candidates. Provider and backend names are never accepted. |
| `optimize` | `balanced` (default), `cost`, `latency` | How admissible candidates are ordered (below). Overrides the profile's `optimize`. |
| `max_cost_usd` | number > 0 | Hard per-task ceiling: candidates whose estimate exceeds it are not admissible, and neither are unpriced ones. Overrides the profile's value. |
| `fallback` | `none` (default for pinned models), `same_model`, `any` | What a pinned request may do when its model cannot serve it. Ignored in auto mode, which always falls back across its candidates. |
| `budget.scope` | 1–128 chars of `[A-Za-z0-9._:/-]` | A client-chosen spend bucket within the key (a project, a customer, a batch). Its spend is tracked and reported. |
| `budget.limit_usd` | number ≥ 0 | A self-imposed cap for the scope. The effective cap is the smaller of this and the operator's scope cap, if any. |

### Optimize modes

| Mode | Order of admissible candidates |
|---|---|
| `balanced` | live health first (success rate and latency; untried candidates are neutral), then the static preference: the key's default backend, its default tag, the most satisfied optional controls (native duration, roles, audio), active before deprecated, and the configuration order. This is the historical auto-router order plus the lifecycle tie-break. |
| `cost` | lowest estimated cost first (unpriced candidates last), then active before deprecated, then live health, then the stable order. |
| `latency` | lowest observed latency first (untried candidates rank as neutral), then health, then the stable order. |

A candidate inside a rate-limit cooldown always sinks to the bottom, whatever the mode; it is still tried
last in case the cooldown is stale.

## Capabilities: native audio

Video models differ in whether they generate sound. The limits catalogue carries
`supports_audio_output` (`true`: the model renders dialogue, effects and music with the picture; `false`: it
is silent). A video request with `parameters.include_audio: true`:

- never routes to a model documented as silent (`supports_audio_output: false`);
- prefers a model documented as audio-capable (an optional-control hit);
- may still route to an undocumented model (`null`), like every other undocumented limit.

Combined with the existing role checks, this is how a client asks for **lip-synced dialogue**: send the
line in the prompt, `include_audio: true`, and the speaker's voice as a `reference_audio` part; only models
that accept reference audio and render audio remain candidates.

## Capabilities: multi-shot and enhancement

Some video models render a whole sequence in one request: several shots separated by hard cuts, described in the
prompt. The catalogue records how many as `max_shots`; clients (Rideo groups a scene's shots by it) keep the
request within `max_shots` and `max_duration_seconds`.

Enhancement models take a `reference_video` and return it upscaled to the requested `dimensions`
(`supports_upscale`) or interpolated to the requested `fps` (`supports_frame_interpolation`, up to `max_fps`).
Clients find them in `GET /v1/models/limits` and pin them. None of the built-in providers documents these yet;
operators declare their own (self-hosted or proxied) models in `catalog.models`:

```yaml
catalog:
  models:
    my-multishot-model: {modality: video, max_shots: 4, max_duration_seconds: 20}
    my-upscaler: {modality: video, supports_upscale: true, supports_frame_interpolation: true, max_fps: 60}
```

## Capabilities: segmentation

Segmentation models take a `reference_video` and a text prompt naming the subject ("the person", "the red car") and
return a **matte**: a grayscale video of the same length, size and frame rate, white where the subject is and black
elsewhere, which editors use as an alpha channel (Rideo's *Remove the background*). They are listed with
`supports_segmentation`; operators declare theirs in `catalog.models`:

```yaml
catalog:
  models:
    my-matting-model: {modality: video, supports_reference_video: true, supports_segmentation: true}
```

## Capabilities: performance

Performance models animate a character from a person's performance (Runway Act-Two, Wan Animate and the like): they
take a driving `reference_video` (expressions, lip movements, head and body motion, timing) and a `first_frame` with
the character, and return the character performing it for the requested length (up to `max_duration_seconds`). They
are listed with `supports_performance`; operators declare theirs in `catalog.models`:

```yaml
catalog:
  models:
    my-performance-model:
      {modality: video, supports_first_frame: true, supports_reference_video: true, supports_performance: true}
```

## Model lifecycle

The catalogue records when a model stops being offered:

| Field | Meaning |
|---|---|
| `deprecated_on` | ISO date the provider announced the shutdown (or deprecation). The model still works but ranks after active models. |
| `retired_on` | ISO date the model stops working. From that date (UTC) it is excluded from auto mode and from `GET /v1/models`. |
| `replacement` | Suggested successor model id, tried first when a pinned request falls back. |

`GET /v1/models/limits` reports `lifecycle: active | deprecated | retired` and the dates, and keeps listing
retired models so clients can see why a pinned id stopped working and migrate. Operators can retire or
deprecate any model in configuration (below) without waiting for a gateway release.

A **pinned** request for a retired model fails with `410 model_retired` (detail names the date and the
replacement) unless it allows `fallback: any`, in which case it is served by the replacement or the best
auto candidate.

## Fallbacks

Auto mode tries its ranked candidates in order. A failure moves to the next candidate when it suggests
instability: rate limits (`429`), timeouts, upstream `5xx` and transport errors. Client errors (`4xx`,
unsupported features, validation) stop immediately, because another backend would reject them too or would
mask a real client mistake. Every attempt is recorded in the selection store, so the next request's
ranking reflects what just happened.

Pinned requests keep their single-attempt behaviour by default. `routing.fallback` widens it:

| `fallback` | Candidates, in order | Typical use |
|---|---|---|
| `none` | the first usable backend serving the model | exact reproduction; the caller handles failures |
| `same_model` | every backend and account that serves the pinned model, health-ranked | keep a pinned look (for example a film's pilot model) while surviving one exhausted key or one flaky backend |
| `any` | `same_model`, then the catalogue's `replacement`, then the auto candidates for the request (pinned model excluded), ranked by `optimize` | stay productive through retirements and capacity outages; the caller re-verifies the output |

With `fallback: any`, a pinned candidate that is not admissible under `max_cost_usd` or the budget is
skipped in favour of cheaper candidates (reason `over_budget`).

The task resource reports what happened, so a client can re-verify or record the model actually used:

```json
"model": "veo-3.1-fast-generate-preview",
"routing": {
  "requested_model": "veo-3.1-generate-preview",
  "fallback": true,
  "fallback_reason": "rate_limited",
  "attempts": 2,
  "optimize": "balanced",
  "estimated_cost": 0.8,
  "budget": {"scope": "rideo:prj_123", "limit_usd": 50, "spent_usd": 12.4, "reserved_usd": 0.8}
}
```

`model` is the model that served the task. Without a fallback it is the id the client sent (aliases are
echoed unchanged); after a fallback it is the served model, and `routing.requested_model` keeps the original.
`fallback_reason` is one of `model_retired`, `rate_limited`, `provider_unavailable`, `timeout` and
`over_budget`.

## Prices and estimates

The **price catalogue** (`mm_gateway/models/pricing.py`) holds list prices in USD next to the limits
catalogue, with `source_urls` and the date they were checked (`as_of`). Only prices published on official
pricing pages are built in; operators set or override any price in configuration (their contract price,
a reseller's price, a self-hosted model's cost).

| Field | Applies to | Meaning |
|---|---|---|
| `per_second` | video, music | price per output second |
| `per_second_tiers` | video | `[{max_longest_side, per_second}]` ascending; the first tier whose `max_longest_side` covers the requested size applies, else the last |
| `per_image` | image | price per output image |
| `per_request` | all | flat price per task (for example per song) |

Estimate of one task:

| Modality | Estimate |
|---|---|
| image | `output_count` (default 1) × `per_image` + `per_request` |
| video | `duration_seconds` (default: the model's minimum duration, else 5 s) × the tier's `per_second` (else `per_second`) + `per_request` |
| music | `duration_seconds` (default 30 s) × `per_second` + `per_request` |

A model with none of these fields is **unpriced**: its estimate is `null`.

`POST /v1/images/estimate`, `POST /v1/videos/estimate` and `POST /v1/music/estimate` take the same body as
the create call and run stages 1–7 without creating a task:

```json
{
  "object": "estimate",
  "modality": "video",
  "currency": "USD",
  "model": "veo-3.1-fast-generate-preview",
  "estimated_cost": 0.8,
  "candidates": [
    {"model": "veo-3.1-fast-generate-preview", "estimated_cost": 0.8, "lifecycle": "active", "admissible": true},
    {"model": "veo-3.1-generate-preview", "estimated_cost": 3.2, "lifecycle": "active", "admissible": false, "reason": "max_cost"},
    {"model": "sora-2", "estimated_cost": null, "lifecycle": "retired", "admissible": false, "reason": "retired"}
  ],
  "budget": {"scope": "rideo:prj_123", "limit_usd": 50, "spent_usd": 12.4, "reserved_usd": 0.0}
}
```

Candidates are listed once per model id (never per backend: backend identity stays private), best-ranked
first. `model` and `estimated_cost` are the candidate a create would try first, or `null` when nothing is
admissible. Clients use this to show a cost before a batch starts.

Every task's `usage` carries its cost once known:

| Member | Meaning |
|---|---|
| `usage.cost` | USD. The provider-reported cost when the provider returns one, else the estimate of the served candidate (for succeeded tasks). |
| `usage.cost_source` | `provider` or `estimate` |
| `usage.currency` | `USD` |

## Budgets and the ledger

Two kinds of caps protect spend; both are checked in stage 6 for every candidate.

| Budget | Set by | Period | Applies to |
|---|---|---|---|
| **Key budget** | operator: `keys[].budget: {limit_usd, period}` | `day`, `month` (calendar, UTC) or `total` | every task created with the key |
| **Scope budget** | client: `routing.budget {scope, limit_usd}`; operator caps: `keys[].budget.scopes_limit_usd` | `total` (lifetime of the gateway's ledger) | tasks that name the scope |

Admission is `spent + reserved + estimate ≤ limit` for every applicable cap. An **unpriced** candidate is
not admissible while any cap applies (`budget.allow_unpriced: true` in configuration opts out), because
an unknown cost cannot be proven to fit.

The **ledger** reserves a task's estimate when the provider accepts it, and settles when the background
monitor sees the task finish: a succeeded task moves its actual cost (provider-reported, else the estimate)
to `spent`; a failed, cancelled or expired task releases the reservation without cost. Reservation makes
concurrent creates safe: ten parallel requests cannot each see the same remaining budget.

`GET /v1/usage` reports the authenticated key's spend:

```json
{
  "object": "usage",
  "currency": "USD",
  "period": {"kind": "month", "start": "2026-10-01T00:00:00Z", "end": "2026-11-01T00:00:00Z"},
  "key": {"spent_usd": 212.4, "reserved_usd": 3.2, "limit_usd": 500, "remaining_usd": 284.4, "tasks": 311},
  "scopes": [{"scope": "rideo:prj_123", "spent_usd": 12.4, "reserved_usd": 0.8, "limit_usd": 50, "remaining_usd": 36.8, "tasks": 19}],
  "models": [{"model": "veo-3.1-generate-preview", "modality": "video", "spent_usd": 160.0, "tasks": 50}]
}
```

`?scope=` narrows `scopes` to one scope. The ledger is process-local like the bundled task store; a
multi-instance deployment injects a shared implementation with `create_app(..., ledger=...)`.

## Configuration

```yaml
routing:
  default_optimize: balanced          # balanced | cost | latency
  profiles:                           # named, server-defined policies for routing.profile
    cheap:   {optimize: cost}
    fast:    {optimize: latency}
    quality: {tags: [quality]}
    capped:  {optimize: cost, max_cost_usd: 2.0, fallback: any}

catalog:
  models:                             # overrides merged over the built-in catalogues
    sora-2: {retired_on: "2026-09-24"}
    my-self-hosted-ltx: {supports_audio_output: true, price: {per_second: 0.01}}
    veo-3.1-generate-preview: {price: {per_second_tiers: [{max_longest_side: 1920, per_second: 0.32}]}}

budget:
  allow_unpriced: false

keys:
  - id: application
    key: ${GATEWAY_API_KEY}
    allow_tags: [production]
    budget: {limit_usd: 500, period: month, scopes_limit_usd: 100}
```

A profile may set `tags` (backends must carry one of them), `optimize`, `max_cost_usd` and `fallback`.
Request members override the profile member by member.

## Errors

| Status | `code` | When |
|---|---|---|
| `410` | `model_retired` | a pinned model is past its `retired_on` date and `fallback` is not `any` |
| `402` | `budget_exceeded` | every candidate that fits would exceed the key or scope budget; the problem carries `budget` (scope, limit, spent, reserved) and the cheapest `estimated_cost` |
| `422` | `cost_limit_exceeded` | every candidate that fits costs more than `max_cost_usd`, or is unpriced |
| `422` | `validation_error` | nothing fits the request's limits (unchanged) |

## Observability

| Metric | Labels | Meaning |
|---|---|---|
| `gateway_routing_decisions_total` | `modality`, `mode` (`auto`/`pinned`), `optimize` | routed creates |
| `gateway_routing_excluded_total` | `modality`, `reason` (`limits`, `retired`, `max_cost`, `unpriced`, `budget`) | candidates filtered out per request |
| `gateway_routing_fallbacks_total` | `modality`, `reason` | tasks served by a candidate other than the first choice |
| `gateway_cost_usd_total` | `key`, `modality`, `model`, `source` | settled spend |
| `gateway_budget_rejections_total` | `key`, `budget` (`key`/`scope`) | creates refused for budget |
| `gateway_budget_spent_usd`, `gateway_budget_limit_usd` | `key` | key budget state for the current period |

Each routed create logs `auto_route_plan` (mode, optimize, candidate count, exclusions by reason, first
choice and its estimate), each attempt `auto_route_attempt_ok` / `auto_route_attempt_failed`, and each
settlement `ledger_settled` (task, cost, source). Scope names appear in logs and in `/v1/usage`, never in
metric labels (their cardinality is unbounded).

## Non-goals

- Currency conversion: all prices and budgets are USD.
- Live price discovery: prices are a static, sourced catalogue plus configuration, like the limits.
- Quality scoring of outputs: clients that need a quality gate (for example Rideo's consistency judge) verify
  results themselves; `routing.fallback` makes the fallback visible so they can.
