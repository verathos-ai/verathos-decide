# Gateway

The gateway serves users from the miners' proven capacity. It runs next to the subnet owner's validator, which is
the auditor: the auditor tells it which endpoints it may use, and checks a sample of the answers it receives.

```
 user ── HTTPS ──► reverse proxy ──► gateway ──── signed request ────► miner endpoint
                                       │   ◄── answers + signed receipt ──┘
                                       │
                                       ├── GET /v1/routes ─────────────► owner validator (auditor)
                                       └── POST /v1/gateway/samples ───►   recomputes a sample
```

| | Code |
|---|---|
| API, request path | `src/verathos/gateway/app.py` |
| Endpoint choice, breakers | `gateway/router.py` |
| Requests to miners, receipt checks | `gateway/upstream.py` |
| Accounts, keys, limits, usage | `gateway/accounts.py`, `gateway/limits.py`, `gateway/usage.py` |
| Routes, samples | `gateway/routes.py`, `gateway/sampler.py`, `protocol/gateway.py` |
| The auditor's side | `throughput/auditor.py` (routes, sample checks), `validator/service.py` |

## How a request is served

1. **Key.** `Authorization: Bearer vd_...`. Keys are looked up by their SHA-256 and cached for 60 s, so a revoked
   key or a changed limit applies within a minute. Repeated unknown keys from one address are rate limited.
2. **Limits.** The account's rate (requests/s and burst, GCRA) and its monthly quota (UTC calendar month).
3. **Request.** A SystemOne request (`protocol/api.py`) for a model the catalog lists as `active` and the routes
   serve, with a state within the model's `max_state_tokens` (4 characters per token, as miners count).
4. **Endpoint.** Two endpoints of the model are drawn in proportion to their proven capacity; the one with fewer
   requests in flight per unit of capacity takes the request. In this process an endpoint takes at most
   `credited rps x max_latency_s / instances` requests at once, what it serves within the latency bound.
5. **Miner.** The request is signed with the gateway's hotkey for that miner. The answer counts only if the miner's
   receipt binds this request and these answers and names the gateway as caller, and the answers are well formed.
6. **Retries.** A failed attempt (connection, timeout, error status, no valid receipt) is retried on another miner,
   up to `max_attempts` within `request_timeout_s`; decision requests have no side effects. A miner that answers
   busy (429, 503) is skipped for that request without counting as a failure. After 3 failures in a row an
   endpoint's circuit breaker opens for 5 s; one probe request then decides, and each failed probe doubles the time
   (up to 120 s).
7. **Response.** The answers, the usage, the miner's signed receipt and the request id. Usage is counted per
   account, key, model and day, and written every 5 s.

## Answer checks

The gateway's hotkey is listed in the catalog (`gateways`), so a miner can tell user traffic from validator
traffic. To keep it from answering users with a cheaper model, the gateway keeps one exchange per endpoint and
sampling window (`sample_window_s`, 10 minutes), drawn uniformly from the window's answers after they are in, and
sends the window's picks to the auditor. The auditor recomputes up to `gateway_checks_per_epoch` (2) per endpoint and
epoch, between tests:
- a wrong answer is evidence, like one in a throughput test: every validator checks the receipt (issued to a catalog
  gateway) and its own reference, and the hotkey is reset and put on probation (`SCORING.md`);
- the differences are tallied per endpoint over the last `drift_window_tests` epochs, apart from the test answers'
  tally, so correct test answers do not hide drifted answers to users; beyond the model's `mean_abs`, the endpoint's
  credit is removed (and with it its routes).

Signed malformed answers are proof on their own and always go to the auditor.

## Routes

`GET /v1/routes` on the owner's validator: the endpoints listed in an accepted manifest now, credited in the current
verdicts (qualified, not drifted), of `active` models, whose hotkey has no proven wrong answer there and that are
not on probation at the owner's validator; with their credited capacity and test latency. The table is signed by
the auditor hotkey. The gateway polls it every 30 s, uses only tables signed by the subnet owner's hotkey and newer
than its own, and keeps the last good one on disk. A new endpoint is routable once its first throughput test is
credited.

## Setup

1. A hotkey for the gateway (no registration needed): `btcli wallet new_hotkey`, or a dedicated wallet holding only
   the hotkey. It is the most exposed key of the subnet, so it is not the owner hotkey.
2. List it in the catalog's `gateways` and commit the catalog (`SUBNET_ADMIN.md`). Miners serve it from their next
   catalog check.
3. Install: `VIRTUAL_ENV=$PWD/.venv uv pip install -e ".[server,chain,gateway]"`.
4. PostgreSQL (SQLite only for development):

   ```bash
   sudo -u postgres createuser -P gateway && sudo -u postgres createdb -O gateway gateway
   ```

5. `gateway.json`:

   ```json
   {
     "network": "finney",
     "netuid": 96,
     "wallet_name": "gateway",
     "hotkey_name": "default",
     "database_url": "postgresql+asyncpg://gateway:<password>@127.0.0.1/gateway",
     "port": 8080,
     "metrics_port": 9100
   }
   ```

   | Field | Default | Meaning |
   |---|---|---|
   | `listen_host`, `port` | 127.0.0.1, 8080 | where the gateway listens, behind the TLS reverse proxy |
   | `metrics_port` | - | Prometheus metrics on 127.0.0.1 |
   | `auditor_url` | owner's axon | the owner validator's service |
   | `instances` | 1 | gateway processes sharing the miners' capacity |
   | `trusted_proxies` | localhost | networks whose `X-Forwarded-For` is believed |
   | `request_timeout_s`, `attempt_timeout_s`, `max_attempts` | 10, 4, 3 | deadline per request, per attempt, attempts |
   | `max_in_flight` | 4096 | requests in flight per process; more get 503 |
   | `max_body_bytes` | 1 MiB | |
   | `sample_window_s` | 600 | one exchange per endpoint and window goes to the auditor |
   | `state_dir`, `catalog_dir` | `~/.verathos/gateway`, `~/.verathos/catalog` | last routes, catalogs |

6. Schema: `verathos-gateway --config gateway.json db upgrade` (after every update; the gateway does not start on
   an older schema).
7. Run `verathos-gateway --config gateway.json serve` under a supervisor, behind a reverse proxy that terminates
   TLS. `GET /ready` answers 200 once the gateway has routes and finds its hotkey among the catalog's gateways.

### More than one process

One process keeps rate limits, in-flight counts and breakers exactly. For more throughput, run several (each with
its own `port` and the same `instances`), and have the reverse proxy send each API key to one of them, for example
with nginx:

```nginx
upstream gateway { hash $http_authorization consistent; server 127.0.0.1:8080; server 127.0.0.1:8081; }
```

Each process then uses `1 / instances` of every endpoint's capacity, and per-key limits hold exactly. A monthly quota
is enforced from the database's usage plus each process's own counts, so it can be exceeded by what other processes
served in the last few seconds.

## API

`POST /v1/systemone`:

```bash
curl -s https://<gateway>/v1/systemone -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" -d '{
  "model": "kev-4b",
  "state": "Hi, we were billed twice for March. Please refund today or we cancel.",
  "questions": {"team": {"type": "choice", "criteria": {"billing": "Charges", "technical": "Bugs"}},
                "churn": {"type": "noul", "instructions": "Does the customer threaten to cancel?"}}}'
```

```json
{"id": "6f1c...", "model": "kev-4b",
 "answers": {"team": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.97, "technical": 0.03}},
             "churn": {"type": "noul", "noul": 0.91}},
 "usage": {"input_tokens": 41, "output_tokens": 0},
 "receipt": {"receipt": {"miner_hotkey": "5F...", "endpoint_id": "gpu0", "caller_hotkey": "5D...", "...": "..."},
             "signature": "..."}}
```

`GET /v1/models` lists the models served now; `GET /health` answers while the process runs.

Every response carries `X-Request-Id` (the client's, if it sent a valid one). Errors are
`{"error": {"type": ..., "message": ...}}`:

| Status | Type | When |
|---|---|---|
| 400 | `invalid_request` | not JSON, or not a SystemOne request |
| 401 | `unauthorized` | missing, unknown, revoked or expired key, or a suspended account |
| 404 | `model_not_found` | a model the gateway does not serve |
| 413 | `too_large` | body or state over the limits |
| 429 | `rate_limited` | over the account's rate; `Retry-After` |
| 429 | `quota_exceeded` | the month's quota is used up |
| 503 | `busy` | no capacity free for the model now; `Retry-After` |
| 503 | `unavailable` | no miner answered in time, or accounts cannot be checked; `Retry-After` |

## Accounts and keys

```bash
verathos-gateway accounts create --name acme --rate 50 --burst 100 [--monthly-quota 1000000]
verathos-gateway accounts set --name acme --rate 100        # also --burst, --monthly-quota, --status suspended
verathos-gateway keys create --account acme --name prod     # prints the key once
verathos-gateway keys list --account acme
verathos-gateway keys revoke --prefix vd_1a2b3c4
verathos-gateway usage --account acme --since 2026-10-01
```

## Metrics

`gateway_requests_total{model,outcome}`, `gateway_request_seconds{model}`, `gateway_upstream_seconds{model}`,
`gateway_upstream_failures_total{kind}`, `gateway_retries_total`, `gateway_in_flight`, `gateway_routes{model}`,
`gateway_routes_age_seconds`, `gateway_breakers_open`, `gateway_usage_flush_failures_total`,
`gateway_samples_total{outcome}`.
