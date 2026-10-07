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
   key or a changed limit applies within a minute. An address (IPv6: a /64) sending invalid keys is limited to 2 per
   second (bursts of 20) before any lookup.
2. **Limits.** The account's rate (requests/s and burst, GCRA), applied to each of its keys, and its monthly quota
   (UTC calendar month) for the account.
3. **Request.** Strict JSON (no NaN or infinities): a SystemOne request (`protocol/api.py`) for a model the catalog
   lists as `active` and the routes serve, with a state within the model's `max_state_tokens` (4 characters per
   token, as miners count), and questions whose answers fit a response (64 KiB). Requests no miner could answer are
   refused here, so they cannot count against miners.
4. **Endpoint.** Two endpoints of the model are drawn in proportion to their proven capacity; the one with fewer
   requests in flight per unit of capacity takes the request. In this process an endpoint takes at most
   `credited rps x max_latency_s / instances` requests at once, what it serves within the latency bound.
5. **Miner.** The request is signed with the gateway's hotkey for that miner. The answer counts only if the miner's
   receipt binds this request and these answers and names the gateway as caller, and the answers are well formed.
6. **Retries.** A failed attempt is retried on another miner, up to `max_attempts` within `request_timeout_s`;
   decision requests have no side effects. A failure that is the miner's (no connection, a response without a valid
   receipt, a compressed or oversized response, a refusal of the gateway) counts against it at once; one that the
   request itself may cause (a timeout, an error status) counts only if another miner then answers the same request.
   A miner that answers busy (429, 503) is skipped for one second, and a refusal of the request (another 4xx) counts
   against nobody. After 3 failures in a row an endpoint's circuit breaker opens for 5 s; one probe request then
   decides, and each failed probe doubles the time (up to 120 s).
7. **Response.** The answers, the input tokens the miner signed in its receipt, the receipt itself and the request
   id. Usage is counted per account, key, model and day, and written every 5 s.

## Answer checks

The gateway's hotkey is listed in the catalog (`gateways`), so a miner can tell user traffic from validator
traffic. To keep it from answering users with a cheaper model, the gateway keeps one exchange per endpoint and
sampling window (`sample_window_s`, 10 minutes), drawn uniformly from the window's answers after they are in, and
sends the window's picks to the auditor. The auditor keeps up to 8 per endpoint and epoch and recomputes a random
`gateway_checks_per_epoch` (2) of them, between tests:
- a wrong answer is evidence: the auditor publishes the miner's signed receipt (not the request), every validator
  checks that the receipt was issued to a catalog gateway, and the hotkey is reset and put on probation
  (`SCORING.md`);
- the differences are tallied per endpoint over the last `drift_window_tests` epochs, apart from the test answers'
  tally, so correct test answers do not hide drifted answers to users; beyond the model's `mean_abs`, the endpoint's
  credit is removed (and with it its routes).

Signed malformed answers are proof on their own: they always go to the auditor, and are kept for the next window if
a post fails.

The gateway also reports, every 5 s, how many requests each endpoint answered it per second, and their input tokens.
Miners serve the gateway before the auditor's test traffic; what an endpoint answers the gateway inside a test's
measured window counts as delivered, so serving users never costs a miner credit (`THROUGHPUT_AUDIT.md`). The input
tokens per model are the users' demand, which, at each model's catalog `price`, sets the models' emission shares
(`SCORING.md`).

## Privacy of user requests

A user's request is seen by the gateway, by the miner that answers it, and, for the few sampled ones, by the auditor
until their check: the auditor then deletes the request and the answers and keeps only the miner's receipt (hashes of
the request and the answers) and the differences it measured. Nothing a user sent is published: evidence of a wrong
answer to a user carries only the receipt. Validators check that the miner answered the gateway and take the
auditor's word for the wrongness, as they take it for capacity; test evidence, whose requests are synthetic, stays
fully checkable by every validator.

## Routes

`GET /v1/routes` on the owner's validator: the endpoints listed in an accepted manifest now, credited in the current
verdicts (the previous epoch's while the current ones wait for a test to settle; qualified, not drifted), of `active`
models, whose hotkey has no wrong answer proven there or since, and is not on probation at the owner's validator;
with their credited capacity and test latency. The table is signed by
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
   | `admin_host`, `admin_port`, `admin_token_file` | 127.0.0.1, -, - | the admin API (see below); off without a port |
   | `state_dir`, `catalog_dir` | `~/.verathos/gateway`, `~/.verathos/catalog` | last routes, catalogs |

6. Schema: `verathos-gateway --config gateway.json db upgrade` (after every update; the gateway does not start on
   an older schema).
7. Run `verathos-gateway --config gateway.json serve` under a supervisor, behind a reverse proxy that terminates
   TLS, as an active and a standby process (below). `GET /ready` answers 200 once the gateway has routes and finds
   its hotkey among the catalog's gateways.

### Restarts without downtime

Run two processes with the same configuration except `port`, `admin_port`, `metrics_port` and `state_dir`: the
active one, and a standby the reverse proxy uses only while the active one is down. Both keep `instances: 1`, since
only one of them serves at a time.

```nginx
upstream gateway {
    server 127.0.0.1:8080 max_fails=1 fail_timeout=5s;
    server 127.0.0.1:8081 backup;
    keepalive 64;
}
# in the server block that terminates TLS
location / {
    proxy_pass http://gateway;
    proxy_http_version 1.1;
    proxy_set_header Connection "";
    proxy_set_header X-Forwarded-For $remote_addr;
    proxy_next_upstream error timeout non_idempotent;
    proxy_next_upstream_tries 2;
    proxy_connect_timeout 1s;
}
```

Restart (and update) the standby first, then the active one, each once the other is ready:

```bash
systemctl restart verathos-gateway-standby; until curl -sf http://127.0.0.1:8081/ready; do sleep 1; done
systemctl restart verathos-gateway;         until curl -sf http://127.0.0.1:8080/ready; do sleep 1; done
```

On SIGTERM a process stops accepting connections, so the proxy sends new requests to the other one; it finishes
the requests in flight (at most `request_timeout_s`), writes usage, samples and served counts, and exits. A request
whose process fails before answering goes to the other process once. A process is ready as soon as it has started:
it serves the routes it kept on disk while it fetches new ones. A restart of the auditor does not reach users: the
gateways keep their routes until it serves new ones. A key's rate limit starts afresh on the other process.

### More than one process

One process keeps rate limits, in-flight counts and breakers exactly. For more throughput, run several (each with
its own `port` and the same `instances`), and have the reverse proxy send each API key to one of them, for example
with nginx:

```nginx
upstream gateway { hash $http_authorization consistent; server 127.0.0.1:8080; server 127.0.0.1:8081; }
```

Each process then uses `1 / instances` of every endpoint's capacity, and each key's rate holds exactly (an account's
keys may land on different processes). Restart them one at a time; with `proxy_next_upstream` as above, the proxy
moves a stopped process's keys to the others meanwhile. A monthly quota is enforced from the database's usage plus each process's own
counts, so it can be exceeded by what other processes served in the last few seconds.

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
| 400 | `invalid_request` | not JSON (NaN and infinities included), not a SystemOne request, or refused by the miners |
| 401 | `unauthorized` | missing, unknown, revoked or expired key, or a suspended account |
| 404 | `model_not_found` | not an active model of the catalog |
| 413 | `too_large` | body, state, or the answers the questions would need, over the limits |
| 429 | `rate_limited` | over the key's rate, or too many invalid keys from the address; `Retry-After` |
| 429 | `quota_exceeded` | the month's quota is used up |
| 500 | `internal` | a fault of the gateway |
| 503 | `busy` | no capacity free for the model now; `Retry-After` |
| 503 | `unavailable` | no miner serves the model right now, none answered in time, or accounts cannot be checked; `Retry-After` |

## Accounts and keys

```bash
verathos-gateway accounts create --name acme --rate 50 --burst 100 [--monthly-quota 1000000]
verathos-gateway accounts set --name acme --rate 100        # also --burst, --monthly-quota, --status suspended
verathos-gateway keys create --account acme --name prod     # prints the key once
verathos-gateway keys list --account acme
verathos-gateway keys revoke --account acme --prefix vd_1a2b3c4d5e6f7
verathos-gateway usage --account acme --since 2026-10-01
```

## Admin API

For the operators only: inspection, and requests sent past the router or to one chosen endpoint. Off unless
`admin_port` is set. It listens on a socket of its own, `admin_host:admin_port` (loopback by default; a private
address at most, never `0.0.0.0`), so the reverse proxy never reaches it, and on the public port its routes answer
404. Every call needs `Authorization: Bearer <token>`, the token read from `admin_token_file` at start:

```bash
umask 077; openssl rand -hex 32 > ~/.verathos/gateway/admin.token   # at least 32 characters, mode 600
```

The gateway refuses to start with a shorter token or a file others can read. Every call is logged, wrong tokens
with the caller's address.

| Call | Returns |
|---|---|
| `GET /admin/v1/state` | the routes in use (epoch, age, endpoints, catalog), active models, limits, and what waits to be written or sent (usage rows, samples, served counts) |
| `GET /admin/v1/routes` | every routable endpoint as the router sees it: credited rate, latency, cap, in flight, busy time, breaker |
| `POST /admin/v1/systemone` | a SystemOne request, served as described below |

```bash
curl -s http://127.0.0.1:8090/admin/v1/systemone -H "Authorization: Bearer $(cat ~/.verathos/gateway/admin.token)" \
  -d '{"request": {"model": "kev-4b", "state": "...", "questions": {...}},
       "target": {"hotkey": "5F...", "endpoint_id": "gpu0"},
       "count": false}'
```

- `target`: the one endpoint to send it to, one attempt, outside the router's accounting (its load and breaker are
  untouched); it must be a routable endpoint of the model. Without it, the request is routed as users' requests are.
- `count`: whether the exchange counts as user traffic: served counts (test windows, demand) and the sampler. Off by
  default. Admin requests are never billed to an account.
- The reply holds the endpoint, its URL, the answers, input tokens, the receipt, the miner's response as received,
  every attempt (endpoint, outcome, failure kind and blame, milliseconds) and the total time; 502 with the attempts
  if no endpoint answered.

## Metrics

`gateway_requests_total{model,outcome}`, `gateway_request_seconds{model}`, `gateway_upstream_seconds{model}`,
`gateway_upstream_failures_total{kind}`, `gateway_retries_total`, `gateway_in_flight`, `gateway_routes{model}`,
`gateway_routes_age_seconds`, `gateway_breakers_open`, `gateway_usage_flush_failures_total`,
`gateway_samples_total{outcome}`.
