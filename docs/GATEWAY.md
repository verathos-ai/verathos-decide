# Gateway

The gateway serves users from the miners' proven capacity. It runs next to the subnet owner's validator, which is
the auditor: the auditor tells it which endpoints it may use, and checks a sample of the answers it receives.

```
 user ── HTTPS ──► reverse proxy ──► gateway ── signed request, TLS ──► miner endpoint
                                       │   ◄── answers + signed receipt ──┘
                                       │
                                       ├── GET /v1/routes (TLS) ────────► owner validator (auditor)
                                       └── POST /v1/gateway/samples ───►   recomputes a sample
```

Every connection the gateway makes to miners and the auditor is TLS. They are reached by IP address with self-signed
certificates, each accepted by its pin (`DISCOVERY.md`): a miner machine's from the route the auditor signed, the
auditor's from the committed catalog (`auditor.tls`), which the gateway follows on chain. Catalogs, which their
digest proves, are fetched accepting any certificate.

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
   second (bursts of 20) before any lookup. The hashes of the keys not revoked are loaded every minute (and a key
   made through this process's admin API at once); a key not among them is looked up only within a budget all
   clients share (20 per second), so random keys from many addresses never load the database. While that budget is
   spent, a key made through another process is refused until the next load.
2. **Limits.** The account's rate (requests/s and burst, GCRA), applied to each of its keys, and its monthly quota
   (UTC calendar month) for the account. A key or credit check that needs the database waits for it at most 5 s
   (then 503), and every database statement is bounded (10 s, 5 s waiting for a lock), so a stalled database
   refuses requests instead of holding them; the miners' deadline (`request_timeout_s`) starts after these checks.
3. **Request.** Strict JSON (no NaN or infinities): a SystemOne request (`protocol/api.py`) for a model the catalog
   lists as `active` and the routes serve, whose state, question ids, instructions and criteria together are within
   the model's `max_state_tokens` (4 characters per token, as miners count; JSON as canonical JSON), and questions
   whose answers fit a response (64 KiB), and whose state is not in the form of the subnet's test requests or
   canaries (`throughput.workload.in_mix_form`, `validator.canaries.canary_like`: a miner relaying them through the
   gateway would have its answers to them counted as users'). Requests no miner could answer are refused here, so
   they cannot count against miners.
4. **Endpoint.** Two endpoints of the model are drawn in proportion to their proven capacity; the one with fewer
   requests in flight per unit of capacity takes the request. In this process an endpoint takes at most
   `credited rps x max_latency_s / instances` requests at once, what it serves within the latency bound.
5. **Miner.** The request is signed with the gateway's hotkey for that miner. The answer counts only if the miner's
   receipt binds this request and these answers and names the gateway as caller, and the answers are well formed:
   one distribution per question over its options, the `noul`, `choice` and `score` users read derived from it (p(true),
   the most likely option, the expected level, within 0.001, a score also within the 4-decimal rounding of the
   probabilities it sums), and no fields beyond an
   answer type's own (`protocol/api.py`).
6. **Retries.** A failed attempt is retried on another miner, up to `max_attempts` within `request_timeout_s`;
   decision requests have no side effects. A failure that is the miner's (no connection, a response without a valid
   receipt, a compressed or oversized response, a refusal of the gateway) counts against it at once; one that the
   request itself may cause (a timeout, an error status) counts only if another miner then answers the same request.
   A connection the gateway itself cannot make (out of descriptors or buffers, all its connections to miners in use,
   no route out) counts against nobody. A miner that answers busy (429, 503) is skipped for one second. A refusal of
   the request (another 4xx) counts against it if another miner then answers the same request; the gateway has
   checked everything a miner checks, so a refusal no other miner shares is also reported as refusing users (without
   opening its breaker), and the user gets 503, not 400. A slow endpoint gets fewer requests on its own: they stay in
   flight longer. After 3 failures in a row an endpoint's circuit breaker opens for 5 s; one probe request then
   decides, and each failed probe doubles the time (up to 120 s). A probe that times out stays the one probe until
   its request is settled: it fails if another miner answers the request.
7. **Response.** The answers, the input tokens the miner signed in its receipt (at most one per byte of the
   request, at least one per 8 characters of the state; the auditor checks sampled counts against its reference,
   too many or too few being a wrong answer), the receipt itself and the request id. Usage is counted per account,
   key, model and day, and written every 5 s, with every account's answered requests and input tokens per model and
   hour (the network's totals).
8. **Report.** Every 5 s the gateway tells the auditor, per endpoint and second, the user requests it sent it and those
   it answered (with input tokens, the paid ones apart, and durations), how often it considered it for a request (drew
   it), and how often it was not ready then (busy, behind an open breaker, or its latest answer or timed-out attempt
   took at least twice the latency bound it declared its capacity at, `max_latency_s`, 2 s) or refused the request
   (answered busy or failed, refused a request another miner answered, or refused one alone). An answer counts in the
   second its request was sent. Drawn while full counts neither way while its latest answer came within `max_latency_s`;
   full beyond it, it serves less than its capacity at the bound, so it was not ready. Requests that are not user
   traffic (admin requests not counted) are left out. The auditor publishes each endpoint's tally with its credit, and
   validators score it (`SCORING.md`).

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

During a throughput test the auditor counts what an endpoint answers users only once it has checked one of those
answers (`THROUGHPUT_AUDIT.md`): a miner knows when a test runs. So the gateway follows the public test schedule on
the chain, and from when it sees a test's start block until the test's traffic ends it draws, per endpoint, one of
the exchanges it sends the endpoint, uniformly as above; once the test's sessions end (`session_s` after the start)
it posts each pick marked with the test's id, and again on failure for 5 minutes.

The gateway also reports, every 5 s, how many requests each endpoint answered it per second, and their input tokens.
Miners serve the gateway before the auditor's test traffic; what an endpoint answers the gateway inside a test's
measured window counts as delivered, so serving users never costs a miner credit (`THROUGHPUT_AUDIT.md`). The input
tokens per model are the users' demand; of them, the paid ones (`paid_input_tokens`) set, at each model's catalog
`price`, the models' emission shares, and per endpoint its part of the usage pool (`SCORING.md`). Usage is paid
when its account is prepaid, has topped up, and its free credit is spent: its balance is no more than its top-ups.
Free credit (a sign-up grant, the demo) is spent first, as the totals count paid usage, so an account that only ever
had free credit is unpaid, one that bought credit is paid once its grants are used, and accounts that are not prepaid
(invoiced, internal) are unpaid. Admin requests are never paid.

## Privacy of user requests

A user's request is seen by the gateway, by each miner it is sent to (the one that answers it, and any that failed
before it, up to `max_attempts`), and, for the few sampled ones, by the auditor
until their check: the auditor then deletes the request and the answers and keeps only the miner's receipt (hashes of
the request and the answers) and the differences it measured. Nothing a user sent is published: evidence of a wrong
answer to a user carries only the receipt. Validators check that the miner answered the gateway and take the
auditor's word for the wrongness, as they take it for capacity; test evidence, whose requests are synthetic, stays
fully checkable by every validator.

Between the gateway and the miners, and to the auditor, requests and answers travel over TLS, each certificate
accepted by its pin only: whoever is on the path reads nothing, and a machine presenting another certificate gets no
request. When the gateway runs next to the auditor, `auditor_url` can name its loopback or private address
(`https://127.0.0.1:<port>`); the auditor's service is reached there with the same pin.

## Routes

`GET /v1/routes` on the owner's validator: the endpoints listed in an accepted manifest now that proved control at
their URL (`DISCOVERY.md`), credited in the current verdicts (the previous epoch's while the current ones wait for a
test to settle; qualified, not drifted), of `active` models, whose hotkey has no wrong answer proven there or since,
and is not on probation at the owner's validator; with their URL (the machine serving the endpoint, which need not be
the axon), the pin of that machine's certificate, credited capacity and test latency. The gateway uses a route only at
an https URL a manifest may list, on a public address; any other is logged and left out, so not even a signed table
can send it into a private network. While the verdicts cover no confirmed test (the auditor missed the
tests or they were void), the credits come from the latest confirmed test the auditor holds, whatever its age. The
table is signed by the auditor hotkey. The gateway polls it every 30 s, uses only tables signed by the subnet owner's
hotkey, newer than its own and issued at most 5 minutes ahead of its clock, and keeps the last good one on disk. A new
endpoint is routable once its first throughput test is credited. While the auditor serves no newer table the gateway
keeps using its own; older than 3 epochs, the routes are stale: the log says so every 10 minutes, and `GET /ready`
(still 200) shows it (`"routes": {"age_s": ..., "stale": true}`). A newer table that drops every route of a model its
catalog still has active is refused, with an error in the log, and the current one kept: an auditor that lost its
view of the miners does not take a model off the gateway. Once the current table is stale, the newer one is taken.

## Setup

1. A hotkey for the gateway (no registration needed): `btcli wallet new_hotkey`, or a dedicated wallet holding only
   the hotkey. It is the most exposed key of the subnet, so it is not the owner hotkey.
2. List it in the catalog's `gateways` with the public addresses its requests to miners come from (`{"hotkey": ...,
   "egress": [...]}`) and commit the catalog (`SUBNET_ADMIN.md`). Miners serve it from their next catalog check.
   Miners' firewalls drop connections from any other address once they enforce (`MINING.md`, Firewall): list every
   address the gateway's requests leave from, and commit a catalog with the new ones before moving it.
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
   | `listen_host`, `port` | 127.0.0.1, 8080 | where the gateway listens, behind the TLS reverse proxy; an address off loopback only with `tls_cert_file` |
   | `tls_cert_file`, `tls_key_file` | -, - | a certificate (PEM, with its chain) and key the gateway serves its listeners with, admin API included; needed for any listener off loopback |
   | `metrics_port` | - | Prometheus metrics on 127.0.0.1 |
   | `auditor_url` | owner's axon | the owner validator's service, `https://ip:port`, reached with the catalog's auditor pin; its loopback or private address when it runs next to the gateway (see privacy) |
   | `instances` | 1 | gateway processes sharing the miners' capacity |
   | `trusted_proxies` | localhost | networks whose `X-Forwarded-For` is believed |
   | `request_timeout_s`, `attempt_timeout_s`, `max_attempts` | 10, 4, 3 | the miners' deadline per request, per attempt, attempts |
   | `max_in_flight` | 4096 | requests in flight per process; more get 503 |
   | `max_body_bytes` | 1 MiB | |
   | `max_in_flight_bytes` | 256 MiB | request bodies held in this process at once; more get 503 |
   | `sample_window_s` | 600 | one exchange per endpoint and window goes to the auditor |
   | `admin_host`, `admin_port`, `admin_token_file` | 127.0.0.1, -, - | the admin API (see below); off without a port |
   | `accounts_token_file`, `accounts_prefix` | -, `u_` | the account service's token and the accounts it reaches (see below) |
   | `accounts_max_credit`, `accounts_max_credit_per_hour` | 100 USD, 1,000 USD | the account service's credit caps, in micro-USD (per process) |
   | `state_dir`, `catalog_dir` | `~/.verathos/gateway`, `~/.verathos/catalog` | last routes, catalogs |

6. Schema: `verathos-gateway --config gateway.json db upgrade` (after every update; the gateway does not start on
   an older schema).
7. Run `verathos-gateway --config gateway.json serve` under a supervisor, behind a reverse proxy that terminates
   TLS, as an active and a standby process (below). `GET /ready` answers 200 once the gateway has routes and finds
   its hotkey among the catalog's gateways. Every request holds a connection from the proxy and one to a miner: at
   start the gateway raises its soft open-file limit to the hard one, keeps its connections to each miner machine
   within what `max_in_flight` leaves of it, and warns when that is too little. systemd's default soft limit is 1024, so give the
   unit a higher one:

   ```ini
   [Service]
   User=gateway
   ExecStart=/opt/verathos/.venv/bin/verathos-gateway --config /etc/verathos/gateway.json serve
   Restart=always
   LimitNOFILE=65536
   ```

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
the requests in flight (their checks and `request_timeout_s`), writes usage, samples and served counts (10 s each
at most; what cannot be written is logged), and exits. A request whose process fails before answering goes to the
other process once. A process is ready as soon as it has started: it serves the routes it kept on disk while it
fetches new ones. A restart of the auditor does not reach users: the gateways keep their routes until it serves new
ones. A key's rate limit starts afresh on the other process.

### More than one process

One process keeps rate limits, in-flight counts and breakers exactly. For more throughput, run several (each with
its own `port` and the same `instances`), and have the reverse proxy send each API key to one of them, for example
with nginx:

```nginx
upstream gateway { hash $http_authorization consistent; server 127.0.0.1:8080; server 127.0.0.1:8081; }
```

Each process then uses `1 / instances` of every endpoint's capacity, and each key's rate holds exactly (an account's
keys may land on different processes). Restart them one at a time; with `proxy_next_upstream` as above, the proxy
moves a stopped process's keys to the others meanwhile. A monthly quota is enforced from the database's usage plus
each process's own counts; a process reads the account's month with its key, at most a minute before using it, so a
quota can be exceeded by what other processes served in that time.

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
 "usage": {"input_tokens": 41},
 "receipt": {"receipt": {"miner_hotkey": "5F...", "endpoint_id": "gpu0", "caller_hotkey": "5D...", "...": "..."},
             "signature": "..."}}
```

`GET /v1/models` (computed at most every 15 s) lists the catalog's active models, each with its state limit
(`max_state_tokens`), the input it accepts (`input_modalities`: `["text"]`, a state of plain text or JSON, for every
model), price (`price.usd_per_million_input_tokens`; `usd_per_million_output_tokens` is 0), license, weights
(`weights.repo`, `weights.revision`, `base_model`), whether it is `available` now (a routable endpoint that is not
failing), and its `state` and `latency_ms` as `GET /v1/status` gives them.
`GET /health` answers while the process runs.

`GET /v1/status` (public; computed at most every 15 s and cacheable as long) says per model what users get from
this gateway process:

```json
{"updated_at": "2026-10-07T17:40:00+00:00", "since": "2026-10-07T16:15:00+00:00",
 "models": [{"id": "kev-4b", "state": "operational", "endpoints": 3, "capacity_rps": 42.0,
             "latency_ms": {"p50": 251, "p95": 410, "window_s": 900},
             "success_rate_1h": 0.998, "busy_rate_1h": 0.0, "requests_1h": 5212, "availability_24h": 0.999}]}
```

- `state`: `unavailable` without a routable endpoint or with every endpoint failing; `degraded` when under 98% of the
  last 15 minutes' user requests were answered (of those not refused for want of capacity) or more than half the
  endpoints are failing; else `operational`.
- `latency_ms`: p50 and p95 of the answers to users, retries included, over `window_s`: the last 15 minutes, or,
  when the latest 20 answers (all of the last 24 hours' when fewer) go back further, the fewest whole hours that
  hold them; null without answers in the last 24 hours.
- `success_rate_1h`: answered among answered and failed user requests of the last hour (null under 20 of them);
  `busy_rate_1h`: the share of the hour's user requests refused for want of free capacity (503 `busy`; null under 20
  requests); `requests_1h`: answered, failed and busy.
- `availability_24h`: the share of minutes the model was not unavailable, of the minutes recorded.
- In memory: the figures cover `since` (the process start) until now.

Every response to `POST /v1/systemone` carries `X-Request-Id` (the client's, if it sent a valid one). Errors are
`{"error": {"type": ..., "message": ...}}`:

| Status | Type | When |
|---|---|---|
| 400 | `invalid_request` | not JSON (NaN and infinities included), or not a SystemOne request |
| 401 | `unauthorized` | missing, unknown, revoked or expired key, or a suspended account |
| 402 | `insufficient_credit` | a prepaid account's balance is used up |
| 404 | `model_not_found` | not an active model of the catalog |
| 413 | `too_large` | body, state, or the answers the questions would need, over the limits |
| 429 | `rate_limited` | over the key's rate, or too many invalid keys from the address; `Retry-After` |
| 429 | `quota_exceeded` | the month's quota is used up; `Retry-After` until the next month |
| 500 | `internal` | a fault of the gateway |
| 503 | `busy` | no capacity free for the model now, or the gateway is at capacity (`max_in_flight`, `max_in_flight_bytes`); `Retry-After` |
| 503 | `unavailable` | no miner serves the model right now, none answered in time or the miners refused the request, every miner of the model is failing (`Retry-After: 30`), accounts cannot be checked, or the gateway has no catalog yet (starting); `Retry-After` |

## Accounts and keys

```bash
verathos-gateway accounts create --name acme --rate 50 --burst 100 [--monthly-quota 1000000] [--prepaid yes]
verathos-gateway accounts set --name acme --rate 100        # also --burst, --monthly-quota, --status suspended, --prepaid
verathos-gateway keys create --account acme --name prod     # prints the key once
verathos-gateway keys list --account acme
verathos-gateway keys revoke --account acme --prefix vd_1a2b3c4d5e6f7
verathos-gateway usage --account acme --since 2026-10-01
verathos-gateway credits add --account acme --usd 5 --kind topup --reference <payment id>
verathos-gateway credits list --account acme
```

### Credit

An answer costs its input tokens times the model's catalog `price` (USD per million input tokens). The cost is
written with the usage every 5 s, in whole micro-USD per account, key, model and day (fractions are carried to the
next write, so answers cost what their tokens do), and taken from the account's balance in the same transaction.
Every account has a balance: the credit recorded minus the cost of its usage. A prepaid account (`--prepaid yes`) is
served only while its balance is positive; others are served regardless (their balance just goes negative).

Credit is a ledger. Each entry has a `kind` (`grant`: free credit, `topup`: a payment, `adjust`: an operator's
correction, the only kind that may be negative) and a `reference` naming the payment or grant it records. A
reference is recorded once: adding the same entry again changes nothing, and a reference already used for another
entry is refused. So a payment processor's retries never add a payment twice.

Each process reads a prepaid account's balance at most 5 s before using it and subtracts what it charged since and
what its requests in flight may cost at most (one token per byte of the request), reserved when a request is accepted
and settled to its real cost when it ends. New credit therefore applies within 5 s (at once when recorded through
this process), and other processes' charges within about 10 s (they write every 5 s). A request is accepted while
that remainder is positive, so a balance ends below zero by at most one request's cost per process.

## Admin API

For the operators and the service that signs users up: inspection, requests routed as users' are or sent to one
chosen endpoint, and accounts, keys, usage and credit. Off unless
`admin_port` is set. It listens on a socket of its own, `admin_host:admin_port` (loopback by default; a private
address at most, never `0.0.0.0`, and only with `tls_cert_file`), so the reverse proxy never reaches it, and on the
public port its routes answer 404. Every call needs `Authorization: Bearer <token>`, the token read from `admin_token_file` at start:

```bash
umask 077; openssl rand -hex 32 > ~/.verathos/gateway/admin.token   # at least 32 characters, mode 600
```

The gateway refuses to start with a shorter token or a file others can read. Every call is logged, wrong tokens
with the caller's address, and every credit with the token and address that recorded it.

The service that signs users up (the web app) gets a token of its own, `accounts_token_file` (made the same way, and not
the admin token), whose scope is narrower: only the accounts named with `accounts_prefix` (`u_`), which it creates
prepaid; their keys, usage and ledger; changes other than `prepaid`; credit of kind `grant` or `topup`, at most
`accounts_max_credit` per entry and `accounts_max_credit_per_hour` within an hour (per process; recording an entry again
under its reference takes no more); and the network's usage totals and series. Anything else, including requests, state
and routes, answers 403, and reaching a cap is logged as an error. A leaked web app token can then neither give an
account unlimited service nor mint more than the hourly cap.

| Call | Returns |
|---|---|
| `GET /admin/v1/state` | the routes in use (epoch, age, endpoints, catalog), the models routed, limits, and what waits to be written or sent (usage rows, samples, served counts) |
| `GET /admin/v1/routes` | every routable endpoint as the router sees it: credited rate, latency, cap, in flight, busy time, breaker, its latest failure |
| `POST /admin/v1/systemone` | a SystemOne request, served as described below |
| `POST /admin/v1/accounts` | `{"name", "rate_rps", "burst", "monthly_quota", "prepaid"}` -> 201 and the account; 409 if the name is taken. Names: `[A-Za-z0-9_.:@-]{1,128}` |
| `GET /admin/v1/accounts/{name}` | the account: limits, status, `prepaid`, `balance` (micro-USD, as written) |
| `PATCH /admin/v1/accounts/{name}` | any of `status` (`active`, `suspended`), `rate_rps`, `burst`, `monthly_quota` (null: unlimited), `prepaid` |
| `POST /admin/v1/accounts/{name}/keys` | `{"name", "expires_at"}` -> 201 with `key`, returned only here |
| `GET /admin/v1/accounts/{name}/keys` | the keys: prefix, name, created, expires, revoked |
| `DELETE /admin/v1/accounts/{name}/keys/{prefix}` | revokes the key (within the key cache's minute) |
| `GET /admin/v1/accounts/{name}/usage?since=YYYY-MM-DD` | per day and model: answered and failed requests, input tokens, cost (micro-USD); default since the first of the month |
| `POST /admin/v1/accounts/{name}/credits` | `{"amount" (micro-USD), "kind", "reference", "note"}` -> 201 with the entry and the new balance; 200 if `reference` already records this entry; 409 if it records another |
| `GET /admin/v1/accounts/{name}/credits?limit=100` | the balance and the ledger, newest first |
| `GET /admin/v1/usage/totals` | the network's totals, below |
| `GET /admin/v1/usage/series?days=30&bucket=day` | the network's usage per day or week, below |

Each gateway process serves the admin API on its own `admin_port`; with an active and a standby process, a client
of the account calls tries the other one when one does not answer (a restart). Every write is in the shared
database, so either process gives the same answers.

`GET /admin/v1/usage/totals` sums every account's usage as written (every 5 s), all time and in the last 24 hours
(the current UTC hour and the 23 before it) and 30 days (today and the 29 UTC days before), in all and per model;
amounts in micro-USD:

```json
{"as_of": "2026-10-08T17:42:10+00:00",
 "since": {"last_24h": "2026-10-07T18:00:00+00:00", "last_30d": "2026-09-09T00:00:00+00:00"},
 "requests": {"all_time": 1204311, "last_24h": 48211, "last_30d": 903554},
 "input_tokens": {"all_time": 291004533, "last_24h": 11320444, "last_30d": 218700120},
 "models": {"kev-4b": {"requests": {"all_time": 980112, "last_24h": 40012, "last_30d": 731020},
                       "input_tokens": {"all_time": 233100422, "last_24h": 9310022, "last_30d": 175300111}}},
 "paid_usage_micros": {"all_time": 9120000, "last_30d": 6400000},
 "topups_micros": {"all_time": 25000000, "last_30d": 15000000},
 "paying_accounts": {"all_time": 31, "last_30d": 12}}
```

- `requests`, `input_tokens`: answered requests and their input tokens, of every account (paid, free, the demo,
  internal); admin requests are not usage. Usage from before the hourly totals counts at its day's first hour.
- `paid_usage_micros`: usage paid with bought credit: per account, its usage cost beyond its free credit (grants,
  counted as spent first), up to its top-ups; in the last 30 days, what that grew by. Grants, adjustments and the
  usage they pay are never counted.
- `topups_micros`, `paying_accounts`: the top-ups recorded and the accounts with at least one, all time and in the
  last 30 days.

`GET /admin/v1/usage/series` gives the same usage over time, per UTC day (`bucket=day`, the default) or week
(`bucket=week`, Monday to Sunday): the buckets that hold the last `days` days, whole, up to the current one; without
`days`, all time, from the bucket of the first usage or top-up. At most 400 buckets: more answers 400, and all time
keeps the latest 400. Every bucket is listed, with every model answered in the range:

```json
{"as_of": "2026-10-08T17:42:10+00:00", "bucket": "day", "since": "2026-10-02", "first_day": "2026-08-20",
 "models": ["kev-4b", "kev-9b"],
 "buckets": [{"start": "2026-10-02", "requests": {"kev-4b": 31240, "kev-9b": 4410},
              "input_tokens": {"kev-4b": 7100300, "kev-9b": 1200410},
              "paid_usage_micros": 210000, "topups_micros": 5000000}]}
```

- `first_day`: the first UTC day with usage or a top-up; null before any.
- `requests`, `input_tokens`: as in the totals, per model.
- `paid_usage_micros`: per account, its usage cost in day order: the first part, as much as its grants, is free; the
  next, as much as its top-ups, is paid, on the days it was spent. The days add up to the totals' all-time paid
  usage, and to their last 30 days' unless credit was recorded after the spending it covers: a top-up counts in the
  totals when it is recorded and here on the days it was spent, and a grant makes the earliest paid spending free,
  after the fact.
- `topups_micros`: the top-ups recorded in the bucket.

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

The log says when an endpoint's breaker opens (with its latest failure: the kind and a status or exception class)
and closes, and each kind of failed attempt per endpoint at most once a minute, with how often it happened (an
unexpected error each time, with its traceback); it never
holds a key or what a user sent.
