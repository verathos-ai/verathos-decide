# Throughput audit

Answer verification proves that an endpoint serves the right model. Scores scale with how much it serves, so that
is checked too: every endpoint declares the requests per second it serves, and once per epoch the auditor sends
every endpoint its declared rate, all at the same moment, and credits each endpoint with what it delivers.

Because every endpoint is tested at once, hardware behind several endpoints or hotkeys must serve the sum of their
rates at the same time. A GPU behind two endpoints delivers its capacity once, split between them; a pool of
machines behind many hotkeys delivers its capacity once, however it is spread. There are no GPU classes: what counts
is proven throughput, so miners choose hardware by throughput per cost.

One validator runs the tests: the auditor, by default the subnet owner's validator (the catalog can name another
hotkey). It publishes signed verdicts every validator applies. Every setting lives in the catalog the subnet owner
commits on chain.

| | Code |
|---|---|
| Schedule, policy, test sessions | `src/verathos/throughput/policy.py`, `throughput/follow.py` |
| Standard request mix | `throughput/workload.py` |
| The miner's measurement | `throughput/measure.py` |
| Admission on the miner (tests yield to other traffic) | `miner/admission.py`, `miner/app.py` |
| Auditor and load generator | `throughput/auditor.py`, `throughput/loadgen.py`, `native/loadgen` |
| Verdicts | `throughput/feed.py`, `validator/throughput.py` |

## Declared capacity

Each catalog model defines a standard request mix (`workload`): a range of words per state and a set of questions.
Every request of the mix is unique (a random ticket id and random words from a fixed vocabulary), so nothing can be
served from a cache.

On start the miner waits until each engine answers, then measures all its endpoints in the same rounds, so
endpoints that share hardware compete as they do in a test:

1. warm-up runs of 8 s at concurrency `min(16, max_concurrency)` until every endpoint's last two runs agree within
   5% (at most 12 runs);
2. one run per concurrency step (1, 2, 4, 8, ... up to `max_concurrency`); an endpoint stops climbing once its
   throughput falls below 90% of its best or its latency (concurrency / throughput) exceeds `max_latency_s`
   (default 2 s);
3. five runs at each endpoint's knee: the lowest concurrency within 3% of its best throughput.

The declared `capacity_rps` is `capacity_margin` (default 0.9) times the median of the five runs. Because the
latency at the declared rate stays within `max_latency_s`, and a test warms up for at least that long, a test
measures every honest endpoint in steady state. A measurement
takes a few minutes. It is stored in `state_dir/capacity.json` together with a fingerprint of what it depends on (the
endpoint's model, engine URL and `max_concurrency`, `max_latency_s`, the engine's `GET /v1/models` document, and the
request mix) and is reused until one of them changes. `--remeasure` measures again, for example after a hardware change.

The miner never sends an engine more requests at once than its knee concurrency; further requests wait in the
miner (up to four per engine slot, at most 30 s).

## Schedule

Everyone derives the test from the chain, and nobody knows it before its selection block appears:

- Eligible blocks of epoch `e` are `e x epoch_blocks + 1` up to the last block from which the test traffic still
  ends inside the epoch.
- Block `b` draws a hit when `H("VERATHOS_THROUGHPUT_TEST_V1" || b || hash(b)) x (last - b + 1) < 2^256`. The
  epoch's first hit selects the test; the last eligible block always hits, so there is exactly one test per epoch,
  uniformly placed.
- Traffic starts at `start_block = selection block + lead_blocks` (default 5 blocks, about 60 s) and lasts
  `warmup_s + test_s` (3 s + 8 s).
- Validators hold their canaries from `start_block - 1` to `start_block + traffic_blocks + 1`, where
  `traffic_blocks` is the traffic duration in blocks, rounded up.

## A test

1. **Targets.** At the selection block the auditor takes every open endpoint listed in an accepted manifest with a
   declared capacity and a model that is not retired. Each is tested at its declared rate, but at most
   `rate_growth` (2) times what it delivered in its last counted test, and at most the model's `first_test_rps`
   before its first. An endpoint cannot make the auditor send more than it has shown it serves.
2. **Sessions.** For each endpoint the auditor signs one session: netuid, auditor, miner hotkey, endpoint, test id
   and expiry, sent as `X-Verathos-Session` (base64 of the canonical JSON) and `X-Verathos-Session-Signature`.
   Every request carries a fresh `X-Verathos-Request-Id`. The miner checks the session once and refuses a request id
   it has seen; its receipts name the auditor as caller and the request id as nonce.
3. **Traffic.** At the start block the load generator sends each endpoint its rate, open loop (sending never waits
   for answers), for `warmup_s + test_s`. Delivered = responses (HTTP 200 with a JSON body) completed inside the
   last `test_s` seconds, divided by `test_s`.
4. **Samples.** For `samples_per_test` (10) random delivered responses per endpoint the auditor checks the receipt:
   signed by the miner, binding the request and the answers, naming the auditor and the request id. Delivered is
   scaled by the share of valid receipts. The answers of every valid sample are recomputed by the auditor's
   reference engine; an answer that fails the hard check (`SCORING.md`) is a wrong answer, proven by the miner's own
   signature. The request, the response and the receipt are kept as evidence.
5. **Credit.** The tested rate if delivered reaches `pass_fraction` (0.95) of it, else the delivered rate. A result
   with a wrong answer is `invalid` and credits nothing.
6. **Confirmation.** Once the selection block is final with the hash it was selected by, the test is confirmed;
   otherwise it is void. A test is also void if the load generator fails, starts more than 1 s late, or the
   auditor observes the start block more than one block late. One endpoint's result is void if more than 1% of its
   sends left the generator late.

Outcomes per endpoint: `passed` (credit = tested rate), `partial` (credit = delivered), `invalid`, `void`.

## Other traffic goes first

The miner admits requests from validators and gateways ahead of test requests: a test request waits while any
other request waits, and a request that finds the queue full takes the place of the newest waiting test request. A
test never delays other traffic; test requests beyond the endpoint's spare capacity are not delivered, and the
credit shows it.

## Verdicts

For scoring epoch `E` the auditor publishes, signed by its hotkey, at `GET /v1/throughput/{E}` on its axon:

- per endpoint, the credit of its latest counted result in the confirmed tests of epochs `E - credit_ttl_epochs` to
  `E - 1`, and whether it has qualified (at least `qualification_tests` confirmed tests with a credit);
- the evidence of every wrong answer in those tests.

The verdicts are served once those tests are settled (404 before). Validators wait up to 5 minutes for them, then
use the last verdicts they hold while those still cover a test in the window. A validator applies a piece of
evidence only if the receipt in it binds the request and the answers to the miner's hotkey and, when it runs a
reference engine for the model, its own reference also proves the answers wrong. Each piece applies once.

How credits and wrong answers enter the score is in `SCORING.md`.

## What it catches

| Attempt | Result |
|---|---|
| Two endpoints on one GPU, each declaring the GPU's capacity | tested together, each delivers half and is credited half |
| A pool of machines behind many endpoints or hotkeys | tested together, the pool delivers its capacity once |
| Declaring more than the endpoint serves | credited what it delivers; declaring more gains nothing |
| Fast junk responses | receipts that do not bind scale delivered down; a signed wrong answer invalidates the hotkey |
| Caching or sharing answers | every request is unique |
| Holding capacity back until the test | the test block is unknown until about 60 s before; the endpoint must serve users with the same hardware |

## Policy

`throughput_audit` in the catalog:

| Setting | Default | Meaning |
|---|---|---|
| `mode` | `enforce` | `observe`: validators count declared capacities and apply no evidence; tests still run |
| `auditor_hotkey` | subnet owner | hotkey that runs the tests and signs the verdicts |
| `lead_blocks` | 5 | blocks from the selection block to the start of the traffic |
| `max_latency_s` | 2 | highest latency at which a miner's measurement counts throughput |
| `warmup_s`, `test_s` | 3, 8 | unmeasured and measured traffic, seconds; `warmup_s` is at least `max_latency_s` |
| `request_timeout_s` | 20 | per-request timeout of the load generator |
| `pass_fraction` | 0.95 | share of the tested rate an endpoint must deliver to be credited the full rate |
| `samples_per_test` | 10 | responses per endpoint whose receipts and answers are checked |
| `capacity_margin` | 0.9 | share of the measured median a miner declares |
| `qualification_tests` | 1 | confirmed tests with a credit before an endpoint is scored |
| `credit_ttl_epochs` | 2 | how many closed epochs back a test still counts |
| `rate_growth` | 2.0 | tested rate at most this times the last delivered rate |

Per model: `workload` (the request mix), `score_per_rps` and `first_test_rps`.

## Running the auditor

The auditor needs:
- the load generator: `verathos-build-loadgen` (Rust) builds `native/loadgen` and installs it next to the code;
- a reference engine per model on a GPU: about 10 checks per endpoint and test, and a reference GPU checks Kev-4B
  in fp32 at 10-16 requests/s (24 CPU cores: about 0.13/s);
- a public IP and port for its service, and a steady connection: one machine sends the whole network's test
  traffic in one burst.

Its database is `state_dir/throughput.sqlite` (tables `tests` and `results`, the last 30 epochs).
