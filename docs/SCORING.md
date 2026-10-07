# Answer verification and scoring

An epoch is `epoch_blocks` blocks (catalog, default 180, about 36 minutes), independent of the subnet tempo: every
validator scores and sets weights once per epoch, within the chain's weight rate limit (100 blocks on SN96). The
settings below are the catalog's `scoring` section and each model's `tolerance`, so every validator applies the
same ones.

## Why answers can be checked

A decision model answers a typed request (a state and a set of questions) with a probability distribution per
question, from one forward pass. That pass is nearly deterministic across GPUs, Apple Silicon and CPU, so the full
probability vector fingerprints the exact weights. Measured on Kev-4B/9B (RTX 4090, RTX 5090, M1 Max, CPU fp32; 420
probabilities per run): honest pairs differed by at most 0.017 per probability, with a mean of 0.0014-0.0023; Kev-9B
answering for Kev-4B had a mean difference of 0.084 and exceeded 0.10 on every request.

The validator therefore recomputes a sample of requests itself, in fp32, and compares.

## Canaries

Each epoch the validator derives a pool of `canary_pool_size` (16) requests per model from a private secret, the
epoch and the model (support tickets and code-review requests with generated details and a varying subset of questions). Miners
cannot predict them. Canaries are signed by the validator's hotkey, so a miner can tell validator traffic from
gateway traffic; the checks here cover what validators are served, and answers to other callers are not verified.
The reference engine answers each canary once; every endpoint of that model receives its own random subset of
`canaries_per_endpoint` (10) canaries, spread over the first `canary_spread_fraction` (0.6) of the epoch. Canaries
wait while a throughput test runs (`THROUGHPUT_AUDIT.md`). If the reference engine cannot answer, the model's
endpoints are not audited that epoch and keep their score, for at most 3 epochs in a row.

Every request is signed by the validator and bound to the miner and a nonce. Every answer comes with a receipt
signed by the miner hotkey that binds the request, the answers, the endpoint, the model, the caller, the nonce, the
netuid and the mechanism. A request the endpoint declines (error status or timeout) counts as unanswered. A reply
without a valid receipt, or whose answers are malformed (a probability that is not a number in [0, 1]), fails the
request as a hard failure.

## Comparison

Per request (hard check). The request fails if:
- question ids or option keys differ from the reference;
- any probability differs from the reference by more than `hard_abs` (default 0.10); or
- the top answer changes where the reference was confident (its margin over the runner-up is at least
  `flip_margin`, default 0.20).

Per endpoint (statistical check). Over the endpoint's last `statistical_window_epochs` (3) audited epochs, the mean
absolute difference over all compared probabilities must stay below `mean_abs` (default 0.005) once at least
`min_probabilities` (default 100) have been compared. This catches lower-precision substitutes that stay under the
hard bound on single requests.

Tolerances are per model and set in the catalog.

## Score

Per endpoint and epoch:
- zero, with probation for every endpoint of the hotkey and their averages reset to zero, if a request failed the
  hard check, the statistical check failed, or the throughput auditor proved a wrong answer the hotkey signed in a
  test (see `THROUGHPUT_AUDIT.md`);
- zero if the endpoint failed the identity challenge, answered nothing, or has not yet proven capacity in a
  throughput test;
- otherwise `score_per_rps(model) x proven capacity x answered / sent x latency factor`. The proven capacity is the
  auditor's credit in requests/s of the model's standard mix, at most what the endpoint declares now (in `observe`
  mode the declaration itself, up to the model's `first_test_rps`). `score_per_rps` makes models of different cost
  per request comparable; `candidate` models have none. The latency factor is `sqrt(median of the model's scored
  endpoints / own latency)`, capped to [0.7, 1.3], where an endpoint's latency is its median latency in its
  throughput test, as the auditor's verdicts state it, so every validator applies the same factor. Each endpoint
  weighs in the median by its proven capacity, so splitting capacity over more endpoints does not move it. In
  `observe` mode there is no latency factor.

Probation lasts `probation_epochs` (6) epochs, the failing one included, whatever the later results; a new failure
restarts the count, and the hotkey's statistical windows start over. An endpoint a hotkey adds while one of its
endpoints is on probation inherits the remaining probation, so renaming an endpoint does not escape it. Scores enter an exponential moving average (`ema_alpha`,
0.2); endpoints that disappear decay toward zero and are forgotten once negligible.

## Weights

A UID's value is the sum of its endpoints' EMAs, normalised over all UIDs and set on mechanism 1 once per epoch.
Endpoints whose UID now belongs to another hotkey are ignored. If no endpoint has a positive score, the weight goes
to the subnet owner's UID.
