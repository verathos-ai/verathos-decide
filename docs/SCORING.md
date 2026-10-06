# Answer verification and scoring

An epoch is `epoch_blocks` blocks (catalog, default 180, about 36 minutes), independent of the subnet tempo: every
validator scores and sets weights once per epoch, within the chain's weight rate limit (100 blocks on SN96).

## Why answers can be checked

A decision model answers a typed request (a state and a set of questions) with a probability distribution per
question, from one forward pass. That pass is nearly deterministic across GPUs, Apple Silicon and CPU, so the full
probability vector fingerprints the exact weights. Measured on Kev-4B/9B (RTX 4090, RTX 5090, M1 Max, CPU fp32; 420
probabilities per run): honest pairs differed by at most 0.017 per probability, with a mean of 0.0014-0.0023; Kev-9B
answering for Kev-4B had a mean difference of 0.084 and exceeded 0.10 on every request.

The validator therefore recomputes a sample of requests itself, in fp32, and compares.

## Canaries

Each epoch the validator derives a pool of `pool_size` requests per model from a private secret, the epoch and the
model (support tickets and code-review requests with generated details and a varying subset of questions). Miners
cannot predict them. Canaries are signed by the validator's hotkey, so a miner can tell validator traffic from
gateway traffic; the checks here cover what validators are served, and answers to other callers are not verified.
The reference engine answers each canary once; every endpoint of that model receives its own random subset of
`per_endpoint` canaries, spread over the first `audit_fraction` of the epoch. If the reference engine cannot answer,
the model's endpoints are not audited that epoch and keep their score.

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

Per endpoint (statistical check). Over the endpoint's last `window_epochs` audited epochs, the mean absolute
difference over all compared probabilities must stay below `mean_abs` (default 0.005) once at least `min_probabilities` (default 100)
have been compared. This catches lower-precision substitutes that stay under the hard bound on single requests.

Tolerances are per model and set in the catalog.

## Score

Per endpoint and epoch:
- zero, with probation, if a request failed the hard check, the statistical check failed, or the capacity audit
  gated the endpoint;
- zero if the endpoint failed the identity challenge or answered nothing, and, while the capacity audit gates
  (`score_gate`), if the audit cannot cover it (not CUDA, or its GPU has no class in the catalog) or the endpoint
  is in its first `qualification_epochs` epochs (see `CAPACITY_AUDIT.md`: the audit needs that time to catch an
  endpoint sharing a GPU);
- otherwise `score_weight(model) x answered / sent x latency factor`, where the latency factor is
  `sqrt(median latency of the model's passing endpoints / own median latency)`, capped to [0.7, 1.3]. Only
  endpoints that passed this epoch set the median.

Probation lasts `probation_epochs` epochs, the failing one included, whatever the later results; a new failure
restarts the count. An endpoint a hotkey adds while one of its endpoints is on probation inherits the remaining
probation, so renaming an endpoint does not escape it. Scores enter an exponential moving average (`ema_alpha`);
endpoints that disappear decay toward zero and are forgotten once negligible.

## Weights

A UID's value is the sum of its endpoints' EMAs, normalised over all UIDs and set on mechanism 1 once per epoch.
Endpoints whose UID now belongs to another hotkey are ignored. If no endpoint has a positive score, the weight goes
to the subnet owner's UID.
