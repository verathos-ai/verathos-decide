# Answer verification and scoring

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
cannot predict them and cannot tell them from other traffic. The reference engine answers each canary once; every
endpoint of that model receives its own random subset of `per_endpoint` canaries, spread over the first
`audit_fraction` of the epoch.

Every request is signed by the validator and bound to the miner, endpoint and a nonce; every answer comes with a
receipt signed by the miner hotkey that binds the request, the answers, the model and the catalog. A missing or
invalid receipt counts as an unanswered request.

## Comparison

Per request (hard check). The request fails if:
- question ids or option keys differ from the reference;
- any probability differs from the reference by more than `hard_abs` (default 0.10); or
- the top answer changes where the reference was confident (its margin over the runner-up is at least
  `flip_margin`, default 0.20).

Per endpoint (statistical check). Over a rolling window of `window_epochs` epochs, the mean absolute difference over
all compared probabilities must stay below `mean_abs` (default 0.005) once at least `min_probabilities` (default 100)
have been compared. This catches lower-precision substitutes that stay under the hard bound on single requests.

Tolerances are per model and set in the catalog.

## Score

Per endpoint and epoch:
- zero, with probation, if a request failed the hard check, the statistical check failed, or the capacity audit
  gated the endpoint;
- zero if the endpoint failed the identity challenge, answered nothing, or (in capacity `score_gate` mode) declares
  no calibrated GPU class;
- otherwise `score_weight(model) x answered / sent x latency factor`, where the latency factor is
  `sqrt(median latency of the model's endpoints / own median latency)`, capped to [0.7, 1.3].

An endpoint on probation scores zero for `probation_epochs` epochs, whatever its later results; a new failure
restarts the count. Scores enter an exponential moving average (`ema_alpha`); endpoints that disappear decay toward
zero.

## Weights

A UID's weight is the sum of its endpoints' EMAs, normalised over all UIDs, set on mechanism 1. Endpoints whose UID
now belongs to another hotkey are ignored. If no endpoint has a positive score, the weight goes to the subnet owner's
UID.
