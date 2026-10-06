# Capacity audit

Answer verification proves an endpoint serves the right model, not that it has its own GPU. Without a hardware
check one GPU could back many hotkeys or endpoints (sybils) and collect several rewards for one machine. The capacity
audit closes that gap: endpoints are drafted at random, together with endpoints that share their IP or domain, and
each drafted endpoint must run a GPU workload calibrated for its declared GPU class inside a deadline, beside its
resident model, and then prove the work. Endpoints that share hardware cannot all finish in time; their misses
accumulate until they are gated.

One validator evaluates audits: the auditor, by default the subnet owner's validator (the catalog can name another
hotkey). It publishes the endpoints it gates, signed by its hotkey, and every other validator applies them. The
policy lives in the catalog (`capacity_audit`), so the subnet owner commits it on chain with the model list.

## Roles and code

| | Code |
|---|---|
| Public derivations (windows, cohorts, seeds, workload rows) | `src/verathos/capacity/core.py` |
| Miner worker and GPU bench | `capacity/miner.py`, `capacity/bench/` |
| Auditor, ingest and verdicts | `capacity/auditor.py`, `capacity/ingest.py`, `capacity/verdicts.py` |
| Holds on other validators | `capacity/holds.py`, `validator/capacity.py` |
| Proof verification | `capacity/proof/`, `capacity/zk/`, `native/gemm` |

## Slots

A slot is one endpoint of one hotkey listed in an accepted manifest: `slot_id = H(netuid, hotkey, endpoint_id)`.
The auditor drafts from every open CUDA endpoint that declares `gpu_name` and `vram_gb`. Validators re-discover
miners every 10 minutes for this list, so a new endpoint joins the audit within minutes. An endpoint whose declared
GPU is not a calibrated class cannot be drafted; in `score_gate` mode it scores zero.

## Windows and cohorts

Miners and the auditor follow the chain head and derive everything from public data, so they agree without
messages:

- A block `B_select` opens a window if `H(block hash, block, tempo, windows_per_epoch)` falls below
  `windows_per_epoch / tempo`, and `B_proof` still falls in the same tempo.
- `B_start = B_select + lead_blocks`, `B_proof = B_start + proof_challenge_delay_blocks`.
- Cohort seed `H(B_select hash, epoch)`. Each slot is drafted independently with probability
  `p = min(cohort_fraction / windows_per_epoch, max_drain_fraction)`: a share `1 - group_stress_fraction` of `p`
  from the slot's own id, the rest from its group tokens (IP, /24, host, registered domain, hotkey, alone and with
  the model). Slots behind one IP or domain are therefore often drafted together.
- A slot drafted in a window stays out of the cohorts until `payload_cooldown_blocks` after that window's `B_proof`;
  miners and the auditor apply the same rule.

## One audit

| Step | Miner | Auditor |
|---|---|---|
| `B_select` | drafts its slot, marks the endpoint busy (`503`) for the hold, prelaunches the bench on the endpoint's GPU | drafts the same slots, records the window |
| `B_start` | releases the bench with `proof_seed = H(B_start hash, slot_id)` | records when it observed `B_start` |
| timed work | runs the workload rows of its GPU class, committing a Merkle root per round; sends a pass-0 receipt and, when done, a final receipt with the combined transcript root | final receipt on time if it arrives within `deadline_s + transport_grace_s` of its own `B_start` observation, and before it has observed `B_proof` |
| `B_proof` | challenge `H(transcript root, B_proof hash, lease, slot_id)`; builds the sampled proof; sends it | proof must arrive within `payload_deadline_s` of `B_proof`; verified against the slot's scheduled workload, the proof seed and the challenge |

All artifacts are signed by the miner hotkey (sr25519) and sent to the auditor's axon.

### The workload

Each GPU class has a calibrated row (`DEFAULT_GPU_CLASSES` in `core.py`): a pass count of int8 GEMMs on an 8960 x
8960 workspace that an unshared GPU of that class finishes in about two thirds of the 30 s deadline. Datacenter rows
add transition rounds (memory-bound state mixing), a tail lane and an FP64 lane that consumer cards cannot run at
the required rate. Measured on an RTX 5090: 337 passes in 20.4 s; the proof is 257 KB and verifies in 58 ms.

### The proof

The timed run commits every round root; nothing about the sample is known until `B_proof`. The proof then opens:
- one sampled round per capacity lane: the challenged Q = A x B block with its Merkle path, a sumcheck proof over the
  Mersenne-61 field that the block equals the product of its A rows and B columns, spot checks of those operands,
  and a GEMM sumcheck over the Pallas scalar field whose operand claims the auditor checks against bands it
  recomputes from the public workspace seed (or, for transition rounds, against Merkle-opened state blocks);
- for transition rounds, one sampled state transition with its source blocks;
- for the FP64 lane, one sampled output block, checked entry by entry as exact dot products.

Lanes are bound into one combined transcript root, committed in the signed final receipt before `B_proof`.

## Outcomes

| Outcome | When |
|---|---|
| `passed` | final on time, proof verified |
| `timing_miss` | final receipt after the deadline |
| `no_show` | no final receipt by the deadline |
| `missing_proof` | final on time, no proof within `payload_deadline_s` of `B_proof` |
| `invalid` | proof fails verification, pass-0 root changed, or the final receipt was sent after `B_proof` |
| `neutral` | the auditor observed a window block late, could not verify, or the window blocks did not finalize unchanged |

Outcomes count only once all three window blocks are finalized with the hashes the auditor observed. Neutral
outcomes never count against a miner.

## Gating

In `score_gate` mode the auditor gates an endpoint when, over the last `repeat_window_epochs` epochs:
- `invalid` >= `invalid_proof_misses_for_zero_score` (default 1), or
- `missing_proof` + `no_show` >= `hard_proof_misses_for_zero_score` (default 2), or
- `timing_miss` >= `timing_misses_for_zero_score` (default 2).

Timing misses and no-shows of the running epoch are not counted until it closes. A gated endpoint scores zero and
enters probation like an endpoint that failed answer verification. Evidence ages out of the window; nothing else
clears it.

The auditor serves the gated endpoints of an epoch at `GET /capacity/v1/verdicts?epoch=N`, signed by its hotkey.
Other validators fetch them from the auditor's axon each epoch, verify the signature, and gate the same endpoints;
if the feed is unreachable or invalid they gate nothing that epoch.

In `observe` mode (the initial setting) the audit runs and records outcomes but gates nothing.

## Policy

| Setting | Default | |
|---|---|---|
| `mode` | `observe` | `observe` or `score_gate` |
| `auditor_hotkey` | subnet owner | hotkey that evaluates audits |
| `windows_per_epoch` | 5 | expected windows per tempo |
| `cohort_fraction` | 0.3 | expected share of slots audited per tempo |
| `max_drain_fraction` | 0.1 | cap on the per-window draft probability |
| `group_stress_fraction` | 0.35 | share of the draft probability drawn from group tokens |
| `lead_blocks`, `proof_challenge_delay_blocks` | 5, 4 | `B_select` to `B_start` to `B_proof` |
| `deadline_s`, `transport_grace_s` | 30, 3 | must be shorter than `proof_challenge_delay_blocks` blocks |
| `payload_deadline_s` | 90 | proof delivery after `B_proof` |
| `max_proof_payload_bytes` | 32 MiB | |
| `repeat_window_epochs` | 10 | evidence window |

Overrides go in the catalog's `capacity_audit` object; the catalog rejects unknown settings and inconsistent
timing.
