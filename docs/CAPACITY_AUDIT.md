# Capacity audit

Answer verification proves an endpoint serves the right model, not that it has its own GPU. Without a hardware
check one GPU could back many hotkeys or endpoints (sybils) and be scored several times for one machine. The capacity
audit closes that gap: endpoints are drafted at random, together with endpoints that share their hotkey or IP, and
each drafted endpoint must run a GPU workload calibrated for its declared GPU class inside a deadline, beside its
resident model, and then prove the work. Endpoints that share hardware cannot all finish in time; their misses
accumulate until they are gated.

One validator evaluates audits: the auditor, by default the subnet owner's validator (the catalog can name another
hotkey). It publishes the endpoints it gates, signed by its hotkey, and every other validator applies them. The
policy, including the calibrated GPU classes, lives in the catalog (`capacity_audit`), so the subnet owner commits
it on chain with the model list. Miners and validators check the committed catalog every minute.

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
miners every 2 minutes for this list, so a new endpoint joins the audit within minutes. An endpoint whose declared
GPU has no class in the catalog cannot be drafted; in `score_gate` mode it scores zero, as do non-CUDA open
endpoints. Endpoint URLs are on the hotkey's axon (validators reject other manifests), so the grouping tokens below
come from chain data a miner cannot choose per endpoint. A slot's bench lease binds only its identity and epoch,
`H(netuid, hotkey, endpoint_id, epoch)`; to change hardware, use a new endpoint id.

## Windows and cohorts

Miners and the auditor follow the chain head and derive everything from public data, so they agree without
messages:

- Epochs are the subnet's scoring epochs: `epoch = block // epoch_blocks` (catalog, default 180 blocks).
- A block `B_select` opens a window if `H(block hash, block, epoch_blocks, windows_per_epoch)` falls below
  `windows_per_epoch / epoch_blocks`, and `B_proof` still falls in the same epoch.
- `B_start = B_select + lead_blocks`, `B_proof = B_start + proof_challenge_delay_blocks`.
- Cohort seed `H(B_select hash, epoch)`. With `t = draft_probability`, a slot is drafted if its own id draws a hit
  with probability `t x (1 - group_stress_fraction)`, or if any of its group tokens does, each with probability
  `t x group_stress_fraction`. The tokens are the hotkey, the axon IP and its /24, each alone and with the model;
  a token hit drafts every slot that carries it, so endpoints of one hotkey or behind one IP are drafted together.
- A slot drafted in a window stays out of the cohorts until `payload_cooldown_blocks` after that window's `B_proof`;
  miners and the auditor apply the same rule. Both replay the recent blocks when they start, so slots drafted
  before a restart still count as busy; a window whose `B_start` has passed by the time a miner processes it only
  marks the slot busy. The auditor keeps a hotkey's last slots while a manifest fetch fails and its commitment is
  unchanged.

## One audit

| Step | Miner | Auditor |
|---|---|---|
| `B_select` | drafts its slot, prelaunches the bench on the endpoint's GPU (the endpoint keeps serving) | drafts the same slots, records the window |
| `B_start` | stops serving the endpoint (`503`) and releases the bench with `proof_seed = H(B_start hash, slot_id)` | records when it observed `B_start` |
| timed work | runs the workload row of its GPU class, committing a Merkle root per round; when done, sends a final receipt with the combined transcript root and the `B_start` hash it used, and serves again | final receipt on time if it arrives within `deadline_s + transport_grace_s` of its own `B_start` observation, and before it has observed `B_proof` |
| `B_proof` | challenge `H(transcript root, B_proof hash, lease, slot_id)`; builds the sampled proof; sends it with the `B_proof` hash it used | proof must arrive within `payload_deadline_s` of `B_proof`; verified against the slot's scheduled workload, the proof seed and the challenge |

All artifacts are signed by the miner hotkey (sr25519), bind the slot's lease and the digest of its workload row,
and are sent to the auditor's axon. Final receipts are retried until the deadline, and never after the miner has
seen `B_proof`; proofs travel on a separate lane, so a proof upload never delays another endpoint's receipt. The
auditor takes the arrival time when a request arrives. A proof upload names its slot in headers signed by the
slot's hotkey; the auditor refuses it before reading the body unless that slot is waiting for a proof and has no
other upload in progress.

### The workload

Each GPU class has a calibrated row (`capacity_audit.gpu_classes` in the catalog): a pass count of int8 GEMMs on an
8960 x 8960 workspace that an unshared GPU of that class finishes in about two thirds of the 30 s deadline. Datacenter rows
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
| `missing_proof` | final on time, no proof within `payload_deadline_s` of `B_proof`; or an artifact built on another `B_start` or `B_proof` hash than the canonical one (the miner's node followed a fork) |
| `invalid` | proof fails verification, or the final receipt arrived after `B_proof` |
| `neutral` | see below |

Outcomes count only once all three window blocks are finalized with the hashes the auditor observed. Validator-side
gaps never count against a miner; the slot is neutral when:
- the auditor observed `B_start` only after the next block (its timing would be off);
- the auditor could not follow the window from its selection block: it started or restarted (it replays recent
  blocks, so the slots still count as busy, as on the miners), or its block feed stalled past `B_start`;
- the slot was still open when the auditor restarted (artifacts were refused while it was down);
- the window opened while a newly committed policy propagates, from one catalog check before the auditor adopted
  it until two checks plus one hold after, since miners adopt it up to a check apart;
- verification failed on the auditor's side, or the window blocks did not finalize unchanged.

## Gating

In `score_gate` mode the auditor gates an endpoint in epoch E when, over the `repeat_window_epochs` epochs before
E:
- `invalid` >= `invalid_proof_misses_for_zero_score` (default 1), or
- `missing_proof` + `no_show` >= `hard_proof_misses_for_zero_score` (default 2), or
- `timing_miss` >= `timing_misses_for_zero_score` (default 2).

The same rules apply to each hotkey over the evidence of all its endpoints, including ones it renamed or removed,
with the miss thresholds multiplied by its current number of endpoints (`invalid` is not scaled); when they fire,
every current endpoint of the hotkey is gated. Renaming an endpoint therefore never clears evidence, while a few
honest misses spread over a large hotkey do not gate it.

Only closed epochs count. The last windows of an epoch settle (proof, verification, finality) within a few
minutes of its end, and validators ask after their canary phase, so they all get the same set. A gated endpoint scores zero and enters probation like an endpoint that failed answer
verification; new endpoints of the same hotkey inherit that probation. Evidence ages out of the window; nothing
else clears it.

The auditor serves the gated endpoints of an epoch at `GET /capacity/v1/verdicts?epoch=N` (epochs that have
started only), signed by its hotkey. Other validators fetch them from the auditor's axon each epoch, after their
canary phase, verify the signature, and gate the same endpoints; if the feed is unreachable or invalid they gate
nothing that epoch.

In `observe` mode the audit runs and records outcomes but gates nothing, and endpoints need no qualification; it is
a switch for the subnet owner if the audit misbehaves, not a normal operating mode.

## Downtime, detection and qualification

An endpoint stops serving only during its timed work: from `B_start` until its final receipt, about 20 s on a
calibrated GPU and at most `deadline_s + transport_grace_s`. Validators hold their canaries for a drafted endpoint
during those blocks (one block before `B_start` to one after the deadline), so an audit never costs an endpoint
score.

A sybil is caught only when two endpoints sharing one GPU are drafted into the same window: both then need about
40 s and miss the deadline. Gating takes two such windows (the timing threshold), and verdicts count closed epochs
only, so add one epoch. A new endpoint is therefore scored only after `qualification_epochs` epochs; most sybils are
gated before they are ever scored. Renaming an endpoint restarts its qualification, and new endpoints of a hotkey on
probation inherit the probation.

Measured with the selection code (20,000 windows per row), 180-block epochs and `windows_per_epoch` 4, for the
hardest case of two endpoints on one GPU: endpoints of different hotkeys on different IPs (no shared grouping
token), plus the cheaper shapes:

| `draft_probability` | drafted per window | downtime | 2 hotkeys, 2 IPs: epochs to gate, median / 90% | 2 hotkeys, 1 IP | 1 hotkey, 1 IP |
|---|---|---|---|---|---|
| 0.15 | 35% | 2.3% | 3.5 / 7.8 | 2.0 / 4.2 | 1.5 / 3.2 |
| **0.25 (default)** | 52% | 3.3% | 1.5 / 3.2 | 1.2 / 2.5 | 1.0 / 2.0 |
| 0.35 | 64% | 4.1% | 1.0 / 2.0 | 1.0 / 1.8 | 0.8 / 1.5 |

Downtime is the share of time an endpoint does not serve (about 35 s per audit). With the default, an endpoint is
audited about twice per epoch and a two-endpoint sybil is gated within 3.2 epochs in 90% of cases (4.2 with the
closed-epoch lag), hence the default of 4 qualification epochs (about 2.4 hours). More endpoints per GPU are caught
faster.

## Policy

| Setting | Default | |
|---|---|---|
| `mode` | `score_gate` | `score_gate` or `observe` |
| `auditor_hotkey` | subnet owner | hotkey that evaluates audits |
| `windows_per_epoch` | 4 | expected windows per epoch |
| `draft_probability` | 0.25 | `t`, the probability of each draw (slot id, each grouping token) per window |
| `group_stress_fraction` | 0.35 | share of `t` drawn on grouping tokens instead of the slot id |
| `qualification_epochs` | 4 | epochs a new endpoint is checked before it is scored |
| `lead_blocks`, `proof_challenge_delay_blocks` | 5, 4 | `B_select` to `B_start` to `B_proof` |
| `deadline_s`, `transport_grace_s` | 30, 3 | must be shorter than `proof_challenge_delay_blocks` blocks |
| `payload_deadline_s` | 90 | proof delivery after `B_proof` |
| `max_proof_payload_bytes` | 32 MiB | |
| `repeat_window_epochs` | 10 | evidence window |
| `gpu_classes` | none | calibrated workload rows; only these GPU classes can be audited |

Settings go in the catalog's `capacity_audit` object (the epoch length is the catalog's `epoch_blocks`); the catalog
rejects unknown settings, duplicate GPU classes and inconsistent timing. Changing `epoch_blocks` renumbers epochs,
which starts the capacity evidence afresh. A GPU class row needs `gpu_name`, `vram_gb` and `capacity_passes`; datacenter rows also set
the tail, FP64 and transition settings (see the shipped catalog).
