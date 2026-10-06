# Testnet runs

## 2026-10-06: full subnet run with the capacity audit gating (testnet 405)

Four rented hosts, each with its own public IP (the validator needs one: a miner on the same NAT cannot reach it
through the public address):

| Role | Hardware | Hotkey (UID) | Endpoints |
|---|---|---|---|
| Honest miner | RTX 5090 | `test_miner96/rehearsal-aug18` (12) | `gpu0` Kev-4B |
| Answer cheater | RTX 5090 | `test_miner96/default` (1) | `gpu0` Kev-0.8B advertised as `kev-4b` |
| Sybil | one RTX 5090 | `test_sn96/decide-sybil` (14) | `gpu0` and `gpu1`, both Kev-4B on the same GPU, declared as two 5090s |
| Auditor validator | A100 | `test_sn96` owner (0) | CPU fp32 Kev-4B reference; serves the catalog and the audit |
| Follower validator | same host | `test_vali96/default`, dry run | applies the auditor's verdicts |

Catalog committed by the owner hotkey with `capacity_audit` in `score_gate` and a raised audit rate (30 windows
per tempo, every slot drafted). The miners fetched it from the owner's axon and published their manifests by
themselves.

Capacity audit, first four windows (seconds from the auditor's `B_start` observation to the final receipt; proof
delivery after `B_proof`):

| Endpoint | Outcome (each window) | Final receipt | Proof |
|---|---|---|---|
| Honest `gpu0` | passed x4 | 20.4-21.3 s | 1.1-1.7 s |
| Cheater `gpu0` (real 5090) | passed x4 | 20.9-21.8 s | 1.2-1.5 s |
| Sybil `gpu0`, `gpu1` (one GPU) | timing_miss x4 each | 43.3-44.5 s | - |

All windows were confirmed final. Epoch results, identical on the auditor and on the follower (which gated the
sybil from the auditor's signed verdict feed):

| Endpoint | Canaries | Mean abs diff | Result | Score |
|---|---|---|---|---|
| Honest `gpu0` | 10/10 | 0.0014 | pass | 1.00 |
| Cheater `gpu0` | 10/10 | 0.139 | fail_hard, probation | 0 |
| Sybil `gpu0`, `gpu1` | 7/10 each | 0.0015 | capacity gated (timing), probation | 0 |

The auditor set weights on mechanism 1 for UID 12 only. Found in this run and fixed afterwards: the auditor held
canaries only while an audit was open, so 3 of the sybil's canaries hit its audit-hold `503`s (holds are now timed
like the miners'); a cached empty auditor URL stopped receipt delivery until restart.

## 2026-10-06: Clef-flash drift

`verathos-engine clef` (`Cloudflare/clef-flash` at `17f0b0ad`, torch 2.10, transformers 5.18, reference PyTorch
kernels) on an A100 80GB in bfloat16, on the same A100 in float32, and on CPU in float32; the 30 tickets with six
questions used for Kev (420 probabilities):

| Pair | Max abs diff | Mean | Median | Changed choices |
|---|---|---|---|---|
| A100 bf16 vs CPU fp32 | 0.0085 | 0.0010 | 0.0004 | 0 |
| A100 fp32 vs CPU fp32 | 0.0000 | 0.0000 | 0.0000 | 0 |

Honest drift is below Kev's (max 0.017), so the default tolerances (`hard_abs` 0.10, `mean_abs` 0.005) fit Clef.

## 2026-10-06: first validator epoch on mechanism 1 (testnet 405)

Owner setup with `scripts/testnet_owner_ops.py` (owner coldkey from the OS keyring): `max_allowed_uids` 256 -> 128
(chain rule: max UIDs x mechanisms <= 256), mechanism count 1 -> 2, emission split 90/10 ([58982, 6553]).

Miner (one RTX 5090, hotkey `test_miner96/rehearsal-aug18`, UID 12): `gpu0` Kev-4B, `gpu1` Kev-0.8B advertised as
`kev-4b`. Validator `python -m verathos.validator --once` as UID 0 (`test_sn96`, validator permit) with a CPU fp32
Kev-4B reference engine on another machine; 10 canaries per model, 6 per endpoint.

| Endpoint | Answered | Identity | Mean abs diff | Verdict | Score |
|---|---|---|---|---|---|
| gpu0 (Kev-4B) | 6/6 | verified | 0.0014 | insufficient (below 100 probabilities) | 1.03 |
| gpu1 (Kev-0.8B as kev-4b) | 6/6 | verified | 0.1335 | fail_hard | 0, probation |

`set_weights(mechid=1)` succeeded; 405 has commit-reveal on (reveal period 1 epoch), and the commit appeared in
mechanism 1's timelocked queue, separate from the existing validators' commits on mechanism 0. Epoch wall time
1 min 45 s, mostly CPU reference computation.

Follow-up: the statistical bound needs more probabilities than one epoch of 6 canaries provides; tallies should
accumulate over a rolling window of epochs.

## 2026-10-06: miner smoke test with a disguised model (testnet 405)

Setup:
- One rented host with two RTX 5090s, Kev engines from `jaredpalmer/kev` (bf16, fused DeltaNet, CUDA graphs).
- Miner `python -m verathos.miner` with hotkey `test_miner96/rehearsal-aug18` (UID 12): one public port, two
  endpoints, both advertised as `kev-4b`.
  - `gpu0` runs Kev-4B (honest).
  - `gpu1` runs Kev-9B behind the `kev-4b` label (cheater).
- On start the miner published its axon (`serve_axon`) and manifest digest (`set_commitment`).
- Probe `scripts/probe_miner.py` from another machine, signing as the subnet owner hotkey (configured as a
  permitted caller). Reference answers: `tests/fixtures/drift/drift-kev-4b-CPUfp32.json` (Kev-4B, CPU fp32, same
  30 tickets).

| Endpoint | Manifest | Identity challenge | Receipts | Mean abs diff | Hard failures | Verdict |
|---|---|---|---|---|---|---|
| gpu0 (Kev-4B) | accepted | verified | 30/30 valid and binding | 0.0015 | 0/30 | pass |
| gpu1 (Kev-9B as kev-4b) | accepted | verified | 30/30 valid and binding | 0.0908 | 30/30 | fail_hard |

Round trip median about 330 ms from the probe host (network distance; engine time about 35 ms).

Issues found and fixed during the run:
- `bt.Axon` needs the wallet's `coldkeypub.txt` (public) next to the hotkey; without it `serve_axon` failed and the
  miner kept serving. The miner now stops when `serve_axon` or `set_commitment` fails.
- The bittensor SDK leaves non-daemon threads that keep a script alive after `main()`; the probe exits explicitly.

## 2026-10-06: discovery and catalog anchoring (testnet 405)

Miner hotkey `test_miner96/rehearsal-aug18` (UID 12), owner hotkey `test_sn96` (UID 0), manifest served from a
temporary host, bittensor SDK 10.3.2, a discovery script since folded into the miner, validator and `verathos-subnet`.

| Case | Result |
|---|---|
| Signed manifest (2 endpoints), axon set, digest committed | accepted, both endpoints read, catalog matched |
| Catalog digest committed by the owner hotkey | accepted |
| New validly signed manifest served without committing | rejected: digest differs from commitment |
| Manifest content edited after signing | rejected: signature does not verify |
| Manifest signed by another hotkey served at the axon | rejected: hotkey mismatch |
| New manifest committed and served | accepted, visible 3 blocks after submission |
| Full discovery pass over all 14 UIDs (commitment pre-filter) | 3 s, one manifest fetched |

Commitment space used: 200 of 3,100 bytes (miner, two commits), 100 (owner, one commit).

## 2026-10-06: capacity audit on an RTX 5090

`verathos-build-native` on a clean `nvidia/cuda:12.8.2-devel-ubuntu22.04` box (torch 2.10.0+cu128, Rust 1.85);
test suite 72 passed. Results:

| Run | Timed GPU work | Deadline | Proof build | Payload | Verification |
|---|---|---|---|---|---|
| Calibrated 5090 row, 337 passes of 8960x8960 int8 (`scripts/capacity_roundtrip.py`) | 20.4 s | 30 s | 0.1 s | 257 KB | ok in 58 ms |
| Datacenter shape at small pass counts: transition mix 3 / fanout 64, tail lane, FP64 lane | 2.8 s | n/a | 5.6 s | 11.6 MB | ok in 178 ms |
| Miner hot-start file protocol with the real bench (start seed, final timing, challenge) | 20.5 s | 30 s | | | ok, derived as the auditor does |

Peak GPU memory about 0.6 GB, so the bench fits beside a resident model.
