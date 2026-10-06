# Hot Capacity Workspace

This directory contains local protocol prototypes for Verathos hot-capacity
audits. It is not subnet integration code.

## Files

- `bench_workspace.py`: capacity lane benchmark/prototype. The current strongest
  capacity workload is the transition/fanout v3 path, enabled with
  `--transition-mix-rounds > 0` and `--transition-fanout > 1`.
- `bench_fp64_identity.py`: FP64 server-identity lane benchmark/prototype. It
  commits FP64 output roots during timed GPU work, then builds a post-timing
  Fiat-Shamir sampled proof for one output block.
- `workspace_kernel.cu` and `binding.cpp`: shared CUDA extension used by both
  lanes.

## Build

```bash
cd scripts/hot_capacity_workspace
python3 build.py build_ext --inplace
```

## Capacity Lane

Use the v3 transition/fanout path to test whether a claimed lease can remain hot
while other leases are due in the same audit window.

Example smoke run:

```bash
python3 bench_workspace.py \
  --matrix-dim 1024 \
  --passes 2 \
  --rounds 1 \
  --block-size 64 \
  --transition-mix-rounds 3 \
  --transition-fanout 64 \
  --verify-proof \
  --out /tmp/hot_capacity_v3_capacity
```

Tamper-check an emitted final JSON:

```bash
python3 check_capacity_v3_tamper.py \
  /tmp/hot_capacity_v3_capacity/lease-000_final.json \
  --out /tmp/hot_capacity_v3_tamper.json
```

## Combined Audit

Use this harness to test the actual local protocol shape: v3 capacity and FP64
identity in one committed audit transcript with one post-timing sample seed.
The default mode is sequential workspace reuse: capacity roots are committed,
the capacity arena is released/reused, FP64 roots are committed, and sampled
proof openings are reconstructed post-timing one lane at a time. Do not use
simultaneous workspace residency for production memory accounting.

```bash
python3 bench_combined.py \
  --capacity-matrix-dim 512 \
  --capacity-passes 1 \
  --capacity-rounds 1 \
  --capacity-block-size 64 \
  --transition-mix-rounds 2 \
  --transition-fanout 8 \
  --fp64-matrix-dim 512 \
  --fp64-passes 1 \
  --fp64-rounds 1 \
  --fp64-block-size 64 \
  --spot-checks 1 \
  --out /tmp/hot_capacity_combined.json
```

`bench_combined.py` derives:

```text
combined_transcript = H(capacity_roots || fp64_roots || parameters)
sample_seed = H(B_proof || combined_transcript)
```

The same `sample_seed` selects the v3 capacity opening and the FP64 identity
opening. This is the shape to port later.

The capacity lane can be split into a main segment and an optional tail segment.
Use the main segment for the strongest transition/fanout workload, and use the
tail only to fill residual timing budget when main-pass granularity is too
coarse. The tail has its own committed root chain, parameters, sampled opening,
and verification result; it is not hidden inside the main proof.

Run the compact validation matrix on real hardware with:

```bash
python3 validate_combined.py \
  --out-dir /tmp/hot_capacity_combined_validation \
  --capacity-workspace-mib 500 \
  --capacity-passes 1 \
  --capacity-rounds 1 \
  --capacity-warmup-passes 1 \
  --transition-mix-rounds 3 \
  --transition-fanout 64 \
  --capacity-tail-enabled \
  --capacity-tail-profile-passes 4 \
  --capacity-tail-transition-mix-rounds 1 \
  --capacity-tail-transition-fanout 8 \
  --fp64-matrix-dim 4096 \
  --fp64-passes 1 \
  --fp64-rounds 1 \
  --fp64-warmup-passes 1 \
  --run-simultaneous-diagnostic
```

For server-class calibration, derive the fused pass counts from one shared
target instead of assigning each lane a full window:

```bash
python3 validate_combined.py \
  --out-dir /tmp/hot_capacity_combined_server_mix \
  --capacity-workspace-mib 500 \
  --capacity-profile-passes 4 \
  --capacity-passes 1 \
  --capacity-rounds 1 \
  --capacity-warmup-passes 1 \
  --transition-mix-rounds 3 \
  --transition-fanout 64 \
  --fp64-matrix-dim 4096 \
  --fp64-profile-passes 64 \
  --fp64-passes 1 \
  --fp64-rounds 1 \
  --fp64-warmup-passes 1 \
  --auto-mix-passes \
  --target-timed-s 20.0 \
  --fp64-target-fraction 0.3333333333
```

This profiles several passes per lane, allocates the shared target, optionally
fills remaining capacity budget with the lower-cost tail, and then validates
the fused sequential workload with the derived pass counts. Do not derive
calibration rows from one-pass FP64 profiles; CUBLAS/launch overhead distorts
the per-pass estimate.

The validation matrix runs capacity-only, FP64-only, fused sequential, and
optionally fused simultaneous diagnostics. The expected production property is
that fused sequential timing is close to the sum of the separate lanes while
live allocation stays close to the larger individual lane, not the sum.
Use the reported preallocated timed relation for calibration. Prefer
`timed_cuda_component_s` for profile-derived pass counts; `timed_wall_s`,
allocation, cache management, warmup, and post-timing proof assembly are
diagnostics/operational overhead, not pass-count threshold components.
Record `capacity_params`, `capacity_tail_params`, `fp64_params`,
`capacity_tail_verify_ok`, and `timed_cuda_component_s` for any row that uses
the tail.

The combined harness is for server-identity rows. For non-FP64 GPU claims,
use the capacity-only lane and assign the full target to capacity. A small FP64
diagnostic lane can be added later if evidence shows it catches real spoofing,
but it is not part of the default non-server policy.

## FP64 Identity Lane

Use this lane only for GPU classes where FP64 is a server-class feature, such as
A100/H100/H200/B200 and close descendants. Do not apply it to L40S, RTX 6000
Ada, RTX PRO 6000, RTX 4090, RTX 5090, or other non-FP64-class inference GPUs.

Example smoke run:

```bash
python3 bench_fp64_identity.py \
  --matrix-dim 4096 \
  --block-size 64 \
  --passes 1 \
  --rounds 1 \
  --spot-checks 1 \
  --verify-proof \
  --tamper-test \
  --out /tmp/fp64_identity_4096.json
```

The FP64 lane local message order is:

1. Miner derives deterministic FP64 A/B matrices from the audit seed and runs
   timed GPU work.
2. During the timed path, CUDA commits the FP64 output C Merkle root for every
   pass/round.
3. Miner emits final timing plus the C-root chain.
4. After `B_proof`/Fiat-Shamir challenge is known, the miner recomputes only the
   sampled round, opens one C block, and sends the Merkle path plus sampled C
   values.
5. Validator recomputes the challenge, checks the root path, and recomputes one
   or more deterministic FP64 dot products on CPU.

The verifier intentionally does not recompute the full GEMM.

For production, FP64 should be fused into the same audit-window transcript as
the v3 capacity lane, not run as an unrelated one-off timing test:

- v3 capacity roots and FP64 C roots are committed before final timing;
- the same post-timing unpredictable `B_proof`/block challenge samples both
  relations;
- the deadline applies to the combined calibrated workload;
- GPU policy decides whether an endpoint must satisfy capacity only or capacity
  plus FP64 identity.

The standalone FP64 harness is a local relation/proof prototype and calibration
tool. Its one-round timings are not decision-grade by themselves.
