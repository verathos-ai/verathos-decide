"""Prove-and-verify round trip of the capacity audit bench on this machine's GPU.

Runs the combined bench with the calibrated workload the policy assigns to the given GPU class, then reports the
bench's own verification of the resulting proof. Needs the native builds: native/pcs (cargo build --release),
verathos/zk/cuda and verathos/capacity/bench (python build.py build_ext --inplace).

    python scripts/capacity_roundtrip.py "NVIDIA GeForce RTX 5090" 32
"""
import json, secrets, subprocess, sys, time
from verathos.capacity import core
from verathos.capacity.proof.combined import CURRENT_COMBINED_PROOF_PROTOCOL_VERSION
policy = core.AuditPolicy()
ok, why, row = core.gpu_support_status(sys.argv[1], int(sys.argv[2]), policy)
if not ok:
    raise SystemExit(f"{sys.argv[1]}: {why}")
spec = core.gpu_workload_spec(row)
slot = core.Slot(405, "5HTestHotkey", "gpu0", "http://203.0.113.5:8091/e/gpu0", "kev-4b", sys.argv[1], int(sys.argv[2]))
lease = core.lease_id(slot, 22660)
proof_seed = core.derive_proof_seed(secrets.token_hex(32), slot.slot_id, 0)
cmd = [sys.executable, "-m", "verathos.capacity.bench.bench_combined", "--lease-id", lease, "--gpu-index", "0",
       "--seed-hex", proof_seed, "--b-proof-hex", secrets.token_hex(32), "--out", "capacity_roundtrip.json",
       "--proof-protocol-version", str(CURRENT_COMBINED_PROOF_PROTOCOL_VERSION)]
for k, v in spec.items():
    if k not in {"workload_version", "pass_count", "proof_protocol_version"}:
        cmd += [f"--{k.replace('_', '-')}", str(v)]
print("supported:", ok, why, "| passes:", spec["pass_count"], "| deadline_s:", row.deadline_s, flush=True)
t0 = time.time()
r = subprocess.run(cmd, capture_output=True, text=True)
print("exit", r.returncode, "wall", round(time.time() - t0, 1), "s", flush=True)
print(r.stderr[-1500:] if r.returncode else "", flush=True)
report = json.load(open("capacity_roundtrip.json"))
print({k: report.get(k) for k in ("verify_ok", "proof_format", "timed_cuda_component_s", "proof_payload_bytes", "proof_payload_s", "verify_s")})
