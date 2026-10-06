"""Prove-and-verify round trip of the capacity audit bench on this machine's GPU.

Runs the bench with the workload of the given GPU class in the shipped catalog (or, with ``--passes``, the consumer
row shape at that pass count, for calibrating a new class) and a random proof seed and B_proof seed, then prints the
bench's timing and its own verification of the proof. Needs ``verathos-build-native``.

    python scripts/capacity_roundtrip.py "NVIDIA GeForce RTX 5090" 32
    python scripts/capacity_roundtrip.py "NVIDIA GeForce RTX 5080" 16 --passes 200
"""

import argparse
import json
import secrets
import subprocess
import sys
import tempfile

from verathos import catalog
from verathos.capacity import core

parser = argparse.ArgumentParser()
parser.add_argument("gpu_name")
parser.add_argument("vram_gb", type=int)
parser.add_argument("--passes", type=int, help="consumer row shape at this pass count instead of the calibrated row")
args = parser.parse_args()

if args.passes:
    row = core.GpuClass(args.gpu_name, args.vram_gb, capacity_passes=args.passes, capacity_rounds=8)
else:
    policy = catalog.load().audit_policy()
    ok, reason, row = core.gpu_support_status(args.gpu_name, args.vram_gb, policy)
    if not ok:
        raise SystemExit(f"{args.gpu_name}: {reason}; calibrate it with --passes")
spec = core.gpu_workload_spec(row)
slot = core.Slot(405, "roundtrip-hotkey", "gpu0", "http://203.0.113.5:8091/e/gpu0", "kev-4b", args.gpu_name,
                 args.vram_gb)
cmd = [sys.executable, "-m", "verathos.capacity.bench.combined", "--lease-id", core.lease_id(slot, 1),
       "--workload", json.dumps(spec), "--out-dir", tempfile.mkdtemp(prefix="capacity_roundtrip_"),
       "--seed-hex", core.derive_proof_seed(secrets.token_hex(32), slot.slot_id, 0),
       "--b-proof-hex", secrets.token_hex(32)]
print(f"{args.gpu_name}: {spec['pass_count']} passes, deadline {catalog.load().audit_policy().deadline_s:.0f} s",
      flush=True)
result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode:
    raise SystemExit(result.stderr[-3000:])
print(result.stdout)
