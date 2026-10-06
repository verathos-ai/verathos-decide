"""Prove-and-verify round trip of the capacity audit bench on this machine's GPU.

Runs the bench with the calibrated workload the policy assigns to the given GPU class and a random proof seed and
B_proof seed, then prints the bench's timing and its own verification of the proof. Needs ``verathos-build-native``.

    python scripts/capacity_roundtrip.py "NVIDIA GeForce RTX 5090" 32
"""

import json
import secrets
import subprocess
import sys
import tempfile
from pathlib import Path

from verathos.capacity import core

gpu_name, vram_gb = sys.argv[1], int(sys.argv[2])
ok, reason, row = core.gpu_support_status(gpu_name, vram_gb, core.AuditPolicy())
if not ok:
    raise SystemExit(f"{gpu_name}: {reason}")
spec = core.gpu_workload_spec(row)
slot = core.Slot(405, "roundtrip-hotkey", "gpu0", "http://203.0.113.5:8091/e/gpu0", "kev-4b", gpu_name, vram_gb)
lease = core.lease_id(slot, 1)
out_dir = Path(tempfile.mkdtemp(prefix="capacity_roundtrip_"))
cmd = [sys.executable, "-m", "verathos.capacity.bench.combined", "--lease-id", lease, "--workload", json.dumps(spec),
       "--out-dir", str(out_dir), "--seed-hex", core.derive_proof_seed(secrets.token_hex(32), slot.slot_id, 0),
       "--b-proof-hex", secrets.token_hex(32)]
print(f"{gpu_name}: {spec['pass_count']} passes, deadline {row.deadline_s:.0f} s", flush=True)
result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode:
    raise SystemExit(result.stderr[-3000:])
print(result.stdout)
