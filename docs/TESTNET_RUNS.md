# Testnet runs

## 2026-10-06: miner smoke test with a disguised model (testnet 405)

Setup:
- One rented host with two RTX 5090s, Kev engines from `jaredpalmer/kev` (bf16, fused DeltaNet, CUDA graphs).
- Miner `python -m decide.miner` with hotkey `test_miner96/rehearsal-aug18` (UID 12): one public port, two
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

See `DISCOVERY.md`.
