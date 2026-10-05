# verathos-decide

Decision-model network on Bittensor: miners serve open decision models (typed questions in, calibrated
probabilities out, SystemOne-compatible API); validators verify answers by reference recomputation.

Status: early. Implemented so far:

| Module | Purpose |
|---|---|
| `decide/protocol/api.py` | SystemOne request/response types; answers reduced to probability distributions |
| `decide/protocol/manifest.py` | Signed miner manifests and their on-chain commitment |
| `decide/protocol/signing.py`, `canonical.py` | sr25519 hotkey signatures over canonical JSON |
| `decide/verify/compare.py` | Answer comparison: per-request hard bound, per-endpoint statistical bound |
| `decide/catalog/` | Model catalog and its owner commitment |
| `scripts/testnet_discovery.py` | Chain-native discovery exercised on a live network |

Docs: `docs/DISCOVERY.md` (discovery design and testnet results).

```bash
uv venv -p 3.12 .venv && VIRTUAL_ENV=$PWD/.venv uv pip install -e ".[test]"
.venv/bin/python -m pytest
```
