# Verathos

Decision-model network on Bittensor: miners serve open decision models (typed questions in, calibrated
probabilities out, SystemOne-compatible API); validators verify answers against reference computations.

Status: early development on testnet.

| Package | Purpose |
|---|---|
| `verathos/protocol/` | Wire types: SystemOne API, signed manifests, receipts, request signing, canonical encoding |
| `verathos/verify/` | Answer comparison: per-request hard bound, statistical bound |
| `verathos/catalog/` | Model catalog and its owner commitment |
| `verathos/miner/` | Miner service: manifest and endpoints behind one public port, chain publishing |
| `verathos/validator/` | Discovery, canary audits, scoring, weights |

Docs: `docs/DISCOVERY.md` (chain-native discovery), `docs/TESTNET_RUNS.md` (testnet results).

```bash
uv venv -p 3.12 .venv && VIRTUAL_ENV=$PWD/.venv uv pip install -e ".[server,chain,test]"
.venv/bin/python -m pytest
python -m verathos.miner --config miner.json
python -m verathos.validator --config validator.json
```
