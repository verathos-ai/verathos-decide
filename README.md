# Verathos

Decision-model network on Bittensor. Miners serve open decision models (typed questions in, calibrated
probabilities out, SystemOne-compatible API); validators verify answers against reference computations and run
the capacity audit against sybil endpoints.

Status: development on testnet.

```
src/verathos/
  protocol/    wire types: SystemOne API, signed manifests, receipts, request signing
  catalog/     model catalog and its owner commitment
  miner/       miner service: manifest and endpoints behind one public port
  validator/   discovery, canaries, answer comparison, scoring, weights
  capacity/    capacity audit: protocol core, proofs, GPU bench, zk primitives
native/pcs/    Rust PCS library used by the capacity proofs
scripts/       operator and test tools
docs/          design notes and testnet results
```

```bash
uv venv -p 3.12 .venv && VIRTUAL_ENV=$PWD/.venv uv pip install -e ".[server,chain,test]"
.venv/bin/python -m pytest
verathos-miner --config miner.json
verathos-validator --config validator.json
```

Capacity-audit native code is built from source on the machine that runs it:

```bash
(cd native/pcs && cargo build --release)                                   # Rust >= 1.85
(cd src/verathos/capacity/zk/cuda && python build.py build_ext --inplace)   # needs libssl-dev
(cd src/verathos/capacity/bench && python build.py build_ext --inplace)
```

Docs: `docs/DISCOVERY.md` (chain-native discovery), `docs/TESTNET_RUNS.md` (testnet results).
