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
native/gemm/   Rust GEMM sumcheck prover used by the capacity proofs
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
VIRTUAL_ENV=$PWD/.venv uv pip install -e ".[capacity]"
verathos-build-native               # miners: CUDA toolkit, Rust (rustup), libssl-dev
verathos-build-native --validator   # audit validator: Rust, libssl-dev
```

Docs: `docs/DISCOVERY.md` (chain-native discovery), `docs/TESTNET_RUNS.md` (testnet results).
