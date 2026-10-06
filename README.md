# Verathos

A Bittensor subnet for decision models: models that read a state (a ticket, a code change, a document) and answer
typed questions with calibrated probabilities in one forward pass. Miners serve catalog models on their GPUs behind a
SystemOne-compatible API; validators recompute a hidden sample of requests in fp32 and compare the full probability
vectors, which fingerprint the exact weights; a capacity audit drafts endpoints at random to run a calibrated GPU
workload, so one GPU cannot pose as many.

Discovery, the model catalog and the audit policy live in native chain state (axons and commitments); there is no
EVM and no central registry.

## Documentation

| | |
|---|---|
| [Mining](docs/MINING.md) | hardware, install, engines, configuration, running a miner |
| [Validating](docs/VALIDATING.md) | reference engines, configuration, running a validator and the auditor |
| [Scoring](docs/SCORING.md) | how answers are verified and scored |
| [Capacity audit](docs/CAPACITY_AUDIT.md) | the hardware audit: windows, workload, proofs, gating, policy |
| [Discovery](docs/DISCOVERY.md) | manifests, commitments and the catalog on chain |
| [Subnet administration](docs/SUBNET_ADMIN.md) | mechanisms, catalog releases, audit rollout |
| [Testnet runs](docs/TESTNET_RUNS.md) | measured results |

## Layout

```
src/verathos/
  protocol/    wire types: SystemOne API, signed manifests, receipts, request signing
  catalog/     the model catalog and audit policy committed by the owner
  miner/       miner service: manifest and endpoints behind one public port
  validator/   discovery, canaries, answer comparison, scoring, weights
  capacity/    capacity audit: public derivations, miner worker, GPU bench, auditor, proofs
  subnet/      chain helpers miners and validators share, and the verathos-subnet admin CLI
native/gemm/   Rust GEMM sumcheck prover used by the capacity proofs
scripts/       miner probe and capacity-bench round trip
docs/
```

## Development

```bash
uv venv -p 3.12 .venv
VIRTUAL_ENV=$PWD/.venv uv pip install -e ".[server,chain,capacity,test]"
.venv/bin/verathos-build-native --validator     # CPU parts; without --validator (CUDA_HOME set) also the GPU kernels
.venv/bin/python -m pytest
.venv/bin/ruff check src tests scripts
```

The test suite runs on CPU. It includes an end-to-end capacity audit (miner workers, ingest, auditor) on a CPU
reference of the GPU bench.

## License

MIT
