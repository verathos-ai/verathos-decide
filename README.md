# Verathos

A Bittensor subnet for decision models: models that read a state (a ticket, a code change, a document) and answer
typed questions with calibrated probabilities in one forward pass. Miners serve catalog models on their GPUs behind a
SystemOne-compatible API; validators recompute a hidden sample of requests in fp32 and compare the full probability
vectors, which fingerprint the exact weights; a throughput audit sends every endpoint its declared request rate at
the same moment, so endpoints score by the capacity they prove and one machine cannot pose as many. The subnet's
gateway serves users from that proven capacity, and a sample of the answers users get is recomputed too.

Discovery, the model catalog and the audit policy live in native chain state (axons and commitments); there is no
EVM and no central registry.

## Run a miner or a validator

On Ubuntu 22.04 or 24.04, with a registered wallet's hotkey on the machine (`hotkeys/<hotkey>` and `coldkeypub.txt`
under `~/.bittensor/wallets/<wallet>/`; the coldkey stays elsewhere):

```bash
# a miner: NVIDIA GPUs, a public IP address and one open TCP port
curl -LsSf https://raw.githubusercontent.com/<org>/verathos/main/scripts/install.sh | bash -s -- miner

# a validator: a hotkey with a validator permit; the reference models run on GPUs or in RAM
curl -LsSf https://raw.githubusercontent.com/<org>/verathos/main/scripts/install.sh | bash -s -- validator
```

The command installs what the machine needs, asks a few questions (each has a default), and runs everything as
system services. Run it again to update. What it asks, what it installs, how to check, update and uninstall, and
the manual installation: [Mining](docs/MINING.md#quick-start-one-command),
[Validating](docs/VALIDATING.md#quick-start-one-command).

Models of the shipped catalog (the installers read the catalog the subnet owner commits on chain):

| Model | Status | Miner: GPU memory, bfloat16 | Validator reference, float32: RAM or GPU memory |
|---|---|---|---|
| `kev-4b` | active | about 15 GiB | about 21 GiB or 27 GiB |
| `kev-9b` | active | about 26 GiB | about 42 GiB or 48 GiB |
| `clef-flash` | candidate | about 21 GiB | about 41 GiB or 47 GiB |

`active` models are scored; `candidate` models are checked by validators but score zero. A miner serves one model per
GPU: by default the first active model that fits, and its plan lists the others that fit; `--model` names one model
for every GPU, or one per GPU in order (`--model kev-4b,clef-flash`). A validator runs each active model's reference
on a GPU while one has room, smallest first, and in RAM otherwise; `--candidates` adds the candidates' references
where memory is left.

## Documentation

| | |
|---|---|
| [Mining](docs/MINING.md) | hardware, one-command and manual install, engines, configuration, running a miner |
| [Validating](docs/VALIDATING.md) | one-command and manual install, reference engines, configuration, the auditor |
| [Scoring](docs/SCORING.md) | how answers are verified and scored |
| [Throughput audit](docs/THROUGHPUT_AUDIT.md) | declared capacity, the network-wide tests, verdicts, policy |
| [Gateway](docs/GATEWAY.md) | user traffic: keys, limits, routing, answer checks, running it |
| [Discovery](docs/DISCOVERY.md) | manifests, commitments and the catalog on chain |
| [Subnet administration](docs/SUBNET_ADMIN.md) | mechanisms, catalog releases, audit rollout |

## Layout

```
src/verathos/
  protocol/    wire types: SystemOne API, signed manifests, receipts, request signing
  catalog/     the model catalog and audit policy committed by the owner
  miner/       miner service: manifest and endpoints behind one public port
  validator/   discovery, canaries, answer comparison, scoring, weights
  throughput/  throughput audit: schedule, sessions, request mix, miner measurement, auditor, verdicts
  gateway/     verathos-gateway: user traffic to the proven capacity
  engine/      verathos-engine for model families served from their repository code (Clef)
  subnet/      chain helpers miners and validators share, and the verathos-subnet admin CLI
  install/     guided setup: verathos-miner init, verathos-validator init, uninstall
native/loadgen/  Rust load generator the auditor drives the tests with
scripts/       installer, miner probe
docs/
```

## Development

```bash
uv venv -p 3.12 .venv
VIRTUAL_ENV=$PWD/.venv uv pip install -e ".[server,chain,gateway,test]"
.venv/bin/python -m pytest
.venv/bin/ruff check src tests scripts
(cd native/loadgen && cargo test)
```

The test suite runs on CPU. It includes an end-to-end throughput test (miners, auditor, verdicts) against
in-process miners.

## License

MIT
