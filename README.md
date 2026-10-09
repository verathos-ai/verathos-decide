# Verathos

A Bittensor subnet for decision models: models that read a state (a ticket, a code change, a document) and answer
typed questions with calibrated probabilities in one forward pass. Miners serve catalog models on their GPUs behind a
SystemOne-compatible API; validators recompute a hidden sample of requests in fp32 and compare the full probability
vectors, which fingerprint the exact weights; a throughput audit sends every endpoint its declared request rate at
the same moment, so endpoints score by the capacity they prove and one machine cannot pose as many. The subnet's
gateway serves users from that proven capacity, and a sample of the answers users get is recomputed too.

Discovery, the model catalog and the audit policy live in native chain state (axons and commitments); there is no
EVM and no central registry. Every connection between machines is TLS: miners and validators serve self-signed
certificates whose pins the chain-authenticated documents name (manifests, the catalog, validator commitments).

## Run a miner or a validator

On Ubuntu 22.04 or 24.04, with a registered wallet's hotkey on the machine (`hotkeys/<hotkey>` and `coldkeypub.txt`
under `~/.bittensor/wallets/<wallet>/`; the coldkey stays elsewhere):

```bash
# a miner: NVIDIA GPUs, a public IP address and one open TCP port (TLS)
curl -LsSf https://raw.githubusercontent.com/<org>/verathos/main/scripts/install.sh | bash -s -- miner

# a light validator: a hotkey with a validator permit, no GPU; it sets the weights the auditor publishes
curl -LsSf https://raw.githubusercontent.com/<org>/verathos/main/scripts/install.sh | bash -s -- validator

# a full validator: it checks the miners' answers itself on reference models, on GPUs or in RAM
curl -LsSf https://raw.githubusercontent.com/<org>/verathos/main/scripts/install.sh | bash -s -- validator --full

# a reference machine: a GPU machine serving a full validator's engines over TLS, to its address only; the validator
# then runs on a machine without a GPU (validator --full --remote-engines reference-engines.json)
curl -LsSf https://raw.githubusercontent.com/<org>/verathos/main/scripts/install.sh | bash -s -- reference \
    --allow <validator ip>
```

Light validators follow the auditor's results; full validators check answers independently
([Two modes](docs/VALIDATING.md#two-modes)).

The command installs what the machine needs, asks a few questions (each has a default), and runs everything as
system services; with `--no-service` it needs no root and prints the commands for pm2 or another process manager.
Miners and validators then install each new release by themselves (`--no-auto-update` turns that off); running the
command again updates them too. What it asks, what it installs, how to check, update and uninstall, and
the manual installation: [Mining](docs/MINING.md#quick-start-one-command),
[Validating](docs/VALIDATING.md#quick-start-one-command). On a machine of your own with Docker, a miner, a light or
full validator and a reference machine also run as containers: [Docker](docs/DOCKER.md).

Models of the shipped catalog (the installers read the catalog the subnet owner commits on chain):

| Model | Status | Miner: GPU memory, bfloat16 | Validator reference, float32: RAM or GPU memory |
|---|---|---|---|
| `kev-4b` | active | about 15 GiB | about 21 GiB or 27 GiB |
| `kev-9b` | active | about 26 GiB | about 42 GiB or 48 GiB |
| `clef-flash` | active | about 21 GiB | about 41 GiB or 47 GiB |

Clef-Flash reads images as well as text ([Images](docs/GATEWAY.md#images)); answers to requests with images are checked against its float32 reference on a GPU.

All three are active from the start: they share the emission equally while users pay for none of them, and what users
pay moves the shares ([Models](docs/SCORING.md#models)). The scores pay the catalog's miner share of the miners'
emission, the [usage pool](docs/SCORING.md#usage-pool) pays the hotkeys that served paying users on top when the
catalog sets one, and validators set the rest on the subnet owner's UID, which the chain does not pay out
([Weights](docs/SCORING.md#weights)). `active` models are scored; `candidate` models, which a catalog can list before
activating them, are checked by validators but score zero. A miner serves one model per GPU: by default the active
model that pays best on it now, from each model's share of the emission and the capacity serving it on the network
([Models](docs/MINING.md#models)), and its plan lists the others that fit; `--model` names one model for every GPU,
or one per GPU in order (`--model kev-4b,clef-flash`). A validator runs each active model's reference on a GPU while
one has room, smallest first, and in RAM otherwise, all resident while they fit and else one at a time, which needs
memory for the largest only; `--candidates` adds the candidates' references where memory is left.

## Documentation

| | |
|---|---|
| [Mining](docs/MINING.md) | hardware, one-command and manual install, engines, configuration, running a miner |
| [Validating](docs/VALIDATING.md) | light and full mode, one-command and manual install, reference engines (on the machine or a GPU machine of their own), configuration, the auditor |
| [Docker](docs/DOCKER.md) | miners, validators and reference machines as containers, from each release's images |
| [Scoring](docs/SCORING.md) | how answers are verified and scored |
| [Throughput audit](docs/THROUGHPUT_AUDIT.md) | declared capacity, the network-wide tests, verdicts, policy |
| [Gateway](docs/GATEWAY.md) | user traffic: keys, limits, routing, answer checks, running it |
| [Discovery](docs/DISCOVERY.md) | manifests, commitments and the catalog on chain; TLS and certificate pins |
| [Subnet administration](docs/SUBNET_ADMIN.md) | mechanisms, catalog releases, audit rollout, code releases and the weights version key |

## Layout

```
src/verathos/
  protocol/    wire types: SystemOne API, signed manifests, receipts, request signing, TLS pins, callers
  catalog/     the model catalog and audit policy committed by the owner
  miner/       miner service: manifest and endpoints behind one public port
  validator/   discovery, canaries, answer comparison, scoring, weights; light mode
  throughput/  throughput audit: schedule, sessions, request mix, miner measurement, auditor, verdicts
  gateway/     verathos-gateway: user traffic to the proven capacity
  engine/      verathos-engine for model families served from their repository code (Clef)
  subnet/      chain helpers miners and validators share, and the verathos-subnet admin CLI
  reference/   a reference machine's front: its engines over TLS to its callers' addresses, with a key
  install/     guided setup: verathos-miner init, verathos-validator init, verathos-reference init, uninstall;
               verathos-container, which sets up and runs a role in a Docker container
  release.py   the release version and the weights version key
  update.py    self-update to each signed release tag
native/loadgen/  Rust load generator the auditor drives the tests with
scripts/       installer, miner probe
docker/        the release's images (Dockerfile), a compose file per role, the build script
release/       allowed_signers: the keys release tags are signed with
docs/
```

## Development

```bash
uv venv -p 3.12 .venv
VIRTUAL_ENV=$PWD/.venv uv pip install -e ".[server,chain,gateway,test]" ruff
.venv/bin/python -m pytest
.venv/bin/ruff check src tests scripts
(cd native/loadgen && cargo test)
```

The test suite runs on CPU. It includes an end-to-end throughput test (miners, auditor, verdicts) against
in-process miners. The Clef engine's tests also need the `clef` extra and the pinned clef-flash release's code,
tokenizer and configs in the Hugging Face cache (not its weights), and the gateway's image tests Linux with Landlock,
as the gateway's image decoders do; without them they are skipped.

## License

MIT
