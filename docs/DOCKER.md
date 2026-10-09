# Docker

Miners, validators and reference machines also run as Docker containers, from images each release ships. The
installer stays the main way to set up a machine (`MINING.md`, `VALIDATING.md`); the containers set themselves up the
same way and run the engines the release ships, unchanged. A miner may serve any engine whose answers pass
verification ([3. Start one engine per GPU](MINING.md#3-start-one-engine-per-gpu)): a miner that changes its engine,
to serve more requests per GPU for one, installs with the installer.

## Requirements

- A Linux machine of your own, bare metal or a virtual machine, with Docker Engine and its Compose plugin
  (`docker compose`). A rented container cannot run Docker inside it, and most GPU marketplaces rent containers: use
  the installer there, which runs in a container.
- For a miner or a reference machine: the NVIDIA driver (CUDA 12.8 or newer) and the NVIDIA Container Toolkit,
  configured for Docker (`sudo nvidia-ctk runtime configure --runtime=docker`, then restart Docker). Check:
  `docker run --rm --gpus all ubuntu nvidia-smi` lists the GPUs.
- Whatever the role needs besides: a registered hotkey, a public IP address and an open TCP port, stake (`MINING.md`,
  `VALIDATING.md`).

## Images

| Image | Holds | Runs |
|---|---|---|
| `ghcr.io/verathos-ai/verathos:<version>` | the release's code | a light validator; a full validator whose references run on a reference machine |
| `ghcr.io/verathos-ai/verathos-engines:<version>` | the code and the engines its shipped catalog pins: the Kev engine at its pinned revision and `verathos-engine` for Clef models, each for miners (bfloat16, with the fused kernels) and for references (float32), installed as the installer installs them | a miner; a reference machine |

Each image is tagged with its release's version (`1.0.0` for `v1.0.0`) and nothing else: there is no `latest` tag. The
images hold no weights: a container downloads the pinned weights at its first start into its data directory. The
engines image is several GB.

## Quick start

Each machine runs one role, from a directory of its own holding the role's compose file and an `.env`:

```bash
mkdir verathos && cd verathos
curl -fsSL -O https://raw.githubusercontent.com/<org>/verathos/main/docker/compose.miner.yml \
    -O https://raw.githubusercontent.com/<org>/verathos/main/docker/.env.example
cp .env.example .env    # then set what the role needs (below)
docker compose up -d
docker compose logs -f
```

`COMPOSE_FILE` in `.env` names the role's file, so `docker compose` needs no `-f`. The compose files are
`compose.miner.yml`, `compose.validator.yml` (light), `compose.validator-full.yml` and `compose.reference.yml`.

### Miner

Register the hotkey ([1. Register](MINING.md#1-register)), copy the hotkey and `coldkeypub.txt` to the machine
(`~/.bittensor/wallets/<wallet>/hotkeys/<hotkey>`, `~/.bittensor/wallets/<wallet>/coldkeypub.txt`) and open the
miner's TCP port. In `.env`: `COMPOSE_FILE=compose.miner.yml`, `WALLET`, `HOTKEY`; optionally `MODEL`, `GPUS`, `PORT`,
`PUBLIC_IP`.

`docker compose up -d` starts the container, which sets up the machine as `install.sh miner --yes` does: it checks the
hotkey and its registration, reads the catalog the subnet owner commits, chooses each GPU's model
([Models](MINING.md#models)), downloads the pinned weights (tens of GB, once), writes `miner.json` and starts one
engine per GPU and the miner. The miner measures its capacity (a few minutes), then publishes its manifest:

```
miner | ... endpoint gpu0 (kev-4b): 14.5 requests/s at concurrency 4 (runs 14.5, 15.0, 15.0, 14.0, 14.0), declares 13.8
miner | ... serving 1 endpoints on port 8091; throughput tests are accepted from 5D...; status is logged once per epoch
```

The log of the start ends with the command that checks the miner from another machine (`curl ... /v1/manifest`).
`NO_PUBLISH=1` tries the machine before registering: the miner serves and measures and writes nothing on chain.
With GPUs on several machines, every machine runs a container: `WORKER=1` (and `NAME`) on each worker, `PEERS` with
every worker's `ip:port` on the primary
([One hotkey, several GPUs and machines](MINING.md#one-hotkey-several-gpus-and-machines)).

### Light validator

Copy the hotkey and `coldkeypub.txt` as for a miner. In `.env`: `COMPOSE_FILE=compose.validator.yml`, `WALLET`,
`HOTKEY`. The container sets the weights the auditor publishes each epoch
([Two modes](VALIDATING.md#two-modes)); it needs no GPU and opens no port:

```
... light mode: setting the weights the subnet owner's validator publishes each epoch, once its document verifies; no requests to miners
```

### Full validator

The container runs the validator; its references run on a reference machine
([Engines on a GPU machine of their own](VALIDATING.md#engines-on-a-gpu-machine-of-their-own)), which can be a
container too, on another machine or on this one ([below](#full-validator-with-its-own-gpu)). In `.env`:
`COMPOSE_FILE=compose.validator-full.yml`, `WALLET`, `HOTKEY`; put the reference machine's `reference-engines.json`
into `REMOTE_ENGINES_DIR` (`./reference-engines`; one file per reference machine). `EGRESS_IPS` names the addresses its
requests to miners leave from if detection does not find them
([Egress addresses](VALIDATING.md#egress-addresses-and-miners-firewalls)). With `PUBLIC_IP` set, it serves its status
on `PORT` (default 8091).

The subnet owner's validator, the auditor, does not run in a container: with the subnet owner's hotkey the container
stops (`the subnet owner's validator is the auditor: install it with the installer`)
([Throughput audit](VALIDATING.md#throughput-audit)).

### Full validator with its own GPU

On a machine with GPUs, a reference machine's container and the full validator's run side by side, each from a
directory of its own. The validator reaches the reference machine's front on this machine's loopback, which the front
always admits:

```bash
mkdir -p reference validator/reference-engines
# reference/.env: COMPOSE_FILE=compose.reference.yml, PUBLIC_IP=127.0.0.1, ALLOW=127.0.0.1, SCREENS=off
(cd reference && docker compose up -d)
# once it logs "into the REMOTE_ENGINES_DIR of each validator machine":
sudo cp reference/data/reference-engines.json validator/reference-engines/
# validator/.env: COMPOSE_FILE=compose.validator-full.yml, WALLET, HOTKEY
(cd validator && docker compose up -d)
```

`PUBLIC_IP=127.0.0.1` puts this machine's loopback into the file's URLs, and `ALLOW=127.0.0.1` drops every other
address, so the front's port stays closed to the network. The screens are the auditor's: `SCREENS=off` leaves the GPUs
to the references. Both containers use the host's network; the reference machine's port (8443) and the validator's
(8091, with `PUBLIC_IP`) differ.

### Reference machine

In `.env`: `COMPOSE_FILE=compose.reference.yml`, `ALLOW` (the validator machines' addresses); optionally `PORT`
(8443), `MODEL`, `SCREENS`, `DEVICE`. It holds no wallet. The container places and starts each model's float32
reference and bfloat16 screen as `install.sh reference` does, starts the front, and writes
`DATA_DIR/reference-engines.json`: copy it into each validator machine's `REMOTE_ENGINES_DIR` and start (or restart)
the validator there. It keeps its certificate and key across restarts and updates, so the file stays valid.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `COMPOSE_FILE` | | the role's compose file |
| `IMAGE_REGISTRY`, `VERATHOS_VERSION` | `ghcr.io/verathos-ai`, - | the images: registry and owner (`local`: the ones `docker/build.sh` built on this machine), and the release |
| `DATA_DIR` | `./data` | configuration, state and weights |
| `NETWORK`, `NETUID` | `finney`, 96 | `test` (405), or a `wss://` subtensor node |
| `WALLETS_DIR`, `WALLET`, `HOTKEY` | `~/.bittensor/wallets`, -, `default` | the wallet whose hotkey and `coldkeypub.txt` the container reads |
| `PUBLIC_IP`, `PORT` | detected; 8091 (miner, validator), 8443 (reference machine) | the address callers reach this machine at, and its port |
| `HF_TOKEN` | | a Hugging Face token for the downloads |
| `MODEL` | the model that pays best on each GPU | miner: one model for every GPU, or one per GPU in order (`kev-4b,clef-flash`), `auto` for the setup's choice; reference machine: the models to serve (default: the active ones) |
| `GPUS` | all | miner: the GPUs to serve on, as `nvidia-smi` numbers them (`0,1`) |
| `FIREWALL` | `on` | miner: `off` lets anyone reach its port ([Firewall](MINING.md#firewall)) |
| `NO_PUBLISH` | | miner: `1` tries the machine without writing on chain |
| `WORKER`, `NAME`, `PEERS` | | miner: a worker machine and the start of its endpoint ids; the primary's workers (`ip:port`, comma-separated) |
| `DRY_RUN` | | validator: `1` computes weights without setting them |
| `REMOTE_ENGINES_DIR`, `EGRESS_IPS` | `./reference-engines`, detected | full validator: the reference machines' files; the addresses its requests leave from |
| `ALLOW`, `SCREENS`, `DEVICE` | -, `on`, `auto` | reference machine: the callers' addresses; the screens; where the references run (`auto`, `cpu`, `gpu`) |
| `ENGINE_AUTO_INSTALL` | `on` | miner, reference machine: install a Kev engine revision the catalog pins and the image does not carry into `DATA_DIR/engines`, and serve with it; `off`: only the image's engines run ([Updates](#updates)) |
| `AUTO_RESTART` | `on` | miner, reference machine: when the catalog pins new weights or another engine for a model served here, set its engines up for it and restart them while the container runs; `off`: at the next start ([Updates](#updates)) |

At each start a container sets up its role from these, as the installer does with `--yes`, and writes its
configuration into `DATA_DIR`: `miner.json`, `validator.json` or `reference.json` (`MINING.md`, `VALIDATING.md`). What
you add there that the variables do not set stays, such as a miner's `gateways`, an endpoint's `max_concurrency` or
`catalog_mirrors`; `docker compose restart` applies it. The containers never update themselves (`auto_update` is
false).

The wallet reaches a container as two files, read-only: the hotkey and `coldkeypub.txt`. A container never reads a
coldkey, and `docker compose up` stops when either file is missing.

## Data

`DATA_DIR` holds everything a machine keeps; keep it, and back it up:

| Path | |
|---|---|
| `miner.json`, `validator.json`, `reference.json` | the configuration |
| `miner/` | the machine's TLS key and certificate (its pin: a new key is a new pin), capacity measurements and the caps tests set, the last manifest, the firewall's state |
| `validator/` | the canary secret, the score history, its TLS key and certificate with a service |
| `reference/`, `reference-engines.json` | the front's TLS key and certificate and the callers' key; the file for the validators (mode 600) |
| `hf/`, `catalog/`, `cache/` | model weights, catalogs, compiled GPU kernels |
| `engines/` | the Kev engine revisions the container installed (`ENGINE_AUTO_INSTALL`), about 8 GB each; one goes once no engine runs from it, such as when the image carries it |
| `install/` | what the next start keeps: each GPU's model and the engines' ports |

The containers run as root, so these files are root's. A miner set up by the installer moves to Docker with its pin
and measurements: copy `~/.verathos/miner/` to `DATA_DIR/miner/` (and `~/.verathos/hf/` to `DATA_DIR/hf/`, which
saves the download), then remove the installer's setup, which frees the port (`install.sh uninstall`; it deletes the
engines and weights in `~/.verathos`).

## Networking and the miner's firewall

The miner, the full validator and the reference machine use the host's network (`network_mode: host`): they listen on
the host's port, with no address translation in between, and see their callers' addresses as they are, which the
miner's firewall and the reference machine's front go by. Open the port for inbound TCP in the host's firewall. The
light validator opens no port and uses Docker's own network.

The miner's firewall ([Firewall](MINING.md#firewall)) works in both of its layers, as for a miner the installer runs
as root: with `NET_ADMIN` (in `compose.miner.yml`) the miner keeps its nftables table on the host, and its own process
filters too. Without `NET_ADMIN`, the miner process filters alone (`firewall: the kernel layer is off`). The table
stays when the container stops; `FIREWALL=off` and `docker compose up -d` remove it, as does
`sudo nft delete table inet verathos_miner`.

To keep the miner off the host's network, replace `network_mode: host` and `cap_add` in `compose.miner.yml` with
`ports: ["8091:8091"]`, the miner's `PORT` on both sides: the miner process then filters alone. The firewall needs the
callers' own addresses, which Docker's port forwarding keeps by default; where Docker's userland proxy forwards a
connection (an IPv6 caller to a container on an IPv4 network, for one) and with rootless Docker, the miner sees
other addresses and its firewall only observes. A published port also bypasses the host's firewall (ufw).

## GPUs

A miner's container serves every GPU it gets (all of the host's, `count: all`), or those `GPUS` names: one engine per
GPU, endpoints `gpu0`, `gpu1`, ..., as the installer sets them up, measured together, since a throughput test sends
every endpoint of the machine its rate at once. One container serves one machine. A reference machine's container
places its engines on the GPUs and in RAM as `install.sh reference` does.

## Updates

A release is a new image tag. To update, set `VERATHOS_VERSION` in `.env` to it, then:

```bash
docker compose pull && docker compose up -d
```

This is the only way a container's image changes: the images carry version tags only, so a container runs the release
`VERATHOS_VERSION` names until it is changed, and an image updater such as watchtower has nothing to follow.

The container starts on the new image and sets up its role again. A miner keeps its pin and reuses its measurement
unless what it depends on changed: an engine's code or revision, or how the miner measures; then it measures again.

The subnet owner's catalog changes need no new image:

- **New weights or another engine for a model served here.** By default (`AUTO_RESTART=on`) the container applies them
  by itself while it runs: it reads the committed catalog every minute, downloads the new weights (and installs the
  engine) while the engines run on, then restarts the engines of that model alone (`catalog | engine-gpu0 restarts
  with what catalog ... pins`); the miner measures again where its rules say so. What fails is tried again 10 minutes
  later, the engines running as they are meanwhile. With `AUTO_RESTART=off`, `docker compose restart` applies it.
- **A Kev engine revision the image does not carry.** By default (`ENGINE_AUTO_INSTALL=on`) the container installs it
  into `DATA_DIR/engines` with the installer's own code and serves with it, at the start and when it follows the
  catalog; a reference machine does the same for its references and screens, so they stay exact. The image of a
  release that carries the revision makes that copy unused, and it goes. With `ENGINE_AUTO_INSTALL=off` only the
  image's engines run: a miner serves the model with the revision its image carries and warns at each start
  (`... this image carries ...`; validators check answers, not engines), and a reference machine leaves the model out
  (references run exactly what the catalog pins).
- **A model the catalog retires.** The miner withdraws its endpoints at once; `docker compose restart` gives their GPUs
  another model.
- **A model on a new engine family.** It needs the image of a release that ships the engine.

A validator whose release the subnet owner retires sets no weights (`this validator's release is too old`): update.
Run a reference machine on the release its validators run.

## Health, restarts and logs

| | |
|---|---|
| status | `docker compose ps`: `healthy` once the role's service answers (a miner its `/health`, a reference machine every engine through its front, a full validator with a service its port); `starting` during a first start's downloads |
| logs | `docker compose logs -f`; each line behind its process's name (`engine-gpu0 \|`, `miner \|`, `front \|`), what following the catalog does behind `catalog \|`; rotated at 50 MB, five files |
| restarts | Docker starts the container again when it exits and after a reboot (`restart: unless-stopped`); within it, an engine or the service that exits starts again 10 s later. A setup that stops (an unregistered hotkey, a model that does not fit) says why and tries again about once a minute |
| firewall | `docker compose exec miner verathos-miner firewall status --config /data/miner.json` |
| stop | `docker compose down`; `DATA_DIR` stays |

## Troubleshooting

| Problem | What to do |
|---|---|
| `could not select device driver "nvidia" with capabilities: [[gpu]]` | the NVIDIA Container Toolkit is missing or not configured for Docker: `sudo nvidia-ctk runtime configure --runtime=docker` and restart Docker |
| `no NVIDIA GPU in the container` | the host's driver: `docker run --rm --gpus all ubuntu nvidia-smi` must list the GPUs |
| `bind source path does not exist: .../hotkeys/<hotkey>` | the hotkey or `coldkeypub.txt` is not under `WALLETS_DIR/WALLET`; with `sudo docker`, `~` is root's home: set `WALLETS_DIR` |
| `required variable WALLET is missing a value` | set it in `.env` |
| `port ... is in use on this machine` | another process holds `PORT`, such as a miner the installer set up: remove it (`install.sh uninstall`) or choose another port |
| `hotkey ... is not registered` | register it; `NO_PUBLISH=1` tries the machine first |
| `firewall observing: requests of ... arrive from 172.17.0.1` | a published port behind Docker's userland proxy: use the host's network |
| `... this image carries ...` | the catalog pins a Kev engine revision the image does not carry, and `ENGINE_AUTO_INSTALL` is off or its install failed (the line before says why): turn it on, or update to the release that ships it |
| `... did not install (...)` | the engine's source, PyPI or the disk (`DATA_DIR` needs about 8 GB per revision): the container tries again 10 minutes later while it follows the catalog, else at the next start |
| `docker: 'compose' is not a docker command` | install Docker's Compose plugin |
| downloads slow or rate limited | set `HF_TOKEN` |

Others are as for the installer ([Mining](MINING.md#troubleshooting), [Validating](VALIDATING.md#troubleshooting)).

## Building the images

The subnet owner builds and publishes each release's images from its signed tag:

```bash
bash docker/build.sh v1.2.0           # local/verathos:1.2.0 and local/verathos-engines:1.2.0
bash docker/build.sh v1.2.0 --push    # ghcr.io/verathos-ai/verathos:1.2.0 and .../verathos-engines:1.2.0, pushed
```

`docker/build.sh` exports the tag (only once a key in `release/allowed_signers` verifies it, as the installer checks
releases), builds `verathos` and `verathos-engines` from `docker/Dockerfile` and tags both with the version, as
`local/...` for the machine that built them (`IMAGE_REGISTRY=local`). Only `--push` tags them for the registry and
pushes them, and only for a release. A branch or commit builds as it is, tagged `<version>-<commit>`;
`--working-tree` builds the checkout, tagged `<version>-dev`.

The engines' environments are a layer of their own, keyed by the Kev revisions the catalog pins and
`verathos-engine`'s requirements: a release that changes neither, built where the previous one was, leaves that layer
as it was, and machines download only the layers that changed. Publish the images of a release that ships another Kev
revision before committing the catalog that pins it.
