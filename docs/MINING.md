# Mining

A miner serves decision models from the catalog (`src/verathos/catalog/catalog.json`) on NVIDIA GPUs, one model
per GPU. Each GPU runs a decision engine; the miner service in front of them publishes a signed manifest, answers
signed requests, and runs the capacity audit when one of its endpoints is drafted.

Validators score an endpoint by comparing its answers with a reference computation (see `SCORING.md`). The
capacity audit (see `CAPACITY_AUDIT.md`) checks that every endpoint has the GPU it declares to itself.

## Requirements

- Linux with NVIDIA drivers, CUDA toolkit 12.8 (`nvcc`), Python 3.12, Rust (via rustup), `build-essential`,
  `pkg-config`, `libssl-dev`, `git`.
- One GPU per endpoint, of a calibrated class:

  | | GPU classes |
  |---|---|
  | Consumer and workstation | RTX 3090, 3090 Ti, 4090, 5090; RTX A4500, A5000, A6000; RTX 4000/5000/6000 Ada; RTX PRO 4000/4500/5000 Blackwell; RTX PRO 6000 Blackwell Server Edition; L4, L40, L40S, A40 |
  | Datacenter (adds an FP64 lane) | A100 40/80 GB, H100 (HBM3, PCIe, NVL), H200, B200 |

  Other GPUs cannot be audited; in `score_gate` mode their endpoints score zero.
- A public IP and one open TCP port for the miner service.
- A hotkey registered on the subnet; the miner machine needs the hotkey and `coldkeypub.txt`, never the coldkey.

## 1. Register

```bash
btcli subnet register --netuid 96 --wallet.name <wallet> --wallet.hotkey <hotkey>
```

Copy `~/.bittensor/wallets/<wallet>/coldkeypub.txt` and `~/.bittensor/wallets/<wallet>/hotkeys/<hotkey>` to the
same paths on the miner machine.

## 2. Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
curl -sSf https://sh.rustup.rs | sh -s -- -y
git clone https://github.com/<org>/verathos && cd verathos
uv venv -p 3.12 .venv
VIRTUAL_ENV=$PWD/.venv uv pip install -e ".[server,chain,capacity]"
CUDA_HOME=/usr/local/cuda .venv/bin/verathos-build-native
```

`verathos-build-native` compiles the capacity-audit code on this machine for the GPUs it sees: the workload
kernels, the block prover and the Rust GEMM prover. Run it again after every update.

## 3. Start one engine per GPU

Kev models run on the Kev engine, which serves `POST /v1/systemone`:

```bash
git clone https://github.com/jaredpalmer/kev && cd kev
uv sync --extra serve -p 3.12 && VIRTUAL_ENV=$PWD/.venv uv pip install flash-linear-attention
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m kev.serve --run jaredpalmer/kev-4b --port 8008
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m kev.serve --run jaredpalmer/kev-9b --port 8009
```

Serve the exact weights the catalog pins (`hf_repo` at `hf_revision`). Any other weights, quantisation or model
fails verification. Keep each engine on its own GPU and bind it to localhost.

## 4. Configure

`miner.json`:

```json
{
  "network": "finney",
  "netuid": 96,
  "wallet_name": "<wallet>",
  "hotkey_name": "<hotkey>",
  "public_host": "203.0.113.5",
  "port": 8091,
  "endpoints": [
    {
      "endpoint_id": "gpu0",
      "model_id": "kev-4b",
      "engine_url": "http://127.0.0.1:8008",
      "cuda_device": 0,
      "hardware": {"kind": "cuda", "device_id": "GPU-6f1c...", "gpu_name": "NVIDIA GeForce RTX 5090", "vram_gb": 32}
    }
  ]
}
```

| Field | Meaning |
|---|---|
| `network` | `finney`, `test`, or a subtensor websocket URL |
| `public_host`, `port` | where validators reach the miner; the miner listens on `listen_host` (default `0.0.0.0`) at `port` |
| `scheme` | `http` (default) or `https` if a TLS proxy terminates in front of the miner |
| `endpoint_id` | 1-32 characters of `a-z`, `0-9`, `-`; unique per miner |
| `model_id` | a catalog model |
| `engine_url`, `engine_api_key` | the endpoint's engine and its bearer token, if it needs one |
| `cuda_device` | CUDA index of the endpoint's GPU; the capacity-audit bench runs on it |
| `hardware.device_id` | the GPU UUID: `nvidia-smi --query-gpu=uuid --format=csv,noheader` |
| `hardware.gpu_name` | exactly `nvidia-smi --query-gpu=name --format=csv,noheader` |
| `hardware.vram_gb` | `memory.total` in GiB, rounded (`nvidia-smi --query-gpu=memory.total --format=csv,noheader`) |
| `max_concurrency` | requests the endpoint accepts at once (default 64) |
| `gateway_hotkeys` | callers allowed in addition to validators with a permit |

## 5. Run

```bash
.venv/bin/verathos-miner --config miner.json
```

On start the miner:
1. checks that its catalog matches the one the subnet owner commits on chain, and exits otherwise (update and
   restart);
2. signs a manifest listing its endpoints, points its axon at `public_host:port` (`serve_axon`) and commits the
   manifest digest (`set_commitment`); both are free and only sent when they change;
3. serves the manifest (`GET /v1/manifest`) and the endpoints (`/e/<endpoint_id>/v1/...`) on one port;
4. follows the chain and runs the capacity audit for its CUDA endpoints.

Run it under a supervisor (systemd, pm2) so it restarts after updates and crashes. Requests are accepted only
from hotkeys holding a validator permit and from `gateway_hotkeys`, each signed and bound to this miner.

## Capacity audit duty

When an endpoint is drafted, the miner answers `503` for that endpoint for a few minutes, runs the calibrated GPU
workload beside the resident model, and sends signed receipts and a proof to the auditor. Nothing needs to be done
by hand, but:
- do not run other GPU work on an endpoint's GPU;
- declare the GPU each endpoint really has: an endpoint that shares a GPU with another, or claims a faster class,
  misses deadlines and is gated;
- keep the clock synchronised (NTP) and the machine responsive; a late final receipt is a miss.

## Updating

```bash
git pull && VIRTUAL_ENV=$PWD/.venv uv pip install -e ".[server,chain,capacity]"
CUDA_HOME=/usr/local/cuda .venv/bin/verathos-build-native
```

Then restart the miner. A release that changes the catalog becomes active when the owner commits the new catalog;
from then on validators accept only manifests built with it.

## Checking a miner

From another machine, with a hotkey the miner accepts (for example a gateway hotkey listed in `gateway_hotkeys`):

```bash
.venv/bin/python scripts/probe_miner.py --network finney --netuid 96 --uid <uid> --endpoint gpu0 \
    --caller-wallet <wallet> --caller-hotkey <hotkey> --reference tests/fixtures/drift/drift-kev-4b-CPUfp32.json
```

It checks the manifest against the chain, the endpoint's identity signature and receipts, and compares 30 answers
with stored CPU fp32 reference answers (the fixture covers `kev-4b`).
