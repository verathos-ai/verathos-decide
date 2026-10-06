# Mining

A miner serves decision models from the catalog (the model list the subnet owner commits on chain; see
`DISCOVERY.md`) on NVIDIA GPUs, one model per GPU. Only `active` catalog models are scored; `candidate` models can be
served and are checked by validators but score zero. Each GPU runs a decision engine; the miner service in front of them publishes a signed manifest, answers
signed requests, and runs the capacity audit when one of its endpoints is drafted.

Validators score an endpoint by comparing its answers with a reference computation (see `SCORING.md`). The
capacity audit (see `CAPACITY_AUDIT.md`) checks that every endpoint has the GPU it declares to itself.

## Requirements

- Linux with NVIDIA drivers, CUDA toolkit 12.8 (`nvcc`), Python 3.12, Rust (via rustup), `build-essential`,
  `pkg-config`, `libssl-dev`, `git`.
- One GPU per endpoint, of a GPU class the catalog lists (shipped catalog):

  | | GPU classes |
  |---|---|
  | Consumer and workstation | RTX 3090, 3090 Ti, 4090, 5090; RTX A4500, A5000, A6000; RTX 4000/5000/6000 Ada; RTX PRO 4000/4500/5000 Blackwell; RTX PRO 6000 Blackwell Server Edition; L4, L40, L40S, A40 |
  | Datacenter (adds an FP64 lane) | A100 40/80 GB, H100 (HBM3, PCIe, NVL), H200, B200 |

  Other GPUs cannot be audited; in `score_gate` mode their endpoints score zero. The subnet owner adds classes by
  publishing a new catalog.
- A public IP address and one open TCP port for the miner service. Validators reach endpoints only at the axon's
  IP and port, so the miner service must be reachable there directly (plain HTTP; requests and receipts are signed).
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

Validators check answers, not engines: any engine that serves `POST /v1/systemone` and returns the reference
probabilities within the model's tolerances passes, and a faster engine scores higher through the latency factor.
Each catalog model names a known-good engine and revision (`engine.source` at `engine.revision`); a different
engine or version is the miner's risk, since answers that drift past the tolerances fail like a wrong model.

What must be exact are the weights: the catalog pins `hf_repo` at `hf_revision`. Download that revision and serve
it from disk, so an update of the model repository never changes what the endpoint serves. Quantised or other
weights fail verification.

Kev models on the Kev engine:

```bash
git clone https://github.com/jaredpalmer/kev && cd kev && git checkout <engine.revision>
uv sync --extra serve -p 3.12 && VIRTUAL_ENV=$PWD/.venv uv pip install flash-linear-attention
.venv/bin/huggingface-cli download jaredpalmer/kev-4b --revision <hf_revision> --local-dir models/kev-4b
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m kev.serve --run models/kev-4b --port 8008
```

Keep each engine on its own GPU and bind it to localhost.

Clef models run on `verathos-engine`, which loads the inference code shipped in the model repository at the pinned
revision. Install it in its own environment:

```bash
uv venv -p 3.12 /opt/clef && VIRTUAL_ENV=/opt/clef uv pip install -e ".[clef]"
CUDA_VISIBLE_DEVICES=2 /opt/clef/bin/verathos-engine clef --repo <hf_repo> --revision <hf_revision> --port 8010
```

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
| `public_host`, `port` | the public IP address and port validators reach the miner at (the axon); the miner listens on `listen_host` (default `0.0.0.0`) at `port` |
| `endpoint_id` | 1-32 characters of `a-z`, `0-9`, `-`; unique per miner |
| `model_id` | a catalog model |
| `engine_url`, `engine_api_key` | the endpoint's engine and its bearer token, if it needs one |
| `cuda_device` | CUDA index of the endpoint's GPU; the capacity-audit bench runs on it |
| `hardware.device_id` | the GPU UUID: `nvidia-smi --query-gpu=uuid --format=csv,noheader` |
| `hardware.gpu_name` | exactly `nvidia-smi --query-gpu=name --format=csv,noheader` |
| `hardware.vram_gb` | `memory.total` in GiB, rounded (`nvidia-smi --query-gpu=memory.total --format=csv,noheader`) |
| `max_concurrency` | requests the endpoint accepts at once (default 64) |
| `gateway_hotkeys` | callers allowed in addition to validators with a permit |
| `catalog_mirrors` | extra catalog sources, URL templates with `{digest}`, tried after the owner's axon |
| `catalog_dir` | local catalog store (default `~/.verathos/catalog`) |

## 5. Run

```bash
.venv/bin/verathos-miner --config miner.json
```

On start the miner:
1. finds the catalog the subnet owner commits on chain (the shipped copy, its local store, the owner's axon or a
   configured mirror; any copy is checked against the committed digest); endpoints whose model the catalog does not
   serve (unknown or `retired`) are left out, and it exits if none remain;
2. checks the capacity-audit setup (GPU fields, bench extension), then signs a manifest listing its endpoints,
   points its axon at `public_host:port` (`serve_axon`) and commits the manifest digest (`set_commitment`); both
   are free and only sent when they change, so a restart with the same configuration writes nothing;
3. serves the manifest (`GET /v1/manifest`) and the endpoints (`/e/<endpoint_id>/v1/...`) on one port;
4. follows the chain and runs the capacity audit for its CUDA endpoints;
5. checks the owner's commitment every minute; when the owner commits a new catalog it re-signs and re-publishes
   the manifest by itself, withdrawing endpoints whose model was retired, and adopts the new audit policy.

Run it under a supervisor (systemd, pm2) so it restarts after updates and crashes. Requests are accepted only
from hotkeys holding a validator permit and from `gateway_hotkeys`, each signed and bound to this miner; other
callers are refused before their request body is read, and bodies are capped at 1 MiB.

## Reading the log

The miner logs its setup once, then what happens to its endpoints. Every two minutes it reads its status from the
capacity auditor's validator (`GET /v1/status/<hotkey>` on its axon, that validator's view of the last epoch) and
logs it once per epoch, plus every capacity audit once its outcome is final:

```
endpoint gpu0: kev-4b (active) on NVIDIA GeForce RTX 5090 32 GB, CUDA device 0, engine http://127.0.0.1:8008
new endpoints are checked for 5 epochs (about 3.0 h) before they are scored
manifest committed on chain (2968f3219dc7)
serving 1 endpoints on port 8091; status is logged once per epoch
capacity audit at block 8163045, gpu0: drafted, timed work starts at block 8163050
capacity audit at block 8163045, gpu0: timed work done in 20.4 s, final receipt sent
capacity audit at block 8163045, gpu0: proof sent
validator's epoch 45351:
  gpu0 (kev-4b): qualifying, checked but not scored until epoch 45354 (in about 96 min); answers pass 10/10, mean diff 0.0014
capacity audit at block 8163045, gpu0: passed, timed work done in 20.4 s
validator's epoch 45354:
  gpu0 (kev-4b): scored 0.99 this epoch, average 0.20; answers pass 10/10, mean diff 0.0014
```

| Endpoint state | Meaning |
|---|---|
| `scored` | checks passed; the score and its average (EMA) |
| `qualifying` | new endpoint, checked but not scored until the epoch shown |
| `candidate model` | the catalog lists the model as a candidate: checked, not scored |
| `answers FAILED` | an answer failed verification; zero score for the probation epochs shown |
| `GATED by the capacity audit` | audits were missed or a proof was invalid; zero score for the probation epochs shown |
| `on probation` | still serving out an earlier failure |
| `not scored, ... cannot be audited` | the declared GPU has no class in the catalog |
| `not in that epoch` | the validator did not accept the manifest in that epoch (published after it started, or rejected: check the validator's reason with `scripts/probe_miner.py`) |

## Capacity audit duty

When an endpoint is drafted, the miner warms the bench up while still serving, answers `503` for that endpoint only
while the timed GPU work runs (about 20 s, beside the resident model), and sends a signed final receipt and a proof
to the auditor. A new endpoint is checked for its first `qualification_epochs` epochs before it is scored. Nothing needs to
be done by hand, but:
- do not run other GPU work on an endpoint's GPU;
- declare the GPU each endpoint really has: an endpoint that shares a GPU with another, or claims a faster class,
  misses deadlines and is gated;
- use a new `endpoint_id` when an endpoint moves to other hardware;
- use a reliable subtensor node: artifacts name the block hashes the miner used, and work built on a fork the
  chain did not keep is a miss;
- keep the machine responsive; a late final receipt is a miss;
- restart the miner between audits: an endpoint drafted when the miner stops misses that audit (a `no_show`).
  The miner logs each drafted endpoint, and an audit takes about two minutes.

## Updating

```bash
git pull && VIRTUAL_ENV=$PWD/.venv uv pip install -e ".[server,chain,capacity]"
CUDA_HOME=/usr/local/cuda .venv/bin/verathos-build-native
```

Then restart the miner. Catalog changes (new models, tolerances, audit policy) need no update: the miner picks up a
new commitment by itself. Releases carry code changes, such as support for a new engine family.

## Checking a miner

From another machine, with a hotkey the miner accepts (for example a gateway hotkey listed in `gateway_hotkeys`):

```bash
.venv/bin/python scripts/probe_miner.py --network finney --netuid 96 --uid <uid> --endpoint gpu0 \
    --caller-wallet <wallet> --caller-hotkey <hotkey> --reference tests/fixtures/drift/drift-kev-4b-CPUfp32.json
```

It checks the manifest against the chain, the endpoint's identity signature and receipts, and compares 30 answers
with stored CPU fp32 reference answers (the fixture covers `kev-4b`).
