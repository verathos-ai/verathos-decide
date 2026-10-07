# Mining

A miner serves decision models from the catalog (the model list the subnet owner commits on chain; see
`DISCOVERY.md`), one model per GPU. Only `active` catalog models are scored; `candidate` models can be served and
are checked by validators but score zero. Each GPU runs a decision engine; the miner service in front of them
measures each endpoint's capacity, publishes a signed manifest and answers signed requests.

Validators check every endpoint's answers against a reference computation (see `SCORING.md`), and the throughput
audit (see `THROUGHPUT_AUDIT.md`) checks the capacity each endpoint declares. An endpoint's score is the capacity it
proves, so hardware pays by its throughput per cost.

## Requirements

- Linux, Python 3.12, `git`, and what the engines need (NVIDIA drivers for GPU engines).
- One GPU per endpoint. Any GPU that runs the model's engine works; a GPU shared by several endpoints is credited
  once, split between them.
- A public IP address and one open TCP port for the miner service. Validators and the auditor reach endpoints only
  at the axon's IP and port, so the miner service must be reachable there directly (plain HTTP; requests and
  receipts are signed). A test sends every endpoint its declared rate for about 20 s: the connection must carry it.
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
git clone https://github.com/<org>/verathos && cd verathos
uv venv -p 3.12 .venv
VIRTUAL_ENV=$PWD/.venv uv pip install -e ".[server,chain]"
```

## 3. Start one engine per GPU

Validators check answers, not engines: any engine that serves `POST /v1/systemone` and returns the reference
probabilities within the model's tolerances passes, and a faster engine proves more capacity. Each catalog model
names a known-good engine and revision (`engine.source` at `engine.revision`); a different engine or version is the
miner's risk, since answers that drift past the tolerances fail like a wrong model.

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
    {"endpoint_id": "gpu0", "model_id": "kev-4b", "engine_url": "http://127.0.0.1:8008"}
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
| `max_concurrency` | highest concurrency the capacity measurement tries (default 256); the engine never gets more requests at once than the concurrency the measurement settles on |
| `gateway_hotkeys` | callers allowed in addition to validators with a permit |
| `catalog_mirrors` | extra catalog sources, URL templates with `{digest}`, tried after the owner's axon |
| `catalog_dir` | local catalog store (default `~/.verathos/catalog`) |
| `state_dir` | capacity measurements and the last published manifest (default `~/.verathos/miner`) |

## 5. Run

```bash
.venv/bin/verathos-miner --config miner.json
```

On start the miner:
1. finds the catalog the subnet owner commits on chain (the shipped copy, its local store, the owner's axon or a
   configured mirror; any copy is checked against the committed digest); endpoints whose model the catalog does not
   serve (unknown or `retired`) are left out, and it exits if none remain;
2. starts its service: the manifest (`GET /v1/manifest`) and the endpoints (`/e/<endpoint_id>/v1/...`) on one port;
3. waits until every engine answers, then measures the endpoints' capacity through its own service, all at once (a
   few minutes; see `THROUGHPUT_AUDIT.md`), or reuses the stored measurements if nothing they depend on changed.
   Meanwhile it serves the manifest it last published. An endpoint that cannot be measured is withdrawn, and the
   miner stops only if none is left;
4. signs a manifest listing its endpoints and their declared capacity, points its axon at `public_host:port`
   (`serve_axon`) and commits the manifest digest (`set_commitment`); both are free and only sent when they change,
   so a restart with the same configuration and measurements writes nothing;
5. checks the owner's commitment and its engines every minute; when the owner commits a new catalog it re-signs
   and re-publishes the manifest by itself, withdrawing endpoints whose model was retired and measuring again if the
   catalog changed what a measurement depends on (a model's request mix or `max_latency_s`); when an engine changed
   (its version, weights or settings), it measures again. A failed measurement keeps the current manifest, and a
   measurement that a throughput test interrupts is retried later.

Run it under a supervisor (systemd, pm2) so it restarts after updates and crashes. Requests are accepted from
hotkeys holding a validator permit, from the subnet's gateways (the catalog's `gateways`) and the miner's own
`gateway_hotkeys`, and as test traffic under a session the auditor signed. Other callers are refused before their
request body is read; bodies are capped at 1 MiB and must arrive within 10 s, and a state may hold at most 4
characters per token of the model's `max_state_tokens`. A validator gets at most 4 requests at once per endpoint
(HTTP 429 beyond that); gateways are not limited. The miner signs only well-formed answers: an engine that returns
anything else gets an error back to the caller, never a signed receipt. The auditor recomputes a sample of the
answers given to the subnet's gateway, as it does test answers: serve users the same model, at the same precision.

## Reading the log

The miner logs its setup once, then what happens to its endpoints. Every two minutes it reads its status from the
auditor's validator (`GET /v1/status/<hotkey>` on its axon, that validator's view of the last epoch) and logs it
once per epoch, plus every throughput test of the hotkey once it is final:

```
measuring the capacity of 1 endpoints (about 3 minutes)
endpoint gpu0 (kev-4b): 19.9 requests/s at concurrency 32 (runs 19.2, 22.9, 19.1, 20.1, 19.9), declares 18.9
manifest committed on chain (2968f3219dc7)
serving 1 endpoints on port 8091; throughput tests are accepted from 5F...; status is logged once per epoch
validator's epoch 45351:
  gpu0 (kev-4b): qualifying, checked but not scored until it passes a throughput test; answers pass 10/10, mean diff 0.0014
throughput test of epoch 45351 (block 8163045), gpu0: passed, delivered 18.9 of 18.9 requests/s, p95 latency 1630 ms; credited 18.9
validator's epoch 45352:
  gpu0 (kev-4b): scored 18.71 this epoch, average 3.74; proven 18.9 of 18.9 requests/s declared; answers pass 10/10, mean diff 0.0014
```

| Endpoint state | Meaning |
|---|---|
| `scored` | checks passed; the score, its average (EMA) and the proven capacity |
| `qualifying` | checked, but not scored until it passes a throughput test |
| `candidate model` | the catalog lists the model as a candidate: checked, not scored |
| `answers FAILED` | an answer failed verification; zero score for the probation epochs shown |
| `WRONG ANSWER in a throughput test` | the auditor proved a wrong answer the hotkey signed; every endpoint of the hotkey restarts from zero after probation |
| `on probation` | still serving out an earlier failure |
| `not in that epoch` | the validator did not accept the manifest in that epoch (published after it started, or rejected: check the validator's reason with `scripts/probe_miner.py`) |

| Test outcome | Meaning |
|---|---|
| `passed` | delivered at least 95% of the tested rate; credited what it delivered |
| `BELOW the tested rate` | delivered less than 95% of it; credited delivered x delivered / tested; the reason names the shortfall |
| `INVALID` | a sampled answer was wrong |
| `not counted` | the test or this endpoint's result was void (the reason says why); earlier credit still counts |

## Throughput tests

Once per epoch, at a block nobody knows in advance, every endpoint gets its declared rate for 19-22 s: a warmup
of random length, then a 16 s measured window. A reply counts if it arrives inside the window, to a request sent at
most 2 s before the window opened, so an endpoint whose latency stays within 2 s has every reply counted. Nothing
needs to be done by hand, but:
- keep the engines warm and their GPUs free of other work: a test delivers what the hardware has at that moment;
- run `--remeasure` after changing an endpoint's hardware; the miner measures again by itself when the engine's
  identity, the endpoint's configuration, the request mix or `max_latency_s` changes;
- serve the catalog's precision: the auditor recomputes test answers, and an endpoint whose answers drift from the
  reference over several tests loses its credit;
- an engine that just restarted runs slow for its first minutes, and a test then credits less;
- requests from validators and gateways are always served before test requests, so a busy endpoint delivers less
  of the test rate.

## Updating

```bash
git pull && VIRTUAL_ENV=$PWD/.venv uv pip install -e ".[server,chain]"
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
