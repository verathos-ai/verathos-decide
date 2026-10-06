# Validating

A validator discovers miners from chain state, sends every endpoint a few canary requests per epoch, compares the
answers with its own reference computation, and sets weights on mechanism 1. How answers are compared and scored
is in `SCORING.md`.

Every validator takes part in the capacity audit (`CAPACITY_AUDIT.md`). The validator whose hotkey is the
auditor (the subnet owner unless the catalog names another hotkey) runs it; every other validator applies the
auditor's signed verdicts. Each validator finds its role at startup from the catalog and the chain.

## Requirements

- Linux, Python 3.12, `git`.
- A hotkey registered on the subnet with a validator permit.
- One reference engine per catalog model. The reference is the model run in fp32; CPU is enough because each
  canary is computed once per epoch and reused for every endpoint. Measured on a rented server CPU: Kev-4B 3.4 s and
  Kev-9B 6.6 s per request; the fp32 weights alone take about 16 GB and 36 GB of RAM. A GPU in fp32 is faster.
- The auditor additionally needs Rust (rustup), `build-essential`, `libssl-dev`, a public IP and one open TCP
  port.
- A public IP of its own: behind most NATs a machine cannot reach other machines that share its public IP through
  that IP, so a validator next to miners on one NAT can neither send them canaries nor receive their audit
  artifacts.

## 1. Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/<org>/verathos && cd verathos
uv venv -p 3.12 .venv
VIRTUAL_ENV=$PWD/.venv uv pip install -e ".[server,chain]"
```

The auditor installs the capacity extra and builds the GEMM verifier library:

```bash
VIRTUAL_ENV=$PWD/.venv uv pip install -e ".[server,chain,capacity]"
.venv/bin/verathos-build-native --validator
```

## 2. Start the reference engines

The reference defines the right answer, so it runs exactly what the catalog pins: the engine at `engine.revision`
and the weights at `hf_revision`, in float32.

```bash
git clone https://github.com/jaredpalmer/kev && cd kev && git checkout <engine.revision>
uv sync --extra serve -p 3.12
.venv/bin/huggingface-cli download jaredpalmer/kev-4b --revision <hf_revision> --local-dir models/kev-4b
CUDA_VISIBLE_DEVICES="" KEV_DTYPE=fp32 .venv/bin/python -m kev.serve --run models/kev-4b --port 8101
```

Run them on CPU without `flash-linear-attention` installed.
Clef models use `verathos-engine` in float32 (see `MINING.md` for the install):

```bash
/opt/clef/bin/verathos-engine clef --repo <hf_repo> --revision <hf_revision> --port 8103 --device cpu --dtype float32
```

## 3. Configure

`validator.json`:

```json
{
  "network": "finney",
  "netuid": 96,
  "wallet_name": "<wallet>",
  "hotkey_name": "<hotkey>",
  "references": {"kev-4b": "http://127.0.0.1:8101", "kev-9b": "http://127.0.0.1:8102"}
}
```

| Field | Default | Meaning |
|---|---|---|
| `mechid` | 1 | mechanism the weights are set on |
| `references` | | model id -> reference engine URL; models without one are not audited by this validator |
| `pool_size` | 16 | canaries per model per epoch, each computed once by the reference |
| `per_endpoint` | 10 | canaries each endpoint receives per epoch |
| `window_epochs` | 3 | epochs over which the statistical bound is evaluated |
| `audit_fraction` | 0.6 | share of the epoch over which canaries are spread |
| `ema_alpha` | 0.2 | weight of the newest epoch in the score EMA |
| `probation_epochs` | 6 | epochs an endpoint scores zero after a verification failure |
| `state_dir` | `~/.verathos/validator` | scores, the canary secret, the auditor database |
| `dry_run` | false | compute weights without setting them |
| `public_host`, `port` | -, 8091 | public IP and port of the validator's service; needed by the subnet owner and the auditor |
| `catalog_mirrors` | [] | extra catalog sources, URL templates with `{digest}`, tried after the owner's axon |
| `catalog_dir` | `~/.verathos/catalog` | local catalog store; the owner's validator serves catalogs from it |

## 4. Run

```bash
.venv/bin/verathos-validator --config validator.json
```

Per epoch (one subnet tempo) the validator:
1. uses the catalog the subnet owner commits (shipped copy, local store, owner's axon, mirrors; any copy is
   checked against the committed digest). The validator checks the commitment every minute in the background and
   applies a new audit policy at once; models and tolerances apply from the next epoch;
2. reads axons and commitments and accepts each miner's manifest if it is signed by the hotkey, matches the
   committed digest, lists its endpoints on the hotkey's axon and was built for the current catalog (or, for one
   tempo after a change, the previous one);
3. challenges each endpoint to sign a nonce with the listing hotkey;
4. derives the epoch's canaries from its secret, computes the references, sends each endpoint its subset spread over
   the first `audit_fraction` of the epoch, and checks every signed receipt;
5. scores (`candidate` models are checked but weighted zero, `retired` models are skipped), updates the EMA and
   probation, and sets weights on `mechid`.

Nothing a miner returns can stop the epoch: replies are size- and time-bounded and every malformed reply is
recorded against that endpoint. Chain errors are retried every block. `--once` runs one epoch immediately and exits. Keep `state_dir` across restarts: it holds the canary secret and the
score history.

## Capacity audit

Every validator derives the audit windows from chain data, like the miners, and holds its canaries for an
endpoint while that endpoint is inside an audit hold. In `score_gate` mode it fetches the auditor's verdicts for
the epoch from the auditor's axon (`GET /capacity/v1/verdicts`), checks they are signed by the auditor hotkey, and
scores the gated endpoints zero with probation. Without a valid feed it gates nothing that epoch. In `score_gate`
mode every validator also scores zero the open endpoints the audit cannot cover (not CUDA, or a GPU without a
class in the catalog).

If the validator's hotkey is the auditor, it also:
- serves the ingest (`/capacity/v1/receipt`, `/capacity/v1/proof`) and the verdicts on `port` and points its
  axon at `public_host:port`, where miners and validators find it;
- follows every block, drafts audit windows, times receipts and verifies proofs;
- in `score_gate` mode, gates endpoints on its own evidence.

The audit database is `state_dir/capacity.sqlite`. Keep the auditor running continuously: windows it did not
follow are void, slots still open when it restarts are neutral, and an outage voids evidence rather than counting
misses. After a restart it replays the recent blocks so the slots of running windows stay busy, as they are on the
miners.
