# Validating

A validator discovers miners from chain state, sends every endpoint a few canary requests per epoch, compares the
answers with its own reference computation, and sets weights on mechanism 1. How answers are compared and scored
is in `SCORING.md`.

Every validator takes part in the throughput audit (`THROUGHPUT_AUDIT.md`). The subnet owner's validator is the
auditor and runs the tests; every other validator applies the auditor's signed verdicts. Each validator finds its
role at startup from the chain.

## Requirements

- Linux, Python 3.12, `git`.
- A hotkey registered on the subnet with a validator permit.
- One reference engine per catalog model. The reference is the model run in fp32; CPU is enough because each
  canary is computed once per epoch and reused for every endpoint, so the reference work does not grow with the
  number of miners: 16 canaries per model and epoch (`canary_pool_size`). They are computed within 30% of the
  canaries' time, about 6.5 minutes per epoch, the models side by side; a slower reference computes fewer of them
  (each endpoint then gets fewer canaries) and never delays the epoch. Measured on a rented server CPU: Kev-4B 3.4 s
  and Kev-9B 6.6 s per request; on 12 cores shared with other work Kev-9B took 16-29 s, which still computes 16
  canaries alone and 8 next to Kev-4B. The fp32 weights alone take about 16 GB and 36 GB of RAM. A GPU in fp32 is
  faster. Rechecking the auditor's evidence on the references takes at most a tenth of the epoch; what is not
  rechecked by then is taken on the auditor's word, as when a reference is down.
- The auditor additionally needs Rust (rustup) and `build-essential` for the load generator, a public IP and one
  open TCP port with an uplink for the network's test traffic, and fast recomputes: 2 answers per endpoint and
  test, within 15 minutes. A bf16 screen per model on a GPU does that (on an A100, Kev-4B recomputes about 6
  answers per second in bf16 and 2 in fp32; the bf16 weights of Kev-4B, Kev-9B and Clef-flash take about 45 GB),
  with the fp32 references confirming what it flags: at 6 per second every endpoint is checked at least once per
  test up to about 5,000 endpoints of the model. Beyond what fits, the checks run every endpoint's first before any
  second, in a random order (`THROUGHPUT_AUDIT.md`).
- A public IP of its own: behind most NATs a machine cannot reach other machines that share its public IP through
  that IP, so a validator next to miners on one NAT cannot send them canaries.

## 1. Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/<org>/verathos && cd verathos
uv venv -p 3.12 .venv
VIRTUAL_ENV=$PWD/.venv uv pip install -e ".[server,chain]"
```

The auditor also builds the load generator:

```bash
curl -sSf https://sh.rustup.rs | sh -s -- -y
.venv/bin/verathos-build-loadgen
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
Clef models use `verathos-engine` in float32 (see `MINING.md` for the install). `clef-flash` is a `candidate`:
its endpoints are checked but not scored, which lets the network measure it before the owner activates it; it
needs about 36 GB of RAM in fp32.

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
| `references` | | model id -> the model's fp32 reference engine: a URL, or `{"url", "api_key"}` for a remote engine behind https (the Kev server takes its key from `KEV_API_KEY`). Every `active` model needs one: the validator does not start without, and stops when a new catalog activates a model it has none for; `candidate` models without one are not audited by this validator |
| `audit_screens` | {} | auditor only: model id -> a bf16 engine (same forms) that screens the sampled test answers; an answer it flags counts as wrong only once the fp32 reference confirms it |
| `state_dir` | `~/.verathos/validator` | scores, the canary secret, the auditor database |
| `dry_run` | false | compute weights without setting them |
| `public_host`, `port` | -, 8091 | public IP and port of the validator's service; needed by the subnet owner and the auditor |
| `catalog_mirrors` | [] | extra catalog sources, URL templates with `{digest}`, tried after the owner's axon |
| `catalog_dir` | `~/.verathos/catalog` | local catalog store; the owner's validator serves catalogs from it |

Scoring settings (canaries, statistical window, EMA, probation) come from the catalog, so every validator applies
the same ones.

## 4. Run

```bash
.venv/bin/verathos-validator --config validator.json
```

Run it under a supervisor (systemd, pm2) that restarts it.

Per epoch (`epoch_blocks` from the catalog, default 180 blocks) the validator:
1. uses the catalog the subnet owner commits (shipped copy, local store, owner's axon, mirrors; any copy is
   checked against the committed digest). The validator checks the commitment every minute in the background and
   applies a new audit policy at once; each epoch is scored under the catalog committed at its first block, so
   models, tolerances and the epoch length change for every validator at the same epoch;
2. reads axons and commitments and accepts each miner's manifest if it is signed by the hotkey, matches the
   committed digest, lists its endpoints on the hotkey's axon and was built for the current catalog (or, for one
   epoch after a change, the previous one);
3. challenges each endpoint to sign a nonce with the listing hotkey;
4. derives the epoch's canaries from its secret, computes the references (within 30% of the canaries' time), sends
   each endpoint its subset of the canaries computed, spread over the rest of the first `canary_spread_fraction` of
   the epoch (holding them while a throughput test runs), and checks every signed receipt;
5. reads the auditor's verdicts for the epoch: each endpoint's proven capacity, what users got from it (readiness
   and latency), each model's user demand, and any proven wrong answers;
6. scores (`candidate` models are checked but weighted zero, `retired` models are skipped), updates the EMA and
   probation, and sets weights on `mechid` (retried once per block, up to 10 times).

Nothing a miner returns can stop the epoch: replies are size- and time-bounded and every malformed reply is
recorded against that endpoint. Chain errors are retried every block, and an epoch that fails part way leaves no
trace in the score history for its retry. `--once` runs one epoch immediately and exits. Keep `state_dir` across
restarts: it holds the canary secret, the score history (including the last epoch scored, so a restart does not
score an epoch twice) and the catalog followed (so manifests built for the previous catalog stay accepted for the
rest of their epoch).

## Throughput audit

Every validator follows the test schedule from chain data and holds its canaries while a test's traffic runs. It
fetches the auditor's verdicts for the epoch from the auditor's axon (`GET /v1/throughput/<epoch>`), checks they are
signed by the auditor hotkey, and scores each endpoint by the capacity credited there for the model it serves now.
It applies a wrong answer proven in a test only after checking the miner's receipt in the evidence and, if it runs
a reference engine for the model, recomputing the answer (if that engine does not answer, it applies the auditor's
verdict). User requests are not published: for a wrong answer to the gateway it checks that the miner's receipt was
issued to a catalog gateway and takes the auditor's word for the wrongness, as it does for capacity. If the verdicts stay unavailable it uses the last ones it holds while they still cover a
recent test; without any, no endpoint has proven capacity that epoch. In `observe` mode it counts declared
capacities, up to the model's `first_test_rps`, and fetches nothing.

Validators that run a service (the subnet owner's and the auditor's) also serve each hotkey's status for its miners
at `GET /v1/status/<hotkey>`: per endpoint the state in the last epoch (scored, qualifying, candidate, failed,
invalid, probation), its checks, declared and proven capacity and scores, and the auditor's recent test results for
the hotkey.

If the validator's hotkey is the auditor, it also:
- serves the verdicts on `port` and points its axon at `public_host:port`, where miners and validators find it;
- re-discovers miners every 2 minutes, so new endpoints join the next test;
- runs every scheduled test with the load generator, checks the sampled answers on its reference engines, and
  confirms tests once their selection block is final;
- serves the subnet's gateway its routes (`GET /v1/routes`) and takes its sampled exchanges
  (`POST /v1/gateway/samples`, signed by a catalog gateway), which it checks between tests (`GATEWAY.md`).

The auditor database is `state_dir/throughput.sqlite`. Keep the auditor running continuously: a test it misses is
not run, and endpoints keep the credit of their last test for `credit_ttl_epochs`. A test running when it restarts
is void.
