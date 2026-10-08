# Validating

A validator sets weights once per epoch, on the mechanism the subnet owner's catalog names (`SUBNET_ADMIN.md`). It
runs in one of two modes (`mode` in `validator.json`; `--light` or `--full` for the installer).

## Two modes

| | Light (the default) | Full |
|---|---|---|
| What it does | sets the weights the subnet owner's validator (the auditor) publishes each epoch, once their signature verifies | discovers miners, sends every endpoint canary requests, compares the answers with its own float32 references, applies the auditor's signed throughput verdicts, scores, sets weights |
| Requests to miners | none | identity challenges and canaries to every endpoint |
| Hardware | a small Linux machine, no GPU | a float32 reference engine per active model, in RAM or on GPUs: all resident (Kev-4B, Kev-9B and Clef-flash: about 103 GiB of RAM) or one at a time (about 42 GiB, for the largest) |
| Stake | a validator permit | a validator permit and the catalog's `validator_min_stake` (10,000), or miners refuse it |
| Download | the code | the code, the engines and the model weights (tens of GB) |

Light validators follow the auditor's results; full validators check the answers independently. The subnet owner's
validator is the auditor and always runs full.

A full validator discovers miners from chain state, sends every endpoint a few canary requests per epoch, compares
the answers with its own reference computation, and sets weights. How answers are compared and scored is in
`SCORING.md`. Every full validator takes part in the throughput audit (`THROUGHPUT_AUDIT.md`): the auditor runs the
tests; every other full validator applies the auditor's signed verdicts. Each validator finds its role at startup
from the chain.

## Requirements

Both modes:
- Linux (the one-command install: Ubuntu 22.04 or 24.04), Python 3.12, `git`.
- A hotkey registered on the subnet with a validator permit, which setting weights takes.

A light validator needs nothing more: no GPU, no reference engine, no stake beyond the permit and no public IP. It
reads the chain and the auditor's `GET /v1/network`, and keeps its clock within 5 minutes of the auditor's (NTP).

A full validator also needs:
- The stake miners require of a validator's hotkey: at least the catalog's
  `scoring.validator_min_stake` of stake (10,000 in the shipped catalog), counted as the metagraph's `S`: alpha stake
  plus TAO stake weighted as for validator permits. Miners refuse any other validator; one below logs a CRITICAL line
  at its start and every epoch, and sends no requests and sets no weights while it stays below. The auditor (the
  subnet owner's validator) is answered whatever its stake. Each validator gets at most the catalog's
  `scoring.validator_requests_per_epoch` requests per endpoint and epoch (by default 4 x `canary_pool_size`, 64),
  more than three times what an epoch's canaries and identity challenges take; a refusal for it in the log is a
  fault of the validator.
- One reference engine per active catalog model, and optionally per candidate model to check its endpoints too. The
  reference is the model run in fp32; CPU is enough because each canary is computed once per epoch and reused for every
  endpoint, so the reference work does not grow with the number of miners: 16 canaries per model and epoch
  (`canary_pool_size`). They are computed within 30% of the canaries' time, about 6.5 minutes per epoch: resident
  references side by side (a candidate's reference on the same CPU shares it with the active ones), or one at a time
  ([References: resident or one at a time](#references-resident-or-one-at-a-time)); a slower reference computes fewer
  of them (each endpoint then gets fewer canaries) and never delays the epoch. Measured on a rented server CPU: Kev-4B
  3.4 s and Kev-9B 6.6 s per request; on 12 cores shared with other work Kev-9B took 16-29 s, which still computes 16
  canaries alone and 8 next to Kev-4B. Clef-flash in fp32 on a CPU is slower: 20-32 s per request of the standard
  mix on 24 cores shared with Kev-9B's reference (Kev-9B 12-25 s there), so on a CPU it computes only part of its
  pool within the time; give its reference a GPU where one has room (about 47 GiB). The fp32 weights alone take about
  16 GB and 36 GB of RAM for the Kev models. A GPU in fp32 is faster. Rechecking the auditor's evidence on the references takes at most a tenth of the epoch; what is not rechecked
  by then is taken on the auditor's word, as when a reference is down.
- The auditor additionally needs Rust (rustup) and `build-essential` for the load generator, a public IP and one
  open TCP port with an uplink for the network's test traffic, and fast recomputes: 2 answers per endpoint and
  test, within 15 minutes. A bf16 screen per model on a GPU does that (on an A100, Kev-4B recomputes about 6
  answers per second in bf16 and 2 in fp32; the bf16 weights of Kev-4B, Kev-9B and Clef-flash take about 45 GB),
  with the fp32 references confirming what it flags: at 6 per second every endpoint is checked at least once per
  test up to about 5,000 endpoints of the model. Beyond what fits, the checks run every endpoint's first before any
  second, in a random order (`THROUGHPUT_AUDIT.md`).
- A public IP of its own: behind most NATs a machine cannot reach other machines that share its public IP through
  that IP, so a validator next to miners on one NAT cannot send them canaries.

## References: resident or one at a time

A full validator runs its references in one of two ways (`reference_mode` in `validator.json`; `--references` for the
installer, which chooses by the memory it finds):

- **resident** (`resident`): every reference runs all the time, each a service; the memory holds them all. Each epoch
  they compute their canaries side by side, and every model is checked every epoch.
- **one at a time** (`sequential`): each epoch the validator starts a model's reference itself (from the weights the
  installer downloaded), computes that model's canaries, stops it, then starts the next model's. The memory needs only
  the largest reference. Starts and stops count in the same reference time as resident references compute in (30% of
  the canaries' time, about 6.5 minutes of a 180-block epoch), so the canaries keep their time and spread.

| References | RAM (or memory on one GPU) |
|---|---|
| resident: Kev-4B, Kev-9B and Clef-flash | about 103 GiB (121 GiB) |
| one at a time | about 42 GiB (48 GiB), for Kev-9B |

These are the free memory the installer asks for; the machine also holds the system and the validator itself. The
installer keeps the references resident while the active models' all fit in the free memory, and runs them one at a
time otherwise, or when asked to with `--references sequential`. The subnet owner's validator (the auditor) checks
test answers at any time and every model each epoch: its references are always resident.

The validator learns how long each model's reference takes to start, to compute a canary and to stop
(`state_dir/references.json`, the largest of each model's last 5 measurements) and plans each epoch with them. A model
needs its start, `canaries_per_endpoint` (10) canaries and its stop; on the server CPU above, 34 s of canaries for
Kev-4B and 66 s for Kev-9B, plus the time to load each model's weights. A model that fits computes its whole pool of
16 canaries while the time the models after it need remains.

**Coverage.** When the reference time does not hold every model, the validator checks those that fit, in an order it
draws each epoch from its canary secret, so miners cannot tell which models it checks. With `k` models to check, a
model that has not run for `k` epochs in a row is overdue: overdue models come first, the longest overdue first; the
others follow in a random order in which a model weighs one plus the epochs it has waited, so a model skipped gets
priority the next time. Models are taken in that order while their time fits in what the ones before leave; the first
model, when overdue, runs even if its time alone exceeds the reference time, and computes the canaries it can. The
longest overdue model runs every epoch, so every model is checked at least once every `2k` epochs (4 epochs for two
models, 6 for three), whatever its time, and more often when several fit in an epoch. Canaries for the models not
checked are not sent, and their endpoints get no identity challenge from this validator that epoch (unless the model
was planned to run and the time ran out).

**Results of the models not checked.** For each model it does not check in an epoch (not drawn, or its reference did not
start or answer in time), the validator takes the results the source publishes: the auditor's, or those of
`source_hotkey`. It reads the source's `GET /v1/network` for the epoch as light validators do (signed by the source
hotkey for this netuid within 5 minutes of its clock, for the catalog's epoch length and this epoch), until 2 minutes
before the epoch ends. Each endpoint of such a model that the source lists enters this validator's scoring by the state
the source gave it:

| Source's state | In this validator's scoring |
|---|---|
| `failed` | failed: score zero, and every endpoint of its hotkey starts probation with its average reset, as for this validator's own failures |
| `probation` | score zero for the epoch |
| `not_audited`, `invalid` | its average is held (an `invalid` hotkey's proven wrong answers come with the auditor's verdicts, as for every model) |
| any other | the score the source gave it: its share of its model's emission in the epoch |

That score enters the endpoint's average, probation and the weights exactly as one the validator computes, and the
validator sets one weight vector from both: checking every model, it scores as with resident references; checking
none, it sets the source's weights, exactly so once its averages hold the source's results alone (an average moves
`ema_alpha`, 0.2, of the way each epoch). Without a document that verifies by then, the endpoints of the models not
checked keep their averages, as endpoints without reference answers (at most 3 epochs in a row). An endpoint of such a
model that the source does not list keeps its average too.

Each epoch logs one line with the models checked and those taken from the source, with why each was not checked, as a
warning when one was not:

```
epoch 45406: references one at a time: checked kev-4b, clef-flash; taken from the subnet owner's validator 5D...: kev-9b (not drawn)
```

and, without the source's results,
`...; not checked, their averages kept (no results of the source: <why>): kev-9b (not drawn)`. Each reference's output
is in `state_dir/references/<model>.log` (its latest start). A reference started for a proven wrong answer to recheck
runs only if its start fits in the recheck's time; else the auditor's word stands, as when a reference is down.

## Quick start (one command)

Copy the hotkey and the coldkey's public key to the validator machine:
`~/.bittensor/wallets/<wallet>/hotkeys/<hotkey>` and `~/.bittensor/wallets/<wallet>/coldkeypub.txt`. The installer
never creates, imports or reads a coldkey. Then, on Ubuntu 22.04 or 24.04, as root or a user with sudo:

```bash
# light (the default for a hotkey that is not the subnet owner's)
curl -LsSf https://raw.githubusercontent.com/<org>/verathos/main/scripts/install.sh | bash -s -- validator

# full
curl -LsSf https://raw.githubusercontent.com/<org>/verathos/main/scripts/install.sh | bash -s -- validator --full
```

It asks for the wallet/hotkey (default: the hotkey on the machine, or the one set up before) and the mode (default:
the one set up before, else light), shows its plan with the mode and what it needs, and asks to go ahead. Without
questions:

```bash
curl -LsSf https://raw.githubusercontent.com/<org>/verathos/main/scripts/install.sh | bash -s -- validator --yes \
    --wallet validator --hotkey default
```

| Option | Meaning |
|---|---|
| `--wallet`, `--hotkey` | the wallet and hotkey |
| `--light`, `--full` | the mode; without either, the one set up before (a validator set up before modes existed is full), else light; the subnet owner's hotkey is always full. The options for references below are for full validators |
| `--device` | where the references run: `auto` (on GPUs while they fit, smallest model first, the rest in RAM), `cpu` or `gpu` |
| `--references` | `resident` (every reference a service) or `sequential` (one at a time, started by the validator); default: as set up before, else resident while the active models' references all fit in the free memory ([References](#references-resident-or-one-at-a-time)) |
| `--candidates` | also run references for the catalog's candidate models, in the memory the active ones leave; a later run keeps them, `--no-candidates` removes them |
| `--dry-run` | compute weights without setting them; `--no-dry-run` sets them again |
| `--network`, `--netuid` | `finney` and 96 by default; `test` (netuid 405), or a subtensor websocket URL |
| `--no-service`, `--source`, `--ref`, `--dir` | as for miners (`MINING.md`) |

What it does (a full validator; light, see below):
1. installs `git`, `curl` and, in a container without systemd, `supervisor`; then uv and Python 3.12;
2. puts the code in `~/verathos` with its Python environment, and `verathos-validator` on the `PATH`;
3. checks the hotkey, its registration and validator permit, and reads the catalog the subnet owner commits;
4. places a float32 reference engine for every `active` model in the free memory, smallest first: on the GPU with
   the most free memory while one has room, otherwise in RAM. With `--candidates` it then places the `candidate`
   models (checked, scored zero) in what is left, and leaves out one that does not fit. When the active models'
   references do not all fit at once, it runs them one at a time, each placed alone in the free memory
   ([References](#references-resident-or-one-at-a-time)). The plan says which, with the memory (`all references
   resident (RAM about 103 GiB)` or `references one at a time (RAM for the largest: about 42 GiB)`), and names every
   model's place and memory;

   | Model | Status | float32 reference in RAM | on a GPU |
   |---|---|---|---|
   | Kev-4B | active | about 21 GiB | about 27 GiB |
   | Kev-9B | active | about 42 GiB | about 48 GiB |
   | Clef-flash | active | about 41 GiB | about 47 GiB |

5. installs each model's engine without the fused kernels (they change float32 results): the Kev engine at
   `engine.revision` for Kev models, `verathos-engine` from this code for Clef models, which runs the model
   repository's own inference code; then downloads the weights at `hf_revision` and the base model at
   `base_revision`; the references run offline from that copy;
6. writes `~/.verathos/validator.json` ([3. Configure](#3-configure)) with the `references` (one at a time: with the
   command that serves each), `reference_mode` and `state_dir` (`~/.verathos/validator`: the canary secret, the score
   history and the last status; keep it);
7. starts the references (`verathos-reference-kev-4b`, ...) on `127.0.0.1` and the validator
   (`verathos-validator`) as services, waits for the references to load, and shows the validator's start. One at a
   time, only the validator is a service: it starts each reference itself when it checks the model, on the same port.

Code, engines, weights and services live where they do for miners (`MINING.md`).

In light mode it skips steps 4 and 5, writes `mode: light` and no references, and starts only the validator. On a
machine that ran a full validator, it removes the reference services; their engines and weights stay until uninstall.
Running it again with `--yes` keeps the mode; `--full` or `--light` changes it.

### Check that it works

A light validator's setup ends with its first line, and each epoch, once the auditor has published its results
(from about 60% into the epoch), it logs the weights it set:

```
light mode: setting the weights the subnet owner's validator publishes each epoch, once its document verifies; no requests to miners
epoch 45406: set the weights of the subnet owner's validator 5D...: 41 UIDs, top 12: 0.081, 3: 0.074, 27: 0.069, 8: 0.052, 30: 0.047
```

A full validator's setup ends with its first lines:

```
throughput tests are run by 5D...; applying its verdicts
epoch 45406: 3 miners, 3 open endpoints
```

At the end of each epoch it logs the emission shares, the miner share, a line per endpoint checked and the weights:

```
  emission shares: clef-flash 33.3%, kev-4b 33.3%, kev-9b 33.3%
  miner share 0.590000: the miners' weights sum to it, the subnet owner's UID gets the rest
  5CzN.../gpu0 epoch_verdict=insufficient answered=7/7 identity=True mean_abs=0.0013 proven_rps=7.6 users=1.00 score=0.191
dry run (mechanism 0 configured dry): weights {0: 0.41, 1: 0.3363, 12: 0.2537}
```

A validator restarted in an epoch it already scored (or, light, whose weights it set) starts with the next one.

| | systemd | supervisord (containers) |
|---|---|---|
| status | `systemctl status verathos-validator` (full, resident: and `verathos-reference-kev-4b verathos-reference-kev-9b`) | `supervisorctl status` |
| validator log | `journalctl -u verathos-validator -f` | `tail -f /var/log/verathos/verathos-validator.log` |
| reference log | `journalctl -u verathos-reference-kev-4b -f`; one at a time: `~/.verathos/validator/references/kev-4b.log` | `tail -f /var/log/verathos/verathos-reference-kev-4b.log`; one at a time: as with systemd |
| restart after editing `validator.json` | `systemctl restart verathos-validator` | `supervisorctl restart verathos-validator` |

### Update

```bash
bash ~/verathos/scripts/install.sh validator --yes
```

updates the code, and the engines and weights when the catalog pins new revisions; it adds a reference when the
catalog activates a model. References keep running unless their revision or settings changed (or, for
`verathos-engine`, its code); the validator restarts.

It keeps the mode, and references one at a time stay so (`--references resident` makes them resident again). Resident
references that no longer fit in the memory, as when the catalog activates another model, run one at a time from then
on. A light validator has only its code to update.

### Uninstall

```bash
bash ~/verathos/scripts/install.sh uninstall
```

stops and removes the services, the code, the engines and the weights. The configuration and state in `~/.verathos`
(with the canary secret and the score history) stay; `--purge` deletes them too. Wallets are never touched.

### Troubleshooting

| Problem | What to do |
|---|---|
| `UID ... has no validator permit yet` | the chain takes weights, and miners answer, only from validators with a permit, which follows the stake |
| `epoch ...: no weights set: no results of the source verified by block ...` | light: the reason follows. The auditor's axon unreachable, or its results not out yet, cost that epoch only, and the weights set before stay on chain; `not signed within 300 s of this machine's clock`: sync the clock (NTP); `it lists no weights` or `it is not signed`: the auditor runs an older release |
| `this is the subnet owner's hotkey: its validator is the auditor ... runs in full mode` | the owner's validator is never light: `install.sh validator --full` |
| `miners refuse validators below validator_min_stake` | add stake to the hotkey until its `S` reaches the catalog's minimum; miners see it within 10 minutes |
| `the kev-9b reference needs about 42 GiB ...; not enough is free` | free memory or add some; `--device` chooses between GPU and RAM |
| `<model> (candidate ...): no reference, not enough memory left` | a candidate's reference is optional: free memory and run the command again, or leave it to other validators |
| `... of 16 canaries without a reference answer this epoch` | the reference's log: it is down, or too slow to compute the canaries within the epoch's reference time ([Requirements](#requirements)) |
| `references one at a time: checked ...; taken from ...` | the reference time held only the models checked; every model is checked at least once every `2k` epochs ([Coverage](#references-resident-or-one-at-a-time)). Resident references (more memory) or a faster machine check more each epoch |
| `the kev-9b reference did not start: ...` | its log, `~/.verathos/validator/references/kev-9b.log`; `another process holds its port`: a reference service of a resident setup still runs, run the installer again |
| `the kev-9b reference needs about 42 GiB of RAM, even one at a time` | the largest reference needs that much free memory: free or add some, or `--device` for a GPU |
| `no reference engine configured for the active models` | the catalog activated a model: run the command again |
| `the subnet's catalog is not available` | the subnet owner's axon serves it; add a mirror to `catalog_mirrors` in `validator.json` |
| `this is the subnet owner's hotkey` | the owner's validator also serves the catalog and runs the throughput audit: set `public_host` and `port`, and build the load generator ([1. Install](#1-install)) |
| `curl: command not found`, `coldkeypub.txt not found`, no services after a container restart, `local changes in ... stop the update`, slow downloads | as for miners (`MINING.md`) |

## Manual installation

### 1. Install

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

A light validator needs nothing else: go on with [3. Configure](#3-configure).

### 2. Start the reference engines (full)

The reference defines the right answer, so it runs exactly what the catalog pins: the engine at `engine.revision`
and the weights at `hf_revision`, in float32.

```bash
git clone https://github.com/jaredpalmer/kev && cd kev && git checkout <engine.revision>
uv sync --extra serve -p 3.12
.venv/bin/huggingface-cli download jaredpalmer/kev-4b --revision <hf_revision> --local-dir models/kev-4b
CUDA_VISIBLE_DEVICES="" KEV_DTYPE=fp32 .venv/bin/python -m kev.serve --run models/kev-4b --port 8101
```

Run them on CPU without `flash-linear-attention` installed.
Clef models use `verathos-engine` in float32, which runs the model repository's own inference code one request at a
time; install it as for miners (`MINING.md`) but without `flash-linear-attention`. `clef-flash` is active: its
reference needs about 41 GiB of RAM (47 GiB on a GPU).

```bash
/opt/clef/bin/verathos-engine clef --repo <hf_repo> --revision <hf_revision> --port 8103 --device cpu --dtype float32
```

References one at a time are not started by hand: give each its command in `references` and set `reference_mode` to
`sequential` ([3. Configure](#3-configure)); the validator runs the command from `directory` with `environment` over
its own, and the engine must serve on the URL's port on `127.0.0.1`:

```json
"reference_mode": "sequential",
"references": {"kev-4b": {"url": "http://127.0.0.1:8101", "directory": "/path/to/kev",
                          "command": [".venv/bin/python", "-m", "kev.serve", "--run", "jaredpalmer/kev-4b@<hf_revision>", "--port", "8101"],
                          "environment": {"CUDA_VISIBLE_DEVICES": "", "KEV_DTYPE": "fp32", "HF_HUB_OFFLINE": "1"}}}
```

Each epoch (and the auditor before each test and round of checks) asks every reference and screen what it serves
(`GET /v1/models`): one that names other weights than the catalog's `hf_revision` is not used, and its model's
endpoints are not audited. `verathos-engine` names its revision, and so does the Kev server run as
`--run <hf_repo>@<hf_revision>` (as `verathos-validator init` sets it up); run from a local directory as above it
names none, so download exactly the pinned revision. When the catalog pins other weights or another engine for a
model in place, the validator logs it as critical: restart the reference on them within `revision_transition_epochs`
(3), the epochs during which the model's answers are not checked.

### 3. Configure

`validator.json` of a light validator:

```json
{
  "network": "finney",
  "netuid": 96,
  "wallet_name": "<wallet>",
  "hotkey_name": "<hotkey>",
  "mode": "light"
}
```

of a full validator:

```json
{
  "network": "finney",
  "netuid": 96,
  "wallet_name": "<wallet>",
  "hotkey_name": "<hotkey>",
  "mode": "full",
  "references": {"kev-4b": "http://127.0.0.1:8101", "kev-9b": "http://127.0.0.1:8102"}
}
```

| Field | Default | Meaning |
|---|---|---|
| `mode` | `full` | `light` or `full` ([Two modes](#two-modes)); the subnet owner's validator does not start in light mode |
| `source_hotkey` | the subnet owner's | light: the validator whose published weights to set; full with references one at a time: the validator whose results the models not checked take. Another validator must hold a validator permit and serve its results (a full validator with `public_host`) |
| `mechid` | the catalog's | mechanism the weights are set on and manifests are accepted for, instead of the catalog's `mechanism`; leave it unset to follow the catalog |
| `references` | {} | full: model id -> the model's fp32 reference engine: a URL, or `{"url", "api_key"}` for a remote engine behind https (the Kev server takes its key from `KEV_API_KEY`), or for references one at a time `{"url", "command", "directory", "environment"}`: the command that serves it at the URL on `127.0.0.1`, which the validator runs when it checks the model (the installer writes them; a reference without a command is taken as running). Every `active` model needs one: the validator does not start without, and stops when a new catalog activates a model it has none for; `candidate` models without one are not audited by this validator. Light validators ignore it |
| `reference_mode` | `resident` | full: `resident` or `sequential` (one at a time, [References](#references-resident-or-one-at-a-time)); the subnet owner's validator does not start one at a time |
| `audit_screens` | {} | auditor only: model id -> a bf16 engine (same forms) that screens the sampled test answers; an answer it flags counts as wrong only once the fp32 reference confirms it |
| `state_dir` | `~/.verathos/validator` | scores, the canary secret, the auditor database; references one at a time: their times (`references.json`) and logs (`references/`); light: the last epoch whose weights it set |
| `dry_run` | false | compute weights without setting them (light: log the weights it would set) |
| `public_host`, `port` | -, 8091 | public IP and port of the validator's service; needed by the subnet owner and the auditor. A full validator that sets them serves its status and the weights it sets there, for light validators that name it in `source_hotkey` |
| `catalog_mirrors` | [] | extra catalog sources, URL templates with `{digest}`, tried after the owner's axon |
| `catalog_dir` | `~/.verathos/catalog` | local catalog store; the owner's validator serves catalogs from it |

Scoring settings (canaries, statistical window, EMA, probation) come from the catalog, so every validator applies
the same ones.

### 4. Run

```bash
.venv/bin/verathos-validator --config validator.json
```

Run it under a supervisor (systemd, pm2) that restarts it: the validator also exits by itself when the subnet owner
hotkey changes, to start again under the new auditor. At start a full validator waits until its reference engines
answer (at most 15 minutes), so references restarted with it can load their models before an epoch needs them.

A light validator, each epoch (`epoch_blocks` from the catalog, default 180 blocks):
1. waits until the source (the subnet owner's validator, or `source_hotkey`) can have scored the epoch: the source
   publishes its results after its canaries, from `canary_spread_fraction` (0.6) of the epoch on;
2. then every 30 s fetches the source's `GET /v1/network` from the axon the chain lists for the source hotkey, and
   sets the weights in it only if the document is signed by the source hotkey under `verathos/network-status/v1` for
   this netuid, was signed within 5 minutes of the validator's clock (the source signs it again every 30 s), is for
   the current or the previous epoch and for a later epoch than the weights it set last;
3. sets exactly those weights (the subnet owner's UID's included; UIDs it does not list get none) on the mechanism of
   the catalog it follows (or `mechid`), as a full validator sets its own: the subnet's weights version key,
   commit-reveal where the subnet has it, retried once per block up to 10 times; and logs the epoch, the source, the
   number of UIDs and the highest weights;
4. without a document that verifies by the same point of the next epoch, when the next results can come, sets
   nothing for the epoch and logs why, once; the weights it set before stay on chain.

It sends no request to any miner and runs no reference engine. With an auditor of an older release, whose document
lists no weights or is not signed, it sets no weights.

Per epoch a full validator:
1. uses the catalog the subnet owner commits (shipped copy, local store, owner's axon, mirrors; any copy is
   checked against the committed digest). The validator checks the commitment every minute in the background and
   applies a new audit policy at once; each epoch is scored under the catalog committed at its first block, so
   models, tolerances and the epoch length change for every validator at the same epoch;
2. reads axons and commitments and accepts each miner's manifest if it is signed by the hotkey, matches the
   committed digest, lists its endpoints at public addresses (the axon or the miner's other machines, `DISCOVERY.md`)
   and was built for the current catalog (or, for one epoch after a change, the previous one) and for its mechanism.
   While a hotkey's axon serves no acceptable manifest, the one it last served stands in for up to 3 epochs, as long
   as the commitment still names it, so the endpoints on the miner's other machines keep their checks;
3. challenges each endpoint at its URL to sign a nonce with the listing hotkey, while it computes the references;
   a host that is not the hotkey's gets no more than one challenge per epoch;
4. derives the epoch's canaries from its secret, computes the references (within 30% of the canaries' time; one at
   a time, those of the models that fit, [References](#references-resident-or-one-at-a-time)), sends each endpoint
   that passed its challenge its subset of the canaries computed, spread over the rest of the first
   `canary_spread_fraction` of the epoch (holding them while a throughput test runs), and checks every signed
   receipt;
5. reads the auditor's verdicts for the epoch: each endpoint's proven capacity, what users got from it (readiness
   and latency), each model's user demand, and any proven wrong answers;
6. one at a time: takes the source's results for the models it did not check;
7. scores (`candidate` models are checked but weighted zero, `retired` models are skipped), updates the EMA and
   probation, scales the miners' weights to the catalog's miner share for the epoch and gives the subnet owner's UID
   the rest (`SCORING.md`), and sets weights on the catalog's mechanism, or `mechid` (retried once per block, up to
   10 times).

Nothing a miner returns can stop the epoch: replies are size- and time-bounded and every malformed reply is
recorded against that endpoint. Chain errors are retried every block, and an epoch that fails part way leaves no
trace in the score history for its retry. `--once` runs one epoch immediately and exits. Keep `state_dir` across
restarts: it holds the canary secret, the score history (including the last epoch scored, so a restart does not
score an epoch twice) and the catalog followed (so manifests built for the previous catalog stay accepted for the
rest of their epoch).

## Throughput audit

Every full validator follows the test schedule from chain data and holds its canaries while a test's traffic runs. It
fetches the auditor's verdicts for the epoch from the auditor's axon (`GET /v1/throughput/<epoch>`), checks they are
signed by the auditor hotkey, and scores each endpoint by the capacity credited there for the model it serves now.
It applies a wrong answer proven in a test only after checking the miner's receipt in the evidence and, if it runs
a reference engine for the model, recomputing the answer (if that engine does not answer, it applies the auditor's
verdict). User requests are not published: for a wrong answer to the gateway it checks that the miner's receipt was
issued to a catalog gateway and takes the auditor's word for the wrongness, as it does for capacity. If the verdicts stay unavailable it uses the last ones it holds while they still cover a
recent test; without any, no endpoint has proven capacity that epoch. In `observe` mode it counts declared
capacities, up to the model's `first_test_rps`, and fetches nothing.

Validators that run a service (the subnet owner's, which is the auditor, and full validators with `public_host`) also
serve each hotkey's status for its miners at `GET /v1/status/<hotkey>`: the UID and its share of the weights set, per
endpoint the state in the last epoch (scored, qualifying, candidate, drifted, failed, invalid, probation), its
checks, declared and proven capacity, how ready users found it, its latency factor and scores, and the hotkey's
results in the auditor's last 10 tests (at most 64 endpoints each). `GET /v1/network` serves every hotkey's status of
the last epoch at once (without the test results, built once per epoch), `weights`, per UID exactly the weights the
validator set after the epoch (the subnet owner's UID's included), `mechanism`, the mechanism they are set on, and
`miner_share`, the epoch's miner share, for network dashboards and light validators. The validator publishes it
before it sets those weights, and a restarted one serves the last epoch's until its next.

Both documents are signed with the validator's hotkey: the status's fields plus `netuid`, `auditor` (the signing
hotkey) and `issued_ms`, signed as canonical JSON (sorted keys, no whitespace) under the domain `verathos/status/v1`
(a hotkey's status) or `verathos/network-status/v1` (the network's) into `signature` (`protocol/status.py`). Miners
act on the test results in their status only once it verifies against the subnet owner hotkey, and light validators
on the weights only once it verifies against their source; a reader that does not check ignores the four fields.

These routes are public and share the auditor's process and uplink. A hotkey's status is built at most every 30 s
(sooner once an epoch is scored or a test recorded, confirmed or voided) and may be cached as long
(`Cache-Control: public, max-age=30`); the network's is built once per epoch and signed again at most every 30 s, so
its `issued_ms` shows it is served now. Each client address (an IPv4 address,
or an IPv6 /64) gets a burst of 10 status requests, then one every 5 s, and HTTP 429 with `Retry-After` beyond, so
about 24 miner machines polling every two minutes fit behind one address; the service tracks at most 4,096 addresses
at once. Behind a reverse proxy on the same host, the client's address comes from `X-Forwarded-For`, which the
service takes from loopback peers only.

If the validator's hotkey is the auditor, it also:
- serves the verdicts on `port` and points its axon at `public_host:port`, where miners and validators find it;
- re-discovers miners every 2 minutes, so new endpoints join the next test;
- runs every scheduled test with the load generator, checks the sampled answers on its reference engines, and
  confirms tests once their selection block is final;
- serves the subnet's gateway its routes (`GET /v1/routes`) and takes its sampled exchanges
  (`POST /v1/gateway/samples`, signed by a catalog gateway), which it checks between tests (`GATEWAY.md`).

The auditor database is `state_dir/throughput.sqlite`. Keep the auditor running continuously: a test it misses is
not run, and endpoints keep the credit of their last test for `credit_ttl_epochs` (the gateway's routes keep it
until the next confirmed test). A test running when it restarts is void.
