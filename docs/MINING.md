# Mining

A miner serves decision models from the catalog (the model list the subnet owner commits on chain; see
`DISCOVERY.md`), one model per GPU. Only `active` catalog models are scored; `candidate` models can be served and
are checked by validators but score zero. Each GPU runs a decision engine; the miner service in front of them
measures each endpoint's capacity, publishes a signed manifest and answers signed requests.

Validators check every endpoint's answers against a reference computation (see `SCORING.md`), and the throughput
audit (see `THROUGHPUT_AUDIT.md`) checks the capacity each endpoint declares. An endpoint's score is the capacity it
proves, so hardware pays by its throughput per cost. The scores split the catalog's miner share of the mechanism's
miner emission (0.59 in the shipped catalog); with the catalog's usage pool, the hotkeys whose endpoints answered
paying users get more on top, by the paid input tokens they answered; validators set the rest as weight on the subnet
owner's UID, which the chain does not pay out (`SCORING.md`).

## Requirements

- Linux (the one-command install: Ubuntu 22.04 or 24.04), Python 3.12, `git`, and what the engines need (NVIDIA
  drivers for GPU engines).
- One GPU per endpoint. Any GPU that runs the model's engine works; a GPU shared by several endpoints is credited
  once, split between them.
- A public IP address and one open TCP port for the miner service, on every machine. Validators, the auditor and
  the gateways reach each endpoint directly at its machine's IP and port, so the miner service must be reachable there
  (TLS only, with the machine's own certificate, which the manifest pins; requests and receipts are signed). A test
  sends every endpoint its declared rate for about 20 s: each machine's connection must carry its own endpoints' rates.
  The miner's firewall lets only those callers and the hotkey's machines reach the port ([Firewall](#firewall)).
- A hotkey registered on the subnet; every miner machine needs the hotkey and `coldkeypub.txt`, never the coldkey.

## Quick start (one command)

Register the hotkey first ([1. Register](#1-register)) and copy the hotkey and the coldkey's public key to the
miner machine: `~/.bittensor/wallets/<wallet>/hotkeys/<hotkey>` and `~/.bittensor/wallets/<wallet>/coldkeypub.txt`.
The installer never creates, imports or reads a coldkey. Then, on the miner machine, as root or a user with sudo
(as any user with `--no-service`):

```bash
curl -LsSf https://raw.githubusercontent.com/<org>/verathos/main/scripts/install.sh | bash -s -- miner
```

After installing the code it asks a few questions, then shows its plan and asks to go ahead before it sets up
engines and services:

| Question | Default |
|---|---|
| Wallet/hotkey | the hotkey on the machine, or the one set up before |
| Role of this machine | primary, or the one set up before ([One hotkey, several GPUs and machines](#one-hotkey-several-gpus-and-machines)) |
| Public IP address of this machine | detected |
| Public TCP port of the miner | 8091 |
| Model for each kind of GPU | the one that pays best now, measured from the network; or the one named before with `--model` ([Models](#models)) |

To answer on the command line instead, add `--yes` and the options:

```bash
curl -LsSf https://raw.githubusercontent.com/<org>/verathos/main/scripts/install.sh | bash -s -- miner --yes \
    --wallet miner --hotkey default --port 8091 --model kev-4b
```

| Option | Meaning |
|---|---|
| `--wallet`, `--hotkey` | the wallet and hotkey |
| `--public-ip`, `--port` | the address validators reach the miner at |
| `--model` | one model for every GPU, or one per GPU in order (`kev-4b,clef-flash`); `auto` leaves a GPU to the setup ([Models](#models)) |
| `--gpus` | the GPUs to serve on (`0,1`; default all) |
| `--network`, `--netuid` | `finney` and 96 by default; `test` (netuid 405), a `wss://` subtensor node, or a node on this machine (`local`, `ws://127.0.0.1:<port>`) |
| `--no-publish` | try the machine before registering: the miner serves and measures, but writes nothing on chain; once the hotkey is registered, run the command again without it |
| `--worker`, `--name` | this machine serves its GPUs for the hotkey's primary machine; its endpoint ids start with `--name` (default: its IP address) ([One hotkey, several GPUs and machines](#one-hotkey-several-gpus-and-machines)) |
| `--peers` | on the primary: every worker's public `ip:port`, comma-separated; replaces the list (`''` for none) |
| `--no-auto-update` | the miner does not update its code by itself to each new release; `--auto-update` turns that on again ([Update](#update)) |
| `--no-firewall` | the miner's port answers anyone; `--firewall` turns the firewall on again ([Firewall](#firewall)) |
| `--no-service` | no root, no system packages and no services: set everything up in the user's home and print the commands to run under pm2 or another process manager you run; `git`, `ssh-keygen` (`openssh-client`), `curl` and a C compiler (`build-essential`) must be there ([Without root](VALIDATING.md#without-root-pm2-or-another-process-manager)) |
| `--source`, `--ref`, `--dir` | the code from another git URL, git bundle or tarball; a branch, tag or commit (default: the newest release; a tag is installed only signed by a release key, a branch, commit or tarball as given: [Update](#update)); another directory than `~/verathos-decide`, which must be missing, empty or a checkout of this code (another checkout is never changed) |

What it does:
1. installs `git`, `openssh-client` (it checks release signatures), `curl`, `build-essential` (the engine compiles
   its GPU kernels on first use), `nftables` (the firewall) and, in a container without systemd, `supervisor` (with
   `--no-service`, nothing); then uv and Python 3.12 in the user's home;
2. puts the code in `~/verathos-decide` (an install from before in `~/verathos` stays there) with its Python
   environment, and `verathos-miner` and `verathos-validator` on the
   `PATH`;
3. checks the hotkey, its registration, the subnet's catalog and the GPUs;
4. installs each chosen model's engine with `flash-linear-attention` at the version the engine pins (the Kev engine
   at the catalog's `engine.revision` for Kev models, `verathos-engine` from this code for Clef models) and downloads
   the weights at `hf_revision` (and the base model at `base_revision`). The engines run offline from that copy, so
   they serve exactly the pinned revisions;
5. writes the configuration, `~/.verathos/miner.json` ([4. Configure](#4-configure)), and sets up the kernel layer of
   the firewall: a root-owned helper and the sudo rule that lets the miner's user run it ([Firewall](#firewall));
6. starts one engine per GPU (`verathos-engine-gpu0`, ...) on `127.0.0.1` and the miner (`verathos-miner`) as
   services, waits for the engines to load, and shows the capacity measurement (a few minutes) until the miner
   serves.

| Path | |
|---|---|
| `~/verathos-decide` | code and its Python environment |
| `~/.verathos/miner.json` | configuration |
| `~/.verathos/miner/` | capacity measurements and the caps tests set on them, the last published manifest and the endpoints this machine serves, the firewall's state and learned addresses |
| `~/.verathos/engines/`, `~/.verathos/hf/` | engines (one per pinned Kev revision, one for Clef) and model weights |
| `/etc/systemd/system/verathos-*.service` | services; in a container `/etc/supervisor/conf.d/verathos-*.conf`, with logs in `/var/log/verathos/` |
| `/usr/local/sbin/verathos-firewall`, `/etc/sudoers.d/verathos-firewall` | the firewall's helper and its sudo rule |

### Models

| Model | Status | GPU memory (bfloat16) | Engine | Relative speed |
|---|---|---|---|---|
| `kev-4b` | active | about 15 GiB | Kev | 1.0 |
| `kev-9b` | active | about 26 GiB | Kev | 0.92 |
| `clef-flash` | active | about 21 GiB | `verathos-engine` (Clef) | 0.72 |

These are the shipped catalog's models; the installer reads the catalog on chain and lists its models with the memory
each needs. All three are active from the start and share the miners' emission equally while users pay for none of them
(`SCORING.md`). Active models are scored. A candidate, which a catalog can list before activating it, is checked by
validators but scores zero until the subnet owner activates it: serving one helps measure it and earns nothing yet. The
relative speed is the catalog's `relative_speed`: the requests/s of the model's standard mix one GPU serves, relative to
the other models, as the subnet owner measured them on one reference GPU (`SUBNET_ADMIN.md`). How the models compare
differs by GPU: kev-9b serves 0.92 times kev-4b's requests on an H100, and clef-flash 0.72 times (its mix with
images; 0.79 times the same mix without them); on an A100 kev-9b serves 0.68-0.82 times and clef-flash 0.65 times
without images, on an RTX 5090 kev-9b 0.55 times.

Each GPU serves one model: by default the active model that pays best on it now. The setup reads the auditor's
network status (`GET /v1/network` on the subnet owner's axon, built once per epoch) and uses it only if the subnet
owner hotkey signed it for this subnet within two epochs of the machine's clock. From it, per active model:

- its share of the emission: its share of the scores, as the auditor published it for its last epoch
  (`SCORING.md`), times the miner share the auditor published with it;
- the capacity proven by the endpoints serving it: the requests/s credited to its scored endpoints, this machine's
  own left out.

A GPU's rate on a model is its speed times the model's relative speed. Its speed comes from the capacity the miner
measured on it before (what it declares, `~/.verathos/miner/capacity.json`), else from a GPU of the same kind on the
machine, else from the network's median scored endpoint. A model's share splits among its endpoints by proven
capacity, so a GPU of rate `x` on a model of share `s` with proven capacity `C` adds about `s x / (C + x)` of the
emission (availability to users and latency left out). Where the hotkey already holds `H` of that capacity (on its
other machines, or on this machine's GPUs placed before), it adds what the hotkey's part gains,
`s ((H + x) / (C + x) - H / C)`. GPUs whose model is named by hand count first; then each GPU in order takes the
active model that adds most, and the GPUs after it count it, so a machine with several GPUs spreads over the models
where that pays more. For example, with each model at a third of the scores (0.197 of the emission at the shipped
miner share of 0.59) and 300, 120 and 400 requests/s proven on kev-4b, kev-9b and clef-flash, an A100 whose
measurement declares 13.9 requests/s of kev-4b adds about `0.197 x 13.9 / 313.9 = 0.87%` there,
`0.197 x 12.8 / 132.8 = 1.9%` on kev-9b (12.8 = 13.9 x 0.92) and `0.197 x 10.0 / 410.0 = 0.48%` on clef-flash
(10.0 = 13.9 x 0.72): it serves kev-9b.

On an update, a GPU keeps the model it serves unless another pays at least 1.2 times as much: a switch downloads the
model and measures again, and the numbers move with every epoch. When another pays at least 1.1 times as much, the
plan says so and how to switch:

```
the network in the auditor's epoch 45412: kev-4b 19.7% of the emission, 300.0 requests/s proven; kev-9b 19.7% of the emission, 240.0 requests/s proven; clef-flash 19.7% of the emission, 400.0 requests/s proven; the median endpoint 14.6 requests/s at relative speed 1
GPU 0 (NVIDIA A100 80GB PCIe, 80 GiB): kev-4b, endpoint gpu0, engine on 127.0.0.1:8008
GPU 0: kev-4b, as set up before: about 0.87% of the emission (its share 19.7%, 300.0 requests/s proven by other endpoints, about 13.9 on this GPU by its kev-4b measurement). Also fit: kev-9b 0.99%, clef-flash 0.48%. kev-9b would pay about 1.14x as much on GPU 0: --model kev-9b to switch
another choice: --model <id> for every GPU, or one per GPU in order, as in --model kev-9b
```

Without the network's numbers (the auditor's validator unreachable or of an older release, a status not signed as
above, or no endpoint scored yet, as on a new subnet), each GPU keeps the model it served before, else takes the
first active model that fits, and the setup says why in one line.

The setup never chooses a candidate. `--model` names one model for every GPU (`--model kev-9b`), or one per GPU in
the order of their numbers, which mixes models on one machine (`--model kev-4b,clef-flash` on two GPUs); `auto`
leaves a GPU to the setup (`--model kev-9b,auto`). On a terminal the installer also asks once per kind of GPU, with
the setup's choice as the default. A model named by hand stays on later runs, whatever pays more, until
`--model auto`.

### Check that it works

The setup ends once the miner serves:

```
endpoint gpu0 (kev-4b): 14.5 requests/s at concurrency 4 (runs 14.5, 15.0, 15.0, 14.0, 14.0), declares 13.8
serving 1 endpoints on port 8091; throughput tests are accepted from 5D...; status is logged once per epoch
```

and prints the commands for its services:

| | systemd | supervisord (containers) |
|---|---|---|
| status | `systemctl status verathos-miner verathos-engine-gpu0` | `supervisorctl status` |
| miner log | `journalctl -u verathos-miner -f` | `tail -f /var/log/verathos/verathos-miner.log` |
| engine log | `journalctl -u verathos-engine-gpu0 -f` | `tail -f /var/log/verathos/verathos-engine-gpu0.log` |
| restart after editing `miner.json` | `systemctl restart verathos-miner` | `supervisorctl restart verathos-miner` |

The setup prints the machine's TLS pin and the command that checks it from another machine,
`curl -sk --pinnedpubkey 'sha256//<pin, base64>' https://<public ip>:<port>/v1/manifest`, which returns the signed
manifest with each endpoint's declared capacity and each machine's pin (`-k` skips the CA check; the pin is the
check); requests without a permitted caller's signature get `403`. Once the firewall enforces, only callers' machines
get through, and the check times out anywhere else. The setup ends with one line on the firewall
([Firewall](#firewall)). Once per epoch the miner logs what the validators saw ([Reading the log](#reading-the-log)).

### Update

Run the command again; `--yes` keeps every earlier answer:

```bash
bash ~/verathos-decide/scripts/install.sh miner --yes
```

It updates the code from where it was installed, and the engines and weights when the catalog pins new revisions.
A GPU whose model the setup chose moves to another model only if that pays at least 1.2 times as much
([Models](#models)). The miner restarts and reuses its capacity measurement while nothing it depends on changed; an
engine restarts only when its revision or settings changed, or for `verathos-engine` (Clef models) its code, which
the miner then measures again. A release that changes how the miner measures makes it measure once more.

A release is a tag `v<version>` signed by a release key, one that `release/allowed_signers` lists. The installer
checks the tag it installs (`--ref`, by default the newest release) against its own copy of the file and stops at one
that is not signed so (`v... is not signed ...: not installing it`); while its copy lists no key, it installs the
default branch. A branch, a commit (`--ref`) or a tarball (`--source`) it installs as given: checking those is yours.

A miner set up by the installer also updates its code by itself (`auto_update` in `miner.json`; `--no-auto-update`
turns it off, `--auto-update` on again). About every hour it reads the release tags (`v<version>`) of the repository
its code came from. The newest release whose tag is signed by a key that `release/allowed_signers` of the running
code lists is installed at a random time within the next hour, so miners do not all restart at once, and never while
a throughput test is selected or its traffic runs, nor while a measurement runs or waits for a test: the miner checks
out the tagged commit, installs it into its Python environment and exits, and its process manager starts the new
release (systemd, supervisord and pm2 restart a process that exits). The restarted miner reuses its capacity
measurement; its engines keep running. It does not update a checkout with local changes (`self-update: local changes
in ...`), to a tag that is not signed by such a key (`self-update: v... is not signed ...`, logged once per tag) or to
a tag whose code names another version; a failed install puts the previous code back. While the file lists no key (`... lists no release signing key`), it installs nothing: run the installer again.
Engines and services stay as they are: a release that changes them, such as `verathos-engine`'s code, says so, and
running the installer again updates them.

### Uninstall

```bash
bash ~/verathos-decide/scripts/install.sh uninstall
```

stops and removes the services, the code, the engines and the weights, and the firewall's table, helper and sudo
rule. The configuration and state in `~/.verathos` stay for a later install; `--purge` deletes them too (other files in
`~/.verathos` stay). Wallets are never touched.

### Troubleshooting

| Problem | What to do |
|---|---|
| `curl: command not found` | minimal images lack it: `apt-get update && apt-get install -y curl`, then the command again |
| `hotkey ... is not registered on netuid 96` | register it ([1. Register](#1-register)) and run the command again; `--no-publish` tries the machine first |
| `coldkeypub.txt not found` | copy `~/.bittensor/wallets/<wallet>/coldkeypub.txt` from the machine holding the wallet |
| `no NVIDIA GPU found` | install the NVIDIA driver; in a container, start it with the GPUs passed through |
| `the NVIDIA driver supports CUDA 12.x` | the engines need a driver for CUDA 12.8 or newer |
| `port ... is in use` | choose another port with `--port` |
| validators cannot reach the miner | the setup's check, `curl -sk --pinnedpubkey ... https://<public ip>:<port>/v1/manifest` (on a worker `/v1/endpoints`), from another machine must answer while the firewall observes: open the port for inbound TCP (`ufw allow <port>/tcp`, the provider's firewall). Behind port forwarding, as in most rented containers, the outside port must be the same as the port inside: each machine is listed at one port. Once the firewall enforces, see `verathos-miner firewall status` |
| `firewall observing: requests of ... arrive from 10.0.2.2, not a public address` | a proxy or port forwarding in front of the miner replaces the callers' addresses with its own (Docker's userland proxy, slirp in rootless containers): the firewall cannot tell callers apart there and only observes. Forward the port keeping the callers' addresses (Docker's default iptables forwarding does) |
| `firewall: the kernel layer is off (...)` | the miner filters connections in its own process, which needs nothing; for the kernel layer too: `verathos-miner firewall install` once, as a user with sudo, then restart the miner |
| `endpoint ... of peer ... left out` | endpoint ids must be unique across the hotkey's machines: give the worker another `--name` |
| `peer ...: its TLS certificate (pin ...) is not the one it names` | the worker's port is served by another process, or its certificate changed while it ran: restart the worker's miner |
| `status unavailable: ... names no TLS pin for the auditor` | the committed catalog does not name the auditor's certificate yet: the miner serves as usual and reads its status once the subnet owner commits one |
| an engine keeps stopping | its log says why; out of GPU memory: another model, or a GPU free of other work |
| `no endpoint could be measured` | the engine logs: the measurement needs the engines answering |
| no services after a container restart | containers start no services by themselves: run the command again, or `supervisord -c /etc/supervisor/supervisord.conf` |
| `local changes in ... stop the update` | the installer manages `~/verathos-decide`: see `git -C ~/verathos-decide status` and discard the changes |
| `v... is not signed ...: not installing it` | the tag is not signed by a release key this installer holds: install another release (`--ref v<version>`), or get the current installer if the subnet owner changed the key ([Update](#update)) |
| `... is a checkout of other code` | the directory holds another repository, such as an earlier Verathos release: the installer leaves it as it is; keep the default `~/verathos-decide` or choose another `--dir` |
| downloads slow or rate limited | use a Hugging Face token: `export HF_TOKEN=<token>` before the command |

## One hotkey, several GPUs and machines

A hotkey serves up to 64 endpoints, one per GPU. On one machine nothing needs doing: the installer sets up an
endpoint per GPU (`gpu0`, `gpu1`, ...).

On several machines, one is the primary and the others are its workers. Every machine runs the engines for its own
GPUs and `verathos-miner`, which measures its endpoints and serves them at its own public IP and port: each machine
needs that port open for inbound TCP, and the hotkey and `coldkeypub.txt` (never the coldkey). The primary publishes
the hotkey's manifest, listing every endpoint at its own machine's address; a worker writes nothing on chain and
serves what it measured and the pin of its own certificate, signed by the hotkey, at `GET /v1/endpoints`, which the
primary reads every minute over TLS, taking it only from a worker that presented that certificate; the manifest
names every machine's pin.
Validators, the auditor and the gateways send each endpoint's traffic, the throughput test included, straight to its
machine: each machine needs the bandwidth of its own endpoints only, and a machine that is down costs only its own.
The firewall of each machine lets the hotkey's other machines through: a worker the primary at the hotkey's axon
address, so the primary's requests must leave from its public IP address (on a primary whose requests leave from
another, set `"firewall": false` on its workers).

A worker is protected exactly like a single miner: it runs the same miner service, which answers only validators
holding a permit and the catalog's minimum stake in the metagraph (read again every 10 minutes), the auditor, the
catalog's gateways and the auditor's signed test sessions, refuses anyone else with 403 before reading the request,
and applies the same per-caller limits. Engines
listen on 127.0.0.1 only. `GET /v1/endpoints` is read-only and holds nothing the manifest does not publish.

Set up the workers first, then the primary with every worker's address:

```bash
# on each worker
curl -LsSf https://raw.githubusercontent.com/<org>/verathos/main/scripts/install.sh | bash -s -- miner --worker \
    --yes --wallet miner --hotkey default --port 8091
# on the primary
curl -LsSf https://raw.githubusercontent.com/<org>/verathos/main/scripts/install.sh | bash -s -- miner --yes \
    --wallet miner --hotkey default --port 8091 --peers 203.0.113.7:8091,198.51.100.7:8091
```

A worker's endpoint ids start with its name (`--name`, default its IP address, as in `203-0-113-7-gpu0`): ids
must be unique across the hotkey's machines. At its end the worker's setup prints the line for the primary. Running
`install.sh miner --yes --peers ...` again on the primary replaces its list; `peers` in its `miner.json` and a
restart of its miner do the same.

The primary takes a worker's endpoints only as the hotkey signed them, for its catalog, at that worker's address,
and leaves out an endpoint whose id another machine already lists (its log says which). It keeps a worker that does
not answer in the manifest for an epoch, then withdraws its endpoints until it is back. While the primary itself is
down, validators keep using the manifest it last served for up to 3 epochs, so the workers' endpoints keep their
checks. Each machine logs the status of its own endpoints; `GET /v1/status/<hotkey>` on the auditor lists all.

## Firewall

The miner's port answers only the network's callers (the auditor, the gateways, the validators the miner serves) and
the hotkey's own machines. A connection from any other address is dropped at its first packet, before TCP, TLS or any
request is read, so scans and connection or handshake floods from the rest of the internet cost the miner nothing.
Nothing needs configuring.

### Who gets through

- The addresses the callers publish: the auditor's and each gateway's in the catalog (`auditor.egress`, `gateways`),
  each validator's with a permit and the catalog's `validator_min_stake` in its chain commitment (`VALIDATING.md`), and
  those of the miner's own `gateways`.
- The addresses the miner learned: where each request came from whose signature (or test session) verified for one of
  those callers, kept for about a week after the caller's last request from there, while it stays a caller. A
  validator whose requests leave from another address than it publishes keeps working this way.
- This machine's own addresses and the hotkey's other machines: on a primary its `peers`, on a worker the hotkey's
  axon (its primary).
- Loopback.

The miner reads the list from the chain every 10 minutes and when the catalog changes. An address that drops out stays
for two more readings, and when the chain cannot be read the last list stays, across restarts too
(`~/.verathos/miner/firewall.json`; the learned addresses are in `firewall-roster.json`).

### Observe, then enforce

A miner starts observing: it learns and reports, and drops nothing. It enforces once all of these hold, and observes
again as soon as one does not:
- the committed catalog's `firewall` is `enforce`: the subnet owner's switch, `observe` until validators publish their
  addresses widely (`SUBNET_ADMIN.md`);
- the list holds the auditor's address;
- a request authenticated as an allowed caller came from a published address, and none came from an address that is
  not public: a proxy or port forwarding in front of the miner that hides the callers' addresses (Docker's userland
  proxy, slirp in rootless containers) keeps it observing;
- the known addresses cover at least half of the stake of the validators that call miners: those that publish a
  validator identity, or whose requests the miner verified (light validators never call miners).

The log says each change once, and where the firewall stands once per epoch:

```
firewall observing: the catalog's firewall is observe: the subnet owner switches it to enforce once the validators publish their addresses; 23 addresses (19 published, 2 learned, 2 of this hotkey's machines); known addresses cover 100% of the stake of the 12 validators that call miners
firewall enforcing: known addresses cover 100% of the stake of the validators that call miners; 23 addresses (...)
firewall: validators that publish no egress address and have not reached this miner, whose requests it drops: 5F... (UID 12)
```

### Two layers

- The kernel's: the nftables table `inet verathos_miner`, whose chain hooks prerouting at priority -300 (before
  connection tracking and any port forwarding), accepts the allowed addresses on the miner's port and drops the rest
  of that port. Other ports, SSH included, are untouched, and other firewalls (ufw) still apply. Each change replaces
  the whole table in one transaction, and a table that went missing (restarting `nftables.service` flushes every
  table) or was changed is put back within 10 minutes. The miner runs as a normal user: the installer adds a
  root-owned helper, `/usr/local/sbin/verathos-firewall`, and a sudo rule that lets the miner's user run exactly its
  `apply <port>`, `status` and `remove`, nothing else. A miner running as root applies the table itself.
- The miner's own, always: a filter on its listening socket drops connections from other addresses before the TCP
  handshake, which takes no privilege, and the service answers any request from another address 403 before reading
  it. Without root (`--no-service`, pm2) or in a container without `NET_ADMIN`, this is the firewall, and the miner says
  so once at start (`firewall: the kernel layer is off (...); connections are filtered in the miner process`).

The kernel sets no per-address rate limits: a throughput test opens many connections at once from the auditor's
address. Within the service a validator gets one request at once per endpoint, and gateways and tests are bounded by
each endpoint's queue (HTTP 429 and 503 beyond; [5. Run](#5-run)). Keep `tcp_syncookies` on (the kernel's default; the
miner warns at start if it is off). A flood that fills the machine's link is beyond any host's reach: only the
provider can absorb it. An empty IPv6 list drops all IPv6 to the port.

### Checking it, turning it off

| | |
|---|---|
| where it stands | `verathos-miner firewall status`: the mode and why, the addresses, the kernel table |
| the kernel table | `sudo nft list table inet verathos_miner` |
| the kernel layer, set up later | `verathos-miner firewall install` once, as a user with sudo, then restart the miner |
| turn it off | `install.sh miner --yes --no-firewall` (removes the table, the helper and the sudo rule, and writes `"firewall": false`), or `"firewall": false` in `miner.json` and a restart of the miner, which removes its table |
| open the port at once | `sudo nft delete table inet verathos_miner`; an enforcing miner puts it back within 10 minutes, so turn the firewall off as well |

## Manual installation

### 1. Register

```bash
btcli subnet register --netuid 96 --wallet.name <wallet> --wallet.hotkey <hotkey>
```

Copy `~/.bittensor/wallets/<wallet>/coldkeypub.txt` and `~/.bittensor/wallets/<wallet>/hotkeys/<hotkey>` to the
same paths on the miner machine.

### 2. Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/<org>/verathos verathos-decide && cd verathos-decide
uv venv -p 3.12 .venv
VIRTUAL_ENV=$PWD/.venv uv pip install -e ".[server,chain]"
```

### 3. Start one engine per GPU

Which model to serve is yours to choose (the one-command setup chooses as in [Models](#models)). Each active model
has its own share of the emission, split among the endpoints serving it by proven capacity (`SCORING.md`); the shares
follow what users pay for each model through the subnet's gateway, and the miner logs them every epoch. What a
request costs differs by GPU (the larger Kev model costs 1.1x the smaller on an H100 and 1.8x on an RTX 5090, for
the same requests), so compare what your hardware proves on each model with how many others serve it.

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

Clef models run on `verathos-engine`, which loads the encoding and the head from the inference code shipped in the
model repository at the pinned revision. On a GPU in bfloat16 it batches the requests that arrive together and runs
the Qwen3.5 backbone on fused kernels, with CUDA graphs for small passes; Clef-flash needs about 21 GiB of GPU memory
(a 24 GB GPU), and capturing the graphs takes up to a minute after loading (if capturing fails, every pass runs
without graphs, which is slower for small batches). Install it in its own environment with the kernels at the
version the engine pins:

```bash
uv venv -p 3.12 /opt/clef && VIRTUAL_ENV=/opt/clef uv pip install -e ".[clef]"
VIRTUAL_ENV=/opt/clef uv pip install "flash-linear-attention==$(sed -n 's/^FLA_VERSION = "\(.*\)".*/\1/p' src/verathos/engine/qwen35.py)"
CUDA_VISIBLE_DEVICES=2 /opt/clef/bin/verathos-engine clef --repo <hf_repo> --revision <hf_revision> --port 8010
```

Clef-flash on one GPU, requests of about 525 tokens with one or two questions, several requests at once:

| GPU | Requests per second | One request alone |
|---|---|---|
| H100 80 GB | 69 | 20 ms |
| A100 80 GB | 25 | 58 ms |
| RTX 5090 | 18 | 96 ms |

The backbone's matrix multiplications are most of the work, so a GPU's bfloat16 throughput sets these figures; GeForce
cards accumulate bfloat16 in fp32 at half their fp16 rate. Answers from all of them stay within 0.016 of the float32
reference.

A model whose catalog entry lists `image` in its `input_modalities` also gets requests with images, which the gateway
has made canonical (`GATEWAY.md`, Images); in the shipped catalog, Clef-Flash does. The bfloat16 engine runs them in
passes of their own, without CUDA graphs: their images through the release's vision tower, whose features take the
image tokens' places, and the rest as for text, with image tokens counted in a pass's token budget. The tower runs in
float16: in bfloat16, as released, answers with images drift from the float32 reference about twice as far as text
answers on average, in float16 no further. A pass whose float16 features are not all finite (an activation beyond
float16's range; the largest of the images measured reach about a fifth of it) runs its images through the tower again
in bfloat16, as released, and the engine logs it. The float32 engine (`--dtype float32`) runs the release's own code on
them. Either engine gives the release only a request's model, state and questions, and images only as decoded pixels:
nothing fetches a URL or opens a path.

Clef-flash with images, requests of 500 to 580 tokens of text with four or five questions and square images, 64 at
once (requests per second) and one alone:

| Images per request | Image tokens | H100 80 GB | RTX 5090 |
|---|---|---|---|
| one of 256 x 256 px | 67 | 49/s, 60 ms | 17/s, 140 ms |
| four of 256 x 256 px | 265 | 33/s, 60 ms | 11/s, 140 ms |
| one of 512 x 512 px | 259 | 36/s, 58 ms | 12/s, 138 ms |
| four of 512 x 512 px | 1,033 | 15.5/s, 77 ms | 4.9/s, 221 ms |
| one of 1024 x 1024 px | 1,027 | 14/s, 87 ms | 5.0/s, 215 ms |
| four of 1024 x 1024 px | 4,105 | 4.1/s, 241 ms | 1.5/s, 681 ms |

The RTX 5090 ran in an older host (PCIe 3.0 x8, a 2016 Xeon), where the vision tower's many small host-side steps cost
more than its GPU work for a 256 px image: a faster CPU serves small images faster.

An image token takes about twice the GPU time of a text token (1.7 to 2.1 times on the H100 for images of 512 px a side
or more, 1.6 to 1.8 on the RTX 5090; an image of 256 px adds about 2 ms besides on the H100). The vision tower adds 0.9
GB of weights: on the H100 PyTorch's memory peaked at 21.4 GB allocated and 22.8 GB reserved with images (21.2 GB and
22.9 GB with text alone), and at 22.5 GB and 24.0 GB for one request at the 16,384-token limit with four 1 MP images
(22.3 GB and 23.9 GB with text only). On the RTX 5090 the engine holds 19.4 GB allocated and 20.2 GB reserved once
loaded, and the CUDA context and libraries 1.2 GB besides. A 24 GB card has 25.8 GB (24,564 MiB), of which the driver
keeps about 0.5 GB on a Blackwell GeForce: with PyTorch's allocator capped at 23.9 GB, what such a card leaves it,
everything ran, the requests at the 16,384-token limit at 23.4 GB reserved and 24.6 GB on the device in all, and
capped at 22.9 GB those requests and requests with four 1 MP images ran as well. Answers with images stay within 0.034
of the float32 reference on both GPUs, with a mean difference of 0.0009, about that of text answers.

In the shipped catalog a quarter of Clef-Flash's request mix carries an image of 256 to 512 px a side ([Throughput
tests](#throughput-tests)). From the figures above, such a request takes about 450 text tokens of GPU time more on the
H100 (about 310 for the image, 139 for its question). The miner's own measurement of the mix on an H100 gave 23.0
requests per second (22.6-23.3 in three measurements) against 25.0 for the same mix without images (24.8-25.8) in one
session; on the RTX 5090 the figures above give about 7.9 against 8.5 (derived, not measured on the mix).

### 4. Configure

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
| `network` | `finney`, `test`, a `wss://` subtensor node, or a node on this machine (`local`, `ws://127.0.0.1:<port>`); plain `ws://` to another machine is refused |
| `mechid` | the mechanism the manifest is for, instead of the one the catalog names (`SUBNET_ADMIN.md`); leave it unset to follow the catalog |
| `public_host`, `port` | the public IP address and port validators reach this machine's endpoints at (on the primary also the axon); the miner listens on `listen_host` (default `0.0.0.0`) at `port` |
| `role` | `primary` (default): publishes the hotkey's manifest; `worker`: serves its endpoints for a primary and writes nothing on chain ([One hotkey, several GPUs and machines](#one-hotkey-several-gpus-and-machines)) |
| `peers` | primary only: the workers' `ip:port`, whose endpoints the manifest lists |
| `endpoint_id` | 1-32 characters of `a-z`, `0-9`, `-`; unique per miner |
| `model_id` | a catalog model |
| `engine_url`, `engine_api_key` | the endpoint's engine and its bearer token, if it needs one: `http://127.0.0.1:<port>` on this machine, or `https://` elsewhere |
| `engine_tls_pin` | an https engine's certificate pin (`DISCOVERY.md`); without one, a public CA's certificate for its hostname |
| `max_concurrency` | highest concurrency the capacity measurement tries (default 256); between measurements, the engine never gets more requests at once than the concurrency the measurement settles on |
| `gateways` | the miner's own gateways besides the catalog's, each `{"hotkey": ..., "egress": [...]}`: served like the catalog's gateways, from the public addresses their requests come from |
| `firewall` | `true` (default): the port answers only the network's callers and the hotkey's machines once the catalog has miners enforce it; `false`: anyone ([Firewall](#firewall)) |
| `catalog_mirrors` | extra catalog sources, https URL templates with `{digest}`, tried after the owner's axon |
| `catalog_dir` | local catalog store (default `~/.verathos/catalog`) |
| `state_dir` | capacity measurements and the caps tests set on them, the last published manifest, and the machine's TLS key and certificate (`tls.key`, `tls.crt`) (default `~/.verathos/miner`) |
| `auto_update` | install each new release by itself outside throughput tests, then exit for the process manager to start it (default false; the installer sets true) ([Update](#update)) |

### 5. Run

```bash
.venv/bin/verathos-miner --config miner.json
```

On start the miner:
1. finds the catalog the subnet owner commits on chain (the shipped copy, its local store, the owner's axon or a
   configured mirror; any copy is checked against the committed digest); endpoints whose model the catalog does not
   serve (unknown or `retired`) are left out, and it exits if none remain;
2. starts its service over TLS: the manifest (`GET /v1/manifest`, on a worker its endpoints, `GET /v1/endpoints`)
   and the endpoints (`/e/<endpoint_id>/v1/...`) on one port, with the machine's certificate (`state_dir/tls.key`,
   `tls.crt`, created at first start and logged with its pin; delete both to rotate it, and the next manifest names
   the new pin);
3. waits until every engine answers, then measures the endpoints' capacity through its own TLS service, as tests
   reach it, all at once (a
   few minutes; see `THROUGHPUT_AUDIT.md`), or reuses the stored measurements if nothing they depend on changed. A
   machine that served endpoints before waits for a throughput test already selected
   ([When the miner measures](#when-the-miner-measures)). Meanwhile it serves the manifest it last published. An
   endpoint that cannot be measured is withdrawn, and the miner stops only if none is left;
4. signs a manifest for the catalog's mechanism listing its endpoints, their declared capacity and its pin (on a
   primary with `peers`, the workers' endpoints and pins too), points its axon at `public_host:port` (`serve_axon`) and commits the manifest digest
   (`set_commitment`); both are free and only sent when they change, so a restart with the same configuration and
   measurements writes nothing. A worker signs only its own endpoints and serves them to its primary at
   `GET /v1/endpoints`;
5. checks the owner's commitment and its engines every minute; when the owner commits a new catalog it re-signs and
   re-publishes the manifest by itself (for the new catalog's mechanism, if it names another), withdrawing endpoints
   whose model was retired and measuring again if the catalog changed what a measurement depends on (a model's
   request mix or `max_latency_s`); when an engine changed (its version, weights or settings), it measures again;
   when an endpoint fell short in its last two throughput tests, it measures again and declares at most what the
   endpoint delivered in them ([Falling short](#falling-short)). A failed measurement keeps the current manifest, a
   measurement that a throughput test interrupts is retried later, and one that waits for a test starts after it.

Run it under a supervisor (systemd, pm2) so it restarts after updates and crashes. Requests are accepted from hotkeys
holding a validator permit and at least the catalog's `scoring.validator_min_stake` of stake (the metagraph's `S`: alpha
stake plus TAO stake weighted as for permits; both read every 10 minutes), from the auditor whatever its stake, from the
subnet's gateways (the catalog's `gateways`) and the miner's own `gateways`, and as test traffic under a session the
auditor signed; while the firewall enforces, only from their addresses ([Firewall](#firewall)). Other callers are
refused before their request body is read; bodies are capped at 1 MiB (for a model whose catalog entry lists `image`, at
its `images.max_request_bytes`, 13 MiB by default) and must arrive within 10 s, and a request's state, question ids,
instructions and criteria together may hold at most 4 characters per token of the model's `max_state_tokens`. Images
reach an engine only for a model that reads them, at most its `images.max_count`, each canonical as the gateway makes it
(an object of `image/png` or `image/jpeg` in base64, 8-bit RGB with nothing but its header and tables before its pixels,
its sides multiples of `images.token_px` and its pixels within the model's budget); anything else gets 422 before the
engine sees it. A validator gets one request at once per endpoint (HTTP 429 beyond that), each no larger than the
largest request of the model's request mix or the largest canary, for a model that reads images an image canary (HTTP
413), at most the catalog's `scoring.validator_requests_per_epoch` per endpoint and epoch (default 4 x
`canary_pool_size`, 64; beyond, HTTP 429 with `Retry-After` until the next epoch), and while the auditor's test traffic
arrives its requests wait behind the test's; gateways and test traffic are not limited. The miner signs only well-formed
answers (one distribution per question over its options, with the `noul`, `choice` and `score` derived from it; fields
other than an answer type's own are dropped before signing): an engine that returns anything else gets an error back to
the caller, never a signed receipt. The auditor recomputes a sample of the answers given to the subnet's gateway, as it
does test answers: serve users the same model, at the same precision. The input tokens a receipt signs are checked
against the reference's count too: users are billed for them and they set each model's share of the emission, so signing
more or fewer than the reference counts is a wrong answer. A request's images count as the release counts them: (width /
32) x (height / 32) + 2 tokens each, and one more for the request.

## Reading the log

The miner logs its setup once, then what happens to its endpoints. Every two minutes each machine reads the hotkey's
status from the auditor's validator (`GET /v1/status/<hotkey>` on its axon over TLS, accepting the certificate the
catalog's `auditor.tls` names; that validator's view of the last epoch,
with the hotkey's results in the auditor's last 10 throughput tests) and logs it once per epoch, plus every
throughput test of its endpoints once it is final (after a start, the latest of each endpoint):

```
measuring the capacity of 1 endpoints (about 3 minutes)
endpoint gpu0 (kev-4b): 19.9 requests/s at concurrency 32 (runs 19.2, 22.9, 19.1, 20.1, 19.9), declares 18.9
manifest committed on chain (2968f3219dc7)
serving 1 endpoints on port 8091; throughput tests are accepted from 5F...; status is logged once per epoch
validator's epoch 45351:
  emission shares: clef-flash 33.3%, kev-4b 33.3%, kev-9b 33.3%
  miner share 59.00%: the scores below are shares of it; the subnet owner's UID gets the rest of the miners' emission
  gpu0 (kev-4b): qualifying, checked but not scored until it passes a throughput test; answers pass 10/10, mean diff 0.0014
throughput test of epoch 45351 (block 8163225), gpu0: passed, delivered 18.9 of 18.9 requests/s, p95 latency 1630 ms; credited 18.9
validator's epoch 45352:
  emission shares: clef-flash 33.3%, kev-4b 33.3%, kev-9b 33.3%
  miner share 59.00%: the scores below are shares of it; the subnet owner's UID gets the rest of the miners' emission
  gpu0 (kev-4b): scored 12.40% of the epoch's emission, average 2.48%; proven 18.9 of 18.9 requests/s declared; ready for users 100%, latency factor 1.00; answers pass 10/10, mean diff 0.0014
```

| Endpoint state | Meaning |
|---|---|
| `scored` | checks passed; the score, its average (EMA), the proven capacity, how often users found it ready and its latency factor |
| `NO CREDIT ... drifted` | the auditor's recomputes of its answers (in tests, or to users) drifted beyond the model's tolerance over recent checks: no credit until they are back within it |
| `qualifying` | checked, but not scored until it passes a throughput test |
| `candidate model` | the catalog lists the model as a candidate: checked, not scored |
| `answers FAILED` | an answer failed verification; zero score for the probation epochs shown |
| `WRONG ANSWER (in a throughput test or to the gateway)` | the auditor proved a wrong answer the hotkey signed; every endpoint of the hotkey restarts from zero after probation |
| `on probation` | still serving out an earlier failure |
| `not in that epoch` | the validator did not accept the manifest in that epoch (published after it started, or rejected: check the validator's reason with `scripts/probe_miner.py`) |

| Test outcome | Meaning |
|---|---|
| `passed` | delivered at least 95% of the tested rate; credited what it delivered |
| `BELOW the tested rate` | delivered less than 95% of it; credited delivered x delivered / tested; the reason names the shortfall. Two in a row lower the declaration ([Falling short](#falling-short)) |
| `INVALID` | a sampled answer was wrong |
| `not counted` | the test or this endpoint's result was void (the reason says why); earlier credit still counts |

The auditor signs each status with the subnet owner hotkey. The miner checks the signature against the owner hotkey
it reads from the chain, and that the status is for its hotkey and netuid, signed within 5 minutes of the machine's
clock (keep it synchronised) and not before a status it already read. Only the test results of such a status count
for [Falling short](#falling-short). Any other status, such as an unsigned one from an auditor of an older release, is
logged all the same, with one warning until a status verifies again:

```
the auditor's status is logged but not acted on: it is not signed (the auditor runs an older release). Until a status verifies, its test results never make the miner measure again or cap a declaration
```

The auditor builds a hotkey's status at most every 30 s (sooner after an epoch or a test) and answers each client
address (an IPv4 address, or an IPv6 /64) a burst of 10 status requests, then one every 5 s, with HTTP 429 and
`Retry-After` beyond: room for about 24 machines polling every two minutes behind one address.

## Throughput tests

Once per epoch, at a block nobody knows in advance, every endpoint gets its declared rate for 19-22 s: a warmup
of random length, then a 16 s measured window. A reply counts if it arrives inside the window, to a request sent at
most 2 s before the window opened, so an endpoint whose latency stays within 2 s has every reply counted. Nothing
needs to be done by hand, but:
- keep the engines warm and their GPUs free of other work: a test delivers what the hardware has at that moment;
- run `--remeasure` after changing an endpoint's hardware; the miner measures again by itself when the engine's
  identity, the endpoint's configuration, the request mix, `max_latency_s` or the way it measures changes, and when
  the endpoint fell short in two tests ([Falling short](#falling-short));
- serve the catalog's precision: the auditor recomputes test answers, and an endpoint whose answers drift from the
  reference over several tests loses its credit;
- for a model whose request mix carries images (the catalog's `workload.images`; in the shipped catalog a quarter of
  Clef-Flash's requests, each with one image of 256 to 512 px a side), the measurement and the tests send them too:
  an endpoint declares and is credited requests per second of the mix with its requests with images. Every such image
  is new, drawn for its request, and the request also asks what colour its background is, so an engine that refuses
  images delivers only the requests without them, and one that answers without reading them gives wrong answers;
- an engine that just restarted runs slow for its first minutes, and a test then credits less;
- requests from gateways are always served before test requests, and what an endpoint answers them during a test
  counts as delivered; validators' requests wait behind test requests while the test's traffic arrives;
- users count between tests too: the share of the times the gateway considers an endpoint while it is not ready
  (busy, failing, or answering slower than twice `max_latency_s`) comes off its score, and its latency factor comes
  from the latency its users see. An endpoint that declares more than it serves is tested at about what it serves,
  and credited less than that.

### When the miner measures

A measurement loads the machine's engines for a few minutes: a test meanwhile would find them busy and credit less,
and the measurement would count less too. So the miner follows the public test schedule on the chain, and whatever
the reason to measure (the start, a new catalog, a changed engine, a shortfall, `--remeasure`), it starts no
measurement from a test's selection block until one block after its traffic: the `lead_blocks` before the traffic
(5 blocks, about 60 s, by default) and the traffic itself. The endpoints keep serving what they declare meanwhile:

```
the throughput test of epoch 45418 is selected, its traffic from block 8175283: measuring after it
measuring the capacity of 1 endpoints (about 3 minutes)
```

Before an epoch's test is selected nobody knows when it comes, so the miner measures. If the test starts during the
measurement, its traffic stops the measurement within one request, inside the test's unmeasured warmup, and the
measurement is tried again after the test. A machine that serves no endpoint yet (its first start) measures at once:
no test reaches it.

### Falling short

The tests measure what an endpoint serves under real conditions, and their result counts over the miner's own
measurement. When, in a status the subnet owner signed ([Reading the log](#reading-the-log)), an endpoint's last two
counted tests (confirmed, not void) were both `BELOW the tested rate`, with less delivered than the rate tested, at
the rate it declares now, the miner measures again by itself ([When the miner measures](#when-the-miner-measures)):
all the machine's endpoints in the same rounds, as at the start, while they keep serving (requests from gateways and
validators go before the measurement's). The endpoint then declares the new measurement with the usual margin, but
at most the higher rate it delivered in those two tests:

```
throughput test of epoch 45418 (block 8175283), gpu0: BELOW the tested rate, delivered 10.4 of 11.4 requests/s, p95 latency 1810 ms; credited 9.5 (delivered 10.4 of 11.4 requests/s asked)
gpu0 fell short in tests 45417 and 45418 (delivered 10.5 and 10.4 of 11.4): measuring again
measuring the capacity of 1 endpoints (about 3 minutes)
endpoint gpu0 (kev-9b): 12.0 requests/s at concurrency 16 (runs 12.1, 11.9, 12.0, 12.0, 11.8), declares 10.5, at most the 10.5 it delivered in tests 45417 and 45418
```

The new declaration is published as any other (on a worker, the primary takes it from `GET /v1/endpoints`). The cap
is kept across restarts and holds until the miner measures for another reason: an engine, an endpoint's
configuration or the request mix changed, or the miner started with `--remeasure`, which is how to raise the
declaration again once what slowed the endpoint is fixed. A further shortfall can only lower it.

If the endpoint delivered less than half the rate tested in both tests, it was as good as down then (stopped, or
unreachable), which says nothing of what it serves when up. The miner measures again all the same, which shows whether
it recovered, but those tests set no cap; a cap set earlier holds:

```
gpu0 delivered 0.3 and 0.0 of 11.4 in tests 45417 and 45418, as if it was down: measuring again without a cap
```

An endpoint makes the miner measure again at most once every 3 epochs, and never twice for the same tests.

## Updating

A miner set up with the one-command install updates itself to each release, and by running it again
([Update](#update)). A manual installation (`auto_update` in `miner.json` makes a git checkout run under a process
manager update itself too):

```bash
git pull && VIRTUAL_ENV=$PWD/.venv uv pip install -e ".[server,chain]"
```

Then restart the miner. Catalog changes (new models, tolerances, audit policy) need no update: the miner picks up a
new commitment by itself. Releases carry code changes, such as support for a new engine family.

## Checking a miner

From another machine, with a hotkey the miner accepts: a gateway listed in the miner's `gateways` with that
machine's address as its `egress`, which the firewall then lets through too:

```bash
.venv/bin/python scripts/probe_miner.py --network finney --netuid 96 --uid <uid> --endpoint gpu0 \
    --caller-wallet <wallet> --caller-hotkey <hotkey> --reference tests/fixtures/drift/drift-kev-4b-CPUfp32.json
```

It checks the manifest against the chain, the endpoint's identity signature and receipts, and compares 30 answers
with stored CPU fp32 reference answers (fixtures for `kev-4b` and `clef-flash`: `drift-clef-flash-CPUfp32.json`).
