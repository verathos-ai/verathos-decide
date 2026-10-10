# AGENTS.md

Verathos is a Bittensor subnet for decision models (netuid 96 on finney; 405 on test). Miners serve the catalog's
models on GPUs behind a SystemOne-compatible API, validators check their answers against float32 references, and the
subnet's gateway serves users. [README.md](README.md) has the overview and the layout of `src/verathos/`.

## A machine set up with the installer

`scripts/install.sh` puts this code in `~/verathos-decide` and manages that checkout: do not edit or commit there.
Local changes stop both the installer's updates and the self-update.

| | |
|---|---|
| set up or update | `bash ~/verathos-decide/scripts/install.sh miner --yes [options]`, likewise `validator` and `reference` (`--help` lists the options) |
| remove | `bash ~/verathos-decide/scripts/install.sh uninstall` (`--purge` also deletes the configuration) |
| configuration and state | `~/.verathos/miner.json`, `validator.json`, and `~/.verathos/miner/`, `validator/` |
| services (systemd) | `verathos-miner`, `verathos-engine-gpu<N>`, `verathos-validator`, `verathos-reference-<model>`; in containers supervisord, logs in `/var/log/verathos/` |
| the miner's firewall | `verathos-miner firewall status` |
| referral codes | `verathos-miner referral code`, `move`, `status`, `dashboard` ([MINING.md](docs/MINING.md#referrals)) |

- Wallets: a machine holds a hotkey and `coldkeypub.txt`, never a coldkey or a mnemonic. Do not create, import or
  copy one here; registering and staking happen where the coldkey is.
- Without a terminal the installer takes every default and asks nothing: pass the choices as options.
- Guides: [MINING.md](docs/MINING.md), [VALIDATING.md](docs/VALIDATING.md), [DOCKER.md](docs/DOCKER.md); how the network
  scores and audits: [SCORING.md](docs/SCORING.md), [THROUGHPUT_AUDIT.md](docs/THROUGHPUT_AUDIT.md).

## Changing the code

```bash
uv venv -p 3.12 .venv
VIRTUAL_ENV=$PWD/.venv uv pip install -e ".[server,chain,gateway,test]" ruff
.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests scripts
(cd native/loadgen && cargo test)
```

- The tests run on CPU; the Clef engine's and the gateway's image tests skip without their extras and Linux with
  Landlock (README.md, Development).
- The docs state what the code does, with its exact commands, options and log lines: change them with the code.
- `scripts/install.sh` keeps a copy of `release/allowed_signers` (`release_signers`); the tests keep them equal.
