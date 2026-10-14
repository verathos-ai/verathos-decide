# Contributing

## Branches

| | |
|---|---|
| `dev` | the default branch: every pull request goes here |
| `main` | releases: maintainers move it to a commit of `dev` that passed CI and tag the release there |

A pull request against `main` fails the `base-branch` check: change its base branch to `dev` (Edit, next to its
title).

Open an issue before a larger change (a feature; a change to the protocol, scoring or the throughput audit), so it is
agreed before the work. Vulnerabilities are reported privately ([SECURITY.md](SECURITY.md)), never in an issue or a pull
request. The shipped catalog (models, tolerances, scoring settings, the audit policy) is the subnet owner's to change
and the release keys in `release/allowed_signers` are the maintainers': propose a change to either in an issue.

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/); for the load generator in `native/loadgen`, Rust through
[rustup](https://rustup.rs), which installs the version its `rust-toolchain.toml` names.

```bash
git clone https://github.com/<you>/verathos-decide && cd verathos-decide   # your fork
uv venv -p 3.12 .venv
VIRTUAL_ENV=$PWD/.venv uv pip install -e ".[server,chain,gateway,test]" ruff==0.16.10
```

## Tests and lint

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests scripts
(cd native/loadgen && cargo test -- --include-ignored)
```

The tests run on CPU. Some need more and are skipped without it (`pytest -rs` lists them with the reason):

| Tests | Need |
|---|---|
| End-to-end throughput tests | the load generator, built by `.venv/bin/verathos-build-loadgen` |
| Clef engine tests | the `clef` extra (`VIRTUAL_ENV=$PWD/.venv uv pip install -e ".[server,chain,gateway,test,clef]" --torch-backend cpu`) and the pinned clef-flash release's code, tokenizer and configs, without its weights, in the Hugging Face cache (downloaded as the `pytest` job does) |
| Firewall tests with nftables | `nft`, and user and network namespaces for an unprivileged user |
| Gateway image tests | Linux with Landlock |
| Docker tests | `docker compose` |
| Clef engine tests on a GPU | `VERATHOS_GPU_TESTS=1`, a CUDA GPU and the pinned weights |
| Real-time load tests (a shortfall measurement and two load generator tests) | an uncontended machine: skipped where `CI=true`, and `cargo test` runs the two only with `-- --include-ignored` |

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs on every push to `dev` and `main` and on every pull
request: ruff (job `ruff`), the tests with all of the above but a GPU and the real-time load tests (`pytest`) and the load
generator's tests but the real-time ones (`cargo-test`): shared runners lag behind the offered rates.

## Pull requests

A pull request is ready to merge when:

- it targets `dev` and its checks pass: `ruff`, `pytest`, `cargo-test` and `base-branch`;
- what it changes is tested: a fix comes with a test that fails without it;
- the docs say what the code now does, with its exact commands, options and log lines;
- what it makes unused is removed with it;
- it contains no keys, tokens, mnemonics or wallets, and no real machine's address: tests and docs use the
  documentation ranges (192.0.2.0/24, 198.51.100.0/24, 203.0.113.0/24, 2001:db8::/32).

Its description says what changes for miners, validators, reference machines, the gateway or the auditor, and how
nodes on the previous release work with nodes on this one while they update. One change per pull request.

Code follows the ruff settings in `pyproject.toml` (line length 120). Comments and docs state what the code does and
why, not its history. `scripts/install.sh` keeps a copy of `release/allowed_signers` (`release_signers`); the tests
keep them equal.

## Commit messages

One line, as `git log` reads: what the code does once the commit is in, in the present tense, naming the commands,
options and files concerned. A fix also says what happened before, after `where`. No prefix and no body:

```
A model whose every miner is failing is unavailable, not busy
verathos-engine logs no line per request, where uvicorn's access log wrote one at INFO for each request
```

Maintainers merge a pull request with a merge commit (`Merge <topic>: <what it does>`) or squash it into one commit
in the same style.

## Releases

Maintainers cut releases:

1. A pull request to `dev` sets `__version__` in `src/verathos/__init__.py`; the release runs on testnet (netuid 405)
   before it is tagged.
2. The commit is tagged `v<version>` with the release key, and the tag is checked against the previous release's
   `release/allowed_signers` ([Signing releases](docs/SUBNET_ADMIN.md#signing-releases)).
3. `main` moves to the commit, a fast-forward that GitHub accepts once CI has passed on it, and the tag is pushed:

   ```bash
   git push origin <commit>:main
   git push origin v<version>
   ```

Miners and validators install a release only from a tag that a key in `release/allowed_signers` signed
([Releases and the weights version key](docs/SUBNET_ADMIN.md#releases-and-the-weights-version-key)).

## License

A contribution is licensed under the [MIT License](LICENSE), as the repository is. There is no contributor license
agreement and no sign-off.

## Conduct

Taking part means following the [code of conduct](CODE_OF_CONDUCT.md).
