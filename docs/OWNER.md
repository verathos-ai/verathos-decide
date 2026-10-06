# Subnet owner operations

The owner sets up the mechanisms, publishes the catalog (models, verification tolerances and capacity-audit
policy) by committing its digest from the owner hotkey, and runs the auditor validator. `verathos-owner` does the
chain side.

```bash
VIRTUAL_ENV=$PWD/.venv uv pip install -e ".[chain,ops]"
.venv/bin/verathos-owner --network finney --netuid 96 status
```

`status` shows the UID cap, the mechanisms and their emission split, the owner hotkey, and whether the catalog it
commits is this release's catalog.

## Mechanisms

The network scores on mechanism 1. Two mechanisms require a UID cap of 128 (max UIDs x mechanisms <= 256):

```bash
.venv/bin/verathos-owner --network finney --netuid 96 setup-mechanisms --split 90,10 --wallet <owner wallet>
```

It lowers the UID cap if needed (only possible with at most 128 registered UIDs), sets the mechanism count to 2 and
the emission split between mechanisms 0 and 1. Calls wait out the admin freeze window at the end of each tempo.

## Catalog and policy changes

`src/verathos/catalog/catalog.json` lists the served models (pinned weights, engine, verification tolerances,
`score_weight`) and the capacity-audit overrides. A change goes out as a release:

1. Change the catalog in a pull request; tests check that the catalog and its audit policy are valid.
2. Tag a release. Miners and validators update (`MINING.md`, `VALIDATING.md`).
3. Commit the new catalog from the owner hotkey:

   ```bash
   .venv/bin/verathos-owner --network finney --netuid 96 commit-catalog --wallet <owner wallet> --hotkey <owner hotkey>
   ```

From the commit on, validators run only with the committed catalog and accept only manifests built with it, and
miners on an older release refuse to start. Announce the release before committing so operators can update in
time.

## Capacity audit rollout

The audit starts in `observe` mode: it runs and records outcomes but gates nothing. Before switching to
`score_gate`, check the outcomes in the auditor's database (`state_dir/capacity.sqlite`, table `slots`): honest
endpoints should be `passed`, and misses should come from identifiable causes. Then set
`"capacity_audit": {"mode": "score_gate"}` in the catalog and release it as above.

New GPU classes need a calibrated row in `DEFAULT_GPU_CLASSES` (`src/verathos/capacity/core.py`): run
`scripts/capacity_roundtrip.py` on an unshared GPU of that class and set the pass count so the timed work takes
about two thirds of the deadline.

## Test network first

Every release and policy change runs on testnet (netuid 405) before mainnet: the same commands with
`--network test --netuid 405`. For unattended test-network operations the owner coldkey mnemonic can come from the
OS keyring (`--keyring SERVICE:USER`) instead of a wallet; mainnet operations use the wallet.
