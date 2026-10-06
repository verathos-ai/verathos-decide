# Subnet administration

The owner sets up the mechanisms, publishes the catalog (models, verification tolerances and capacity-audit
policy) by committing its digest from the owner hotkey, and runs the auditor validator. `verathos-subnet` does the
chain side.

```bash
VIRTUAL_ENV=$PWD/.venv uv pip install -e ".[chain,ops]"
.venv/bin/verathos-subnet --network finney --netuid 96 status
```

`status` shows the UID cap, the mechanisms and their emission split, the owner hotkey, the committed catalog digest
and the digest of the catalog shipped with this code.

## Mechanisms

The network scores on mechanism 1. Two mechanisms require a UID cap of 128 (max UIDs x mechanisms <= 256):

```bash
.venv/bin/verathos-subnet --network finney --netuid 96 setup-mechanisms --split 90,10 --wallet <owner wallet>
```

It lowers the UID cap if needed (only possible with at most 128 registered UIDs), sets the mechanism count to 2 and
the emission split between mechanisms 0 and 1. Calls wait out the admin freeze window at the end of each tempo.

## Catalog: models and policy

The catalog lists the served models (pinned weights, engine and engine revision, verification tolerances,
`score_weight`, status) and the capacity-audit overrides. Miners and validators use whichever catalog the owner
hotkey commits; publishing a new one needs no code release.

Model status: `active` models are scored; `candidate` models can be served and are checked but weighted zero, which
lets a new model run on the network before it is scored; `retired` models are skipped, and miners withdraw endpoints
serving them from their manifests (a miner with no other endpoint stops).

To publish a catalog:

1. Edit a copy of `src/verathos/catalog/catalog.json` (keep the repository copy in sync through a pull request;
   tests validate it). For a new model: measure its drift on the target GPUs against CPU fp32 and set its
   tolerances; start it as `candidate`.
2. Run the owner validator, which serves catalogs from its `catalog_dir` at `GET /v1/catalog/{digest}` on its axon.
3. On the owner validator's machine:

   ```bash
   .venv/bin/verathos-subnet --network finney --netuid 96 commit-catalog --wallet <owner wallet> \
       --hotkey <owner hotkey> --file catalog.json
   ```

   It validates the file, stores it in `catalog_dir` and commits its digest. Mirrors (`catalog_mirrors` in miner and
   validator configs) can serve the same file; every copy is checked against the digest.

Miners and validators check the commitment every minute: miners re-publish their manifests, the auditor applies a
new audit policy at once (windows opened while it propagates are void), and validators use the new models and
tolerances from their next epoch, accepting manifests built for the previous catalog for one more epoch. Announce
model changes before committing so validators can start a reference engine for a new model and miners can deploy
it. A model family whose engine needs new code, or a protocol change, still goes out as a code release.

## Capacity audit rollout

The shipped catalog runs the audit in `score_gate` mode. Check outcomes in the auditor's database
(`state_dir/capacity.sqlite`, table `slots`): honest endpoints should be `passed`, and misses should come from
identifiable causes. If the audit misbehaves (for example a GPU class whose pass count is wrong), publish a catalog
with `"mode": "observe"`: outcomes are still recorded, nothing is gated and endpoints need no qualification.
`docs/CAPACITY_AUDIT.md` gives the trade-off between audit rate, downtime and detection time behind the defaults.

GPU classes are rows of `capacity_audit.gpu_classes` in the catalog. To add one, run
`scripts/capacity_roundtrip.py "<nvidia-smi name>" <vram> --passes N` on an unshared GPU of that class, choose the
pass count so the timed work takes about two thirds of the deadline, and publish a catalog with the new row.

## Test network first

Every release and policy change runs on testnet (netuid 405) before mainnet: the same commands with
`--network test --netuid 405`. For unattended test-network operations the owner coldkey mnemonic can come from the
OS keyring (`--keyring SERVICE:USER`) instead of a wallet; mainnet operations use the wallet.
