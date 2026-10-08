# Subnet administration

The owner sets up the mechanisms, publishes the catalog (models, verification tolerances, scoring settings and the
throughput-audit policy) by committing its digest from the owner hotkey, and runs the auditor validator. `verathos-subnet` does the
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

The catalog lists the served models (pinned weights, engine and engine revision, verification tolerances, the
standard request mix `workload`, `price`, `anchor_share`, `first_test_rps`, status), the epoch length, the scoring
settings
(`scoring`) and the throughput-audit policy (`throughput_audit`). Miners and validators use whichever catalog the
owner hotkey commits; publishing a new one needs no code release.

`price` is what users pay for the model, in USD per million input tokens, and the model's share of the emission
follows what its users pay (`SCORING.md`). `anchor_share` is the model's share while
there is no user demand; the active models' anchors must sum to 1 (`commit-catalog` refuses a catalog whose do not).
Activating a model means giving it an anchor and lowering the others'; every active model keeps at least
`min_model_share`, whatever its anchor. `commit-catalog` also refuses settings this code does not know. `first_test_rps` caps the rate of an endpoint's first test, and of all a new hotkey's
endpoints together; later tests grow from what was delivered.

`gateways` lists the hotkeys of the subnet's gateways (`GATEWAY.md`): miners serve them like validators, without
the per-caller limit, and the auditor checks a sample of their answers. Rotating a gateway hotkey is a catalog
commit.

The auditor is the validator of the subnet owner hotkey. After changing the subnet owner hotkey on chain, restart
every validator so it applies the new auditor's verdicts (miners follow within a minute).

Model status: `active` models are scored; `candidate` models can be served and are checked but weighted zero, which
lets a new model run on the network before it is scored; `retired` models are skipped, and miners withdraw endpoints
serving them from their manifests (a miner with no other endpoint stops).

New weights or a new reference engine for a model are a new model: add it as a `candidate` under a new `model_id`,
activate it, then retire the old one. Changing a model's `hf_revision` or `engine.revision` in place leaves its
answers unchecked for `scoring.revision_transition_epochs` (3) epochs, while miners and references update; after
that, answers from the old weights fail.

To publish a catalog:

1. Edit a copy of `src/verathos/catalog/catalog.json` (keep the repository copy in sync through a pull request;
   tests validate it). For a new model: measure its drift on the target GPUs against CPU fp32 and set its
   tolerances; define its request mix and start it as `candidate`.
2. Run the owner validator, which serves catalogs from its `catalog_dir` at `GET /v1/catalog/{digest}` on its axon.
3. On the owner validator's machine:

   ```bash
   .venv/bin/verathos-subnet --network finney --netuid 96 commit-catalog --wallet <owner wallet> \
       --hotkey <owner hotkey> --file catalog.json
   ```

   It validates the file (an unknown `throughput_audit` setting is an error), stores it in `catalog_dir` and commits
   its digest. Mirrors (`catalog_mirrors` in miner and
   validator configs) can serve the same file; every copy is checked against the digest.

Miners and validators check the commitment every minute: miners re-publish their manifests (measuring again if a
model's request mix changed), the auditor applies a new audit policy at once, and validators score each epoch
under the catalog committed at its first block, accepting manifests built for the previous catalog for one more
epoch. A changed request mix lowers declared capacities while miners measure again, so change it rarely and
announce it. Announce model changes before committing so validators can start a reference engine for a new model
and miners can deploy it: a validator without a reference engine for an `active` model stops. A model family
whose engine needs new code, or a protocol change, still goes out as a code release.

## Throughput audit

The shipped catalog runs the audit in `enforce` mode. Check results in the auditor's database
(`state_dir/throughput.sqlite`, tables `tests`, `results`, `samples` and `verdicts`): honest endpoints should be
`passed`, and `partial`
or `void` results should have identifiable causes. If the audit misbehaves (for example the auditor's connection
cannot carry the network's test traffic), publish a catalog with `"mode": "observe"`: tests still run and are
recorded, validators count declared capacities (up to each model's `first_test_rps`) and apply no evidence. `docs/THROUGHPUT_AUDIT.md` describes every
setting.

## Test network first

Every release and policy change runs on testnet (netuid 405) before mainnet: the same commands with
`--network test --netuid 405`. For unattended test-network operations the owner coldkey mnemonic can come from the
OS keyring (`--keyring SERVICE:USER`) instead of a wallet; mainnet operations use the wallet.
