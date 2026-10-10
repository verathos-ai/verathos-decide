# Subnet administration

The owner sets up the mechanisms, publishes the catalog (models, verification tolerances, scoring settings and the
throughput-audit policy) by committing its digest from the owner hotkey, and runs the auditor validator. `verathos-subnet` does the
chain side.

```bash
VIRTUAL_ENV=$PWD/.venv uv pip install -e ".[chain,ops]"
.venv/bin/verathos-subnet --network finney --netuid 96 status
```

`status` shows the UID cap, the mechanisms and their emission split, the owner hotkey, the committed catalog digest, its
mechanism, auditor and gateways, the digest of the catalog shipped with this code, the current epoch and the committed
catalog's miner share in it, its usage and referral pools with their current periods, the subnet's weights version key
and this release's ([Releases](#releases-and-the-weights-version-key)).

## Mechanisms

The catalog names the mechanism the network runs on (`mechanism`; 1 when a catalog names none): miners publish
their manifests for it and validators set weights on it, so every node follows the committed catalog and installs
need no setting for it. The shipped catalog names mechanism 0, as on SN96: the subnet is full at 256 UIDs, and the
network runs on its only mechanism. A test network's catalog names its own (netuid 405: `"mechanism": 1`).
`commit-catalog` refuses a catalog whose mechanism the subnet does not have (its mechanism count must exceed it).

A second mechanism is only for a subnet that wants one and can have it: two mechanisms require a UID cap of 128
(max UIDs x mechanisms <= 256), so at most 128 registered UIDs.

```bash
.venv/bin/verathos-subnet --network test --netuid 405 setup-mechanisms --split 90,10 --wallet <owner wallet>
```

It lowers the UID cap if needed, sets the mechanism count to 2 and the emission split between mechanisms 0 and 1.
Calls wait out the admin freeze window at the end of each tempo.

Moving the network to another mechanism is a catalog commit like any other: miners re-publish their manifests for
the new mechanism within minutes, validators accept manifests built for the previous catalog (for its mechanism) for
one more epoch and set each epoch's weights on the mechanism of the catalog it is scored under. The weights set on
the previous mechanism stay there until validators set others. A node's `mechid` (miner and validator
configuration) overrides the catalog's; leave it unset.

## Catalog: models and policy

The catalog lists the served models (pinned weights, engine and engine revision, verification tolerances, the standard
request mix `workload`, `price`, `anchor_share`, `first_test_rps`, `relative_speed`, the input each reads with its image
limits, status), the [mechanism](#mechanisms), the epoch length, the scoring settings (`scoring`, including the
[miner share](#miner-share), the [usage pool](#usage-pool) and [referrals](#referrals)) and the throughput-audit policy
(`throughput_audit`).
Miners and validators use whichever catalog the owner hotkey commits; publishing a new one needs no code release.

`price` is what users pay for the model, in USD per million input tokens, and the model's share of the emission
follows what its users pay (`SCORING.md`). `anchor_share` is the model's share while
there is no user demand; the active models' anchors must sum to 1 (`commit-catalog` refuses a catalog whose do not).
Activating a model means giving it an anchor and lowering the others'; every active model keeps at least
`min_model_share`, whatever its anchor. `commit-catalog` also refuses settings this code does not know (it checks the
keys of a model's `images`, of its `workload` and of the workload's `images`, not those inside its `tolerance` and
`engine`). `first_test_rps` caps the rate of an endpoint's first test, and of all a new hotkey's endpoints together;
later tests grow from what was delivered.

`input_modalities` lists what a model reads: `["text"]`, the default and the shipped Kev models', or `["text",
"image"]`, the shipped Clef-Flash's, which opens a model to images (`GATEWAY.md`, Images) within its `images` limits;
any of them may be set, the others keep these defaults:

```json
"input_modalities": ["text", "image"],
"images": {"max_count": 4, "max_bytes": 4194304, "max_total_bytes": 8388608, "max_input_pixels": 16777216,
           "max_pixels": 1048576, "min_pixels": 65536, "token_px": 32, "max_request_bytes": 13631488,
           "canary_share": 0.25}
```

`max_count` images per request, `max_bytes` per image and `max_total_bytes` per request as sent, `max_input_pixels` per
image as sent, `max_pixels` the pixel budget an image is resized into (at least `min_pixels`), `token_px` the side of
one image token, as the model's processor has it, `max_request_bytes` the body of a request with images, and
`canary_share` the share of each validator's canaries of the model that carry an image (`SCORING.md`). Every node
refuses limits that do not hold together (`token_px` squared <= `min_pixels` <= `max_pixels` <= `max_input_pixels`,
`max_bytes` <= `max_total_bytes`, `max_request_bytes` from 1 MiB to 15 MiB, at most 1,048,576 image tokens per request);
`commit-catalog` refuses a modality this code does not serve (`video` among them), `image` for a model whose engine
reads no images (the Kev engine), a `token_px` other than its engine's (32 for Clef), and a pixel budget beyond what its
engine takes as it is (65,536 to 16,777,216 pixels for Clef). Older nodes ignore both fields, and the catalog's digest
covers them, so every node agrees on the catalog. Open a model to images once the gateway, the auditor and the model's
miners run a release that reads them, with engines that serve them (`MINING.md`; a miner of an earlier release may
refuse a validator's image canary as larger than it takes from validators), and the auditor's reference and screen of
the model run on a GPU (`VALIDATING.md`).

`workload` is a model's standard request mix: miners measure and declare their capacity with it, and the auditor's
tests send it (`THROUGHPUT_AUDIT.md`). `state_words` is the range of words of a request's state and `questions` its
questions; for a model that reads images, `images` sets which of its requests carry images, here the shipped
Clef-Flash's:

```json
"images": {"share": 0.25, "count": [1, 1], "sides_px": [256, 288, 320, 352, 384, 416, 448, 480, 512]}
```

`share` of the requests carry `count[0]` to `count[1]` images each, every image of a width and a height drawn from
`sides_px`: canonical PNGs drawn for the request (`throughput/images.py`; a background colour and one to three
rectangles), and such a request also asks what colour the background is (the question `background`, which the mix's
own questions may not take), so its answers depend on its images. Every node refuses images in the mix of a model
that does not read them, more than the model's `images.max_count`, sides that are not multiples of its
`images.token_px` or whose images fall outside its `min_pixels` to `max_pixels`, and images or requests beyond its
byte limits. Capacity is requests/s of the whole mix, so the share sets how much images weigh in an endpoint's
capacity. A quarter, as for the validators' image canaries, keeps an endpoint that refuses images below `pass_fraction`
(it delivers at most 75% of what it is asked), puts a request with images among the two answers a test recomputes in
44% of an endpoint's tests, and adds 1.5 KB to the tests' average request (`THROUGHPUT_AUDIT.md`, Running the
auditor); sides of 256 to 512 px, as the image canaries', keep an image at 66 to 258 tokens and its PNG at 2.0 to 6.5
KB. A changed mix makes its model's miners measure again: publish its `relative_speed` and `first_test_rps` for the
new mix with it.

`relative_speed` is the model's speed relative to the other models: the requests/s of its standard mix one GPU
serves within `max_latency_s`, measured for every model on the same reference GPU and divided by one model's (1.0;
positive, 1.0 when absent). Miners' setups choose each GPU's model by what it would pay there (`MINING.md`), and
estimate a GPU's rate on a model it has not served from its rate on the one it served and this ratio. Measure a new
model on the reference GPU through the miner's own measurement, as miners do, before activating it. The shipped
values, on an H100 80GB, relative to kev-4b, from one session on 2026-10-09: each model measured three times,
interleaved, each right after its engine started, and kev-4b once more at the end; the medians:

| Model | `relative_speed` | Measured |
|---|---|---|
| `kev-4b` | 1.0 | 32.0 requests/s of the standard mix within 2 s (30.9-32.0, at concurrency 16-32; 29.8 at the end) |
| `kev-9b` | 0.81 | 25.9 requests/s (23.6-26.0, at concurrency 8-32); 0.82 on an A100 80GB SXM4, 0.68 on an A100 80GB PCIe, 0.55 on an RTX 5090 |
| `clef-flash` | 0.72 | 23.1 requests/s of its mix with images (22.1-23.2, at concurrency 32); 25.0 of the same mix without images in an earlier session that day (24.8-25.8, kev-4b 31.6-32.0 there); without images 0.65 on an A100 80GB SXM4 (9.5 against kev-4b's 14.2-15.1) |

A single measurement can settle a step below a model's best (kev-9b's 23.6 above, at concurrency 8): take the median
of several. Each model's `first_test_rps` is about 1.7 times its rate on the reference GPU, to a multiple of 5 (55,
45 and 40), so a first test can ask a faster GPU for more than the reference serves. The ratio differs by GPU: a
larger model costs relatively less on a larger GPU. Re-measure on the reference GPU when an engine gets faster, and
replace an estimate once the model's standard mix is measured.

`gateways` lists the subnet's gateways (`GATEWAY.md`), each a `hotkey` and its `egress` addresses, the public IP
addresses its requests to miners come from: miners serve them like validators, without the per-caller limit, and the
auditor checks a sample of their answers. Rotating a gateway hotkey or moving a gateway is a catalog commit.

`auditor` names the auditor's service: `tls`, the pin of its TLS certificate (`DISCOVERY.md`), and `egress`, the
public IP addresses its requests to miners come from (identity challenges, canaries, tests). Miners, validators,
light validators and gateways reach the auditor only with the certificate the committed catalog names.

```json
"auditor": {"tls": "<pin, 64 hex digits>", "egress": ["<auditor ip>"]},
"gateways": [{"hotkey": "<gateway hotkey>", "egress": ["<gateway ip>", "<gateway ipv6>"]}]
```

Addresses are public unicast IP literals written as the chain writes axons (IPv6 in its shortest form), at most 16
per entry. `commit-catalog` refuses a catalog without `auditor.tls` and `auditor.egress`, or with a gateway without
`egress`. With the validators' published addresses they form the callers' allowlist
(`protocol.callers.allowlist`), which miners' firewalls let through. They must be every address the auditor's and the
gateways' requests leave from: an enforcing miner drops the rest, the auditor's tests included. The auditor checks its
own every 5 minutes and logs a CRITICAL line when its requests leave from an address the catalog does not name.

`firewall` switches miners' firewalls (`MINING.md`, Firewall): `observe` (the default) has them learn and report
and drop nothing; `enforce` has each miner drop connections from every address but the callers' once its own checks
pass (the auditor's address known, callers seen arriving from their published addresses, the known addresses covering
at least half of the stake of the validators that call miners). Switch to `enforce` once the full validators holding
most of that stake publish their egress addresses (their commitments, `verathos/validator/v1 <pin> <addresses>`), and
back to `observe` to have every miner stop dropping at its next catalog check.

```json
"firewall": "enforce"
```

`scoring.validator_min_stake` is the stake a validator needs for miners to answer it, in the unit of the
metagraph's `S` (alpha stake plus TAO stake weighted as for validator permits); 0, or a catalog without it, means any
validator with a permit. Miners apply it once they adopt the committed catalog (within a minute) and read the
validators' stakes every 10 minutes. `scoring.validator_requests_per_epoch` caps what any validator may send one
endpoint per epoch (by default 4 x `canary_pool_size`, more than three times what an epoch's canaries and identity
challenges take). Together they keep stakers from using miners as a free inference service; the auditor is answered
whatever its stake, and its test traffic is not counted.

The auditor is the validator of the subnet owner hotkey. After the subnet owner hotkey changes on chain, every full
validator exits at the start of its next epoch so that its supervisor restarts it under the new auditor (the new
owner's validator then runs the tests); light validators take the new owner's validator as their source at their
next fetch, and miners follow within a minute.

Activate a model only once its reference engine reports `usage.input_tokens` (the auditor logs a warning for one that
does not): the input tokens miners sign set the model's demand, and only the reference's count checks them.

Model status: `active` models are scored; `candidate` models can be served and are checked but weighted zero, which
lets a new model run on the network before it is scored; `retired` models are skipped, and miners withdraw endpoints
serving them from their manifests (a miner with no other endpoint logs an error every minute and stops at its next
start).

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

   It validates the file (an unknown `throughput_audit` setting is an error, and so are a mechanism the subnet does not
   have and a model's input this code does not serve), fills in `auditor.tls` and `auditor.egress` where the file names
   none, from the auditor's configuration on this machine (`--validator-config`, default `~/.verathos/validator.json`:
   the certificate in its `state_dir` and its `egress_ips`), refuses an `auditor.tls` other than that certificate's,
   stores the result in the owner validator's `catalog_dir` (`--catalog-dir`, default the `catalog_dir` of
   `--validator-config`, else `~/.verathos/catalog`) and commits its digest. Nodes that start take the committed
   catalog from there (or a mirror) and wait until they can, so run it as the validator's user, or name its
   configuration. Mirrors (`catalog_mirrors` in miner and validator configs) can serve the same file; every copy is checked
   against the digest.

Miners and validators check the commitment every minute: miners re-publish their manifests (measuring again if a
model's request mix changed), the auditor applies a new audit policy at once, and validators score each epoch
under the catalog committed at its first block, accepting manifests built for the previous catalog for one more
epoch. A changed request mix lowers declared capacities while miners measure again, so change it rarely and
announce it. Announce model changes before committing so validators can start a reference engine for a new model
and miners can deploy it: a validator without a reference engine for an `active` model stops. A model family
whose engine needs new code, or a protocol change, still goes out as a code release.

## The auditor's certificate

The auditor's validator creates its TLS key and certificate in its `state_dir` (`tls.key`, `tls.crt`) at first
start, and logs at every start and catalog change whether the committed catalog names its pin; while it does not
(`CRITICAL`), no node reaches the auditor. Catalogs are fetched from the auditor's axon accepting any certificate,
since their digest proves them, so a catalog naming a new pin reaches every node.

To rotate it: stop the auditor, delete `tls.key` and `tls.crt` in its `state_dir`, start it (it logs the new pin),
and run `commit-catalog` on its machine with a catalog file without `auditor.tls`. Nodes follow the commitment
within a minute; until then their requests to the auditor fail and are retried, and gateways keep their routes.

## Miner share

`scoring.miner_share` is the share of the mechanism's miner emission the scores pay the miners. Validators scale the
miners' weights to sum to it and set the rest as weight on the subnet owner's UID (`SCORING.md`). The chain pays the
subnet owner hotkey no incentive: it burns it, or recycles it if the subnet's `RecycleOrBurn` is set to recycle
(AdminUtils `sudo_set_recycle_or_burn`). Without the setting the miners get all of it (1). It is a number in (0, 1],
or a linear change, as in the shipped catalog:

```json
"miner_share": {"from": 0.59, "to": 0.15, "start_epoch": null, "epochs": 2400}
```

The share is `from` until `start_epoch`, then moves linearly to `to` over `epochs` epochs and stays there; while
`start_epoch` is null it stays at `from`. `from` and `to` are in (0, 1] (either may be the larger), `epochs` at least
1. 2400 epochs of 180 blocks are about 60 days. Every validator computes an epoch's share from the catalog and the
epoch number (block // `epoch_blocks`) alone, rounded to 6 decimals, logs it each epoch, and the auditor publishes it
in `GET /v1/network` (`miner_share`); light validators set the auditor's weights as published.

To start the step-down, set `start_epoch` to the epoch it starts at, no earlier than the epoch the commit falls in
(each epoch is scored under the catalog committed at its first block; an earlier one makes the share jump), and
commit the catalog. `status` shows the current epoch and its share, and `commit-catalog` prints the share
in the current epoch and when the change ends. Keep `start_epoch` and `epochs` in later catalogs, or the share jumps
to what the new values give; changing `epoch_blocks` renumbers the epochs, so it moves the change too. A catalog whose
`miner_share` differs from the committed catalog's is refused unless `--change-miner-share` is given: one built from
the shipped file, whose change has no `start_epoch`, would put the share back to `from` at once.

## Usage pool

`scoring.usage_pool` pays the hotkeys that served paying users on top of their scores, out of what the miner share
leaves (`SCORING.md`). Without it, or with a miner share of 1, there is none.

```json
"usage_pool": {"mode": "burned", "fraction": 1.0, "period_epochs": 40,
               "buyback_coldkey": "5...", "excluded": ["5...", "5..."]}
```

| Setting | Meaning |
|---|---|
| `mode` | `burned`: a period's pool is the alpha `buyback_coldkey` burned on the subnet in it; `revenue`: the period's paid usage, converted to alpha |
| `fraction` | in (0, 1]: the share of that alpha the pool pays |
| `period_epochs` | 1 to 400, default 40 (about a day): a period's pool is paid over the epochs of the next |
| `buyback_coldkey` | `burned` mode: the coldkey whose `AlphaBurned` events on the subnet count |
| `excluded` | hotkeys and coldkeys whose endpoints get no usage pay, such as the owner's own miners; their part stays on the owner's UID |

The buyback coldkey is the owner side's own coldkey, published, not the subnet owner's coldkey (`commit-catalog`
refuses that one): it signs every period, so it holds only the revenue it spends. It buys alpha and burns it from
its own stake, with `add_stake` and `burn_alpha`, or `add_stake_burn` in one call. A burn counts in the period whose
blocks it is in. With `fraction` 1 the pool pays the miners what was burned; a part of the purchase kept rather than
burned (a fee) is not paid. In `revenue` mode no coldkey is read: the pool follows the gateway's paid usage whether or
not alpha is bought with it.

To enable it:

1. Keep the miner share below 1: the pool takes at most `1 - miner_share`. Validators of a release without the pool
   ignore the setting and set their weights without it: enable it once the auditor and the full validators run one
   with it (light validators set the auditor's weights, pool included, whatever their release).
2. Publish the buyback coldkey and the excluded keys, and commit a catalog with `usage_pool`. `commit-catalog` prints
   the pool and the current period. The auditor reads the burns from the start of the period it adopts the catalog
   in (from blocks its node still holds); that period's record is the first, paid over the next period.
3. Burn from the buyback coldkey. The auditor logs each period it closes (`usage pool: period ... closed`), and every
   validator each epoch's usage share.

A period is its blocks: a new `period_epochs` or `epoch_blocks` starts periods of other blocks, and a new period that
overlaps one already recorded has no record (its pool is not paid), so nothing counts twice. A new `buyback_coldkey`
applies to the periods closed after the change: only its burns count in them. Run it on the test network first, with
a test buyback coldkey and a short `period_epochs`.

## Referrals

`scoring.referral_pool` pays referrers, the hotkeys whose codes brought accounts that buy credit, on top of their
scores, out of what the miner share leaves (`SCORING.md`, Referrals; `GATEWAY.md`, Referrals). The shipped catalog has
none; `commit-catalog` refuses one together with a usage pool, and a `payout_per_usd` above 0.9.

```json
"referral_pool": {"share": 0.10, "payout_per_usd": 0.3, "window_periods": 30, "period_epochs": 40,
                  "term_days": 365, "account_cap_usd": 10000, "serving_only": false, "excluded": ["5...", "5..."]}
```

To enable it:

1. Keep the miner share below 1, and run the auditor and the full validators on a release with referrals (light
   validators set the auditor's weights whatever their release).
2. Set the gateway's `referral_site` to the web app's origin, and list other subnets' product names in
   `referral_reserved`; the web app registers codes and referrals before the pool is on, and top-ups credited while the
   catalog has no pool never count.
3. Put the owner's own miners in `excluded`, by hotkey or coldkey (the owner's hotkey and its coldkey's hotkeys are
   never paid), and commit a catalog with `referral_pool`. `commit-catalog` prints the pool and the current period.
   The auditor logs each period it closes (`referrals: period ... closed`), and every validator what it pays each
   epoch.

Removing the section stops the pay at once; a top-up decided meanwhile never counts. A new `period_epochs`,
`window_periods` or `epoch_blocks` starts the windows afresh, so no dollar counts twice. Commit a change of `share`,
`payout_per_usd` or `window_periods` early in a period: a full validator refuses a record whose parameters are not its
catalog's, and the record of the period that ends under the old ones is paid under the new. Run it on the test network
first, with a short `period_epochs` and `window_periods`, `term_days` 30 and `account_cap_usd` 1000.

## Throughput audit

The shipped catalog runs the audit in `enforce` mode. Check results in the auditor's database
(`state_dir/throughput.sqlite`, tables `tests`, `results`, `samples` and `verdicts`): honest endpoints should be
`passed`, and `partial`
or `void` results should have identifiable causes. If the audit misbehaves (for example the auditor's connection
cannot carry the network's test traffic), publish a catalog with `"mode": "observe"`: tests still run and are
recorded, validators count declared capacities (up to each model's `first_test_rps`) and apply no evidence. `docs/THROUGHPUT_AUDIT.md` describes every
setting.

## Releases and the weights version key

A release is a version `major.minor.patch`, `__version__` in `src/verathos/__init__.py` (the package version reads
it), and the commit tagged `v<version>`. Validators send `major x 1,000,000 + minor x 1,000 + patch` as the weights
version key (1.0.0: 1,000,000) and log it at start; minor and patch stay below 1000. The chain refuses weights whose
key is below the subnet's `WeightsVersionKey`, so raising it retires every validator on an older release: such a
validator sets no weights and logs `this validator's release is too old to set weights`. A newer release needs no
change on chain. The validators of the earlier Verathos code send 2004: a `WeightsVersionKey` of 1,000,000 retires
all of them while every validator of this code keeps setting weights.

Raise it from the subnet owner coldkey (AdminUtils `sudo_set_weights_version_key`, as
`btcli sudo set --netuid 96 --param weights_version --value 1000000`), and only once the validators that should keep
setting weights run a release at or above it.

To publish a release: set `__version__` to the new version in a commit on `dev`, tag that commit `v<version>` with a
tag signed by the release key ([Signing releases](#signing-releases)), move `main` to it
([Releases](../CONTRIBUTING.md#releases)) and push the tag to the repository installs come
from. Miners and validators set up by the installer (`auto_update`) find it within an hour and install it within the
hour after, each at a random time and a quiet moment of its own (`MINING.md`, `VALIDATING.md`), then restart; a tag
that is not signed by a key their code lists, or whose code names another version, is never installed. The auditor
is updated by hand. A release that changes engines or services needs the installer run again: say so when announcing
it. Run every release on testnet first.

## Signing releases

Miners and validators install a release only when its tag is signed by a release key: one that
`release/allowed_signers` lists, in the code they run (the self-update) or in the installer (`scripts/install.sh`
holds a copy, `release_signers`, which the tests keep the same as the file). A release that changes the file is
therefore checked against the keys of the release before it. The installer checks a tag it installs, `--ref` or its
default (the newest release); a branch, a commit or a tarball it installs as given, which is the operator's to check.

Tags are signed with SSH keys (`git tag -s` with `gpg.format=ssh`) and checked by `git verify-tag` against the
allowed signers file: git 2.34 and OpenSSH 8.2 or newer do both, as Ubuntu 22.04 and later and macOS ship them, with no
other package. GPG is not on macOS, nor in minimal Ubuntu images.

While the file lists no key, as the shipped placeholder, no release can be verified: nodes do not update by
themselves (`self-update: ... lists no release signing key`) and the installer installs the default branch.

The release key signs nothing else and is kept offline like the owner coldkey, with a passphrase (or on a hardware
key: `-t ed25519-sk`). On the machine that keeps it:

```bash
ssh-keygen -t ed25519 -C verathos-release -f ~/.ssh/verathos-release
printf 'verathos-release namespaces="git" %s\n' "$(cut -d' ' -f1,2 ~/.ssh/verathos-release.pub)"
```

The printed line replaces the placeholder line in `release/allowed_signers` and in `release_signers` of
`scripts/install.sh`, committed before the first release is tagged. Sign each release, check it with the keys of the
release before it (the ones nodes hold) and push it:

```bash
git -c gpg.format=ssh -c user.signingkey="$HOME/.ssh/verathos-release" tag -s v1.0.1 -m v1.0.1
git show v1.0.0:release/allowed_signers > /tmp/allowed_signers
git -c gpg.ssh.allowedSignersFile=/tmp/allowed_signers verify-tag v1.0.1   # Good "git" signature for verathos-release
git push origin v1.0.1
```

The first release is checked against the file it carries itself (`git show v1.0.0:release/allowed_signers`). Nodes
installed from code with the placeholder do not update by themselves: their operators run the installer once.

To rotate the key: create the new key and, in a release, replace the old key's line with the new one in both
copies; sign that release with the old key, and every later release with the new one. A node more than one release
behind installs the rotating release first, then the next. Whoever holds a leaked release key can publish releases
that every node trusting it installs: publish the rotating release at once and announce it; a node that installed
another release since is reinstalled with an installer holding the new key.

## Test network first

Every release and policy change runs on testnet (netuid 405) before mainnet: the same commands with
`--network test --netuid 405`. For unattended test-network operations the owner coldkey mnemonic can come from the
OS keyring (`--keyring SERVICE:USER`) instead of a wallet; mainnet operations use the wallet.
