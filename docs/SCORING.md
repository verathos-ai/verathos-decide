# Answer verification and scoring

An epoch is `epoch_blocks` blocks (catalog, default 180, about 36 minutes), independent of the subnet tempo: every
validator scores and sets weights once per epoch, within the chain's weight rate limit (100 blocks on SN96). The
settings below are the catalog's `scoring` section and each model's `tolerance`, so every validator applies the
same ones.

## Why answers can be checked

A decision model answers a typed request (a state and a set of questions) with a probability distribution per
question, from one forward pass. That pass is nearly deterministic across GPUs, Apple Silicon and CPU, so the full
probability vector fingerprints the exact weights. Measured on Kev-4B/9B (RTX 4090, RTX 5090, M1 Max, CPU fp32; 420
probabilities per run): honest pairs differed by at most 0.017 per probability, with a mean of 0.0014-0.0023; Kev-9B
answering for Kev-4B had a mean difference of 0.084 and exceeded 0.10 on every request.

The validator therefore recomputes a sample of requests itself, in fp32, and compares.

## Canaries

Each epoch the validator derives a pool of `canary_pool_size` (16) requests per model from a private secret, the epoch
and the model (support tickets and code-review requests with generated details and a varying subset of questions).
Miners cannot predict them. Canaries are signed by the validator's hotkey, so a miner can tell validator traffic from
gateway traffic; the checks here cover what validators are served, and the throughput auditor checks a sample of the
answers the subnet's gateway receives (`GATEWAY.md`). For a model that reads images, the catalog's `images.canary_share`
(a quarter) of the pool carries an image the validator draws: one to six shapes in named colours, or a small bar chart,
sometimes with a word or number, as a canonical PNG of 256 to 512 px a side, asked two or three questions about what it
shows (a shape of a colour, how many, the largest or the tallest, the background, a word); its state is an ordinary
canary's. An endpoint that answers without reading the images fails them. The reference engine answers each canary once;
every endpoint of that model receives its own random subset of `canaries_per_endpoint` (10) canaries, spread over the
first `canary_spread_fraction` (0.6) of the epoch. Canaries wait while a throughput test runs (`THROUGHPUT_AUDIT.md`).
If the reference engine cannot answer, the model's endpoints are not audited that epoch and keep their score, for at
most 3 epochs in a row. A validator running its references one at a time checks the models whose references fit in the
epoch's reference time, drawn from its secret so that each of `k` models is checked at least once every `2k` epochs, and
takes the others' results from its source (`VALIDATING.md`).

Every request is signed by the validator and bound to the miner and a nonce. Every answer comes with a receipt signed
by the miner hotkey that binds the request, the answers, the endpoint, the model, the caller, the nonce and the
netuid, and names the catalog and mechanism the miner serves under (not checked against the validator's, so answers
given while a new catalog is adopted stay valid). A request the endpoint declines (an error status, a reply that is
not JSON or too large, or no answer within the audit policy's `request_timeout_s`, 10 s) counts as unanswered. A
JSON reply without a valid receipt, or whose answers are malformed (a probability that is not a number in [0, 1]),
fails the request as a hard failure.

## Comparison

Per request (hard check). The request fails if:
- question ids or option keys differ from the reference;
- any probability differs from the reference by more than `hard_abs` (default 0.10); or
- the top answer changes where the reference was confident (its margin over the runner-up is at least
  `flip_margin`, default 0.20).

Per endpoint (statistical check). Over the endpoint's last `statistical_window_epochs` (3) audited epochs, the mean
absolute difference over all compared probabilities must stay below `mean_abs` (default 0.005) once at least
`min_probabilities` (default 100) have been compared. This catches lower-precision substitutes that stay under the
hard bound on single requests.

Tolerances are per model and set in the catalog.

## Score

Per endpoint and epoch:
- zero, with probation for every endpoint of the hotkey and their averages reset to zero, if a request failed the
  hard check, the statistical check failed, or the throughput auditor proved a wrong answer the hotkey signed in a
  test or to the subnet's gateway (see `THROUGHPUT_AUDIT.md`);
- zero if the endpoint did not prove control at its URL by the identity challenge (it then gets no canaries,
  `DISCOVERY.md`), answered nothing, or has not yet proven capacity in a throughput test;
- otherwise its share of its model's pool (see Models): the model's emission share, split among the model's scored
  endpoints in proportion to `proven capacity x answered / sent x availability to users x latency factor`. The
  proven capacity is the auditor's credit in requests/s of the model's standard mix, at most what the endpoint
  declares now (in `observe` mode the declaration itself, up to the model's `first_test_rps`). Availability to users
  is `(ready + 20) / (offered + 20)` over the times the gateways considered the hotkey's endpoints of the model for a
  user request in the epoch before (not ready: busy, failing, or with the latest answer or timed-out attempt at
  least twice the latency bound; requests it answered busy or failed, or refused while another miner answered,
  count as not ready too; see `GATEWAY.md`), with `user_offers_prior` (20) ready offers to start from, so a few
  offers move it little: capacity shown in tests but kept from users does not count, and neither does keeping users
  waiting. The latency factor is `sqrt(median / own latency)`, capped to [0.7, 1.3]. An endpoint's latency is the
  median latency of its hotkey's answers to the model's users once they gave `user_latency_min_answers` (50) in the
  epoch before, compared with the median over the model's scored endpoints measured the same way, and the factor is
  then scaled by `2 - latency / max_latency_s` between `max_latency_s` and twice it (0 beyond): answering users just
  under the gateway's slow bound, which keeps an endpoint ready on part of its GPU, earns next to nothing; else its median
  latency in its throughput test, compared with those measured so, and then at most 1 while users measure others of
  the model. Tests are known shortly ahead and their requests are recognisable, so what users get is measured on
  users, per hotkey and model, so splitting capacity over endpoints hides none of it. Every validator reads the same
  figures from the auditor's verdicts. Each endpoint weighs in a median by its proven capacity, so splitting capacity
  over more endpoints does not move it. In `observe` mode there is no latency factor and no availability to users.
  An epoch's scores sum to 1 when every active model has scored endpoints; a model whose endpoints all score zero
  leaves its share to the others, as the weights are normalised ([Weights](#weights)).

## Models

Each `active` model has a pool: a share of the epoch's emission, of the part the scores pay (the miner share,
[Weights](#weights)). Its endpoints split it as above; `candidate` models have none. The pools follow what users pay
for each model through the subnet's gateway:
- demand: the input tokens of the users' requests per model (as the miners signed them, the tokens of a request's
  images included), once the gateways tell paid from unpaid only those users paid for, averaged over epochs
  (`demand_ema_alpha`, 0.1: about six hours);
- revenue: demand x the model's `price` (USD per million input tokens, what users pay);
- share: the model's revenue share, blended toward the catalog's `anchor_share` while the total demand is below
  `demand_full_tokens_s` (10,000 input tokens/s), and at least `min_model_share` (0.1) for every active model;
- the auditor publishes the shares with its verdicts, moved from the previous epoch's toward these by at most
  `max_share_step` (0.05) for any model, so a burst of demand shifts the emission over hours, not at once.
  Validators score with the shares published for the models they score, else from the published demand.

Without demand the pools are the anchor shares, each at least `min_model_share`: in the shipped catalog a third each
for kev-4b, kev-9b and clef-flash, all three active from the start. Miners choose which models to serve: a model with
few endpoints splits its pool among few, and a crowded one among many, so capacity moves to where users pay. The
miner setup chooses so by default: per GPU, the model that pays best by the published shares and the capacity proven
on each model (`MINING.md`). No conversion between models is needed: what a request costs on a given GPU is the
miner's to weigh.

Probation lasts `probation_epochs` (6) epochs, the failing one included, whatever the later results; a new failure
restarts the count, and the hotkey's statistical windows start over. An endpoint a hotkey adds while one of its
endpoints is on probation inherits the remaining probation, so renaming an endpoint does not escape it. Scores enter an exponential moving average (`ema_alpha`,
0.2); endpoints that disappear decay toward zero and are forgotten once negligible.

## Weights

A UID's value is the sum of its endpoints' EMAs, normalised over all UIDs. Endpoints whose UID now belongs to another
hotkey are ignored. The miners' values are then scaled to sum to the epoch's miner share, the catalog's
`scoring.miner_share` (0.15 in the shipped catalog), the weights of the [usage pool](#usage-pool) and of
[referrals](#referrals) are added, and the subnet owner's UID gets the rest; the weights are set on the catalog's
mechanism (`SUBNET_ADMIN.md`) once per epoch. The chain pays the subnet owner hotkey no incentive: it burns it, or
recycles it under the subnet's `RecycleOrBurn` setting (`SUBNET_ADMIN.md`). The share is a constant or moves linearly
from one value to another over a number of epochs; each epoch's comes from the catalog and the epoch number alone,
rounded to 6 decimals, so every validator sets the same split. If no endpoint has a positive average and neither pool
pays anybody, all weight goes to the subnet owner's UID. If the owner hotkey holds no UID, the miners' weights take
what the pools leave (the pools' are never scaled up), and with no miner weight no weights are set.

Everything above is what a full validator does. A full validator running its references one at a time scores the
endpoints of the models it does not check in an epoch with the source's published results for them: the score the
source gave an endpoint enters its average as one it computed (zero on the source's probation, a failure as its own
failure, held where the source could not check it), so its weights are one vector from both. A light validator
(`VALIDATING.md`) scores nothing: it sets the weights its source (the auditor, or the validator `source_hotkey`
names) publishes for the epoch (`GET /v1/network`, signed), exactly as the source sets them.

## Usage pool

The catalog's `scoring.usage_pool` (absent in the shipped catalog: off) pays the hotkeys whose endpoints answered
paying users, on top of their scores, out of what the miner share and the referral pool's share leave. Time runs in
periods of `period_epochs` epochs (default 40, about a day); a period's pool is sized and recorded when it ends and paid
over the epochs of the next.

- **Size.** The period's paid usage at the models' `price`, counted when it is spent, in alpha at TAO's price in USD
  when the period closed and the subnet's moving alpha price at the period's end, times `fraction`, divided by the
  mechanism's miner emission over a period, at most 1, rounded to 6 decimals: the period's usage share.
- **Cap.** Each epoch of the next period pays the usage share, at most `1 - miner_share` of that epoch less the
  referral pool's `share` (kept for referrals whether or not they fill it); beyond it nothing more is paid.
- **Split.** A hotkey gets `usage share x what users paid for its endpoints' answers in the period / what they paid in
  all`: its paid input tokens at each model's `price`. Paid input tokens are those of answers to accounts paying with
  bought credit (`GATEWAY.md`), as the miners signed them in their receipts; free credit, the demo and admin requests
  count none. Hotkeys the catalog's `excluded` lists, by hotkey or by coldkey, and hotkeys that hold no UID now get
  nothing.

| Setting | Default | Range | Meaning |
|---|---|---|---|
| `fraction` | 0.6 | (0, 0.9] | share of the period's paid usage, in alpha, the pool pays |
| `period_epochs` | 40 | 1 to 400 | epochs per period |
| `excluded` | [] | at most 256 | hotkeys and coldkeys paid nothing |

With a referral pool, `fraction` and its `payout_per_usd` add up to at most 0.9: a referred USD earns its referrer up to
`payout_per_usd` and, spent, the hotkeys that answer it up to `fraction`, so referring and serving oneself loses at
least a tenth of every USD.

The subnet owner's UID keeps whatever the scores and the pools leave: the parts of excluded and departed hotkeys,
and a period's whole pool if nobody paid in it. Without the pool, or without paid usage, the weights are exactly what
they are without it.

The pool follows what users paid, not what is done with the money: the alpha it pays comes from the emission the
owner's UID would burn. The operator's rule turns the revenue into buybacks of the subnet's alpha, most of it burned
and the platform's fee locked, which takes at least as much alpha out of supply as the pool pays; the website's
economics page lists each buyback with its transaction next to the revenue.

The auditor records each period (`THROUGHPUT_AUDIT.md`): the paid input tokens per hotkey and model, the revenue and
the prices, and the miners' emission at the period's end block, and states the record, signed, with every verdicts
document of the next period. Every full validator sets the same weights from it: it recomputes the record's figures
from its inputs, and checks what its own node still holds of the end block's emission and price (a pruned node keeps
about 256 blocks; the rest it takes on the auditor's word, as it takes capacity). A record that fails either pays
nothing at that validator for the period. A light validator sets the weights its source publishes, usage weights
included. The network document states the epoch's `usage_share`, the share the usage weights sum to, and each hotkey's
status its `usage_weight`.

## Referrals

The catalog's `scoring.referral_pool` (absent in the shipped catalog: off) pays referrers, hotkeys whose referral codes
brought accounts that buy credit (`GATEWAY.md`, [Referrals](GATEWAY.md#referrals)), on top of their scores, out of what
the miner share leaves; with a usage pool, its `share` is kept for it. Time runs in periods of `period_epochs` epochs
(default 40, about a day); a period's record is closed when it ends and paid over the epochs of the next.

- **What counts.** The top-ups of a code's referred accounts, at the USD they were credited at, less their refunds,
  each in the period it reaches the auditor: from the referral on, for `term_days`, up to `account_cap_usd` of one
  account's top-ups within any window's length, and not once the code is revoked. Grants, adjustments and spending
  never count, nor a top-up credited while the catalog has no referral pool.
- **Window.** A referrer's sum `R` is its codes' over the last `window_periods` periods (the period closed included),
  at least 0: a refund offsets its other referred top-ups in the window.
- **Cap.** `payout_per_usd x R / (window_periods x p x E)`: the share of a period's miner emission that pays
  `payout_per_usd / window_periods` of `R`, with `E` the mechanism's miner emission over a period in alpha and `p` the
  alpha's price in USD (TAO in USD when the period closed, times the subnet's moving alpha price at its end).
- **Share.** The caps' sum, at most `share`: each referrer gets its cap, or, when the caps exceed `share`, its part of
  `share` in proportion to its cap. Each epoch of the next period pays it, at most what the epoch's miner share leaves
  (every referrer scaled down alike).
- **Who is paid.** Any hotkey that holds a UID, serving or not, except the subnet owner's (its hotkey, and the hotkeys
  of its coldkey) and those the catalog's `excluded` names by hotkey or coldkey (owner-run miners); with
  `serving_only`, only hotkeys with a positive score weight in the epoch. A referrer without a UID is not paid; its
  codes keep counting, and pay resumes once it holds a UID again or its codes move to a registered hotkey of its
  coldkey.

A referred USD counts in at most `window_periods` periods, at most `payout_per_usd / window_periods` of it in each: it
earns at most `payout_per_usd` USD of emission over its window, whatever its timing, the pool's fill, or how it is
split over accounts and codes. A referrer topping up its own account with its own code gets back at most that per
dollar, while the credit it bought can be neither withdrawn nor transferred. The pool is a ceiling, not a spend: the
subnet owner's UID keeps what it does not pay, and the parts of referrers not paid.

| Setting | Default | Range | Meaning |
|---|---|---|---|
| `share` | 0.10 | (0, 0.5] | at most this share of the mechanism's miner emission per period |
| `payout_per_usd` | 0.3 | (0, 0.9] | at most this many USD of emission per referred USD, over its window; with a usage pool, at most 0.9 less its `fraction` |
| `window_periods` | 30 | 1 to 90 | periods a top-up counts in |
| `period_epochs` | 40 | 1 to 400 | epochs per period |
| `term_days` | 365 | 30 to 1,095, or null | days after its referral an account's top-ups count; null: for life |
| `account_cap_usd` | 10,000 | 1,000 to 100,000, or null | at most this much of one account's top-ups counts within a window's length |
| `serving_only` | false | | pay only referrers with a positive score weight in the epoch |
| `excluded` | [] | at most 256 | hotkeys and coldkeys paid nothing |

The auditor records each period (`THROUGHPUT_AUDIT.md`) and states the record, signed, with every verdicts document of
the next period. A full validator recomputes the record's figures, checks that it states the catalog's
`payout_per_usd`, `share` and `window_periods`, and checks what its own node holds of the period's end block (the
emission, the moving alpha price, the subnet owner, each referrer's coldkey). It also compares the record with the
last one it accepted, while their windows overlap: every code's sums in the periods both hold must be as stated there,
and a code now on another hotkey must not have left a hotkey that belongs, at the record's end block, to another
coldkey than its new one. A record that fails pays nothing at that validator for the period. A light validator sets the
weights its source publishes, referral weights included. The network document states the epoch's `referral_share` and
the record it pays (`referral`), each hotkey's `serving_weight`, `usage_weight` and `referral_weight`, and lists
referrers without endpoints too, with their codes' sums (`VALIDATING.md`).
