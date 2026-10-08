# Throughput audit

Answer verification proves that an endpoint serves the right model. Scores scale with how much it serves, so that
is checked too: every endpoint declares the requests per second it serves, and once per epoch the auditor sends
every endpoint its declared rate, all at the same moment, and credits each endpoint with what it delivers.

Because every endpoint is tested at once, hardware behind several endpoints or hotkeys must serve the sum of their
rates at the same time. A GPU behind two endpoints delivers its capacity once, split between them; a pool of
machines behind many hotkeys delivers its capacity once, however it is spread. There are no GPU classes: what counts
is proven throughput, so miners choose hardware by throughput per cost.

One validator runs the tests: the auditor, which is the subnet owner's validator. It publishes signed verdicts
every full validator applies; light validators set the weights it publishes (`VALIDATING.md`). Every setting lives
in the catalog the subnet owner commits on chain.

| | Code |
|---|---|
| Schedule, policy, test sessions | `src/verathos/throughput/policy.py`, `throughput/follow.py` |
| Standard request mix | `throughput/workload.py` |
| The miner's measurement | `throughput/measure.py`, `miner/timing.py` (when), `miner/shortfall.py` (after tests fell short) |
| Admission on the miner (tests yield to other traffic) | `miner/admission.py`, `miner/app.py` |
| Auditor and load generator | `throughput/auditor.py`, `throughput/loadgen.py`, `native/loadgen` |
| Verdicts | `throughput/feed.py`, `validator/throughput.py` |
| A hotkey's signed status (its test results) | `protocol/status.py`, `validator/service.py`, `miner/status.py` |

## Declared capacity

Each catalog model defines a standard request mix (`workload`): a range of words per state and a set of questions.
Every request of the mix is unique (a random ticket id and random words from a fixed vocabulary), so nothing can be
served from a cache.

The miner measures its own capacity. It starts its service first and sends the measurement traffic through it,
the path test traffic takes (under sessions it signs itself), so the service's own limits count too. Once every
engine answers, it measures all its endpoints in the same rounds, so endpoints that share hardware compete as they
do in a test:

1. warm-up runs of 8 s at concurrency `min(16, max_concurrency)`, until every endpoint's last two runs agree within
   5% (at most 13 runs);
2. one run per concurrency step (1, 2, 4, 8, ... up to `max_concurrency`, at most 4,096); an endpoint stops climbing once its
   throughput falls below 90% of its best or its latency (concurrency / throughput) exceeds `max_latency_s`
   (default 2 s);
3. five runs at each endpoint's knee: the lowest concurrency within 3% of its best throughput.

The declared `capacity_rps` is `capacity_margin` (default 0.95) times the median of the five runs, at most what
the endpoint delivered in tests it fell short in (below). A measurement
takes a few minutes. It is stored in `state_dir/capacity.json` together with a fingerprint of what it depends on (the
endpoint's model with its pinned weight and engine revisions, engine URL and `max_concurrency`, `max_latency_s`, the
scalar fields of the engine's `GET /v1/models` document, and the request mix) and is reused until one of them
changes. The miner checks its engines every minute and measures again when one changed; `--remeasure` measures
again, for example after a hardware change. Whatever its reason, a measurement does not start from a test's selection
block until one block after its traffic: the miner reads the schedule (below) before it measures (`miner/timing.py`).
A machine that serves no endpoint yet measures at once, as no test reaches it. A measurement steps aside when the
auditor's test traffic reaches the miner (a test selected after it started), within one request and inside the
test's unmeasured warmup, and is retried after the test. An endpoint whose engine does not answer, or answers a
single request slower than `max_latency_s`, is withdrawn from the manifest; the others are served. Until its first
measurement is done, a restarted miner serves the manifest it last published.

A test measures under real conditions, and its result counts over the miner's measurement. The miner reads its test
results from the auditor's status (`GET /v1/status/<hotkey>`, its results in the auditor's last 10 tests) and acts on
them only once the status verifies: signed by the subnet owner hotkey, for its hotkey and netuid, within 5 minutes of
its clock and not before a status it already read (`protocol/status.py`). When an endpoint's last two counted tests
(confirmed, not void) were both `partial` with less delivered than the rate tested, at the rate it declares now, the
miner measures again (all its endpoints in the same rounds, as above) and declares at most the higher rate the
endpoint delivered in those two tests. The cap is kept across restarts and holds until the miner measures for another
reason (a fingerprint changed, or `--remeasure`); a further shortfall can only lower it, so a declaration falls only
on the tests' evidence and rises only on a measurement they did not cause. Tests in which the endpoint delivered less
than half the rate tested, both of them, show an outage rather than its capacity: the miner measures again but sets
no cap from them. An endpoint makes the miner measure again this way at most once every 3 epochs, never twice for the
same test.

The miner never sends an engine more requests at once than its knee concurrency; further requests wait in the
miner (up to four per engine slot).

## Schedule

Everyone derives the test from the chain, and nobody knows it before its selection block appears:

- Eligible blocks of epoch `e` are `e x epoch_blocks + 1` up to the last block from which the test traffic still
  ends inside the epoch.
- Block `b` draws a hit when `H("VERATHOS_THROUGHPUT_TEST_V1" || b || hash(b)) x (last - b + 1) < 2^256`. The
  epoch's first hit selects the test; the last eligible block always hits, so there is exactly one test per epoch,
  uniformly placed.
- Traffic starts at `start_block = selection block + lead_blocks` (default 5 blocks, about 60 s). It lasts an
  unmeasured warmup, which the auditor draws for each test from [`warmup_s`, 2 x `warmup_s`] (3-6 s), then the
  measured window of `test_s` (16 s).
- Validators hold their canaries from `start_block - 1` to `start_block + traffic_blocks + 1`, where
  `traffic_blocks` is the longest traffic in blocks, rounded up.
- A new policy applies from the next block; a test already selected stays the epoch's only test.

## A test

1. **Targets.** At the selection block the auditor takes every open endpoint listed in an accepted manifest with a
   declared capacity and a model that is not retired, that signed the auditor's identity challenge at its URL in this
   epoch or the last (`DISCOVERY.md`): test traffic goes only to machines that hold the hotkey, wherever the manifest
   points. Each is tested at its declared rate, but at most
   `rate_growth` (2) times what it delivered in its last counted test, or `retest_growth` (1.1) times while one of
   its last `regrow_after` (3) counted tests fell short (at least `min_test_share` (1/8) of the model's
   `first_test_rps`), and at most `first_test_rps` before its first test of the model it serves now. A test probes far beyond what an endpoint has
   shown it serves only while it keeps up: one that declares more than it serves is tested at about what it
   serves, so the users it also serves during the test are not held up, and its credit stays below that. Counted in first tests' worth (rate /
   the model's `first_test_rps`), a hotkey's endpoints together get at most `rate_growth` times what all its
   endpoints delivered last, renamed ones included, and at least one first test's worth; beyond that their rates are
   scaled down together. Neither an endpoint nor a hotkey can make the auditor send much more than it has shown it
   serves.
2. **Sessions.** For each endpoint the auditor signs one session: netuid, auditor, miner hotkey, endpoint, test id
   and expiry, sent as `X-Verathos-Session` (base64 of the canonical JSON) and `X-Verathos-Session-Signature`.
   Every request carries a fresh `X-Verathos-Request-Id`. The miner checks the session once and refuses a request id
   it has seen; its receipts name the auditor as caller and the request id as nonce.
3. **Traffic.** At the start block the load generator sends each endpoint its rate, open loop (sending never waits
   for answers): the warmup, then the measured window. Delivered = replies (HTTP 200, a JSON object of at most
   64 KiB) that arrive inside the window to requests sent no earlier than `max_latency_s` before it opened, per
   second of the window. An endpoint can answer inside the window only what it serves while the window is open, so
   declaring more than it serves gains nothing; an honest endpoint whose latency stays within `max_latency_s` has
   every reply counted. The miner cannot know when the window opens, so it cannot time held-back answers to it.
   Requests time out after `request_timeout_s`.
4. **Samples.** For `samples_per_test` (10) random delivered responses per endpoint the auditor checks the receipt:
   signed by the miner, binding the request and the answers, naming the auditor and the request id. Delivered is
   scaled by the share of valid receipts. Every signed answer must be well formed (its visible fields derived from
   its distribution, `GATEWAY.md`), and the auditor recomputes `answer_checks_per_test` (2) of them, picked at
   random: an answer that fails the hard check (`SCORING.md`) is a wrong answer, proven by the miner's own
   signature. The request and the response as received (answers and receipt) are kept as evidence. A wrong model
   is wrong on every request, so two recomputes per endpoint suffice.
   The auditor may recompute on a bf16 screen (`audit_screens`, about three times faster than fp32); Kev bounds
   bf16 within about 0.03 of fp32, so two bf16 runs can differ by about 0.06, and an answer the screen flags counts
   only once the fp32 reference confirms it. The recomputed answers' differences are also tallied per endpoint: over
   its last `drift_window_tests` tests, their mean must stay within the model's `mean_abs` (the bound the
   validators' canaries apply), so a lower-precision engine serving test traffic does not pass.
5. **Credit.** The tested rate times the share of what the endpoint was asked in the window that it delivered (at
   most 1). Asked: the user requests the gateway sent it in the window (answered or not, see below) and the test's
   own requests sent then, these only up to what the users' answers leave of the tested rate; delivered: the test's
   counted replies and the users' answers. If it delivered less
   than `pass_fraction` (0.95) of what it was asked, the credit is scaled by that share once more: an endpoint that
   declares twice what it serves, tested at its declaration, is credited a quarter of it, so declaring more than an
   endpoint serves never pays. A result with a
   wrong answer is `invalid` and credits nothing; an endpoint whose answers drift credits nothing.
6. **Confirmation.** Once the selection block is final with the hash it was selected by, the test is confirmed;
   otherwise it is void. A test is also void if the load generator fails, starts more than 1 s late, or the
   auditor observes the start block more than one block late. One endpoint's result is void if more than 1% of its
   sends left the generator late, if the generator's own host ran out of sockets or ports for it, or if the auditor
   could not evaluate it; what one endpoint returns affects only its own result. Recomputes stop 15 minutes after
   the traffic, and an engine that fails three times in a row is not asked again in that test, so an unreachable
   screen or reference skips recomputes instead of holding the test open. When not all fit, every endpoint's first
   recompute runs before any second one, the endpoints in a random order drawn for each test: as many endpoints as
   possible are checked, and which ones miss out cannot be known in advance. Sampled user answers are checked in
   the same order (every endpoint's first, then second).

Outcomes per endpoint: `passed` (delivered at least `pass_fraction` of what it was asked), `partial`, `invalid`,
`void`.

## Other traffic goes first

The miner admits requests from validators and gateways ahead of test requests: a test request waits while any
other request waits, and a request that finds the queue full takes the place of the newest waiting test request. A
test request is dropped once its session expires, or when the load generator has given up on it. A test never
delays users. What an endpoint answers the subnet's gateway during the test's window counts as delivered (see
Gateway answers); test requests beyond its spare capacity otherwise go undelivered, and the credit shows it.

A caller other than a gateway (the catalog's `gateways`, or one in the miner's own `gateway_hotkeys`) needs only its
canaries, so it gets one request at once per endpoint, each no larger (as canonical JSON) than the largest request
of the model's request mix (HTTP 413 beyond), at most the catalog's `scoring.validator_requests_per_epoch` per
endpoint and epoch (by default 4 x `canary_pool_size`), and while the auditor's test traffic arrives at the miner
(its last request came within 5 s) its requests wait behind the test's. Validators hold their canaries during a test, so honest
ones never wait; a validator cannot crowd a competitor's test out with priority traffic. The miner signs only well-formed
answers: an engine fault returns an error, never a signed wrong answer.

A test also leaves room for users: it sends each endpoint its tested rate less the rate of user requests the
subnet's gateway sent it over the last minute (ending 10 s before the test, as the gateway's counts arrive), at
least 10% of the tested rate and enough for the test's samples. With users at their usual rate, an endpoint is asked
for its tested rate in all, not more; it is judged against what it was asked (Credit above), so users sending less
in the window cost it nothing within their random spread (two standard deviations around its usual rate of the
minute before, scaled by the dip of all the model's users), and users it refused count against it; an endpoint whose
users went elsewhere beyond that spread (it was slow) is judged against their usual rate. Users sending more than
usual (a burst, or test traffic relayed through the gateway) cost it nothing either: the miner serves them first, so
each answer beyond what the test left of the tested rate takes the place of a test request, and the test's requests
count as asked only up to what the users' answers leave of the tested rate.

## Gateway answers

The subnet's gateway (`GATEWAY.md`) sends the auditor sampled exchanges of its user traffic: one per endpoint and
sampling window, drawn after the answers are in, signed by a gateway the catalog lists. The auditor keeps, for
endpoints it tests, those whose receipt binds the request and the answers and names that gateway (up to 8 per
endpoint and epoch; signed malformed answers always), and recomputes a random `gateway_checks_per_epoch` of them per
endpoint and epoch, between tests (screen and fp32 reference as above). A wrong answer is evidence like one in a
test; so is a receipt signing more or fewer input tokens than the reference counts (beyond 2% and 2 tokens), here
and in tests, since users' input tokens are what they are billed and set the models' emission shares. The
differences form a drift tally of their own over the last `drift_window_tests` epochs: an endpoint whose answers to
users drift beyond the model's `mean_abs` loses its credit, however correct its test answers are. A gateway answer
counts in the epoch its check finished, so an epoch's verdicts depend only on checks finished before it began. A
sampled user request in the form of the test's requests or of a canary is logged as relayed: some miner forwards
what it is sent to the gateway (`throughput.workload.in_mix_form`, `validator.canaries.canary_like`).

The gateway also reports, per endpoint and by the second each request was sent, how many user requests it sent the
endpoint and how many of them it answered (a report is taken once, even if the gateway posts it again). Miners
serve the gateway before test traffic, so what an endpoint answered the gateway for requests sent inside a test's
measured window counts as delivered, next to the test's own replies: serving users never costs credit, and an
answer held back cannot be moved into a window.

Those answers count only once checked: a miner knows when a test runs, and could answer users with junk then and
give the test its GPU. Per endpoint and test, the gateway draws one of the exchanges it sent the endpoint during the
test's traffic and posts it, marked with the test's id, once the test's sessions end. Before evaluating an endpoint
that answered users in the window, the auditor waits for these (up to two blocks after the sessions end) and
recomputes the endpoint's: a wrong answer makes the result `invalid` and is evidence like any wrong answer to a user;
a passing one lets the users' answers count. Without one (none came, or its recompute could not run then) the users'
answers still count for the pass, but the credit is at most what the endpoint's own test replies prove; the
exchange is then checked between tests like a proof.

Tests alone cannot show what users get between them: a test is known shortly before it starts (its selection block
is public) and its requests carry the auditor's session. So the gateway also reports, per endpoint and second, how
often it considered the endpoint for a user request, how often the endpoint was not ready then (busy, failing,
or answering slower than twice `max_latency_s`) or refused the request, and how long its answers took. The auditor sums these per epoch and keeps them in its database.

## Verdicts

For scoring epoch `E` the auditor publishes, signed by its hotkey, at `GET /v1/throughput/{E}` on its axon:

- the credit of every endpoint the latest confirmed test of epochs `E - credit_ttl_epochs` to `E - 1` targeted (an
  endpoint whose result there is void keeps its latest counted result from those epochs), with its model, its
  median latency in that test, whether its answers drifted, and whether it has qualified (at least
  `qualification_tests` confirmed tests with a credit), and what users got from it in epoch `E - 1`: times it was
  considered for a user request, refusals, answers and their median latency. An endpoint the latest test did not target
  has no credit, so leaving a test out does not keep an older credit;
- for every hotkey with a wrong answer in those tests or in the gateway answers it checked in those epochs, the
  evidence of one: from a test, the request and the response as received; from a user request, only the miner's
  signed receipt (user requests are not published).

They also state each model's user demand: the input tokens per second the gateways reported answered, averaged over
epochs (`demand`), and of them those users paid for once the gateways tell them apart (`paid_demand`); and the
models' emission shares that follow (`shares`, `SCORING.md`), moved by at most `max_share_step` per epoch. Once the
gateways tell paid from unpaid, they list per endpoint the input tokens of the answers it gave paying users in epoch
`E - 1`, summed over every catalog gateway's reports (`paid_served`: hotkey, endpoint, model, input tokens).

The verdicts for the current epoch and the two before are served once their tests are settled (404 otherwise);
once served, an epoch's verdicts do not change.
Validators wait up to 5 minutes for them, then use the last verdicts they hold while those still cover a test in
the window. A validator counts a credit only for the model the endpoint serves now, and at most the capacity it
declares now. It applies test evidence only if the receipt in it binds the request and the answers to the miner's
hotkey and, when it runs a reference engine for the model and that engine answers, its own reference also proves the
answers wrong; evidence from a user request only if the receipt in it was issued to one of the catalog's gateways,
taking the auditor's word for the wrongness, as for capacity. Each piece applies once.

How credits and wrong answers enter the score is in `SCORING.md`.

## What it catches

| Attempt | Result |
|---|---|
| Two endpoints on one GPU, each declaring the GPU's capacity | tested together, each delivers half and is credited a quarter of its declaration |
| A pool of machines behind many endpoints or hotkeys | tested together, the pool delivers its capacity once |
| Declaring more than the endpoint serves | credited less than it serves |
| Leaving an endpoint out of a test (withdrawing it at the selection) | no credit until the next test |
| Holding answers back into the measured window | the window opens at a time the miner does not know; requests sent more than `max_latency_s` before it never count |
| Serving test traffic from a lower-precision engine | the recomputed answers drift over a few tests and the credit goes |
| Renaming or adding endpoints to make the auditor send more | a hotkey's test rate grows only from what its endpoints delivered |
| Fast junk responses | receipts that do not bind scale delivered down; a signed wrong answer invalidates the hotkey |
| Answering users with junk during a test, giving the test the GPU | the gateway's sample of the endpoint's answers to users in the test is recomputed before they count; without one the credit is at most its own test replies |
| Responses built to break the auditor | size and nesting limits; each endpoint is evaluated on its own |
| Flooding a competitor's endpoint with validator traffic during its test | one request at once per validator and endpoint, no larger than the model's standard requests, and behind the test's requests while they arrive |
| Flooding it with gateway traffic during its test, or a miner refusing users then | what an endpoint answers the gateway inside the test's window counts as delivered, and displaces the test's requests one for one; refused users count as asked |
| Caching or sharing answers | every request is unique |
| A staker using miners as a free inference service through validator requests | miners answer only validators with the catalog's `validator_min_stake`, each within a budget of requests per endpoint and epoch that canaries and identity challenges need |
| Listing someone else's server as an endpoint, to aim the test's traffic at it | only endpoints that signed the auditor's challenge at their URL are tested; that server gets one challenge per epoch |

What a test proves is that an endpoint delivers its declared rate on about 60 s notice: test traffic is
recognisable by its session, so hardware used for other work between tests could be freed for the test.

## Policy

`throughput_audit` in the catalog:

| Setting | Default | Meaning |
|---|---|---|
| `mode` | `enforce` | `observe`: validators count declared capacities and apply no evidence; tests still run |
| `lead_blocks` | 5 | blocks from the selection block to the start of the traffic |
| `max_latency_s` | 2 | highest latency at which a miner's measurement counts throughput |
| `warmup_s`, `test_s` | 3, 16 | shortest unmeasured traffic (each test draws it from [`warmup_s`, 2 x `warmup_s`]) and the measured window, seconds; `warmup_s` is at least `max_latency_s` |
| `request_timeout_s` | 10 | per-request timeout of the load generator |
| `pass_fraction` | 0.95 | share of the tested rate below which the credit is scaled down once more |
| `samples_per_test` | 10 | responses per endpoint whose receipts are checked |
| `answer_checks_per_test` | 2 | of those, answers the reference recomputes |
| `capacity_margin` | 0.95 | share of the measured median a miner declares |
| `qualification_tests` | 1 | confirmed tests with a credit before an endpoint is scored |
| `credit_ttl_epochs` | 2 | how many closed epochs back a test still counts |
| `rate_growth` | 2.0 | tested rate at most this times the last delivered rate |
| `retest_growth` | 1.1 | ... at most this times it while one of the last `regrow_after` counted tests fell short |
| `regrow_after` | 3 | passed tests in a row after which the tested rate may grow by `rate_growth` again |
| `min_test_share` | 0.125 | ... but at least this share of the model's `first_test_rps` |
| `drift_window_tests` | 8 | tests (and, for gateway answers, epochs) over which an endpoint's recomputed answers must stay within the model's `mean_abs` |
| `gateway_checks_per_epoch` | 2 | sampled gateway answers the auditor recomputes per endpoint and epoch (0: none) |

Per model: `workload` (the request mix) and `first_test_rps`.

## Running the auditor

The auditor needs:
- the load generator: `verathos-build-loadgen` (Rust) builds `native/loadgen` and installs it next to the code;
- the answer recomputes: `answer_checks_per_test` per endpoint and test, within 15 minutes of the traffic. On an
  A100 with Kev-4B's request mix, fp32 recomputes about 2 per second and bf16 about 6, so a bf16 screen covers
  1,000 endpoints in about 6 minutes, with the fp32 reference (CPU is enough) confirming only what it flags. The
  screen can run on another machine behind https with a key; while it is unreachable, recomputes are skipped and
  receipts are still checked;
- a public IP and port for its service, and an uplink for the whole network's test traffic in one burst: about
  6 KB per request, so 1,000 endpoints at 20,000 requests/s take about 1 Gbit/s, and 2-3 CPU cores.

Its database is `state_dir/throughput.sqlite` (tables `tests`, `results`, `samples` (gateway answers), `verdicts`
(as served), `demand` (users' input tokens per model) and `users` (what users got from each endpoint), the last 30
epochs, more than the last 10 tests a hotkey's status lists). A catalog with another `epoch_blocks` renumbers it (a
test by its selection block, the rest by the first block of its epoch; the verdicts served for the old epochs are
dropped), so credits, test rates, demand and wrong answers carry on.
