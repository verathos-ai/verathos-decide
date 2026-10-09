"""References one at a time: the epoch's schedule within the reference time, the random coverage of the models, the
source's results for the models not checked, and the reference processes the validator starts and stops."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from bittensor_wallet import Keypair

from tests.mock_engine import MockEngine
from tests.tlsfake import Presenting, pin
from verathos import catalog
from verathos.miner.app import caller_request_bytes, create_app
from verathos.miner.config import MinerConfig
from verathos.miner.engine import EngineError
from verathos.protocol.manifest import Endpoint, Manifest, SignedManifest, commitment_value
from verathos.protocol.status import NETWORK_DOMAIN, NetworkStatus, sign_document
from verathos.throughput.policy import BLOCK_TIME_S
from verathos.validator import fallback
from verathos.validator.__main__ import Proven, coverage_line, score_epoch, weights_to_set
from verathos.validator.audit import EndpointResult, audit_epoch
from verathos.validator.canaries import canary_pool
from verathos.validator.compare import RequestComparison, Tolerance
from verathos.validator.config import ReferenceEngine, ValidatorConfig
from verathos.validator.discovery import DiscoveredEndpoint, Neuron, discover
from verathos.validator.light import Refused
from verathos.validator.scoring import MAX_HELD_EPOCHS, ScoreState, weights
from verathos.validator.sequential import (
    OnDemand,
    Runner,
    Sequence,
    Slot,
    StartFailed,
    Timings,
    coverage_order,
    fingerprint,
    fitting,
    overdue,
    waited_after,
)
from verathos.validator.status import StatusBoard, build

ROOT = Path(__file__).resolve().parents[1]
AUDITOR_PIN = "ab" * 32  # the auditor's certificate, as the catalog names it
CAT = catalog.Catalog.parse({**catalog.load().document(), "auditor": {"tls": AUDITOR_PIN}})
SECRET = b"k" * 32
OWNER = "owner-hotkey"


def kp() -> Keypair:
    return Keypair.create_from_mnemonic(Keypair.generate_mnemonic())


def model(model_id: str) -> catalog.CatalogModel:
    """A catalog model of that id (the shipped catalog's, else a copy of kev-4b)."""
    return CAT.get(model_id) or CAT.get("kev-4b").model_copy(update={"model_id": model_id})


class Answering(MockEngine):
    """A mock engine taking ``answer_s`` per request."""

    def __init__(self, model_id: str, answer_s: float) -> None:
        super().__init__(model_id)
        self.answer_s = answer_s

    async def answer(self, request):
        await asyncio.sleep(self.answer_s)
        return await super().answer(request)


FAKE = ReferenceEngine(url="http://127.0.0.1:1", command=["serve"])


class Fake(OnDemand):
    """A reference whose start, answers and stop take the given times; counts how many run at once."""

    def __init__(self, model_id: str, *, start_s: float, answer_s: float, stop_s: float, slot: Slot,
                 timings: Timings, crash: bool = False) -> None:
        super().__init__(model(model_id), FAKE, slot=slot, timings=timings)
        self.start_s, self.answer_s, self.stop_s, self.crash = start_s, answer_s, stop_s, crash
        self.up = False
        self.board: dict = slot.__dict__.setdefault("board", {"running": 0, "peak": 0, "events": []})

    def _count(self, change: int) -> None:
        self.board["running"] += change
        self.board["peak"] = max(self.board["peak"], self.board["running"])

    async def _launch(self, deadline):
        self.up = True
        self._count(1)
        self.board["events"].append(("start", self.model_id))
        wait = self.start_s if deadline is None else min(self.start_s, max(deadline - time.monotonic(), 0.0))
        try:
            await asyncio.sleep(wait)
        except BaseException:
            self._kill()
            raise
        if self.crash:
            self._kill()
            raise StartFailed("it exited with code 1")
        if wait < self.start_s:
            self._kill()
            raise StartFailed("it did not answer in time", timed_out=True)
        return Answering(self.model_id, self.answer_s)

    async def _halt(self):
        if self.up:
            await asyncio.sleep(self.stop_s)
            self.timings.record(self.model_id, "stop_s", self.stop_s)
            self.up = False
            self._count(-1)
            self.board["events"].append(("stop", self.model_id))

    def _kill(self):
        if self.up:
            self.up = False
            self._count(-1)


def seeded_timings(path: Path | None = None, **times: tuple[float, float, float]) -> Timings:
    timings = Timings(path)
    for m, (start, answer, stop) in times.items():
        timings.setup(m, fingerprint(FAKE))
        for kind, value in zip(("start_s", "answer_s", "stop_s"), (start, answer, stop)):
            timings.record(m, kind, value)
    return timings


TIMES = {"kev-4b": (0.2, 0.02, 0.04), "kev-9b": (0.4, 0.04, 0.04), "clef-flash": (0.4, 0.04, 0.04)}


def references(slot: Slot, timings: Timings, times=TIMES, **changes) -> dict[str, Fake]:
    return {m: Fake(m, start_s=t[0], answer_s=t[1], stop_s=t[2], slot=slot, timings=timings, **changes.get(m, {}))
            for m, t in times.items()}


def pools(models, size: int = 8, epoch: int = 7) -> dict:
    return {m: canary_pool(SECRET, epoch, m, size) for m in models}


# ---- the schedule within an epoch's reference time ----

def test_references_run_one_at_a_time_and_the_models_that_fit_in_the_reference_time_are_checked(tmp_path) -> None:
    """Each taken model starts, answers its canaries and stops before the next starts; the models are taken in order
    while their start, a canary per endpoint slot and stop fit in what is left; the first computes its whole pool
    while the time the next one needs stays its own; the phase ends by its deadline."""
    timings = seeded_timings(tmp_path / "references.json", **{m: (s, a, st) for m, (s, a, st) in TIMES.items()})
    slot = Slot()
    refs = references(slot, timings)
    order = ["kev-4b", "kev-9b", "clef-flash"]
    sequence = Sequence(refs, order, overdue=set(), per_endpoint=4, timings=timings)
    # kev-4b 0.2 + 4 x 0.02 + 0.04 = 0.32 s, kev-9b and clef-flash 0.4 + 4 x 0.04 + 0.04 = 0.6 s each
    assert [sequence.cost(m) for m in order] == pytest.approx([0.32, 0.6, 0.6])
    budget = 1.1
    assert sequence.expected(order, budget) == ["kev-4b", "kev-9b"]

    async def run():
        started = time.monotonic()
        got = await sequence.answers(pools(order), started + budget)
        return got, time.monotonic() - started

    got, took = asyncio.run(run())
    assert took <= budget + 0.05
    assert slot.board["peak"] == 1 and slot.running is None
    assert slot.board["events"] == [("start", "kev-4b"), ("stop", "kev-4b"), ("start", "kev-9b"), ("stop", "kev-9b")]
    assert sequence.tried == sequence.checked == ["kev-4b", "kev-9b"] and set(got) == {"kev-4b", "kev-9b"}
    assert sequence.why == {"clef-flash": "not drawn"}
    assert all(a is not None for a in got["kev-4b"])  # the whole pool, kev-9b's time reserved
    assert sum(a is not None for a in got["kev-9b"]) >= 4  # at least a canary per endpoint slot

    # what was measured is kept for the next epochs
    again = Timings(tmp_path / "references.json")
    assert again.get("kev-4b", "answer_s") == pytest.approx(0.02, abs=0.01)
    assert again.get("kev-9b", "start_s") == pytest.approx(0.4, abs=0.05)
    assert again.get("clef-flash", "start_s") == 0.4  # seeded, not run


def test_the_times_are_learned_from_the_first_epoch_and_forgotten_when_the_reference_changes(tmp_path) -> None:
    timings = Timings(tmp_path / "references.json")
    slot = Slot()
    refs = references(slot, timings, times={"kev-4b": (0.1, 0.01, 0.02), "kev-9b": (0.2, 0.02, 0.02)})
    sequence = Sequence(refs, ["kev-9b", "kev-4b"], overdue=set(), per_endpoint=4, timings=timings)
    assert sequence.expected(["kev-4b", "kev-9b"], 0.01) == ["kev-9b", "kev-4b"]  # nothing known: both planned
    asyncio.run(sequence.answers(pools(["kev-4b", "kev-9b"]), time.monotonic() + 5.0))
    assert sequence.checked == ["kev-9b", "kev-4b"]
    assert timings.cost("kev-9b", 4) == pytest.approx(0.2 + 4 * 0.02 + 0.02, abs=0.03)
    kept = Timings(tmp_path / "references.json")
    assert kept.cost("kev-4b", 4) == pytest.approx(timings.cost("kev-4b", 4))
    # another command (a new engine, another device): learned again
    Fake("kev-4b", start_s=0, answer_s=0, stop_s=0, slot=slot, timings=kept)  # same setup: kept
    assert kept.get("kev-4b", "start_s") is not None
    OnDemand(model("kev-4b"), ReferenceEngine(url="http://127.0.0.1:1", command=["other"]), slot=slot, timings=kept)
    assert kept.get("kev-4b", "start_s") is None and kept.cost("kev-4b", 4) is None


def test_an_overdue_model_runs_first_and_alone_when_it_does_not_fit_and_a_failed_start_frees_the_time() -> None:
    timings = seeded_timings(**{"kev-4b": (0.1, 0.01, 0.02), "kev-9b": (0.3, 0.1, 0.02)})
    slot = Slot()
    refs = references(slot, timings, times={"kev-4b": (0.1, 0.01, 0.02), "kev-9b": (0.3, 0.1, 0.02)})
    waited = {"kev-9b": 2, "kev-4b": 0}  # two models: overdue after two epochs without a run
    order = coverage_order(["kev-4b", "kev-9b"], waited, SECRET, 9)
    assert order[0] == "kev-9b" and overdue(["kev-4b", "kev-9b"], waited) == {"kev-9b"}
    # kev-9b needs 0.3 + 4 x 0.1 + 0.02 = 0.72 s, more than the 0.6 s there are: it runs alone, with what it computes
    sequence = Sequence(refs, order, overdue={"kev-9b"}, per_endpoint=4, timings=timings)
    got = asyncio.run(sequence.answers(pools(["kev-4b", "kev-9b"]), time.monotonic() + 0.6))
    assert sequence.checked == ["kev-9b"] and 1 <= sum(a is not None for a in got["kev-9b"]) < 4
    assert sequence.why == {"kev-4b": "not drawn"}
    # not overdue, a model that does not fit is skipped and the next one runs
    sequence = Sequence(refs, ["kev-9b", "kev-4b"], overdue=set(), per_endpoint=4, timings=timings)
    asyncio.run(sequence.answers(pools(["kev-4b", "kev-9b"]), time.monotonic() + 0.6))
    assert sequence.checked == ["kev-4b"] and sequence.why == {"kev-9b": "not drawn"}
    # a reference that does not start leaves its time to the next
    refs = references(slot, timings, times={"kev-4b": (0.1, 0.01, 0.02), "kev-9b": (0.05, 0.1, 0.02)},
                      **{"kev-9b": {"crash": True}})
    timings.models["kev-9b"] = {"setup": timings.models["kev-9b"]["setup"]}  # nothing known about it
    sequence = Sequence(refs, ["kev-9b", "kev-4b"], overdue=set(), per_endpoint=4, timings=timings)
    got = asyncio.run(sequence.answers(pools(["kev-4b", "kev-9b"]), time.monotonic() + 0.6))
    assert sequence.tried == ["kev-9b", "kev-4b"] and sequence.checked == ["kev-4b"]
    assert sequence.why == {"kev-9b": "its reference did not start"} and slot.board["running"] == 0


def test_a_start_that_runs_out_of_time_is_stopped_and_counted_as_at_least_that_long() -> None:
    timings = Timings()
    slot = Slot()
    refs = references(slot, timings, times={"kev-4b": (5.0, 0.01, 0.02)})
    sequence = Sequence(refs, ["kev-4b"], overdue=set(), per_endpoint=4, timings=timings)
    asyncio.run(sequence.answers(pools(["kev-4b"]), time.monotonic() + 0.3))
    assert sequence.why == {"kev-4b": "its reference did not start"} and slot.board["running"] == 0
    assert timings.get("kev-4b", "start_s") >= 0.29


# ---- coverage ----

def simulate(models: list[str], cost: dict[str, float], budget: float, secret: bytes, epochs: int) -> list[list[str]]:
    """The models each epoch plans to run, as the validator decides them (all its references starting as measured)."""
    waited: dict[str, int] = {}
    out = []
    for epoch in range(epochs):
        order = coverage_order(models, waited, secret, epoch)
        late = overdue(models, waited)
        taken = fitting(order, cost.get, budget, order[0] if order[0] in late else None)
        out.append(taken)
        waited = waited_after(waited, models, taken, models)
    return out


def longest_wait(runs: list[list[str]], model: str) -> int:
    longest = current = 0
    for taken in runs:
        current = 0 if model in taken else current + 1
        longest = max(longest, current)
    return longest


def test_every_model_is_checked_at_least_once_every_2k_epochs_and_equally_often() -> None:
    cases = [
        (["a", "b"], {"a": 95.0, "b": 320.0}, 389.0),  # one of two per epoch
        (["a", "b", "c"], {"a": 95.0, "b": 186.0, "c": 190.0}, 389.0),  # two of three
        (["a", "b", "c"], {"a": 300.0, "b": 300.0, "c": 300.0}, 389.0),  # one of three
        (["a", "b", "c", "d"], {"a": 150.0, "b": 150.0, "c": 150.0, "d": 150.0}, 389.0),  # two of four
        (["a", "b", "c"], {"a": 500.0, "b": 100.0, "c": 100.0}, 389.0),  # a alone exceeds the time
    ]
    for models, cost, budget in cases:
        runs = simulate(models, cost, budget, SECRET, 3000)
        k = len(models)
        for m in models:
            assert longest_wait(runs, m) <= 2 * k - 1, (models, m)
        share = {m: sum(m in taken for taken in runs) / len(runs) for m in models}
        if len(set(cost.values())) == 1:  # equal times: equally often
            assert max(share.values()) - min(share.values()) < 0.04, share
        assert min(share.values()) >= 1 / (2 * k), share
    # a model that alone exceeds the time runs only when overdue, alone: about once every k + 1 epochs
    runs = simulate(["a", "b", "c"], {"a": 500.0, "b": 100.0, "c": 100.0}, 389.0, SECRET, 3000)
    assert all(taken == ["a"] for taken in runs if "a" in taken)
    assert 0.2 < sum("a" in taken for taken in runs) / len(runs) < 0.3


def test_which_models_are_checked_cannot_be_told_without_the_secret() -> None:
    models, cost, budget = ["a", "b", "c"], {"a": 300.0, "b": 300.0, "c": 300.0}, 389.0
    mine, other = (simulate(models, cost, budget, secret, 600) for secret in (SECRET, b"o" * 32))
    assert mine == simulate(models, cost, budget, SECRET, 600)  # the validator's own, reproducible
    agree = sum(a == b for a, b in zip(mine, other)) / len(mine)
    assert agree < 0.6  # chance agreement, not the same sequence
    # before a model is overdue, what it waited only weighs the draw: from the same state every model can come first
    firsts = [coverage_order(models, {"a": 1, "b": 1, "c": 0}, SECRET, epoch)[0] for epoch in range(900)]
    counts = {m: firsts.count(m) / len(firsts) for m in models}
    assert counts["a"] == pytest.approx(0.4, abs=0.06) and counts["b"] == pytest.approx(0.4, abs=0.06)
    assert counts["c"] == pytest.approx(0.2, abs=0.06)  # weights 2, 2, 1


def test_the_epochs_waited_follow_the_models_run() -> None:
    waited = waited_after({"a": 3, "gone": 2}, ["a", "b", "c"], ["b"], ["a", "b", "c"])
    assert waited == {"a": 4, "b": 0, "c": 1}
    state = ScoreState(waited=waited)
    assert ScoreState(**json.loads(json.dumps(state.__dict__))).waited == waited


# ---- canaries only for the models checked ----

def miner(host: str, keypair: Keypair, validator: Keypair, model_id: str):
    config = MinerConfig.model_validate({
        "netuid": 405, "mechid": 1, "wallet_name": "w", "hotkey_name": "h", "public_host": host, "port": 80,
        "endpoints": [{"endpoint_id": "gpu0", "model_id": model_id, "engine_url": "http://127.0.0.1:1",
                       "max_concurrency": 8}]})
    signed = SignedManifest.create(Manifest(
        netuid=405, mechid=1, hotkey=keypair.ss58_address, catalog_digest=CAT.digest(),
        endpoints=[Endpoint(endpoint_id="gpu0", url=config.endpoint_url("gpu0"), model_id=model_id,
                            capacity_rps=40.0)], tls={config.base_url: pin(host)}), keypair)
    app = create_app(config=config, keypair=keypair, manifest=lambda: signed, engines={"gpu0": MockEngine(model_id)},
                     authorized=lambda hk: hk == validator.ss58_address, auditors=lambda: set(),
                     max_caller_request=lambda m: caller_request_bytes(m, CAT.get(model_id).workload))
    return app, commitment_value(signed.manifest.digest())


class Counting(Presenting):
    """Routes requests to in-process apps by host, each presenting its host's certificate, and counts them per host
    and kind (the last path segment)."""

    def __init__(self, apps: dict[str, object]) -> None:
        super().__init__(apps)
        self.counts: dict[tuple[str, str], int] = {}

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        key = (request.url.host, request.url.path.rsplit("/", 1)[-1])
        self.counts[key] = self.counts.get(key, 0) + 1
        return await super().handle_async_request(request)


def network(validator: Keypair, served: dict[str, str]):
    """In-process miners, one per host, serving the model given for it."""
    apps, neurons, keys = {}, [], {}
    for uid, (host, model_id) in enumerate(served.items(), start=1):
        keys[host] = kp()
        apps[host], commit = miner(host, keys[host], validator, model_id)
        neurons.append(Neuron(uid, keys[host].ss58_address, host, 80, commit))
    return apps, neurons, keys


@pytest.mark.usefixtures("lab_network")
def test_the_endpoints_of_models_not_checked_get_no_canaries_and_no_challenge() -> None:
    validator = kp()
    served = {"10.0.5.1": "kev-4b", "10.0.5.2": "kev-4b", "10.0.5.3": "kev-9b", "10.0.5.4": "kev-9b"}
    apps, neurons, _ = network(validator, served)
    timings = seeded_timings(**{"kev-4b": (0.05, 0.01, 0.01), "kev-9b": (5.0, 0.01, 0.01)})
    slot = Slot()
    refs = references(slot, timings, times={"kev-4b": (0.05, 0.0, 0.01), "kev-9b": (0.05, 0.0, 0.01)})
    transport = Counting(apps)

    async def run():
        async with httpx.AsyncClient(transport=transport) as client:
            endpoints = [e for m in await discover(neurons, 405, client, {CAT.digest(): 1}) for e in m.endpoints()]
            sequence = Sequence(refs, ["kev-4b", "kev-9b"], overdue=set(), per_endpoint=4, timings=timings)
            started = time.monotonic()
            results = await audit_epoch(endpoints, epoch=3, secret=SECRET, keypair=validator, references=refs,
                                        tolerances={"kev-4b": Tolerance(min_probabilities=1)}, client=client,
                                        netuid=405, pool_size=8, per_endpoint=4, budget_s=2.0,
                                        one_at_a_time=sequence)
            return sequence, results, time.monotonic() - started

    sequence, results, took = asyncio.run(run())
    assert took < 2.0 + 0.5  # the canaries' time: references, starts and stops included
    assert sequence.checked == ["kev-4b"] and sequence.why == {"kev-9b": "not drawn"}
    assert {r.endpoint.endpoint.model_id for r in results.values()} == {"kev-4b"}
    assert all(r.sent == r.answered == 4 for r in results.values())
    for host, model_id in served.items():
        canaries, challenges = transport.counts.get((host, "systemone"), 0), transport.counts.get((host, "identity"), 0)
        if model_id == "kev-4b":
            assert (canaries, challenges) == (4, 1)
        else:
            assert (canaries, challenges) == (0, 0)


@pytest.mark.usefixtures("lab_network")
def test_a_validator_checking_every_model_one_at_a_time_scores_as_with_resident_references() -> None:
    validator = kp()
    served = {"10.0.6.1": "kev-4b", "10.0.6.2": "kev-4b", "10.0.6.3": "kev-9b"}
    apps, neurons, keys = network(validator, served)
    tolerances = {m: Tolerance(min_probabilities=1) for m in ("kev-4b", "kev-9b")}

    async def run(sequential: bool):
        async with httpx.AsyncClient(transport=Counting(apps)) as client:
            endpoints = [e for m in await discover(neurons, 405, client, {CAT.digest(): 1}) for e in m.endpoints()]
            if sequential:
                timings = Timings()
                refs = references(Slot(), timings, times={m: (0.02, 0.0, 0.01) for m in ("kev-4b", "kev-9b")})
                sequence = Sequence(refs, coverage_order(list(refs), {}, SECRET, 3), overdue=set(), per_endpoint=4,
                                    timings=timings)
            else:
                refs, sequence = {m: MockEngine(m) for m in ("kev-4b", "kev-9b")}, None
            results = await audit_epoch(endpoints, epoch=3, secret=SECRET, keypair=validator, references=refs,
                                        tolerances=tolerances, client=client, netuid=405, pool_size=8,
                                        per_endpoint=4, budget_s=None, one_at_a_time=sequence)
            return endpoints, results

    outcome = {}
    for sequential in (False, True):
        endpoints, results = asyncio.run(run(sequential))
        state = ScoreState()
        proven = Proven(dict.fromkeys(results, 30.0), {}, {}, {}, set())
        score_epoch(state, endpoints, results, taken=None, shares={"kev-4b": 0.5, "kev-9b": 0.5}, proven=proven,
                    invalid_hotkeys=set(), sc=CAT.scoring, max_latency_s=2.0)
        outcome[sequential] = ({k: (r.sent, r.answered, r.tally.sum_abs, r.tally.count) for k, r in results.items()},
                               weights(state, {n.uid: n.hotkey for n in neurons}))
    assert outcome[True] == outcome[False]
    assert len(outcome[True][0]) == 3 and sum(outcome[True][1].values()) == pytest.approx(1.0)


# ---- the source's results for the models not checked ----

UIDS = {"hA": 1, "hB": 2, "hC": 3, "hD": 4}
SHARES = {"kev-4b": 0.6, "kev-9b": 0.4}


def endpoint(key: str, model_id: str) -> DiscoveredEndpoint:
    hotkey, endpoint_id = key.split("/")
    return DiscoveredEndpoint(UIDS[hotkey], hotkey, "https://x", Endpoint(
        endpoint_id=endpoint_id, url=f"https://x/e/{endpoint_id}", model_id=model_id, capacity_rps=50.0), "ab" * 32)


def result(ep: DiscoveredEndpoint, *, fail: bool = False) -> EndpointResult:
    r = EndpointResult(ep, Tolerance(min_probabilities=1), identity_ok=True, sent=4, answered=4)
    r.tally.add(RequestComparison(0.3 if fail else 0.002, 0.08 if fail else 0.004, 40, 0))
    return r


# The network over nine epochs: what the source checked (its own results), which endpoints it could not audit,
# which hotkeys the auditor proved wrong; endpoints come and go (one first appears unaudited while its hotkey is on
# probation), hotkeys fail and go on probation.
SCRIPT = [
    dict(endpoints={"hA/g0": "kev-4b", "hA/g1": "kev-9b", "hB/g0": "kev-4b", "hC/g0": "kev-9b", "hD/g0": "kev-4b"},
         failing=set(), held=set(), invalid=set(), uncredited={"hD/g0"}),
    dict(endpoints={"hA/g0": "kev-4b", "hA/g1": "kev-9b", "hB/g0": "kev-4b", "hC/g0": "kev-9b", "hD/g0": "kev-4b"},
         failing={"hB/g0"}, held=set(), invalid=set(), uncredited=set()),
    dict(endpoints={"hA/g0": "kev-4b", "hA/g1": "kev-9b", "hB/g0": "kev-4b", "hB/g2": "kev-9b", "hC/g0": "kev-9b",
                    "hD/g0": "kev-4b"},
         failing=set(), held={"hA/g1", "hB/g2"}, invalid=set(), uncredited=set()),
    dict(endpoints={"hA/g0": "kev-4b", "hA/g1": "kev-9b", "hB/g0": "kev-4b", "hB/g1": "kev-9b", "hB/g2": "kev-9b",
                    "hC/g0": "kev-9b", "hD/g0": "kev-4b"},
         failing=set(), held=set(), invalid={"hC"}, uncredited=set()),
    dict(endpoints={"hA/g1": "kev-9b", "hB/g0": "kev-4b", "hB/g1": "kev-9b", "hB/g2": "kev-9b", "hC/g0": "kev-9b",
                    "hD/g0": "kev-4b"},
         failing={"hC/g0"}, held=set(), invalid=set(), uncredited=set()),
    dict(endpoints={"hA/g1": "kev-9b", "hB/g0": "kev-4b", "hB/g1": "kev-9b", "hB/g2": "kev-9b", "hC/g0": "kev-9b",
                    "hD/g0": "kev-4b"},
         failing=set(), held={"hD/g0"}, invalid=set(), uncredited=set()),
    *[dict(endpoints={"hA/g1": "kev-9b", "hB/g0": "kev-4b", "hB/g1": "kev-9b", "hB/g2": "kev-9b", "hC/g0": "kev-9b",
                      "hD/g0": "kev-4b"},
           failing=set(), held=set(), invalid=set(), uncredited=set())] * 3,
]


def epoch_inputs(step: dict):
    endpoints = [endpoint(k, m) for k, m in step["endpoints"].items()]
    results = {e.key: result(e, fail=e.key in step["failing"]) for e in endpoints if e.key not in step["held"]}
    credited = {e.key: 20.0 + UIDS[e.hotkey] for e in endpoints if e.key not in step["uncredited"]}
    latency = {e.key: 300.0 + 100 * UIDS[e.hotkey] for e in endpoints}
    proven = Proven(credited, latency, {}, {k: 0.9 for k in credited}, set())
    return endpoints, results, proven, set(step["invalid"])


def source_epoch(state: ScoreState, epoch: int, step: dict) -> tuple[NetworkStatus, dict[int, str]]:
    """The source validator's epoch: its scoring, then the network document it publishes (as a reader parses it)."""
    endpoints, results, proven, invalid = epoch_inputs(step)
    scores, failed, _, factors = score_epoch(state, endpoints, results, taken=None, shares=SHARES, proven=proven,
                                             invalid_hotkeys=invalid, sc=CAT.scoring, max_latency_s=2.0)
    hotkey_of_uid = {uid: hotkey for hotkey, uid in UIDS.items()}
    w = weights_to_set(weights(state, hotkey_of_uid), hotkey_of_uid, OWNER)
    board = StatusBoard()
    board.publish(build(epoch=epoch, epoch_blocks=CAT.epoch_blocks, endpoints=endpoints, results=results,
                        scores=scores, failed=failed, invalid_hotkeys=invalid, credited=proven.credited,
                        scored_models=set(SHARES), state=state, model_shares=SHARES, weights=w,
                        user_availability=proven.user_availability, latency_factors=factors),
                  epoch=epoch, epoch_blocks=CAT.epoch_blocks, model_shares=SHARES, weights=w)
    return NetworkStatus.model_validate_json(board.network()), hotkey_of_uid


def own_epoch(state: ScoreState, step: dict, checked: set[str], document: NetworkStatus | None,
              hotkey_of_uid: dict[int, str]) -> dict[int, float]:
    """A validator checking the ``checked`` models itself (seeing what the source saw) and taking the others."""
    endpoints, results, proven, invalid = epoch_inputs(step)
    own = {k: r for k, r in results.items() if r.endpoint.endpoint.model_id in checked}
    taken = None if document is None else fallback.take(document, set(SHARES) - checked)
    score_epoch(state, endpoints, own, taken=taken, shares=SHARES, proven=proven, invalid_hotkeys=invalid,
                sc=CAT.scoring, max_latency_s=2.0)
    return weights_to_set(weights(state, hotkey_of_uid), hotkey_of_uid, OWNER)


def test_a_validator_checking_nothing_sets_the_source_weights_and_one_checking_all_its_own() -> None:
    """Every epoch: checking no model gives exactly the weights the source set; checking every model gives exactly
    what scoring without a source gives (the source's own computation); checking one model and taking the other,
    when its own checks see what the source's saw, gives the source's weights too."""
    source, nothing, everything, mixed = ScoreState(), ScoreState(), ScoreState(), ScoreState()
    for epoch, step in enumerate(SCRIPT, start=100):
        document, hotkey_of_uid = source_epoch(source, epoch, step)
        set_by_source = document.weights
        assert own_epoch(nothing, step, set(), document, hotkey_of_uid) == set_by_source
        assert own_epoch(everything, step, set(SHARES), None, hotkey_of_uid) == set_by_source
        assert own_epoch(mixed, step, {"kev-4b"}, document, hotkey_of_uid) == set_by_source
        assert nothing.ema == source.ema and nothing.probation == source.probation
    assert source.probation  # the script's failures are still being served
    assert set_by_source and sum(set_by_source.values()) == pytest.approx(1.0)


def test_how_each_published_result_enters_the_scoring() -> None:
    def status(state: str, score: float = 0.25) -> dict:
        return {"endpoint_id": f"e-{state}", "model_id": "kev-9b", "state": state, "score": score}

    document = NetworkStatus.model_validate({"epoch": 5, "hotkeys": [
        {"hotkey": "hA", "uid": 1, "endpoints": [status(s) for s in ("scored", "qualifying", "failed", "invalid",
                                                                      "probation", "not_audited", "drifted")]
         + [{"endpoint_id": "own", "model_id": "kev-4b", "state": "scored", "score": 0.5}]},
        {"hotkey": "hB", "uid": None, "endpoints": [status("scored")]}]})
    taken = fallback.take(document, {"kev-9b"})
    assert taken.scores == {"hA/e-scored": 0.25, "hA/e-qualifying": 0.25, "hA/e-failed": 0.0,
                            "hA/e-probation": 0.0, "hA/e-drifted": 0.25}
    assert taken.failed == {"hA/e-failed"} and taken.held == {"hA/e-invalid", "hA/e-not_audited"}
    assert set(taken.uids) == set(taken.scores) | taken.held and set(taken.uids.values()) == {1}
    # a failure the source saw puts the whole hotkey on probation here, its own checks included
    state = ScoreState(ema={"hA/own": 0.4, "hA/e-not_audited": 0.3}, uid_of={"hA/own": 1, "hA/e-not_audited": 1})
    own = endpoint("hA/own", "kev-4b")
    score_epoch(state, [own], {"hA/own": result(own)}, taken=taken, shares=SHARES,
                proven=Proven({"hA/own": 30.0}, {}, {}, {}, set()), invalid_hotkeys=set(), sc=CAT.scoring,
                max_latency_s=2.0)
    assert state.probation["hA/own"] == CAT.scoring.probation_epochs - 1 and state.ema["hA/own"] == 0.0
    assert fallback.scores_valid(document) is None
    assert "not a share" in fallback.scores_valid(NetworkStatus.model_validate(
        {"hotkeys": [{"hotkey": "hA", "uid": 1, "endpoints": [status("scored", float("nan"))]}]}))


def test_without_the_source_results_the_models_not_checked_keep_their_averages() -> None:
    """For at most MAX_HELD_EPOCHS epochs in a row, as an endpoint without reference answers."""
    ep = endpoint("hA/g1", "kev-9b")
    state = ScoreState(ema={"hA/g1": 0.4}, uid_of={"hA/g1": 1})
    proven = Proven({}, {}, {}, {}, set())
    for _ in range(MAX_HELD_EPOCHS):
        _, _, held, _ = score_epoch(state, [ep], {}, taken=None, shares=SHARES, proven=proven,
                                    invalid_hotkeys=set(), sc=CAT.scoring, max_latency_s=2.0)
        assert held == {"hA/g1"} and state.ema["hA/g1"] == 0.4
    score_epoch(state, [ep], {}, taken=None, shares=SHARES, proven=proven, invalid_hotkeys=set(), sc=CAT.scoring,
                max_latency_s=2.0)
    assert state.ema["hA/g1"] == pytest.approx(0.32)


class Chain:
    def __init__(self, owner: str) -> None:
        self.owner = owner

    def get_subnet_owner_hotkey(self, netuid: int) -> str:
        return self.owner

    def get_neuron_for_pubkey_and_subnet(self, hotkey: str, netuid: int):
        return SimpleNamespace(uid=0, is_null=False, validator_permit=True,
                               axon_info=SimpleNamespace(ip="198.51.100.20", port=8091))


def signed(signer: Keypair, epoch: int, **changes) -> dict:
    body = NetworkStatus(epoch=epoch, epoch_blocks=CAT.epoch_blocks, weights={1: 1.0}, hotkeys=[
        {"hotkey": "hA", "uid": 1, "endpoints": [{"endpoint_id": "g1", "model_id": "kev-9b", "state": "scored",
                                                  "score": 0.4}]}]).model_dump(mode="json")
    return sign_document({**body, **changes}, signer, netuid=405, domain=NETWORK_DOMAIN, now_ms=int(time.time() * 1000))


def test_the_source_results_are_read_as_light_validators_read_them() -> None:
    owner = kp()
    served: list = []

    def fetch(url, **kwargs):
        assert url == "https://198.51.100.20:8091/v1/network" and kwargs["tls"] == AUDITOR_PIN
        document = served.pop(0) if len(served) > 1 else served[0]
        if isinstance(document, Exception):
            raise document
        return json.loads(json.dumps(document))

    def results(deadline_s: float, poll_s: float = 0.01, source_hotkey=None, hotkey="me"):
        return asyncio.run(fallback.source_results(
            Chain(owner.ss58_address), netuid=405, hotkey=hotkey, source_hotkey=source_hotkey, epoch=8,
            epoch_blocks=CAT.epoch_blocks, deadline=time.monotonic() + deadline_s, cat=CAT, fetch=fetch,
            poll_s=poll_s))

    # the source publishes epoch 8 once it has scored it: asked again until then
    served[:] = [httpx.ConnectError("refused"), signed(owner, 7), signed(owner, 8)]
    source, document = results(1.0)
    assert source == owner.ss58_address and document.epoch == 8
    assert fallback.take(document, {"kev-9b"}).scores == {"hA/g1": 0.4}
    served[:] = [signed(owner, 7)]
    with pytest.raises(Refused, match="not published the results of epoch 8"):
        results(0.05)
    served[:] = [signed(kp(), 8)]
    with pytest.raises(Refused, match="not signed by the source hotkey"):
        results(0.0)
    served[:] = [{**signed(owner, 8), "weights": None}]
    with pytest.raises(Refused, match="signature does not verify"):
        results(0.0)
    with pytest.raises(Refused, match="the source is this validator"):
        results(0.0, hotkey=owner.ss58_address)


# ---- the epoch's line ----

def test_one_line_per_epoch_names_the_models_checked_and_those_taken_from_the_source() -> None:
    sequence = Sequence({}, [], overdue=set(), per_endpoint=4, timings=Timings())
    sequence.checked, sequence.why = ["kev-4b"], {"kev-9b": "not drawn"}
    level, line = coverage_line(9, ["clef-flash", "kev-4b", "kev-9b"], sequence, {"clef-flash": "no reference"},
                                "the subnet owner's validator 5Abc", None)
    assert level == 30 and line == ("epoch 9: references one at a time: checked kev-4b; taken from the subnet "
                                    "owner's validator 5Abc: clef-flash (no reference), kev-9b (not drawn)")
    level, line = coverage_line(9, ["kev-4b", "kev-9b"], sequence, {}, None, "it is not signed")
    assert level == 30 and line.endswith("not checked, their averages kept (no results of the source: it is not "
                                         "signed): kev-9b (not drawn)")
    assert coverage_line(9, ["kev-4b"], sequence, {}, None, None) == (
        20, "epoch 9: references one at a time: checked kev-4b")


# ---- the reference processes ----

SERVER = textwrap.dedent("""
    import sys
    import uvicorn
    from fastapi import FastAPI
    from tests.mock_engine import MockEngine

    engine = MockEngine(sys.argv[2], served={"data": [{"id": sys.argv[2], "revision": sys.argv[3]}]})
    app = FastAPI()

    @app.get("/v1/models")
    async def models():
        return engine.served

    @app.post("/v1/systemone")
    async def systemone(body: dict):
        return await engine.answer(body)

    uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[1]), log_level="warning")
""")


def free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def served_reference(port: int, model_id: str = "kev-4b", revision: str | None = None) -> ReferenceEngine:
    return ReferenceEngine(url=f"http://127.0.0.1:{port}", command=[
        sys.executable, "-c", SERVER, str(port), model_id, revision or CAT.get(model_id).hf_revision],
        directory=str(ROOT), environment={"PYTHONPATH": f"{ROOT / 'src'}{os.pathsep}{ROOT}"})


def gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    status = Path(f"/proc/{pid}/status")
    return status.exists() and "zombie" in status.read_text()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux validators")
def test_the_validator_starts_a_reference_process_answers_with_it_and_stops_it(tmp_path) -> None:
    runner = Runner.at(tmp_path)
    port = free_port()
    reference = runner.references({"kev-4b": served_reference(port)}, CAT, ["kev-4b"])["kev-4b"]
    canary = canary_pool(SECRET, 1, "kev-4b", 1)[0]

    async def run():
        engine = await reference.start(time.monotonic() + 60)
        pid = reference.process.pid
        answer = await engine.answer(canary)
        await reference.stop()
        return pid, answer

    pid, answer = asyncio.run(run())
    assert answer == asyncio.run(MockEngine("kev-4b").answer(canary))
    assert gone(pid) and reference.process is None and runner.slot.running is None
    kept = Timings(tmp_path / "references.json")
    assert all(kept.get("kev-4b", kind) is not None for kind in ("start_s", "answer_s", "stop_s"))
    assert (tmp_path / "references" / "kev-4b.log").exists()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux validators")
def test_a_reference_that_serves_other_weights_or_exits_or_finds_its_port_taken_does_not_start(tmp_path) -> None:
    runner = Runner.at(tmp_path)
    port = free_port()
    wrong = runner.references({"kev-4b": served_reference(port, revision="0" * 40)}, CAT, ["kev-4b"])["kev-4b"]
    with pytest.raises(StartFailed, match="the catalog pins"):
        asyncio.run(wrong.start(time.monotonic() + 60))
    assert wrong.process is None
    exits = ReferenceEngine(url=f"http://127.0.0.1:{free_port()}", command=[
        sys.executable, "-c", "import sys; print('cannot load the weights'); sys.exit(3)"])
    failing = runner.references({"kev-4b": exits}, CAT, ["kev-4b"])["kev-4b"]
    with pytest.raises(StartFailed, match="exited with code 3(.|\n)*cannot load the weights"):
        asyncio.run(failing.start(time.monotonic() + 60))
    holder = subprocess.Popen(served_reference(port).command, cwd=ROOT,
                              env={**os.environ, **served_reference(port).environment})
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                httpx.get(f"http://127.0.0.1:{port}/v1/models", timeout=1)
                break
            except httpx.HTTPError:
                time.sleep(0.2)
        blocked = runner.references({"kev-4b": served_reference(port)}, CAT, ["kev-4b"])["kev-4b"]
        with pytest.raises(StartFailed, match="another process holds its port"):
            asyncio.run(blocked.start(time.monotonic() + 60))
    finally:
        holder.kill()
        holder.wait()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux validators")
def test_a_reference_ends_with_the_validator_that_started_it(tmp_path) -> None:
    """However the validator ends, its reference does not stay behind holding the memory and the port."""
    port = free_port()
    spec = served_reference(port)
    validator = textwrap.dedent(f"""
        import asyncio, json, os, sys, time
        from pathlib import Path
        from verathos import catalog
        from verathos.validator.config import ReferenceEngine
        from verathos.validator.sequential import Runner

        spec = ReferenceEngine.model_validate(json.loads({spec.model_dump_json()!r}))
        runner = Runner.at(Path({str(tmp_path)!r}))
        reference = runner.references({{"kev-4b": spec}}, catalog.load(), ["kev-4b"])["kev-4b"]
        asyncio.run(reference.start(time.monotonic() + 60))
        print(reference.process.pid, flush=True)
        os._exit(1)  # a validator that dies
    """)
    out = subprocess.run([sys.executable, "-c", validator], capture_output=True, text=True, timeout=90,
                         env={**os.environ, "PYTHONPATH": f"{ROOT / 'src'}{os.pathsep}{ROOT}"})
    pid = int(out.stdout.split()[0])
    deadline = time.monotonic() + 15
    while not gone(pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert gone(pid)


def test_a_recheck_starts_a_stopped_reference_only_in_the_time_left_and_one_at_a_time() -> None:
    timings = seeded_timings(**{"kev-4b": (0.05, 0.01, 0.01), "kev-9b": (0.05, 0.01, 0.01)})
    slot = Slot()
    refs = references(slot, timings, times={"kev-4b": (0.05, 0.0, 0.01), "kev-9b": (0.05, 0.0, 0.01)})
    canary = canary_pool(SECRET, 1, "kev-4b", 1)[0]

    async def run():
        refs["kev-4b"].deadline = time.monotonic() + 0.03  # its start takes 0.05 s
        with pytest.raises(EngineError, match="no time left"):
            await refs["kev-4b"].answer(canary)
        assert slot.board["running"] == 0
        for r in refs.values():
            r.deadline = time.monotonic() + 5
        await refs["kev-4b"].answer(canary)
        await refs["kev-9b"].answer(canary | {"model": "kev-9b"})  # stops kev-4b first
        assert slot.running is refs["kev-9b"] and refs["kev-4b"].engine is None
        await slot.release()

    asyncio.run(run())
    assert slot.board["peak"] == 1 and slot.board["running"] == 0


def test_an_engine_the_validator_starts_listens_on_this_machine() -> None:
    assert ReferenceEngine(url="http://127.0.0.1:8101", command=["serve"]).command == ["serve"]
    with pytest.raises(ValueError, match="serves on this machine"):
        ReferenceEngine(url="https://ref.example/x", command=["serve"])
    config = ValidatorConfig.model_validate({"netuid": 96, "wallet_name": "v", "hotkey_name": "h",
                                             "references": {"kev-4b": "http://127.0.0.1:8101"}})
    assert config.reference_mode == "resident"  # configurations from before stay resident


def test_the_epoch_budget_leaves_the_canaries_their_time() -> None:
    """The references' share of the canaries' time is the same one at a time: starts and stops included."""
    from verathos.validator.audit import REFERENCE_SHARE

    budget_s = CAT.epoch_blocks * BLOCK_TIME_S * CAT.scoring.canary_spread_fraction
    assert REFERENCE_SHARE * budget_s == pytest.approx(388.8)  # about 6.5 minutes of a 180-block epoch


# ---- an epoch of a validator running its references one at a time ----

@pytest.mark.usefixtures("lab_network")
def test_an_epoch_checks_what_fits_takes_the_rest_from_the_source_and_sets_one_weight_vector(
        tmp_path, monkeypatch, caplog) -> None:
    import logging

    from verathos.throughput.feed import Credit, Verdicts
    from verathos.validator import __main__ as validator_main
    from verathos.validator.discovery import Identities, ManifestCache

    caplog.set_level(logging.INFO)
    validator, owner = kp(), kp()
    served = {"10.0.7.1": "kev-4b", "10.0.7.2": "kev-9b"}
    apps, miners, keys = network(validator, served)
    hotkeys = [owner.ss58_address, validator.ss58_address, *(keys[h].ss58_address for h in served)]
    commitments = {n.hotkey: n.commitment for n in miners}
    axons = [("198.51.100.20", 8091), ("198.51.100.21", 8091), *((h, 80) for h in served)]
    st = SimpleNamespace(
        metagraph=lambda netuid, lite=True: SimpleNamespace(
            hotkeys=hotkeys, axons=[SimpleNamespace(ip=ip, port=port) for ip, port in axons],
            S=[1e6, 2e4, 0.0, 0.0], validator_permit=[True, True, False, False]),
        get_all_commitments=lambda netuid: commitments,
        get_subnet_owner_hotkey=lambda netuid: owner.ss58_address,
        get_neuron_for_pubkey_and_subnet=lambda hotkey, netuid: SimpleNamespace(
            is_null=False, validator_permit=True, axon_info=SimpleNamespace(ip="198.51.100.20", port=8091)))
    kev4, kev9 = (f"{keys[h].ss58_address}/gpu0" for h in served)
    credits = [Credit(hotkey=key.split("/")[0], endpoint_id="gpu0", model_id=m, credited_rps=20.0, latency_ms=500.0,
                      test_epoch=6, qualified=True) for key, m in ((kev4, "kev-4b"), (kev9, "kev-9b"))]

    async def verdicts(client, epoch):
        return Verdicts(netuid=405, auditor=owner.ss58_address, epoch=epoch, credits=credits, invalid=[])

    async def proven_invalid(evidence, references, cat, budget_s=None):
        return []

    throughput = SimpleNamespace(auditor_hotkey=owner.ss58_address, manifests=ManifestCache(1000.0),
                                 update_slots=lambda *args: None, holding=lambda: False, identities=Identities(),
                                 verdicts=verdicts, proven_invalid=proven_invalid)
    # a short epoch: canaries over a twentieth of one block's time; on mechanism 1, the miners' manifests'
    cat = CAT.model_copy(update={"mechanism": 1, "scoring": CAT.scoring.model_copy(update={
        "canary_spread_fraction": 0.05, "canary_pool_size": 4, "canaries_per_endpoint": 2})})
    current = SimpleNamespace(catalog=cat, for_epoch=lambda st, block: cat, accepted=lambda: {CAT.digest(): 1})
    cfg = ValidatorConfig(netuid=405, wallet_name="v", hotkey_name="h", state_dir=tmp_path,
                          reference_mode="sequential", references={m: FAKE.model_dump() for m in ("kev-4b", "kev-9b")})
    timings = seeded_timings(**{"kev-4b": (0.02, 0.0, 0.01), "kev-9b": (60.0, 1.0, 0.01)})  # kev-9b never fits

    class FakeRunner(Runner):
        def references(self, specs, cat, models):
            return {m: Fake(m, start_s=0.02, answer_s=0.0, stop_s=0.01, slot=self.slot, timings=self.timings)
                    for m in models}

    document = sign_document(NetworkStatus(epoch=7, epoch_blocks=CAT.epoch_blocks, weights={3: 1.0}, hotkeys=[
        {"hotkey": kev9.split("/")[0], "uid": 3, "endpoints": [
            {"endpoint_id": "gpu0", "model_id": "kev-9b", "state": "scored", "score": 0.4, "sent": 10,
             "answered": 10, "verdict": "pass"}]}]).model_dump(mode="json"), owner, netuid=405,
        domain=NETWORK_DOMAIN, now_ms=int(time.time() * 1000))
    fetched: list = []

    async def from_source(*args, **kwargs):
        def fetch(url, **kw):
            fetched.append(url)
            if isinstance(fetched_document[0], Exception):
                raise fetched_document[0]
            return json.loads(json.dumps(fetched_document[0]))
        return await fallback.source_results(*args, **kwargs, fetch=fetch)

    transport = Counting(apps)
    monkeypatch.setattr(validator_main.tls, "client", lambda limits: httpx.AsyncClient(transport=transport))
    monkeypatch.setattr(validator_main, "source_results", from_source)
    weights_set: list = []
    monkeypatch.setattr(validator_main, "set_weights",
                        lambda cfg, st, wallet, w, mechid: weights_set.append(w) if mechid == 1 else None)

    def epoch(state: ScoreState, board: StatusBoard) -> None:
        asyncio.run(validator_main.run_epoch(cfg, st, SimpleNamespace(hotkey=validator), SECRET, state, 7, 1,
                                             throughput, current, board, FakeRunner(Slot(), timings, None)))

    fetched_document = [document]
    state, board = ScoreState(), StatusBoard()
    epoch(state, board)
    assert fetched == ["https://198.51.100.20:8091/v1/network"]
    line = (f"epoch 7: references one at a time: checked kev-4b; taken from the subnet owner's validator "
            f"{owner.ss58_address}: kev-9b (not drawn)")
    assert any(r.levelname == "WARNING" and r.message == line for r in caplog.records)
    assert state.waited == {"kev-4b": 0, "kev-9b": 1}
    assert state.ema[kev9] == pytest.approx(CAT.scoring.ema_alpha * 0.4) and state.ema[kev4] > 0
    (w,) = weights_set
    assert set(w) == {0, 2, 3} and sum(w.values()) == pytest.approx(1.0)  # the owner's UID beyond the miner share
    assert w[0] == pytest.approx(1 - cat.scoring.miner_share_at(7))
    assert transport.counts.get(("10.0.7.2", "systemone"), 0) == transport.counts.get(("10.0.7.2", "identity"), 0) == 0
    assert transport.counts[("10.0.7.1", "systemone")] == 2
    published = {h.hotkey: h for h in NetworkStatus.model_validate_json(board.network()).hotkeys}
    (taken,) = published[kev9.split("/")[0]].endpoints
    assert (taken.state, taken.score, taken.sent, taken.verdict) == ("scored", 0.4, 10, "pass")

    # without the source's results, kev-9b's endpoint keeps its average
    caplog.clear()
    fetched_document[0] = httpx.ConnectError("refused")
    weights_set.clear()
    epoch(held := ScoreState(ema={kev9: 0.3}, uid_of={kev9: 3}), StatusBoard())
    assert held.ema[kev9] == 0.3 and held.held == {kev9: 1}
    assert any(r.levelname == "WARNING" and "not checked, their averages kept (no results of the source: the "
               "source's axon https://198.51.100.20:8091 did not serve it" in r.message and r.message.endswith(
                   "kev-9b (not drawn)") for r in caplog.records)
