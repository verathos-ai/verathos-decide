"""Full validator loop against in-process miners: discovery, identity, canaries, receipts, comparison, scoring.

Miners are real miner apps served over an in-memory transport; engines are deterministic mocks. The honest miner
uses the reference engine's model; the cheater uses a different one behind the same model label.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from bittensor_wallet import Keypair
from fastapi import Request

from tests.mock_engine import MockEngine
from verathos import catalog
from verathos.miner.app import caller_request_bytes, create_app
from verathos.miner.config import MinerConfig
from verathos.protocol.manifest import Endpoint, Manifest, SignedManifest, commitment_value
from verathos.validator.audit import audit_epoch
from verathos.validator.canaries import assignment, canary_pool
from verathos.validator.compare import Tolerance, Verdict
from verathos.validator.discovery import Neuron, discover
from verathos.validator.scoring import ScoreState, epoch_scores, record_window, update, weights

pytestmark = pytest.mark.usefixtures("lab_network")


class HostRouter(httpx.AsyncBaseTransport):
    """Routes requests to in-process ASGI apps by host name."""

    def __init__(self, apps: dict[str, object]) -> None:
        self._t = {host: httpx.ASGITransport(app=app) for host, app in apps.items()}

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return await self._t[request.url.host].handle_async_request(request)


def kp() -> Keypair:
    return Keypair.create_from_mnemonic(Keypair.generate_mnemonic())


def miner(host: str, keypair: Keypair, validator: Keypair, engine: MockEngine):
    config = MinerConfig.model_validate({
        "netuid": 405, "mechid": 1, "wallet_name": "w", "hotkey_name": "h", "public_host": host, "port": 80,
        "endpoints": [{"endpoint_id": "gpu0", "model_id": "kev-4b", "engine_url": "http://unused",
                       "max_concurrency": 8}]})
    digest = catalog.load().digest()
    signed = SignedManifest.create(Manifest(
        netuid=405, mechid=1, hotkey=keypair.ss58_address, catalog_digest=digest,
        endpoints=[Endpoint(endpoint_id="gpu0", url=config.endpoint_url("gpu0"), model_id="kev-4b",
                            capacity_rps=40.0)]), keypair)
    kev = catalog.load().get("kev-4b")
    app = create_app(config=config, keypair=keypair, manifest=lambda: signed, engines={"gpu0": engine},
                     authorized=lambda hk: hk == validator.ss58_address, auditors=lambda: set(),
                     max_caller_request=lambda m: caller_request_bytes(m, kev.workload))
    return app, commitment_value(signed.manifest.digest())


def test_canary_pool_is_deterministic_and_assignment_rotates() -> None:
    secret = b"s" * 32
    assert canary_pool(secret, 7, "kev-4b", 5) == canary_pool(secret, 7, "kev-4b", 5)
    assert canary_pool(secret, 7, "kev-4b", 5) != canary_pool(secret, 8, "kev-4b", 5)
    assert canary_pool(secret, 7, "kev-4b", 5) != canary_pool(b"t" * 32, 7, "kev-4b", 5)
    a = assignment(secret, 7, "hk/gpu0", 20, 6)
    assert len(a) == 6 and a != assignment(secret, 8, "hk/gpu0", 20, 6)


def test_a_slow_reference_shrinks_the_canary_pool_instead_of_delaying_the_epoch() -> None:
    """Reference work per epoch is the canary pool per model, whatever the number of endpoints, and it runs within a
    share of the epoch's audit time: what a slow reference cannot compute by then is left out of the pool, and every
    endpoint still gets canaries from what was computed."""
    import time as time_module

    class SlowReference(MockEngine):
        async def answer(self, request):
            await asyncio.sleep(0.15)
            return await super().answer(request)

    validator = kp()
    miners_ = {f"10.0.1.{i}": (kp(), MockEngine("kev-4b")) for i in range(30)}
    apps, neurons = {}, []
    for uid, (host, (keypair, engine)) in enumerate(miners_.items()):
        apps[host], commit = miner(host, keypair, validator, engine)
        neurons.append(Neuron(uid, keypair.ss58_address, host, 80, commit))

    async def run():
        async with httpx.AsyncClient(transport=HostRouter(apps)) as client:
            endpoints = [e for m in await discover(neurons, 405, 1, client, {catalog.load().digest()})
                         for e in m.endpoints()]
            started = time_module.monotonic()
            results = await audit_epoch(endpoints, epoch=1, secret=b"k" * 32, keypair=validator,
                                        references={"kev-4b": SlowReference("kev-4b")},
                                        tolerances={"kev-4b": Tolerance(min_probabilities=1)}, client=client,
                                        netuid=405, mechid=1, pool_size=16, per_endpoint=6, budget_s=2.0)
            return results, time_module.monotonic() - started

    results, took = asyncio.run(run())
    assert took < 2.0 + 1.0  # the budget, not 16 x 0.15 s of references plus the canaries' spread
    assert len(results) == 30 and all(0 < r.sent <= 4 and r.answered == r.sent for r in results.values())
    assert all(r.tally.verdict() is not Verdict.FAIL_HARD for r in results.values())


def test_epoch_scores_honest_and_zeroes_cheater() -> None:
    validator, honest_kp, cheat_kp, ghost_kp = kp(), kp(), kp(), kp()
    honest, cheat, ghost = "10.0.0.1", "10.0.0.2", "10.0.0.3"
    honest_app, honest_commit = miner(honest, honest_kp, validator, MockEngine("kev-4b"))
    cheat_app, cheat_commit = miner(cheat, cheat_kp, validator, MockEngine("kev-4b", salt="other-model"))
    ghost_app, _ = miner(ghost, ghost_kp, validator, MockEngine("kev-4b"))
    neurons = [
        Neuron(1, honest_kp.ss58_address, honest, 80, honest_commit),
        Neuron(2, cheat_kp.ss58_address, cheat, 80, cheat_commit),
        Neuron(3, ghost_kp.ss58_address, ghost, 80, None),  # serves a manifest but never committed it
    ]

    async def run():
        transport = HostRouter({honest: honest_app, cheat: cheat_app, ghost: ghost_app})
        async with httpx.AsyncClient(transport=transport) as client:
            miners = await discover(neurons, 405, 1, client, {catalog.load().digest()})
            endpoints = [e for m in miners for e in m.endpoints()]
            results = await audit_epoch(endpoints, epoch=1, secret=b"k" * 32, keypair=validator,
                                        references={"kev-4b": MockEngine("kev-4b")},
                                        tolerances={"kev-4b": Tolerance(min_probabilities=20)}, client=client,
                                        netuid=405, mechid=1, pool_size=10, per_endpoint=6)
            return miners, results

    miners, results = asyncio.run(run())
    assert {m.uid for m in miners} == {1, 2}
    honest = results[f"{honest_kp.ss58_address}/gpu0"]
    cheat = results[f"{cheat_kp.ss58_address}/gpu0"]
    assert honest.identity_ok and cheat.identity_ok
    assert honest.answered == honest.sent == 6 and not honest.errors
    assert honest.tally.verdict() is Verdict.PASS
    assert cheat.tally.verdict() is Verdict.FAIL_HARD

    state = ScoreState()
    record_window(state, results, 3)
    scores, failed = epoch_scores(results, shares={"kev-4b": 1.0}, credited=dict.fromkeys(results, 40.0),
                                  latency_ms={}, state=state)
    update(state, results, scores, failed)
    w = weights(state, {1: honest_kp.ss58_address, 2: cheat_kp.ss58_address})
    assert w == {1: pytest.approx(1.0)}
    assert state.probation == {f"{cheat_kp.ss58_address}/gpu0": 5}  # the failing epoch is the first of six

    # probation holds even if the cheater behaves next epoch
    update(state, results, {k: 1.0 for k in scores}, set())
    assert state.ema[f"{cheat_kp.ss58_address}/gpu0"] == 0.0
    assert state.probation[f"{cheat_kp.ss58_address}/gpu0"] == 4


def test_a_lost_challenge_at_the_axon_is_retried_and_a_canary_counts_only_within_the_request_timeout() -> None:
    """At the host the hotkey's manifest was just fetched from, a lost identity challenge is tried again before each
    canary; a miner holding canaries cannot stretch the epoch: one unanswered within request_timeout_s counts as
    unanswered."""

    class Slow(MockEngine):
        async def answer(self, request):
            await asyncio.sleep(0.5)
            return await super().answer(request)

    validator = kp()
    keys = {"lost": kp(), "slow": kp()}
    apps, neurons = {}, []
    for uid, (host, name, engine) in enumerate((("10.0.2.1", "lost", MockEngine("kev-4b")),
                                                ("10.0.2.2", "slow", Slow("kev-4b")))):
        apps[host], commit = miner(host, keys[name], validator, engine)
        neurons.append(Neuron(uid, keys[name].ss58_address, host, 80, commit))

    class FirstChallengeLost(httpx.AsyncBaseTransport):
        def __init__(self) -> None:
            self.inner, self.lost = HostRouter(apps), set()

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/v1/identity") and request.url.host not in self.lost:
                self.lost.add(request.url.host)
                return httpx.Response(504)
            return await self.inner.handle_async_request(request)

    async def run():
        async with httpx.AsyncClient(transport=FirstChallengeLost()) as client:
            endpoints = [e for m in await discover(neurons, 405, 1, client, {catalog.load().digest()})
                         for e in m.endpoints()]
            return await audit_epoch(endpoints, epoch=1, secret=b"k" * 32, keypair=validator,
                                     references={"kev-4b": MockEngine("kev-4b")},
                                     tolerances={"kev-4b": Tolerance(min_probabilities=1)}, client=client,
                                     netuid=405, mechid=1, pool_size=4, per_endpoint=2, request_timeout_s=0.2)

    results = asyncio.run(run())
    lost, slow = (results[f"{keys[n].ss58_address}/gpu0"] for n in ("lost", "slow"))
    assert lost.identity_ok and lost.answered == lost.sent == 2
    assert slow.identity_ok and (slow.sent, slow.answered) == (2, 0)
    assert slow.errors == ["request failed: TimeoutError"] * 2


class Counting(httpx.AsyncBaseTransport):
    """Routes like HostRouter and counts the requests per host and kind (the last path segment)."""

    def __init__(self, apps: dict[str, object]) -> None:
        self.inner, self.seen = HostRouter(apps), {}

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        kind = (request.url.host, request.url.path.rsplit("/", 1)[-1])
        self.seen[kind] = self.seen.get(kind, 0) + 1
        return await self.inner.handle_async_request(request)


def multi_host_miner(hotkey: Keypair, validator: Keypair, hosts: dict[str, list[str]], listed: dict[str, str]):
    """A hotkey's miner services: ``hosts`` maps each host it runs to the endpoints served there, ``listed`` maps
    every endpoint its manifest lists to the host named in it (which may be someone else's)."""
    digest = catalog.load().digest()
    kev = catalog.load().get("kev-4b")
    signed = SignedManifest.create(Manifest(
        netuid=405, mechid=1, hotkey=hotkey.ss58_address, catalog_digest=digest,
        endpoints=[Endpoint(endpoint_id=e, url=f"http://{host}:80/e/{e}", model_id="kev-4b", capacity_rps=40.0)
                   for e, host in listed.items()]), hotkey)
    apps = {}
    for host, served in hosts.items():
        config = MinerConfig.model_validate({
            "netuid": 405, "mechid": 1, "wallet_name": "w", "hotkey_name": "h", "public_host": host, "port": 80,
            "endpoints": [{"endpoint_id": e, "model_id": "kev-4b", "engine_url": "http://unused"} for e in served]})
        apps[host] = create_app(config=config, keypair=hotkey, manifest=lambda: signed,
                                engines={e: MockEngine("kev-4b") for e in served},
                                authorized=lambda hk: hk == validator.ss58_address, auditors=lambda: set(),
                                serving=lambda served=served: served,
                                max_caller_request=lambda m: caller_request_bytes(m, kev.workload))
    return apps, commitment_value(signed.manifest.digest())


def test_endpoints_on_several_machines_are_checked_where_they_are_and_another_server_gets_one_challenge() -> None:
    """One hotkey, endpoints on its axon and on a worker machine, each reached at its own URL; two more of its
    endpoints name a server that is not the hotkey's (another miner's): that server gets one identity challenge per
    epoch from the validator, however often the validator looks, and no canary."""
    from verathos.validator.discovery import Identities

    validator, hotkey, other = kp(), kp(), kp()
    axon, worker, victim = "10.0.3.1", "10.0.3.2", "10.0.3.3"
    apps, commit = multi_host_miner(hotkey, validator, {axon: ["gpu0"], worker: ["w-gpu0", "w-gpu1"]},
                                    {"gpu0": axon, "w-gpu0": worker, "w-gpu1": worker, "v0": victim, "v1": victim})
    apps[victim], _ = miner(victim, other, validator, MockEngine("kev-4b"))
    transport, identities = Counting(apps), Identities()

    async def epoch(n: int):
        async with httpx.AsyncClient(transport=transport) as client:
            (found,) = await discover([Neuron(1, hotkey.ss58_address, axon, 80, commit)], 405, 1, client,
                                      {catalog.load().digest()})
            return found, await audit_epoch(found.endpoints(), epoch=n, secret=b"k" * 32, keypair=validator,
                                            references={"kev-4b": MockEngine("kev-4b")},
                                            tolerances={"kev-4b": Tolerance(min_probabilities=1)}, client=client,
                                            netuid=405, mechid=1, pool_size=4, per_endpoint=3,
                                            identities=identities)

    found, results = asyncio.run(epoch(7))
    assert {e.host for e in found.endpoints()} == {f"http://{h}:80" for h in (axon, worker, victim)}
    by_id = {r.endpoint.endpoint.endpoint_id: r for r in results.values()}
    for e in ("gpu0", "w-gpu0", "w-gpu1"):
        assert by_id[e].identity_ok and by_id[e].answered == by_id[e].sent == 3
    assert transport.seen[worker, "systemone"] == 6  # the worker's endpoints were sent their canaries directly
    for e in ("v0", "v1"):
        assert not by_id[e].identity_ok and by_id[e].sent == 0 and by_id[e].errors == ["identity not proven at its URL"]
    assert transport.seen[victim, "identity"] == 1 and (victim, "systemone") not in transport.seen

    asyncio.run(epoch(7))  # the same epoch again (the auditor's refresh shares the record)
    assert transport.seen[victim, "identity"] == 1 and transport.seen[worker, "identity"] == 2
    asyncio.run(epoch(8))
    assert transport.seen[victim, "identity"] == 2 and (victim, "systemone") not in transport.seen


def test_a_manifest_accepted_before_stands_in_while_the_axon_is_out() -> None:
    """The machine serving the manifest goes down: the endpoints on the hotkey's other machine keep their canaries,
    while the commitment still names that manifest and for at most the cache's hold time."""
    from verathos.validator.discovery import Identities, ManifestCache

    validator, hotkey = kp(), kp()
    axon, worker = "10.0.5.1", "10.0.5.2"
    apps, commit = multi_host_miner(hotkey, validator, {axon: ["gpu0"], worker: ["w-gpu0"]},
                                    {"gpu0": axon, "w-gpu0": worker})
    cache, identities, down = ManifestCache(hold_s=3600.0), Identities(), set()

    class Transport(httpx.AsyncBaseTransport):
        def __init__(self) -> None:
            self.inner = HostRouter(apps)

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            if request.url.host in down:
                raise httpx.ConnectError("down")
            return await self.inner.handle_async_request(request)

    def epoch(n: int, commitment: str = commit):
        async def run():
            async with httpx.AsyncClient(transport=Transport()) as client:
                found = await discover([Neuron(1, hotkey.ss58_address, axon, 80, commitment)], 405, 1, client,
                                       {catalog.load().digest()}, cache)
                results = await audit_epoch([e for m in found for e in m.endpoints()], epoch=n, secret=b"k" * 32,
                                            keypair=validator, references={"kev-4b": MockEngine("kev-4b")},
                                            tolerances={"kev-4b": Tolerance(min_probabilities=1)}, client=client,
                                            netuid=405, mechid=1, pool_size=4, per_endpoint=2,
                                            identities=identities)
                return found, {r.endpoint.endpoint.endpoint_id: r for r in results.values()}

        return asyncio.run(run())

    (found,), results = epoch(1)
    assert found.axon == f"http://{axon}:80" and all(r.answered == 2 for r in results.values())
    down.add(axon)
    (found,), results = epoch(2)
    assert found.axon is None and [e.endpoint_id for e in found.manifest.endpoints] == ["gpu0", "w-gpu0"]
    assert results["w-gpu0"].identity_ok and results["w-gpu0"].answered == 2
    assert not results["gpu0"].identity_ok and results["gpu0"].sent == 0
    assert epoch(3, commitment_value("1" * 64))[0] == []  # the hotkey committed another manifest
    cache.hold_s = 0.0
    assert epoch(4)[0] == []  # held long enough


def test_a_validator_notices_when_the_subnet_owner_hotkey_moves() -> None:
    """Its role (auditor or follower) and the auditor it trusts are set at its start: a new owner hotkey means a
    restart."""
    from types import SimpleNamespace

    from verathos.validator.__main__ import auditor_moved

    def chain(owner):
        return SimpleNamespace(get_subnet_owner_hotkey=lambda netuid: owner)

    assert auditor_moved(chain("new"), 405, "old") == "new"
    assert auditor_moved(chain("old"), 405, "old") is None
    assert auditor_moved(chain(None), 405, "old") is None  # the chain did not say


def test_weights_ignore_reassigned_uids() -> None:
    state = ScoreState(ema={"hkA/gpu0": 0.5, "hkB/gpu0": 0.5}, uid_of={"hkA/gpu0": 1, "hkB/gpu0": 2})
    assert weights(state, {1: "hkA", 2: "someone-else"}) == {1: pytest.approx(1.0)}


def test_window_catches_subtle_drift_no_single_epoch_shows() -> None:
    """Each epoch alone has too few probabilities for the statistical bound; three epochs together fail it."""
    from verathos.validator.audit import EndpointResult
    from verathos.validator.compare import RequestComparison
    from verathos.validator.discovery import DiscoveredEndpoint

    tol = Tolerance(min_probabilities=100, mean_abs=0.005)
    ep = DiscoveredEndpoint(1, "hkA", "http://x", Endpoint(endpoint_id="gpu0", url="http://x/e/gpu0", model_id="kev-4b",
                                                           capacity_rps=40.0))
    state = ScoreState()
    verdicts = []
    for _ in range(3):
        r = EndpointResult(ep, tol, identity_ok=True, sent=10, answered=10, latencies_ms=[40.0] * 10)
        # mean 0.01, no hard fail
        r.tally.add(RequestComparison(max_abs=0.03, sum_abs=0.6, count=60, confident_flips=0))
        record_window(state, {"hkA/gpu0": r}, 3)
        scores, failed = epoch_scores({"hkA/gpu0": r}, shares={"kev-4b": 1.0}, credited={"hkA/gpu0": 40.0},
                                      latency_ms={}, state=state)
        verdicts.append(bool(failed))
    assert verdicts == [False, True, True]  # 60 probabilities: insufficient; 120 and 180: mean 0.01 > 0.005


def test_score_is_proven_capacity_and_a_proven_wrong_answer_resets_the_hotkey() -> None:
    """Score scales with the capacity the throughput audit credited (none: zero); a wrong answer proven in a test
    fails every endpoint of the hotkey, resets their averages and starts probation."""
    from verathos.validator.audit import EndpointResult
    from verathos.validator.discovery import DiscoveredEndpoint

    def result(hotkey: str, endpoint_id: str, uid: int) -> EndpointResult:
        ep = DiscoveredEndpoint(uid, hotkey, "http://x", Endpoint(endpoint_id=endpoint_id,
                                url=f"http://x/e/{endpoint_id}", model_id="kev-4b", capacity_rps=50.0))
        return EndpointResult(ep, Tolerance(min_probabilities=1), identity_ok=True, sent=4, answered=4,
                              latencies_ms=[40.0] * 4)

    results = {k: result(*k.split("/"), uid) for uid, k in enumerate(("hkA/gpu0", "hkB/gpu0", "hkC/gpu0",
                                                                     "hkC/gpu1"), start=1)}
    state = ScoreState(ema={"hkC/gpu0": 30.0, "hkC/gpu1": 30.0, "hkC/old": 30.0})
    record_window(state, results, 3)
    credited = {"hkA/gpu0": 50.0, "hkC/gpu0": 40.0, "hkC/gpu1": 40.0}
    scores, failed = epoch_scores(results, shares={"kev-4b": 1.0}, credited=credited, latency_ms={},
                                  state=state, failed_hotkeys={"hkC"})
    assert scores == {"hkA/gpu0": pytest.approx(1.0), "hkB/gpu0": 0.0, "hkC/gpu0": 0.0, "hkC/gpu1": 0.0}
    assert failed == {"hkC/gpu0", "hkC/gpu1"}
    update(state, results, scores, failed, alpha=0.2, probation_epochs=6, reset_hotkeys={"hkC"})
    assert state.ema["hkA/gpu0"] == pytest.approx(0.2)
    assert {k: state.ema.get(k, 0.0) for k in ("hkC/gpu0", "hkC/gpu1", "hkC/old")} == dict.fromkeys(
        ("hkC/gpu0", "hkC/gpu1", "hkC/old"), 0.0)
    assert state.probation == {"hkC/gpu0": 5, "hkC/gpu1": 5, "hkC/old": 5}


def test_credited_capacity() -> None:
    from verathos.throughput.feed import Credit, Verdicts
    from verathos.validator.__main__ import credited_capacity
    from verathos.validator.discovery import DiscoveredEndpoint

    endpoints = [DiscoveredEndpoint(1, "hkA", "http://x", Endpoint(endpoint_id=e, url=f"http://x/e/{e}",
                                    model_id=model, capacity_rps=rps))
                 for e, model, rps in (("a", "kev-4b", 30.0), ("b", "kev-4b", 50.0), ("c", "kev-4b", 50.0),
                                       ("d", "kev-9b", 50.0))]

    def credit(hotkey: str, endpoint_id: str, qualified: bool = True, model_id: str = "kev-4b") -> Credit:
        return Credit(hotkey=hotkey, endpoint_id=endpoint_id, model_id=model_id, credited_rps=45.0,
                      latency_ms=800.0, test_epoch=8, qualified=qualified)

    verdicts = Verdicts(netuid=405, auditor="aud", epoch=9, invalid=[], credits=[
        credit("hkA", "a"), credit("hkA", "b", qualified=False), credit("hkB", "a"), credit("hkA", "d")])
    cat = catalog.load()
    # capped at today's declaration; unqualified, untested and re-modelled endpoints have no credit
    assert credited_capacity(endpoints, verdicts, cat, "enforce") == ({"hkA/a": 30.0}, {"hkA/a": 800.0}, {},
                                                                      {"hkA/a": 1.0}, set())
    assert credited_capacity(endpoints, None, cat, "enforce") == ({}, {}, {}, {}, set())
    # observe mode counts declarations, but no more than a model's first test
    assert credited_capacity(endpoints, None, cat, "observe") == ({"hkA/a": 30.0, "hkA/b": 50.0, "hkA/c": 50.0,
                                                                   "hkA/d": 25.0}, {}, {}, {}, set())
    drifted = Verdicts(netuid=405, auditor="aud", epoch=9, invalid=[], credits=[
        credit("hkA", "a").model_copy(update={"credited_rps": 0.0, "drift": True})])
    assert credited_capacity(endpoints, drifted, cat, "enforce").drifted == {"hkA/a"}
    # what users got counts once there is enough of it
    users = Verdicts(netuid=405, auditor="aud", epoch=9, invalid=[], credits=[
        credit("hkA", "a").model_copy(update={"user_offers": 100, "user_refusals": 40, "user_answers": 60,
                                              "user_latency_ms": 300.0}),
        credit("hkA", "c").model_copy(update={"user_offers": 5, "user_refusals": 5, "user_answers": 10,
                                              "user_latency_ms": 900.0})])
    proven = credited_capacity(endpoints, users, cat, "enforce")
    # per hotkey and model, splitting hides nothing: 60 of 105 offers ready, with 20 ready ones to start with
    assert proven.user_availability == {k: pytest.approx(80 / 125) for k in ("hkA/a", "hkA/c")}
    assert proven.user_latency_ms == {"hkA/a": 300.0, "hkA/c": 300.0}  # 70 answers, weighted by them


def test_latency_factor_comes_from_the_test_latencies() -> None:
    from verathos.validator.audit import EndpointResult
    from verathos.validator.discovery import DiscoveredEndpoint

    def result(endpoint_id: str) -> EndpointResult:
        ep = DiscoveredEndpoint(1, "hkA", "http://x", Endpoint(endpoint_id=endpoint_id, url=f"http://x/e/{endpoint_id}",
                                model_id="kev-4b", capacity_rps=20.0))
        # canary latencies (the validator's own view) no longer matter
        return EndpointResult(ep, Tolerance(min_probabilities=1), identity_ok=True, sent=4, answered=4,
                              latencies_ms=[9_999.0] * 4)

    results = {k: result(k.split("/")[1]) for k in ("hkA/fast", "hkA/mid", "hkA/slow")}
    state = ScoreState()
    record_window(state, results, 3)
    scores, _ = epoch_scores(results, shares={"kev-4b": 1.0}, credited=dict.fromkeys(results, 20.0),
                             latency_ms={"hkA/fast": 250.0, "hkA/mid": 1000.0, "hkA/slow": 4000.0}, state=state)
    assert scores == {"hkA/fast": pytest.approx(26 / 60), "hkA/mid": pytest.approx(20 / 60),
                      "hkA/slow": pytest.approx(14 / 60)}  # the pool split by capacity x latency factor


def test_what_users_got_from_an_endpoint_counts_once_there_is_enough_of_it() -> None:
    """An endpoint that refuses users loses that share of its score, and the latency its users saw replaces its test
    latency, compared among endpoints measured the same way: tests alone cannot show either (they are known shortly
    ahead and recognisable)."""
    import math

    from verathos.validator.audit import EndpointResult
    from verathos.validator.discovery import DiscoveredEndpoint

    def result(endpoint_id: str) -> EndpointResult:
        ep = DiscoveredEndpoint(1, "hkA", "http://x", Endpoint(endpoint_id=endpoint_id, url=f"http://x/e/{endpoint_id}",
                                model_id="kev-4b", capacity_rps=20.0))
        return EndpointResult(ep, Tolerance(min_probabilities=1), identity_ok=True, sent=4, answered=4)

    results = {k: result(k.split("/")[1]) for k in ("hkA/fast", "hkA/hoarder", "hkA/quiet")}
    state = ScoreState()
    record_window(state, results, 3)
    scores, _ = epoch_scores(results, shares={"kev-4b": 1.0}, credited=dict.fromkeys(results, 20.0),
                             latency_ms=dict.fromkeys(results, 400.0), state=state,
                             user_latency_ms={"hkA/fast": 200.0, "hkA/hoarder": 800.0},
                             user_availability={"hkA/hoarder": 0.5})
    # users' median 500 ms: fast at the 1.3 cap, the hoarder sqrt(500 / 800) and half its offers refused; the quiet
    # endpoint, measured by its test alone, is compared with no one
    weights_ = {"hkA/fast": 20 * 1.3, "hkA/hoarder": 20 * 0.5 * math.sqrt(500 / 800), "hkA/quiet": 20.0}
    assert scores == {k: pytest.approx(w / sum(weights_.values())) for k, w in weights_.items()}


def test_answering_users_near_twice_the_latency_bound_earns_next_to_nothing() -> None:
    """An endpoint answering users just under the gateway's slow bound (twice max_latency_s) keeps all its draws
    ready on half its GPU; its latency factor falls linearly from the bound to zero at twice it, not to a 0.7 floor."""
    import math

    from verathos.validator.audit import EndpointResult
    from verathos.validator.discovery import DiscoveredEndpoint

    def result(endpoint_id: str) -> EndpointResult:
        ep = DiscoveredEndpoint(1, "hkA", "http://x", Endpoint(endpoint_id=endpoint_id, url=f"http://x/e/{endpoint_id}",
                                model_id="kev-4b", capacity_rps=20.0))
        return EndpointResult(ep, Tolerance(min_probabilities=1), identity_ok=True, sent=4, answered=4)

    results = {k: result(k.split("/")[1]) for k in ("hkA/quick", "hkA/bound", "hkA/dawdler")}
    state = ScoreState()
    record_window(state, results, 3)
    users = {"hkA/quick": 1000.0, "hkA/bound": 2000.0, "hkA/dawdler": 3900.0}
    scores, _ = epoch_scores(results, shares={"kev-4b": 1.0}, credited=dict.fromkeys(results, 20.0),
                             latency_ms=dict.fromkeys(results, 400.0), state=state, user_latency_ms=users,
                             max_latency_s=2.0)
    # median 2000 ms: the quick one at the 1.3 cap, the dawdler sqrt(2000 / 3900) and then 5% of that
    factors = {"hkA/quick": 1.3, "hkA/bound": 1.0, "hkA/dawdler": math.sqrt(2000 / 3900) * (2 - 3.9 / 2)}
    assert scores == {k: pytest.approx(f / sum(factors.values())) for k, f in factors.items()}
    assert scores["hkA/dawdler"] < 0.02


def test_a_model_whose_endpoints_all_refuse_users_scores_zero_and_stops_nothing() -> None:
    from verathos.validator.audit import EndpointResult
    from verathos.validator.discovery import DiscoveredEndpoint

    def result(endpoint_id: str) -> EndpointResult:
        ep = DiscoveredEndpoint(1, "hkA", "http://x", Endpoint(endpoint_id=endpoint_id, url=f"http://x/e/{endpoint_id}",
                                model_id="kev-4b", capacity_rps=20.0))
        return EndpointResult(ep, Tolerance(min_probabilities=1), identity_ok=True, sent=4, answered=4)

    results = {k: result(k.split("/")[1]) for k in ("hkA/a", "hkA/b")}
    state = ScoreState()
    record_window(state, results, 3)
    scores, _ = epoch_scores(results, shares={"kev-4b": 1.0}, credited=dict.fromkeys(results, 20.0),
                             latency_ms={}, state=state, user_availability=dict.fromkeys(results, 0.0))
    assert scores == dict.fromkeys(results, 0.0)


def test_averages_of_another_score_scale_start_over(tmp_path) -> None:
    path = tmp_path / "scores.json"
    path.write_text(json.dumps({"ema": {"hk/gpu0": 18.7}, "probation": {"hk/gpu1": 3}}))  # proven requests/s
    state = ScoreState.load(path)
    assert state.ema == {} and state.probation == {"hk/gpu1": 3}
    state.ema = {"hk/gpu0": 0.4}
    state.save(path)
    assert ScoreState.load(path).ema == {"hk/gpu0": 0.4}


def test_splitting_capacity_over_endpoints_does_not_move_the_latency_median() -> None:
    """The typical latency follows proven capacity, not the number of endpoints: a slow miner listing its capacity
    as many endpoints is still slow (testnet 405, epoch 45364: 17.1 rps at 362 ms against 9.5 rps at 6 s)."""
    from verathos.validator.audit import EndpointResult
    from verathos.validator.discovery import DiscoveredEndpoint

    def result(hotkey: str, endpoint_id: str) -> EndpointResult:
        ep = DiscoveredEndpoint(1, hotkey, "http://x", Endpoint(endpoint_id=endpoint_id,
                                url=f"http://x/e/{endpoint_id}", model_id="kev-4b", capacity_rps=20.0))
        return EndpointResult(ep, Tolerance(min_probabilities=1), identity_ok=True, sent=4, answered=4)

    for parts in (2, 10):
        slow = [f"gpu{i}" for i in range(parts)]
        results = {"fast/gpu0": result("fast", "gpu0")} | {f"slow/{e}": result("slow", e) for e in slow}
        credited = {"fast/gpu0": 17.1} | {f"slow/{e}": 9.5 / parts for e in slow}
        latency = {"fast/gpu0": 362.0} | {f"slow/{e}": 5_960.0 for e in slow}
        state = ScoreState()
        record_window(state, results, 3)
        scores, _ = epoch_scores(results, shares={"kev-4b": 1.0}, credited=credited, latency_ms=latency,
                                 state=state)
        weights_ = {"fast": 17.1, "slow": 9.5 * 0.7}  # the slow side at the 0.7 floor however it is split
        assert scores["fast/gpu0"] == pytest.approx(weights_["fast"] / sum(weights_.values()))
        assert sum(v for k, v in scores.items() if k.startswith("slow/")) == pytest.approx(
            weights_["slow"] / sum(weights_.values()))


def test_the_weighted_median_is_the_ordinary_median_for_equal_weights() -> None:
    from verathos.validator.scoring import weighted_median

    assert weighted_median([(250.0, 1.0), (4000.0, 1.0)]) == 2125.0
    assert weighted_median([(250.0, 1.0), (1000.0, 1.0), (4000.0, 1.0)]) == 1000.0
    assert weighted_median([(250.0, 3.0), (1000.0, 1.0), (4000.0, 1.0)]) == 250.0
    assert weighted_median([(4000.0, 2.0), (250.0, 1.0)]) == 4000.0


def test_a_canary_failure_puts_the_whole_hotkey_on_probation() -> None:
    """Moving the hardware to a spare endpoint of the same hotkey does not escape probation."""
    from verathos.validator.audit import EndpointResult
    from verathos.validator.discovery import DiscoveredEndpoint

    def result(endpoint_id: str) -> EndpointResult:
        ep = DiscoveredEndpoint(1, "hkA", "http://x", Endpoint(endpoint_id=endpoint_id, url=f"http://x/e/{endpoint_id}",
                                model_id="kev-4b", capacity_rps=20.0))
        return EndpointResult(ep, Tolerance(), identity_ok=True, sent=4, answered=4, latencies_ms=[40.0] * 4)

    state = ScoreState(ema={"hkA/gpu0": 10.0, "hkA/spare": 0.0}, window={"hkA/gpu0": [[1.0, 10.0]]})
    update(state, {"hkA/gpu0": result("gpu0"), "hkA/spare": result("spare")}, {"hkA/gpu0": 0.0, "hkA/spare": 0.0},
           {"hkA/gpu0"}, probation_epochs=6)
    assert state.probation == {"hkA/gpu0": 5, "hkA/spare": 5} and state.ema["hkA/gpu0"] == 0.0
    assert "hkA/gpu0" not in state.window  # the failing epoch's data does not extend the probation


def test_endpoints_are_held_for_a_few_epochs_at_most() -> None:
    from verathos.validator.scoring import MAX_HELD_EPOCHS

    state = ScoreState(ema={"hkA/gpu0": 0.5}, uid_of={"hkA/gpu0": 1})
    for _ in range(MAX_HELD_EPOCHS):
        update(state, {}, {}, set(), held={"hkA/gpu0"})
    assert state.ema == {"hkA/gpu0": 0.5}
    update(state, {}, {}, set(), held={"hkA/gpu0"})
    assert state.ema["hkA/gpu0"] == pytest.approx(0.4)


def test_a_restarted_validator_does_not_score_the_same_epoch_twice(tmp_path) -> None:
    state = ScoreState(last_epoch=41)
    state.save(tmp_path / "scores.json")
    assert ScoreState.load(tmp_path / "scores.json").last_epoch == 41


def answers_challenges(app, keypair: Keypair, endpoint_id: str = "gpu0") -> None:
    """Give ``app`` an identity route that signs challenges with ``keypair``, as a miner holding the hotkey does."""
    from verathos.miner.app import IDENTITY_DOMAIN, identity_payload
    from verathos.protocol import signing
    from verathos.protocol.requests import HDR_HOTKEY

    @app.post(f"/e/{endpoint_id}/v1/identity")
    async def identity(request: Request):
        payload = identity_payload(endpoint_id, (await request.json())["challenge"], request.headers[HDR_HOTKEY])
        return {"signature": signing.sign(keypair, IDENTITY_DOMAIN, payload)}


def test_hostile_replies_fail_the_endpoint_without_aborting_the_epoch() -> None:
    """A malformed identity reply costs the endpoint its canaries; NaN probabilities and junk receipts from an
    endpoint that proved control are recorded against it."""
    from fastapi import FastAPI
    from fastapi.responses import PlainTextResponse

    from verathos.validator.discovery import DiscoveredEndpoint

    validator, hostile_kp = kp(), kp()
    apps = {"10.0.0.9": FastAPI(), "10.0.0.10": FastAPI()}
    answers_challenges(apps["10.0.0.9"], hostile_kp)

    @apps["10.0.0.10"].post("/e/gpu0/v1/identity")
    async def identity():
        return PlainTextResponse('{"signature": 5}', media_type="application/json")

    for app in apps.values():
        @app.post("/e/gpu0/v1/systemone")
        async def systemone():
            return PlainTextResponse('{"answers": {"q": {"noul": NaN}}, "receipt": []}',
                                     media_type="application/json")

    signed, unsigned = (DiscoveredEndpoint(1, hostile_kp.ss58_address, f"http://{host}:80", Endpoint(
        endpoint_id="gpu0", url=f"http://{host}:80/e/gpu0", model_id="kev-4b", capacity_rps=40.0))
        for host in apps)

    async def run(ep):
        async with httpx.AsyncClient(transport=HostRouter(apps)) as client:
            return (await audit_epoch([ep], epoch=1, secret=b"k" * 32, keypair=validator,
                                      references={"kev-4b": MockEngine("kev-4b")},
                                      tolerances={"kev-4b": Tolerance()}, client=client, netuid=405, mechid=1,
                                      pool_size=4, per_endpoint=3))[ep.key]

    result = asyncio.run(run(signed))
    assert result.identity_ok
    assert result.sent == 3 and len(result.tally.hard_failures) == 3
    assert result.tally.verdict() is Verdict.FAIL_HARD
    result = asyncio.run(run(unsigned))
    assert not result.identity_ok and result.sent == 0 and not result.tally.hard_failures


def test_non_finite_or_out_of_range_probabilities_are_malformed() -> None:
    from verathos.protocol.api import distributions

    for bad in ({"q": {"noul": float("nan")}}, {"q": {"probabilities": {"a": "0.5"}}}, {"q": {"noul": 1.5}},
                {"q": {"noul": True}}, {"q": []}, []):
        with pytest.raises(ValueError):
            distributions(bad)
    assert distributions({"q": {"noul": 0.25}}) == {"q": {"false": 0.75, "true": 0.25}}


def test_renamed_endpoint_inherits_probation() -> None:
    from verathos.validator.audit import EndpointResult
    from verathos.validator.discovery import DiscoveredEndpoint

    def result(endpoint_id: str) -> EndpointResult:
        ep = DiscoveredEndpoint(1, "hkA", "http://x", Endpoint(
            endpoint_id=endpoint_id, url=f"http://x/e/{endpoint_id}", model_id="kev-4b", capacity_rps=40.0))
        return EndpointResult(ep, Tolerance(), identity_ok=True, sent=4, answered=4, latencies_ms=[40.0] * 4)

    state = ScoreState()
    update(state, {"hkA/gpu0": result("gpu0")}, {"hkA/gpu0": 0.0}, {"hkA/gpu0"}, probation_epochs=6)
    update(state, {"hkA/renamed": result("renamed")}, {"hkA/renamed": 1.0}, set(), probation_epochs=6)
    assert state.ema["hkA/renamed"] == 0.0 and state.probation["hkA/renamed"] == 4


def test_gone_endpoints_are_forgotten() -> None:
    state = ScoreState(ema={"hkA/gpu0": 1e-5}, uid_of={"hkA/gpu0": 1}, window={"hkA/gpu0": [[0.0, 1.0]]})
    update(state, {}, {}, set())
    assert state.ema == {} and state.uid_of == {} and state.window == {}


def test_endpoints_the_validator_could_not_audit_keep_their_score() -> None:
    state = ScoreState(ema={"hkA/gpu0": 0.5}, uid_of={"hkA/gpu0": 1})
    update(state, {}, {}, set(), held={"hkA/gpu0"})
    assert state.ema == {"hkA/gpu0": 0.5}
    update(state, {}, {}, set())
    assert state.ema["hkA/gpu0"] == pytest.approx(0.4)


def test_huge_numbers_and_deep_json_are_malformed_not_fatal() -> None:
    """Replies built to make parsing raise unusual exceptions still fail the request, and discovery survives a
    manifest nested too deeply to parse."""
    from fastapi import FastAPI
    from fastapi.responses import PlainTextResponse

    from verathos.protocol.api import distributions
    from verathos.validator.discovery import DiscoveredEndpoint

    with pytest.raises(ValueError):
        distributions({"q": {"noul": 10**400}})

    validator, hostile_kp = kp(), kp()
    app = FastAPI()
    answers_challenges(app, hostile_kp)

    @app.get("/v1/manifest")
    async def manifest():
        return PlainTextResponse("[" * 50000, media_type="application/json")

    @app.post("/e/gpu0/v1/systemone")
    async def systemone():
        return PlainTextResponse('{"answers": {"q": {"noul": 1' + "0" * 400 + '}}, "receipt": {}}',
                                 media_type="application/json")

    ep = DiscoveredEndpoint(1, hostile_kp.ss58_address, "http://10.0.0.9:80", Endpoint(
        endpoint_id="gpu0", url="http://10.0.0.9:80/e/gpu0", model_id="kev-4b", capacity_rps=40.0))

    async def run():
        async with httpx.AsyncClient(transport=HostRouter({"10.0.0.9": app})) as client:
            found = await discover([Neuron(1, hostile_kp.ss58_address, "10.0.0.9", 80,
                                           commitment_value("0" * 64))], 405, 1, client, {"0" * 64})
            results = await audit_epoch([ep], epoch=1, secret=b"k" * 32, keypair=validator,
                                        references={"kev-4b": MockEngine("kev-4b")},
                                        tolerances={"kev-4b": Tolerance()}, client=client, netuid=405, mechid=1,
                                        pool_size=4, per_endpoint=3)
            return found, results

    found, results = asyncio.run(run())
    assert found == []
    assert results[ep.key].sent == 3 and len(results[ep.key].tally.hard_failures) == 3


def test_chain_trouble_fetching_verdicts_is_an_unavailable_feed() -> None:
    from verathos.throughput import feed
    from verathos.throughput.policy import AuditPolicy
    from verathos.validator.throughput import Throughput

    class BrokenChain:
        def auditor_url(self):
            raise ConnectionError("node down")

    t = Throughput(netuid=405, mechid=1, policy=AuditPolicy(), auditor_hotkey="aud", chain=BrokenChain())
    with pytest.raises(feed.VerdictsUnavailable, match="ConnectionError"):
        asyncio.run(t._fetch(None, 9))


def test_models_share_the_emission_by_what_users_pay_with_the_anchors_as_the_fallback() -> None:
    from verathos.validator.scoring import model_shares

    cat = catalog.load()  # anchors 50/50; prices kev-4b 0.04, kev-9b 0.09; full at 10,000 tokens/s; floor 0.1
    assert model_shares({}, cat) == {"kev-4b": pytest.approx(0.5), "kev-9b": pytest.approx(0.5)}
    full = model_shares({"kev-4b": 30_000.0, "kev-9b": 10_000.0}, cat)  # revenue 1.2 against 0.9
    assert full == {"kev-4b": pytest.approx(1.2 / 2.1), "kev-9b": pytest.approx(0.9 / 2.1)}
    half = model_shares({"kev-4b": 5_000.0}, cat)  # half way: 0.5 x anchors + 0.5 x all to kev-4b
    assert half == {"kev-4b": pytest.approx(0.75), "kev-9b": pytest.approx(0.25)}
    lonely = model_shares({"kev-4b": 50_000.0}, cat)  # all demand on one model: the other keeps its floor
    assert lonely == {"kev-4b": pytest.approx(0.9), "kev-9b": pytest.approx(0.1)}
    assert set(model_shares({"clef-flash": 50_000.0}, cat)) == {"kev-4b", "kev-9b"}  # candidates: none
    skewed = cat.model_copy(update={"models": [m.model_copy(update={"anchor_share": 0.6 if m.model_id == "kev-4b"
                                                                    else 0.2}) for m in cat.models]})
    assert model_shares({}, skewed) == {"kev-4b": pytest.approx(0.75), "kev-9b": pytest.approx(0.25)}  # normalized
    with pytest.raises(ValueError):
        skewed.anchor_shares(strict=True)  # a published catalog's anchors must sum to 1


def test_each_model_pool_is_split_by_its_own_endpoints() -> None:
    from verathos.validator.audit import EndpointResult
    from verathos.validator.discovery import DiscoveredEndpoint

    def result(hotkey: str, model: str) -> EndpointResult:
        ep = DiscoveredEndpoint(1, hotkey, "http://x", Endpoint(endpoint_id="gpu0", url="http://x/e/gpu0",
                                model_id=model, capacity_rps=50.0))
        return EndpointResult(ep, Tolerance(min_probabilities=1), identity_ok=True, sent=4, answered=4)

    results = {"a/gpu0": result("a", "kev-4b"), "b/gpu0": result("b", "kev-4b"), "c/gpu0": result("c", "kev-9b")}
    state = ScoreState()
    record_window(state, results, 3)
    scores, _ = epoch_scores(results, shares={"kev-4b": 0.4, "kev-9b": 0.6},
                             credited={"a/gpu0": 30.0, "b/gpu0": 10.0, "c/gpu0": 5.0}, latency_ms={}, state=state)
    assert scores == {"a/gpu0": pytest.approx(0.3), "b/gpu0": pytest.approx(0.1), "c/gpu0": pytest.approx(0.6)}


def test_the_validator_starts_once_its_reference_engines_are_up(monkeypatch) -> None:
    """References restarted with the validator load their models first; an epoch started before them would compute
    no canary. Any HTTP reply counts as up; one that stays down delays the start by the wait's limit only."""
    import socket
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from verathos.validator import __main__ as validator_main
    from verathos.validator.config import ReferenceEngine

    server = HTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)  # replies 501 to everything: up
    threading.Thread(target=server.serve_forever, daemon=True).start()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        closed = s.getsockname()[1]
    references = {"kev-4b": ReferenceEngine(url=f"http://127.0.0.1:{server.server_port}"),
                  "kev-9b": ReferenceEngine(url=f"http://127.0.0.1:{closed}")}
    try:
        assert asyncio.run(validator_main.references_down(references)) == ["kev-9b"]
    finally:
        server.shutdown()

    checks = iter([["kev-4b", "kev-9b"], ["kev-9b"], []])
    calls = []

    async def loading(refs):
        calls.append(refs)
        return next(checks)

    monkeypatch.setattr(validator_main, "references_down", loading)
    monkeypatch.setattr(validator_main.time, "sleep", lambda s: None)
    validator_main.wait_for_references(references)
    assert len(calls) == 3

    async def never_up(refs):
        return ["kev-9b"]

    monkeypatch.setattr(validator_main, "references_down", never_up)
    validator_main.wait_for_references(references, timeout_s=0.0)  # gives up and starts
