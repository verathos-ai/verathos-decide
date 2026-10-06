"""The throughput audit: schedule, sessions, admission, the miner's measurement, and the auditor end to end against
in-process miners (honest, over-declaring, sharing one engine, and answering with another model)."""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import random

import httpx
import pytest
from bittensor_wallet import Keypair

from tests.fake_loadgen import make_runner
from tests.mock_engine import MockEngine
from verathos import catalog
from verathos.miner.admission import Admission
from verathos.miner.app import create_app
from verathos.miner.config import MinerConfig
from verathos.protocol.manifest import Endpoint, Manifest, SignedManifest
from verathos.throughput import loadgen
from verathos.throughput.auditor import Auditor, Slot, planned_rate
from verathos.throughput.measure import MeasurementStore, Target, measure_all, measure_and_store, stored
from verathos.throughput.policy import (
    AuditPolicy,
    Schedule,
    ScheduledTest,
    eligible_range,
    policy_from,
)
from verathos.throughput.workload import Workload, make_request
from verathos.validator.throughput import Throughput, slots

KEV = catalog.load().get("kev-4b")


def kp() -> Keypair:
    return Keypair.create_from_mnemonic(Keypair.generate_mnemonic())


def block_hash(block: int) -> bytes:
    return hashlib.sha256(f"block {block}".encode()).digest()


class FakeChain:
    def __init__(self, head: int) -> None:
        self.head_block, self.finalized = head, head

    def head(self) -> tuple[int, bytes]:
        return self.head_block, block_hash(self.head_block)

    def block_hash(self, block: int) -> bytes:
        return block_hash(block)

    def finalized_block(self) -> int:
        return self.finalized


class CapacityEngine:
    """Answers like ``engine`` but completes at most ``rps`` requests per second, however many are in flight."""

    def __init__(self, engine: MockEngine, rps: float, concurrency: int = 4) -> None:
        self.engine, self.service_s, self.calls = engine, concurrency / rps, 0
        self._slots = asyncio.Semaphore(concurrency)

    async def answer(self, request):
        self.calls += 1
        async with self._slots:
            await asyncio.sleep(self.service_s)
            return await self.engine.answer(request)


# ---- schedule and policy ----

def test_exactly_one_test_per_epoch_uniformly_placed() -> None:
    policy = AuditPolicy()
    rng = random.Random(1)
    positions = []
    for epoch in range(2000):
        schedule = Schedule(policy)
        first, last = eligible_range(epoch, policy)
        tests = [t for b in range(epoch * 180, (epoch + 1) * 180) if (t := schedule.observe(b, rng.randbytes(32)))]
        assert len(tests) == 1 and first <= tests[0].selection_block <= last
        assert tests[0].start_block + policy.traffic_blocks + 1 < (epoch + 1) * 180  # traffic ends inside the epoch
        positions.append(tests[0].selection_block - first)
    assert sum(positions) / len(positions) == pytest.approx((last - first) / 2, rel=0.08)


def test_validators_hold_canaries_around_the_test() -> None:
    policy = AuditPolicy()
    schedule = Schedule(policy)
    test = next(t for b in range(180, 360) if (t := schedule.observe(b, block_hash(b))))
    held = [b for b in range(180, 360) if schedule.held(b)]
    assert held == list(range(test.start_block - 1, test.start_block + policy.traffic_blocks + 2))


def test_a_policy_change_never_selects_a_second_test_in_an_epoch() -> None:
    chain = FakeChain(head=180 + 170)
    from verathos.throughput.follow import ScheduleFollower

    follower = ScheduleFollower(chain, AuditPolicy())
    (test,) = follower.poll()
    follower.set_policy(dataclasses.replace(AuditPolicy(), lead_blocks=2))
    chain.head_block += 5
    assert follower.poll() == [] and follower.schedule.held(test.start_block)


def test_policy_comes_from_the_catalog_and_is_checked() -> None:
    assert policy_from({"pass_fraction": 0.9, "samples_per_test": 20}, 360).samples_per_test == 20
    for bad in ({"unknown": 1}, {"epoch_blocks": 100}, {"pass_fraction": 1.5}, {"rate_growth": 0.5},
                {"test_s": 400.0}, {"answer_checks_per_test": 11}):
        with pytest.raises(ValueError):
            policy_from(bad, 180)
    with pytest.raises(ValueError, match="too short"):
        policy_from({"lead_blocks": 60}, 50)
    assert catalog.load().audit_policy() == AuditPolicy()


def test_requests_are_unique_and_follow_the_mix() -> None:
    workload = Workload(state_words=(5, 9), questions=KEV.workload.questions)
    rng = random.Random(3)
    requests = [make_request("kev-4b", workload, rng) for _ in range(200)]
    assert len({r["state"] for r in requests}) == 200
    assert all(5 <= len(r["state"].split()) - 2 <= 9 for r in requests)
    with pytest.raises(ValueError):
        Workload(state_words=(9, 5), questions=KEV.workload.questions)


def test_tested_rate_grows_from_what_was_delivered() -> None:
    slot = Slot("hk", "gpu0", "http://x/e/gpu0", "kev-4b", declared_rps=400.0)
    policy = AuditPolicy()
    assert planned_rate(slot, 50.0, None, policy) == 50.0
    assert planned_rate(slot, 50.0, 90.0, policy) == 180.0
    assert planned_rate(slot, 50.0, 5.0, policy) == 10.0
    assert planned_rate(slot, 50.0, 0.0, policy) == 6.25  # an endpoint that delivered nothing is probed
    assert planned_rate(dataclasses.replace(slot, declared_rps=30.0), 50.0, 90.0, policy) == 30.0


def test_credit_is_what_was_delivered_and_over_declaring_costs() -> None:
    from verathos.throughput.auditor import credit

    policy = AuditPolicy()
    assert credit(40.0, 40.0, policy) == 40.0
    assert credit(40.0, 38.5, policy) == 38.5  # inside the pass band: what was delivered
    assert credit(40.0, 41.0, policy) == 40.0
    assert credit(40.0, 20.0, policy) == 10.0  # declared twice what it serves: credited a quarter of the declaration
    # declaring more than an endpoint serves never pays
    serves = 20.0
    assert max((credit(d, min(d, serves), policy), d) for d in (15.0, 20.0, 21.0, 25.0, 40.0))[1] in (20.0, 21.0)


# ---- the miner ----

def test_admission_serves_regular_requests_before_test_traffic() -> None:
    async def run():
        admission = Admission(1, wait_s=0.5)
        assert await admission.acquire(test=False)
        order = []

        async def wait(name: str, test: bool) -> None:
            if await admission.acquire(test=test):
                order.append(name)
                await asyncio.sleep(0.01)
                admission.release()

        waiters = [asyncio.create_task(wait("test", True))]
        await asyncio.sleep(0.01)
        waiters.append(asyncio.create_task(wait("regular", False)))
        await asyncio.sleep(0.01)
        admission.release()
        await asyncio.gather(*waiters)
        assert order == ["regular", "test"] and admission.inflight == 0

        # a full queue: a regular request takes the newest test request's place; a test request is refused
        assert await admission.acquire(test=False)
        tests = [asyncio.create_task(admission.acquire(test=True)) for _ in range(4)]
        await asyncio.sleep(0.01)
        assert not await admission.acquire(test=True)
        regular = asyncio.create_task(admission.acquire(test=False))
        await asyncio.sleep(0.01)
        assert tests[-1].done() and not tests[-1].result()
        admission.release()
        assert await regular
        admission.release()
        assert [await t for t in tests[:3]] == [True, False, False]  # one slot, then the 0.5 s wait ends
        admission.release()
        assert admission.inflight == 0

    asyncio.run(run())


def test_measurement_finds_capacity_and_shares_of_one_engine(tmp_path) -> None:
    async def run():
        alone = CapacityEngine(MockEngine("kev-4b"), rps=200)
        shared = CapacityEngine(MockEngine("kev-4b"), rps=200)
        targets = [Target("alone", "kev-4b", alone, KEV.workload, 8, "f1"),
                   Target("a", "kev-4b", shared, KEV.workload, 8, "f2"),
                   Target("b", "kev-4b", shared, KEV.workload, 8, "f3")]
        store = MeasurementStore(tmp_path / "capacity.json")
        measured, failures = await measure_and_store(targets, store, max_latency_s=1.0, run_s=0.25)
        assert failures == {} and stored(targets, store) == measured
        changed = [dataclasses.replace(targets[0], fingerprint="new")] + targets[1:]
        assert stored(changed, store) is None
        return measured

    m = asyncio.run(run())
    assert m["alone"].measured_rps == pytest.approx(200, rel=0.2)
    assert m["a"].measured_rps + m["b"].measured_rps == pytest.approx(200, rel=0.2)
    assert m["alone"].declared(0.95) == round(m["alone"].measured_rps * 0.95, 2)


def test_fingerprint_ignores_engine_counters() -> None:
    from verathos.throughput.measure import fingerprint

    kev = {"models": [{"name": "kev-latest", "run": "jaredpalmer/kev-4b", "dtype": "bfloat16", "temperature": 2.41,
                       "batches": {"count": 255, "queued": 0}, "prefix_cache": {"hits": 0, "misses": 255}}]}
    later = {"models": [{**kev["models"][0], "batches": {"count": 9000, "queued": 3}}]}
    other = {"models": [{**kev["models"][0], "dtype": "float16"}]}
    config = {"model_id": "kev-4b", "engine_url": "http://127.0.0.1:8008"}
    assert fingerprint(config, kev, KEV.workload) == fingerprint(config, later, KEV.workload)
    assert fingerprint(config, kev, KEV.workload) != fingerprint(config, other, KEV.workload)
    assert fingerprint(config, None, KEV.workload) == fingerprint(config, "not a document", KEV.workload)


def test_an_endpoint_that_cannot_be_measured_is_left_out() -> None:
    from verathos.miner.engine import EngineError
    from verathos.throughput import measure

    class Down:
        async def answer(self, request):
            raise EngineError("connection refused")

    down = Target("down", "kev-4b", Down(), KEV.workload, 8, "f")
    with pytest.raises(measure.MeasurementFailed, match="not answering"):
        asyncio.run(measure.wait_ready(down, timeout_s=0))
    up = dataclasses.replace(down, endpoint_id="up", engine=MockEngine("kev-4b"))
    slow = dataclasses.replace(down, endpoint_id="slow",
                               engine=CapacityEngine(MockEngine("kev-4b"), rps=4, concurrency=1))
    measured, failures = asyncio.run(measure_all([down, up, slow], 0.1, run_s=0.2, ready_timeout_s=0))
    assert set(measured) == {"up"}
    assert failures["down"].startswith("not answering") and failures["slow"] == "one request takes longer than 0.1 s"


def test_declared_capacity_stays_within_the_latency_bound() -> None:
    """Past saturation more concurrency only adds latency: the measurement settles at the knee, well inside the
    bound."""
    from verathos.throughput.measure import knee

    target = Target("gpu0", "kev-4b", CapacityEngine(MockEngine("kev-4b"), rps=160, concurrency=8), KEV.workload, 256,
                    "f")
    measured, _ = asyncio.run(measure_all([target], 0.2, run_s=0.25))
    m = measured["gpu0"]
    assert 8 <= m.concurrency <= 16 and m.measured_rps == pytest.approx(160, rel=0.15)
    assert knee([(4, 80.0), (8, 158.0), (16, 161.0), (32, 160.0)]) == (8, 158.0)


# ---- the auditor end to end ----

class HostRouter(httpx.AsyncBaseTransport):
    def __init__(self, apps: dict[str, object]) -> None:
        self._t = {host: httpx.ASGITransport(app=app) for host, app in apps.items()}

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return await self._t[request.url.host].handle_async_request(request)


def miner_app(host: str, keypair: Keypair, auditor: Keypair, engines: dict[str, object], declared: float,
              port: int = 80):
    config = MinerConfig.model_validate({
        "netuid": 405, "mechid": 1, "wallet_name": "w", "hotkey_name": "h", "public_host": host, "port": port,
        "endpoints": [{"endpoint_id": e, "model_id": "kev-4b", "engine_url": "http://unused", "max_concurrency": 64}
                      for e in engines]})
    signed = SignedManifest.create(Manifest(
        netuid=405, mechid=1, hotkey=keypair.ss58_address, catalog_digest=catalog.load().digest(),
        endpoints=[Endpoint(endpoint_id=e, url=config.endpoint_url(e), model_id="kev-4b", capacity_rps=declared)
                   for e in engines]), keypair)
    app = create_app(config=config, keypair=keypair, manifest=lambda: signed, engines=engines,
                     authorized=lambda hk: False, auditor=lambda: auditor.ss58_address)
    return app, signed.manifest


def test_auditor_credits_what_each_endpoint_delivers_and_proves_wrong_answers(tmp_path) -> None:
    from verathos.validator.discovery import DiscoveredMiner

    auditor_kp = kp()
    keys = {name: kp() for name in ("honest", "over", "shared", "wrong")}
    hosts = {name: f"10.0.0.{i}" for i, name in enumerate(keys, start=1)}
    policy = dataclasses.replace(AuditPolicy(), max_latency_s=0.5, warmup_s=0.5, test_s=2.0, samples_per_test=5)
    chain = FakeChain(head=3 * 180 + 40)

    class CountingReference(MockEngine):
        calls = 0

        async def answer(self, request):
            CountingReference.calls += 1
            return await super().answer(request)

    async def run():
        reference = CountingReference("kev-4b")
        shared = CapacityEngine(MockEngine("kev-4b"), rps=40)
        engines = {"honest": {"gpu0": CapacityEngine(MockEngine("kev-4b"), rps=80)},
                   "over": {"gpu0": CapacityEngine(MockEngine("kev-4b"), rps=20)},
                   "shared": {"gpu0": shared, "gpu1": shared},
                   "wrong": {"gpu0": CapacityEngine(MockEngine("kev-4b", salt="kev-4b-int4"), rps=80)}}
        apps, miners = {}, []
        for uid, name in enumerate(keys, start=1):
            apps[hosts[name]], manifest = miner_app(hosts[name], keys[name], auditor_kp, engines[name], 40.0)
            miners.append(DiscoveredMiner(uid, keys[name].ss58_address, f"http://{hosts[name]}:80", manifest))
        auditor = Auditor(netuid=405, mechid=1, keypair=auditor_kp, chain=chain, policy=policy,
                          catalog=catalog.load, references={}, db_path=tmp_path / "throughput.sqlite",
                          runner=make_runner(HostRouter(apps)))
        auditor.update_slots(slots(miners))
        selection = chain.head_block - policy.lead_blocks
        test = ScheduledTest(3, selection, block_hash(selection).hex(), chain.head_block)
        auditor.follower.poll()
        await auditor.run_test(test, {"kev-4b": reference})
        # two recomputed answers per endpoint at most (the wrong one stops at its first)
        assert 4 <= CountingReference.calls <= 2 * 5
        auditor.settle()
        chain.head_block = 4 * 180 + 1
        assert auditor.verdicts(5) is None  # epoch 5 has not started
        auditor.follower.poll()
        verdicts = auditor.verdicts(4).verdicts

        validator = Throughput(netuid=405, mechid=1, policy=policy, auditor_hotkey=auditor_kp.ss58_address,
                               chain=chain)
        accepted = await validator.proven_invalid(verdicts.invalid, {"kev-4b": reference}, catalog.load())
        agrees = await validator.proven_invalid(verdicts.invalid, {"kev-4b": MockEngine("kev-4b", salt="kev-4b-int4")},
                                                catalog.load())
        tampered = verdicts.invalid[0].model_copy(deep=True)
        tampered.response["answers"] = (await reference.answer(tampered.request))["answers"]
        forged = await validator.proven_invalid([tampered], {}, catalog.load())
        return auditor, verdicts, accepted, agrees, forged

    auditor, verdicts, accepted, agrees, forged = asyncio.run(run())
    credits = {f"{c.hotkey[:6]}/{c.endpoint_id}": c for c in verdicts.credits}
    by_name = {name: k.ss58_address[:6] for name, k in keys.items()}
    honest = credits[f"{by_name['honest']}/gpu0"]
    assert honest.credited_rps == pytest.approx(40.0, rel=0.05) and honest.qualified and honest.test_epoch == 3
    # delivers about half its declaration, so it is credited about a quarter of it
    assert 5 < credits[f"{by_name['over']}/gpu0"].credited_rps < 15
    shares = [credits[f"{by_name['shared']}/{e}"].credited_rps for e in ("gpu0", "gpu1")]
    assert sum(shares) < 30  # one engine cannot pass for two
    wrong = credits[f"{by_name['wrong']}/gpu0"]
    assert wrong.credited_rps == 0.0 and not wrong.qualified
    assert [(e.hotkey, e.endpoint_id) for e in verdicts.invalid] == [(keys["wrong"].ss58_address, "gpu0")]
    assert "differ from the reference" in verdicts.invalid[0].reason
    assert len(accepted) == 1 and agrees == [] and forged == []

    outcomes = {o.outcome for o in auditor.outcomes(keys["over"].ss58_address, 4)}
    assert outcomes == {"partial"}
    assert auditor.store.last_delivered()[f"{keys['honest'].ss58_address}/gpu0"] == pytest.approx(40, rel=0.2)


def test_a_test_whose_selection_block_is_not_final_is_void(tmp_path) -> None:
    chain = FakeChain(head=3 * 180 + 40)
    auditor = Auditor(netuid=405, mechid=1, keypair=kp(), chain=chain, policy=AuditPolicy(), catalog=catalog.load,
                      references={}, db_path=tmp_path / "t.sqlite")
    reorged = ScheduledTest(3, 3 * 180 + 30, "00" * 32, 3 * 180 + 35)
    auditor.store.add_test(reorged)
    auditor.store.add_results(reorged.test_id, [])
    auditor.follower.poll()
    auditor.settle()
    assert auditor.store.unconfirmed() == []
    assert not auditor.store.pending(3, 3)
    chain.head_block = 4 * 180
    auditor.follower.poll()
    assert auditor.verdicts(4).verdicts.credits == []


@pytest.mark.skipif(not loadgen.BINARY_PATH.exists(), reason="verathos-build-loadgen has not been run")
def test_auditor_with_the_load_generator_over_tcp(tmp_path) -> None:
    """The real load generator against miner services on local ports: the honest endpoint passes, the
    over-declaring one is credited less than it serves."""
    import socket
    import threading
    import time

    import uvicorn

    from verathos.validator.discovery import DiscoveredMiner

    auditor_kp, servers, miners = kp(), [], []
    policy = dataclasses.replace(AuditPolicy(), max_latency_s=0.5, warmup_s=1.0, test_s=3.0, samples_per_test=5)
    for uid, rps in ((1, 80), (2, 20)):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        keypair = kp()
        engines = {"gpu0": CapacityEngine(MockEngine("kev-4b"), rps=rps)}
        app, manifest = miner_app("127.0.0.1", keypair, auditor_kp, engines, 40.0, port=port)
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
        threading.Thread(target=server.run, daemon=True).start()
        servers.append(server)
        miners.append(DiscoveredMiner(uid, keypair.ss58_address, f"http://127.0.0.1:{port}", manifest))
    while not all(s.started for s in servers):
        time.sleep(0.05)

    chain = FakeChain(head=3 * 180 + 40)
    auditor = Auditor(netuid=405, mechid=1, keypair=auditor_kp, chain=chain, policy=policy, catalog=catalog.load,
                      references={}, db_path=tmp_path / "throughput.sqlite")
    auditor.update_slots(slots(miners))
    selection = chain.head_block - policy.lead_blocks
    auditor.follower.poll()
    try:
        asyncio.run(auditor.run_test(ScheduledTest(3, selection, block_hash(selection).hex(), chain.head_block),
                                     {"kev-4b": MockEngine("kev-4b")}))
    finally:
        for server in servers:
            server.should_exit = True
    honest, over = (auditor.store.outcomes(m.hotkey, 0)[0] for m in miners)
    assert honest.outcome == "passed" and honest.credited_rps == pytest.approx(40.0, rel=0.05), honest
    # delivers what is left after the warmup backlog, about 17 of 40, and is credited 17 x 17 / 40
    assert over.outcome == "partial" and 3 < over.credited_rps < 12, over
