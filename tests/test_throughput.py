"""The throughput audit: schedule, sessions, admission, the miner's measurement, and the auditor end to end against
in-process miners (honest, over-declaring, sharing one engine, and answering with another model)."""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
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
from verathos.throughput.auditor import Auditor, Checks, Slot, planned_rate
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
    for bad in ({"pass_fraction": 1.5}, {"rate_growth": 0.5}, {"test_s": 400.0}, {"answer_checks_per_test": 11},
                {"retest_growth": 3.0}, {"regrow_after": 0}):
        with pytest.raises(ValueError):
            policy_from(bad, 180)
    # settings this code does not know: an error when publishing, ignored when following
    for unknown in ({"future_setting": 1}, {"epoch_blocks": 100}):
        with pytest.raises(ValueError, match="unknown"):
            policy_from(unknown, 180, strict=True)
        assert policy_from(unknown, 180) == AuditPolicy()
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
    assert planned_rate(slot, 50.0, 90.0, policy, fell_short=True) == 99.0  # probes only 10% beyond after a shortfall


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
                     authorized=lambda hk: False, auditors=lambda: {auditor.ss58_address})
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
        await auditor.run_test(test, Checks({"kev-4b": reference}, {}))
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
        response = json.loads(tampered.response)
        response["answers"] = (await reference.answer(tampered.request))["answers"]
        tampered.response = json.dumps(response)
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
    assert sum(shares) < 40  # one engine (40 requests/s) cannot pass for two
    wrong = credits[f"{by_name['wrong']}/gpu0"]
    assert wrong.credited_rps == 0.0 and not wrong.qualified
    assert [(e.hotkey, e.endpoint_id) for e in verdicts.invalid] == [(keys["wrong"].ss58_address, "gpu0")]
    assert "differ from the reference" in verdicts.invalid[0].reason
    assert len(accepted) == 1 and agrees == [] and forged == []

    outcomes = {o.outcome for o in auditor.outcomes(keys["over"].ss58_address, 4)}
    assert outcomes == {"partial"}
    assert auditor.store.last_delivered()[f"{keys['honest'].ss58_address}/gpu0"][1] == pytest.approx(40, rel=0.2)
    # the over-declarer is probed next only 10% beyond what it delivered, the honest endpoint up to twice
    last = auditor.store.last_delivered(auditor.policy.regrow_after)
    over, honest = f"{keys['over'].ss58_address}/gpu0", f"{keys['honest'].ss58_address}/gpu0"
    assert last[over][2] and not last[honest][2]
    rates = {t.slot.key: t.tested_rps for t in auditor.targets(auditor.policy)}
    assert rates[over] == pytest.approx(min(1.1 * last[over][1], 80.0), rel=0.02)


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
                                     Checks({"kev-4b": MockEngine("kev-4b")}, {})))
    finally:
        for server in servers:
            server.should_exit = True
    honest, over = (auditor.store.outcomes(m.hotkey, 0)[0] for m in miners)
    assert honest.outcome == "passed" and honest.credited_rps == pytest.approx(40.0, rel=0.05), honest
    # delivers what is left after the warmup backlog, about 17 of 40, and is credited 17 x 17 / 40
    assert over.outcome == "partial" and 3 < over.credited_rps < 12, over


def test_a_screen_flag_counts_only_once_the_fp32_reference_confirms_it() -> None:
    request = make_request("kev-4b", KEV.workload, random.Random(5))

    async def run():
        honest = (await MockEngine("kev-4b").answer(request))["answers"]
        wrong = (await MockEngine("kev-4b", salt="other").answer(request))["answers"]
        tol = KEV.tolerance
        screened = Checks({"kev-4b": MockEngine("kev-4b")},
                          {"kev-4b": MockEngine("kev-4b", salt="bf16 noise")}).for_test()

        async def fault(checks, answers, recompute_answers=True):
            return (await checks.check("kev-4b", request, answers, tol, recompute_answers=recompute_answers))[0]

        assert await fault(screened, honest) is None  # flagged by the noisy screen, not confirmed
        assert "differ from the reference" in await fault(screened, wrong)
        assert await fault(screened, wrong, recompute_answers=False) is None  # form only
        assert await fault(screened, {"q": []}, recompute_answers=False) == "malformed answers"
        assert await fault(Checks({}, {"kev-4b": MockEngine("kev-4b")}).for_test(), wrong) is None  # unconfirmable

    asyncio.run(run())


def test_reference_engines_take_a_url_or_a_keyed_tls_endpoint() -> None:
    from verathos.validator.config import ValidatorConfig

    base = {"netuid": 405, "wallet_name": "w", "hotkey_name": "h"}
    cfg = ValidatorConfig.model_validate({**base, "references": {"kev-4b": "http://127.0.0.1:8101"},
                                          "audit_screens": {"kev-4b": {"url": "https://gpu.example.org:8443",
                                                                       "api_key": "k"}}})
    assert cfg.references["kev-4b"].api_key is None and cfg.audit_screens["kev-4b"].api_key == "k"
    with pytest.raises(ValueError, match="needs https"):
        ValidatorConfig.model_validate({**base, "references": {"kev-4b": {"url": "http://203.0.113.7:8101",
                                                                          "api_key": "k"}}})


# ---- review fixes: the auditor stays robust whatever endpoints do ----

def _result(target_id: str, *, completed: int = 160, samples: list | None = None, errors: dict | None = None,
            late_sends: int = 0, sent: int = 160):
    from verathos.throughput.loadgen import TargetResult
    return TargetResult.model_validate({"id": target_id, "sent": sent, "skipped": 0, "completed": completed,
                                        "latency_ms": {"p50": 400.0, "p95": 900.0, "max": 1200.0},
                                        "errors": errors or {}, "late_sends": late_sends, "samples": samples or []})


def _auditor(tmp_path, **policy_overrides):
    policy = dataclasses.replace(AuditPolicy(), **policy_overrides)
    return Auditor(netuid=405, mechid=1, keypair=kp(), chain=FakeChain(head=3 * 180 + 40), policy=policy,
                   catalog=catalog.load, references={}, db_path=tmp_path / "t.sqlite")


def test_a_hostile_response_voids_nothing_but_its_own_endpoint(tmp_path) -> None:
    from verathos.throughput.auditor import Target

    auditor = _auditor(tmp_path)
    test = ScheduledTest(3, 581, block_hash(581).hex(), 586)
    target = Target(Slot("hk", "gpu0", "http://x/e/gpu0", "kev-4b", 10.0), 10.0)
    deep = '{"answers": ' + "[" * 300 + "]" * 300 + ', "receipt": {}}'
    sample = {"request": {"model": "kev-4b"}, "request_id": "ab" * 16, "response": deep}

    async def run(r):
        return await auditor.evaluate(test, target, r, Checks({}, {}).for_test(), auditor.policy)

    hostile = asyncio.run(run(_result("hk/gpu0", samples=[sample])))
    assert hostile.outcome == "partial" and hostile.delivered_rps == 0.0  # no valid receipt: nothing delivered
    assert asyncio.run(run(_result("hk/gpu0", late_sends=5))).outcome == "void"
    assert asyncio.run(run(_result("hk/gpu0", errors={"local": 3}))).reason.startswith("the auditor's load generator")

    async def broken(*args, **kwargs):
        raise RuntimeError("unexpected")

    auditor._evaluate = broken
    assert asyncio.run(run(_result("hk/gpu0"))).outcome == "void"


def test_a_hotkey_cannot_reset_its_test_rate_by_renaming_endpoints(tmp_path) -> None:
    from verathos.throughput.auditor import Result, Target

    auditor = _auditor(tmp_path)
    old = Slot("hk", "gpu0", "http://x/e/gpu0", "kev-4b", 50.0)
    test = ScheduledTest(3, 581, block_hash(581).hex(), 586)
    auditor.store.add_test(test)
    auditor.store.add_results(test.test_id, [Result(Target(old, 50.0), "partial", "", 2.0, 0.16)])
    # the old endpoint delivered 2 requests/s; eight renamed endpoints are tested at 2 x 2 in total
    auditor.update_slots([Slot("hk", f"new{i}", f"http://x/e/new{i}", "kev-4b", 50.0) for i in range(8)])
    rates = [t.tested_rps for t in auditor.targets(auditor.policy)]
    assert sum(rates) == pytest.approx(50.0, abs=0.1)  # at least one first test's worth, never 8 x 50
    auditor.update_slots([Slot("fresh", f"g{i}", f"http://y/e/g{i}", "kev-4b", 50.0) for i in range(8)])
    assert sum(t.tested_rps for t in auditor.targets(auditor.policy)) == pytest.approx(50.0, abs=0.1)


def test_answers_drifting_across_tests_lose_their_credit(tmp_path) -> None:
    from verathos.throughput.auditor import Result, Target

    auditor = _auditor(tmp_path, drift_window_tests=3)
    slot = Slot("hk", "gpu0", "http://x/e/gpu0", "kev-4b", 20.0)
    tol = KEV.tolerance
    for epoch, block in ((1, 200), (2, 380), (3, 560)):
        test = ScheduledTest(epoch, block, block_hash(block).hex(), block + 5)
        auditor.store.add_test(test)
        # each test recomputed 40 probabilities, off by twice the model's mean tolerance
        auditor.store.add_results(test.test_id, [Result(Target(slot, 20.0), "passed", "", 20.0, 20.0, 400.0,
                                                        drift_sum=40 * 2 * tol.mean_abs, drift_count=40)])
        auditor.store.set_state(test.test_id, "confirmed")
    (credit_,) = auditor.store.credits(2, 3, auditor.policy, lambda m: tol)
    assert credit_.drift and credit_.credited_rps == 0.0 and credit_.latency_ms == 400.0


def test_served_verdicts_do_not_change_with_the_policy(tmp_path) -> None:
    auditor = _auditor(tmp_path)
    auditor.follower.poll()
    chain = auditor.chain
    chain.head_block = 4 * 180 + 1
    auditor.follower.poll()
    first = auditor.verdicts(4)
    auditor.set_policy(dataclasses.replace(auditor.policy, credit_ttl_epochs=5))
    assert auditor.verdicts(4) is first


def test_an_auditor_database_of_another_schema_starts_over(tmp_path) -> None:
    import sqlite3

    from verathos.throughput.auditor import SCHEMA_VERSION, Store

    db = sqlite3.connect(tmp_path / "t.sqlite")
    db.executescript("CREATE TABLE tests (test_id TEXT); CREATE TABLE results (old TEXT); PRAGMA user_version = 1;")
    db.close()
    store = Store(tmp_path / "t.sqlite")
    assert store._query("PRAGMA user_version") == [(SCHEMA_VERSION,)]
    assert store.last_delivered() == {}


def test_the_ladder_reaches_max_concurrency_and_a_measurement_steps_aside() -> None:
    from verathos.throughput import measure

    target = Target("gpu0", "kev-4b", MockEngine("kev-4b"), KEV.workload, 48, "f")
    assert measure._ladder(target) == [1, 2, 4, 8, 16, 32, 48]
    with pytest.raises(measure.MeasurementInterrupted):
        asyncio.run(measure.measure_all([target], 1.0, run_s=0.1, stop=lambda: True))


def test_under_pressure_recomputes_reach_as_many_endpoints_as_they_can_in_an_unpredictable_order() -> None:
    """When the reference cannot recompute every endpoint's answers before the deadline, every endpoint's first check
    runs before any second one, and which endpoints miss out differs from test to test."""
    import secrets as secrets_module
    import time as time_module

    from verathos.throughput.auditor import Checks
    from verathos.validator.compare import Tolerance

    class Reference(MockEngine):
        async def answer(self, request):
            await asyncio.sleep(0.01)
            return await super().answer(request)

    request = {"model": "kev-4b", "state": "hello", "questions": {"q": {"type": "noul", "instructions": "x"}}}

    async def one_test() -> dict[int, int]:
        answers = (await MockEngine("kev-4b").answer(request))["answers"]
        recomputes = Checks({"kev-4b": Reference("kev-4b")}, {}).for_test(budget_s=0.15)
        endpoints, checked = 200, {}
        order = list(range(endpoints))
        secrets_module.SystemRandom().shuffle(order)

        async def endpoint(e: int, place: int) -> None:
            for round_ in range(2):
                _, comparison = await recomputes.check("kev-4b", request, answers, Tolerance(),
                                                       recompute_answers=True, rank=round_ * endpoints + place)
                checked[e] = checked.get(e, 0) + (comparison is not None)

        started = time_module.monotonic()
        await asyncio.gather(*(endpoint(e, place) for e, place in zip(range(endpoints), order)))
        assert time_module.monotonic() - started < 1.0  # what does not fit is skipped, nothing waits it out
        return {e: n for e, n in checked.items() if n}

    first, second = asyncio.run(one_test()), asyncio.run(one_test())
    assert 20 < len(first) < 200 and all(n == 1 for n in first.values())  # coverage first: no second checks
    assert set(first) != set(second)  # and not the same endpoints every time
