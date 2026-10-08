"""An endpoint whose last two throughput tests fell short measures again and declares at most what it delivered."""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import threading
import time
from types import SimpleNamespace

import httpx
import pytest
import uvicorn
from bittensor_wallet import Keypair

from tests.test_miner_publisher import CapacityEngine, free_port, scheduled
from tests.test_throughput import FakeChain
from verathos import catalog
from verathos.miner import __main__ as miner_main
from verathos.miner import status as miner_status
from verathos.miner.app import create_app
from verathos.miner.config import MinerConfig
from verathos.miner.peers import Peers, merged
from verathos.miner.shortfall import COOLDOWN_EPOCHS, OUTAGE_SHARE, Shortfalls
from verathos.miner.status import StatusWatcher
from verathos.miner.timing import Deferral, MeasurementDeferred, MeasurementWindow
from verathos.protocol.manifest import Endpoint
from verathos.protocol.receipts import SignedReceipt
from verathos.protocol.requests import sign_request
from verathos.protocol.status import STATUS_DOMAIN, HotkeyStatus, ThroughputOutcome, sign_document
from verathos.throughput import measure
from verathos.throughput.follow import ScheduleFollower
from verathos.throughput.measure import Measurement, MeasurementInterrupted

CATALOG = catalog.load()
REQUEST = {"model": "kev-4b", "state": "Hi, we were billed twice for March.",
           "questions": {"churn": {"type": "noul", "instructions": "Does the customer threaten to cancel?"}}}


def kp() -> Keypair:
    return Keypair.create_from_mnemonic(Keypair.generate_mnemonic())


def outcome(epoch: int, kind: str = "partial", delivered: float = 10.5, *, rate: float = 11.4,
            tested: float | None = None, state: str = "confirmed", endpoint_id: str = "gpu0") -> ThroughputOutcome:
    """A test result as the auditor's status lists it; ``rate``: the declared rate, tested unless ``tested``."""
    tested = rate if tested is None else tested
    return ThroughputOutcome(endpoint_id=endpoint_id, epoch=epoch, start_block=epoch * 180 + 40, state=state,
                             outcome=kind, declared_rps=rate, tested_rps=tested, delivered_rps=delivered,
                             credited_rps=round(tested * min(delivered / tested, 1.0) ** 2, 2))


def test_two_tests_short_of_the_rate_tested_trigger_a_measurement_and_voids_do_not_count(tmp_path) -> None:
    shortfalls = Shortfalls(tmp_path / "shortfalls.json")
    declared = {"gpu0": 11.4}
    due = shortfalls.due([outcome(45416, "passed", 11.4), outcome(45417, delivered=10.5),
                          outcome(45418, delivered=10.4)], declared)
    assert [(s.endpoint_id, [t.epoch for t in s.tests], s.cap_rps) for s in due] == [("gpu0", [45417, 45418], 10.5)]
    assert due[0].line() == ("gpu0 fell short in tests 45417 and 45418 (delivered 10.5 and 10.4 of 11.4): measuring "
                             "again")
    mixed = [outcome(45417, delivered=10.5, tested=11.0), outcome(45418, delivered=10.4)]
    assert "(delivered 10.5 of 11.0 and 10.4 of 11.4)" in shortfalls.due(mixed, declared)[0].line()

    not_due = {
        "one shortfall": [outcome(45417, "passed", 11.4), outcome(45418)],
        "a single test": [outcome(45418)],
        "a pass between": [outcome(45416), outcome(45417, "passed", 11.4), outcome(45418)],
        "a wrong answer": [outcome(45417, "invalid", 0.0), outcome(45418)],
        "the latest not confirmed yet": [outcome(45416), outcome(45417),
                                         outcome(45418, "passed", 11.4, state="unconfirmed")],
        "unconfirmed": [outcome(45417), outcome(45418, state="unconfirmed")],
        "another declaration": [outcome(45417, rate=12.0), outcome(45418, rate=12.0)],
        "the tested rate delivered (users sent more)": [outcome(45417, delivered=11.4),
                                                        outcome(45418, delivered=11.5)],
        "another endpoint": [outcome(45417, endpoint_id="gpu1"), outcome(45418, endpoint_id="gpu1")],
    }
    for why, tests in not_due.items():
        assert shortfalls.due(tests, declared) == [], why

    # A void test, or a result void for this endpoint, is not counted: the shortfalls around it are the last two.
    void_between = [outcome(45415), outcome(45416, state="void"), outcome(45417, "void", 0.0),
                    outcome(45418, delivered=10.4)]
    due = shortfalls.due(void_between, declared)
    assert [t.epoch for t in due[0].tests] == [45415, 45418]
    shortfalls.record(due)
    assert shortfalls.cap("gpu0").rps == 10.5 and shortfalls.cap("gpu0").epochs == (45415, 45418)


def test_an_endpoint_is_measured_again_at_most_once_per_cooldown_and_never_twice_for_a_test(tmp_path) -> None:
    path = tmp_path / "shortfalls.json"
    shortfalls = Shortfalls(path)
    first = [outcome(45417), outcome(45418, delivered=10.4)]
    shortfalls.record(shortfalls.due(first, {"gpu0": 11.4}))
    assert shortfalls.due(first, {"gpu0": 11.4}) == []  # the same tests, even at the same declaration
    restarted = Shortfalls(path)  # kept across restarts
    assert restarted.cap("gpu0").rps == 10.5
    at_cap = {"gpu0": 10.5}
    soon = first + [outcome(45419, delivered=9.6, rate=10.5), outcome(45420, delivered=9.5, rate=10.5)]
    assert 45420 < 45418 + COOLDOWN_EPOCHS and restarted.due(soon, at_cap) == []
    later = first + [outcome(45420, delivered=9.6, rate=10.5), outcome(45421, delivered=9.5, rate=10.5)]
    due = restarted.due(later, at_cap)
    assert [t.epoch for t in due[0].tests] == [45420, 45421] and due[0].cap_rps == 9.6
    straddling = first + [outcome(45421, delivered=9.6, rate=10.5)]  # only one test after the last measurement
    assert restarted.due(straddling, at_cap) == []

    path.write_text("not json")
    assert Shortfalls(path).cap("gpu0") is None  # a damaged file caps nothing


def test_tests_in_which_the_endpoint_was_as_good_as_down_set_no_cap_but_start_the_cooldown(tmp_path) -> None:
    shortfalls = Shortfalls(tmp_path / "shortfalls.json")
    declared = {"gpu0": 11.4}
    half = OUTAGE_SHARE * 11.4

    def due(*delivered: float, tested: tuple[float | None, ...] = (None, None), first: int = 45417):
        """The shortfall of two tests at the declared rate, tested at it unless ``tested``."""
        [s] = shortfalls.due([outcome(first + i, delivered=d, rate=declared["gpu0"], tested=t)
                              for i, (d, t) in enumerate(zip(delivered, tested))], declared)
        return s

    outage = due(0.3, 0.0)
    assert outage.outage and outage.cap_rps is None
    assert outage.line() == ("gpu0 delivered 0.3 and 0.0 of 11.4 in tests 45417 and 45418, as if it was down: "
                             "measuring again without a cap")
    slowdown = due(10.5, 10.4)
    assert not slowdown.outage and slowdown.cap_rps == 10.5 and "fell short" in slowdown.line()
    assert due(half, 0.0).cap_rps == half  # at the share: a slowdown, capped at what it delivered
    assert due(half - 0.01, 0.0).cap_rps is None  # below it in both tests: an outage
    assert due(half - 0.01, 6.0).cap_rps == 6.0  # one test at half or more: a slowdown
    assert due(5.0, 5.0, tested=(9.0, 11.4)).cap_rps == 5.0  # each test against its own tested rate
    assert due(4.0, 5.0, tested=(9.0, 11.4)).cap_rps is None

    shortfalls.record([due(10.5, 10.4)])
    assert shortfalls.cap("gpu0").rps == 10.5
    declared["gpu0"] = 10.5
    shortfalls.record([due(0.3, 0.0, first=45421)])  # an outage keeps the earlier cap
    assert (shortfalls.cap("gpu0").rps, shortfalls.cap("gpu0").epochs) == (10.5, (45417, 45418))
    restarted = Shortfalls(tmp_path / "shortfalls.json")
    assert restarted.cap("gpu0").epochs == (45417, 45418)
    down_again = [outcome(45421, delivered=0.3, rate=10.5), outcome(45422, delivered=0.0, rate=10.5),
                  outcome(45423, delivered=0.0, rate=10.5), outcome(45424, delivered=0.0, rate=10.5)]
    assert restarted.due(down_again, declared) == []  # cooldown: not before epoch 45425
    assert restarted.due(down_again + [outcome(45425, delivered=0.0, rate=10.5)], declared)[0].outage


def publisher_with(tmp_path, rps: dict[str, float], *, role: str = "primary", host: str = "203.0.113.5",
                   port: int = 8091) -> tuple[miner_main.Publisher, list[bool]]:
    """A publisher whose measurement gives ``rps`` per endpoint (stored, and reused unless asked to measure again);
    -> (publisher, the ``remeasure`` flag of each measurement)."""
    config = MinerConfig.model_validate({
        "netuid": 405, "wallet_name": "w", "hotkey_name": "h", "public_host": host, "port": port, "role": role,
        "state_dir": str(tmp_path),
        "endpoints": [{"endpoint_id": e, "model_id": "kev-4b", "engine_url": "http://127.0.0.1:9"} for e in rps]})
    keypair = kp()
    publisher = miner_main.Publisher(config, SimpleNamespace(hotkey=keypair), CATALOG, publish_on_chain=True,
                                     subtensor=SimpleNamespace(get_subnet_owner_hotkey=lambda netuid: "auditor"))
    calls: list[bool] = []

    async def measuring(endpoints, cat, remeasure, stop):
        calls.append(remeasure)
        kept = publisher.store.load()
        if not remeasure and all(e.endpoint_id in kept for e in endpoints):
            return {e.endpoint_id: kept[e.endpoint_id] for e in endpoints}, {}, False
        measured = {e.endpoint_id: Measurement(e.endpoint_id, "f", rps[e.endpoint_id], 8, (rps[e.endpoint_id],) * 5,
                                               time.time()) for e in endpoints}
        publisher.store.save(measured)
        return measured, {}, True

    publisher._measure = measuring
    return publisher, calls


def declared(publisher: miner_main.Publisher) -> dict[str, float]:
    return {e.endpoint_id: e.capacity_rps for e in publisher.local.host.endpoints}


def test_the_new_declaration_is_at_most_what_the_tests_saw_until_a_measurement_for_another_reason(
        tmp_path, caplog) -> None:
    caplog.set_level(logging.INFO, logger="verathos.miner")
    rps = {"gpu0": 12.0, "gpu1": 12.0}
    publisher, calls = publisher_with(tmp_path, rps)
    assert publisher.adopt(CATALOG) and declared(publisher) == {"gpu0": 11.4, "gpu1": 11.4}
    publisher.note_tests([outcome(45417, delivered=10.5), outcome(45418, delivered=10.4),
                          outcome(45418, "passed", 11.4, endpoint_id="gpu1")])
    publisher.note_test_traffic()
    assert not publisher.measure_shortfalls()  # not while a test runs
    publisher.test_seen = float("-inf")
    assert publisher.measure_shortfalls() and calls[-1] is True  # every endpoint of the machine, in the same rounds
    assert declared(publisher) == {"gpu0": 10.5, "gpu1": 11.4}  # still measures 12.0, but the tests saw 10.5
    assert publisher.signed.manifest.endpoints[0].capacity_rps == 10.5
    messages = [r.getMessage() for r in caplog.records]
    assert "gpu0 fell short in tests 45417 and 45418 (delivered 10.5 and 10.4 of 11.4): measuring again" in messages
    assert any(m.startswith("endpoint gpu0 (kev-4b): 12.0 requests/s") and m.endswith(
        "declares 10.5, at most the 10.5 it delivered in tests 45417 and 45418") for m in messages), messages
    count = len(calls)
    assert not publisher.measure_shortfalls() and len(calls) == count  # those tests were at 11.4, not at 10.5

    restarted, calls = publisher_with(tmp_path, rps)
    assert restarted.adopt(CATALOG) and calls == [False]  # the stored measurement, with its cap
    assert declared(restarted) == {"gpu0": 10.5, "gpu1": 11.4}

    rps["gpu0"] = 10.0  # the cap bounds, it does not set: a lower measurement declares less
    restarted.note_tests([outcome(45421, delivered=11.0, endpoint_id="gpu1"),
                          outcome(45422, delivered=11.1, endpoint_id="gpu1")])
    assert restarted.measure_shortfalls()
    assert declared(restarted) == {"gpu0": 9.5, "gpu1": 11.1}  # gpu0's cap holds too: measured for a shortfall

    assert restarted.adopt(CATALOG, remeasure=True)  # measured for another reason (--remeasure, an engine changed)
    assert declared(restarted) == {"gpu0": 9.5, "gpu1": 11.4}
    rps["gpu0"] = 12.0
    assert restarted.adopt(CATALOG, remeasure=True) and declared(restarted) == {"gpu0": 11.4, "gpu1": 11.4}
    assert Shortfalls(tmp_path / "shortfalls.json").cap("gpu0") is None


def test_a_measurement_a_test_interrupts_is_tried_again(tmp_path) -> None:
    publisher, calls = publisher_with(tmp_path, {"gpu0": 12.0})
    assert publisher.adopt(CATALOG)
    measuring = publisher._measure

    async def interrupted(*args, **kwargs):
        raise MeasurementInterrupted()

    publisher._measure = interrupted
    publisher.note_tests([outcome(45417), outcome(45418)])
    assert publisher.measure_shortfalls() and publisher.interrupted
    assert publisher.shortfalls.cap("gpu0") is None and declared(publisher) == {"gpu0": 11.4}
    publisher._measure = measuring
    assert publisher.measure_shortfalls() and declared(publisher) == {"gpu0": 10.5}


def test_an_endpoint_that_was_down_in_both_tests_is_measured_again_without_a_cap(tmp_path, caplog) -> None:
    caplog.set_level(logging.INFO, logger="verathos.miner")
    rps = {"gpu0": 12.0}
    publisher, calls = publisher_with(tmp_path, rps)
    assert publisher.adopt(CATALOG)
    rps["gpu0"] = 11.0  # back up, a little slower than before
    publisher.note_tests([outcome(45417, delivered=0.3), outcome(45418, delivered=0.0)])
    assert publisher.measure_shortfalls() and calls[-1] is True
    assert declared(publisher) == {"gpu0": 10.45} and publisher.shortfalls.cap("gpu0") is None
    messages = [r.getMessage() for r in caplog.records]
    assert ("gpu0 delivered 0.3 and 0.0 of 11.4 in tests 45417 and 45418, as if it was down: measuring again without "
            "a cap") in messages
    assert any(m.startswith("endpoint gpu0 (kev-4b): 11.0 requests/s") and m.endswith("declares 10.4")
               for m in messages), messages
    count = len(calls)
    publisher.note_tests([outcome(45417, delivered=0.3, rate=10.45), outcome(45418, delivered=0.0, rate=10.45)])
    assert not publisher.measure_shortfalls() and len(calls) == count  # never twice for the same tests


def test_a_changed_declaration_is_published_by_the_primary_and_served_by_a_worker(tmp_path, monkeypatch,
                                                                                  lab_network) -> None:
    published = []
    monkeypatch.setattr(miner_main, "committed_catalog", lambda st, netuid: CATALOG.digest())
    monkeypatch.setattr(miner_main, "publish", lambda st, wallet, config, signed: published.append(signed.manifest))
    chain = SimpleNamespace(get_subnet_owner_hotkey=lambda netuid: "auditor")
    primary, _ = publisher_with(tmp_path / "primary", {"gpu0": 12.0})
    primary.engines_changed = lambda: False
    assert primary.adopt(CATALOG)
    primary.publish(chain)
    store = catalog.CatalogStore(tmp_path / "catalog")
    assert primary.follow(chain, store) is chain and len(published) == 1  # nothing changed: nothing published
    primary.note_tests([outcome(45417), outcome(45418, delivered=10.4)])
    assert primary.follow(chain, store) is chain
    assert [e.capacity_rps for e in published[-1].endpoints] == [10.5] and len(published) == 2
    assert json.loads((tmp_path / "primary" / "manifest.json").read_text())["manifest"]["endpoints"][0][
        "capacity_rps"] == 10.5

    worker, _ = publisher_with(tmp_path / "worker", {"w-gpu0": 12.0}, role="worker", host="198.51.100.7", port=20017)
    worker.engines_changed = lambda: False
    assert worker.adopt(CATALOG)
    hotkey = worker.wallet.hotkey
    app = create_app(config=worker.config, keypair=hotkey, manifest=lambda: worker.signed, engines={},
                     authorized=lambda hk: False, auditors=worker.auditors, host_endpoints=lambda: worker.local)
    peers = Peers(["http://198.51.100.7:20017"], hotkey=hotkey.ss58_address, netuid=405, mechid=1)

    def primary_lists() -> list[Endpoint]:
        async def poll():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as client:
                await peers.poll(client, CATALOG.digest(), 3600.0)

        asyncio.run(poll())
        return merged([], peers.peers, CATALOG)[0]

    assert [e.capacity_rps for e in primary_lists()] == [11.4]
    worker.note_tests([outcome(45417, endpoint_id="w-gpu0"), outcome(45418, delivered=10.4, endpoint_id="w-gpu0")])
    assert worker.follow(chain, store) is chain and not worker.publish_on_chain
    assert [(e.endpoint_id, e.capacity_rps) for e in primary_lists()] == [("w-gpu0", 10.5)]


def test_only_tests_in_a_status_the_subnet_owner_signed_make_the_miner_measure_again(tmp_path, monkeypatch) -> None:
    publisher, calls = publisher_with(tmp_path, {"gpu0": 12.0})
    assert publisher.adopt(CATALOG)
    owner, hotkey = kp(), publisher.wallet.hotkey.ss58_address
    publisher.auditor = owner.ss58_address
    status = HotkeyStatus(hotkey=hotkey, tests=[outcome(45417), outcome(45418)]).model_dump(mode="json")
    served = [status]  # an auditor of an older release: unsigned
    monkeypatch.setattr(miner_status, "get_json", lambda url, **kwargs: served[-1])
    watcher = StatusWatcher(hotkey=hotkey, netuid=405, auditor_url=lambda: "http://192.0.2.1:8091",
                            auditor_hotkey=lambda: publisher.auditor, listed=publisher.own_ids,
                            on_tests=publisher.note_tests)
    watcher.poll_once()
    assert publisher.tests == [] and not publisher.measure_shortfalls() and len(calls) == 1
    served.append(sign_document(status, kp(), netuid=405, domain=STATUS_DOMAIN, now_ms=int(time.time() * 1000)))
    watcher.poll_once()  # signed, but not by the subnet owner
    assert publisher.tests == [] and not publisher.measure_shortfalls() and len(calls) == 1
    served.append(sign_document(status, owner, netuid=405, domain=STATUS_DOMAIN, now_ms=int(time.time() * 1000)))
    watcher.poll_once()
    assert publisher.tests == [outcome(45417), outcome(45418)]
    assert publisher.measure_shortfalls() and declared(publisher) == {"gpu0": 10.5}


def test_a_measurement_for_a_shortfall_waits_for_a_selected_test_and_a_first_start_measures_at_once(
        tmp_path, monkeypatch, caplog) -> None:
    caplog.set_level(logging.INFO, logger="verathos.miner")
    monkeypatch.setattr(miner_main, "measure_and_store", functools.partial(measure.measure_and_store, run_s=0.2))
    monkeypatch.setattr(miner_main, "wait_ready", functools.partial(measure.wait_ready, timeout_s=0.5))
    keypair, port, policy = kp(), free_port(), CATALOG.audit_policy()
    config = MinerConfig.model_validate({
        "netuid": 405, "wallet_name": "w", "hotkey_name": "h", "public_host": "203.0.113.5", "port": port,
        "listen_host": "127.0.0.1", "state_dir": str(tmp_path),
        "endpoints": [{"endpoint_id": "gpu0", "model_id": "kev-4b", "engine_url": "http://127.0.0.1:9",
                       "max_concurrency": 8}]})
    publisher = miner_main.Publisher(config, SimpleNamespace(hotkey=keypair), CATALOG, publish_on_chain=False,
                                     subtensor=SimpleNamespace(get_subnet_owner_hotkey=lambda netuid: "auditor"))
    test = scheduled(1, policy)
    chain = FakeChain(head=test.selection_block)  # the test of epoch 1 is selected, its traffic still to come
    publisher.window = MeasurementWindow(ScheduleFollower(chain, policy))
    engine = CapacityEngine(rps=100)
    app = create_app(config=config, keypair=keypair, manifest=lambda: publisher.signed, engines={"gpu0": engine},
                     authorized=lambda hk: False, auditors=publisher.auditors, admissions=publisher.admissions,
                     serving=lambda: publisher.serving)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.05)
    try:
        assert publisher.adopt(CATALOG)  # a first start measures at once: no test reaches this machine yet
        rate = declared(publisher)["gpu0"]
        publisher.note_tests([outcome(45417, delivered=round(0.8 * rate, 2), rate=rate),
                              outcome(45418, delivered=round(0.7 * rate, 2), rate=rate)])
        calls = engine.calls
        for _ in range(2):  # two rounds of the miner's watch
            assert publisher.measure_shortfalls() and publisher.interrupted
        assert engine.calls == calls + 2  # one readiness request each, no measurement
        assert publisher.shortfalls.cap("gpu0") is None and declared(publisher)["gpu0"] == rate
        chain.head_block = test.start_block + 1  # its traffic runs
        assert publisher.measure_shortfalls() and publisher.interrupted and engine.calls == calls + 3
        messages = caplog.messages
        assert sum("fell short in tests 45417 and 45418" in m for m in messages) == 1  # logged once while it waits
        assert messages.count(f"the throughput test of epoch 1 is selected, its traffic from block {test.start_block}: "
                              "measuring after it") == 1
        chain.head_block = test.start_block + policy.traffic_blocks + 2  # its traffic is over
        assert publisher.measure_shortfalls() and not publisher.interrupted
        assert engine.calls > calls + 50 and declared(publisher)["gpu0"] == round(0.8 * rate, 2)
    finally:
        server.should_exit = True


def test_while_a_measurement_waits_the_miner_says_once_why_it_is_wanted(tmp_path, monkeypatch, caplog) -> None:
    caplog.set_level(logging.INFO, logger="verathos.miner")
    publisher, _ = publisher_with(tmp_path, {"gpu0": 12.0})
    assert publisher.adopt(CATALOG)
    measuring, waiting = publisher._measure, "the throughput test of epoch 1 is selected: measuring after it"

    async def deferred(*args, **kwargs):
        raise MeasurementDeferred(Deferral(1, waiting))

    publisher._measure, publisher.engines_changed = deferred, lambda: True
    chain, store = SimpleNamespace(get_subnet_owner_hotkey=lambda netuid: "auditor"), catalog.CatalogStore(tmp_path)
    monkeypatch.setattr(miner_main, "committed_catalog", lambda st, netuid: CATALOG.digest())
    monkeypatch.setattr(miner_main, "publish", lambda st, wallet, config, signed: None)
    publisher.publish(chain)
    caplog.clear()
    for _ in range(3):  # rounds of the miner's watch
        assert publisher.follow(chain, store) is chain and publisher.interrupted
    assert caplog.messages == ["an engine changed; measuring again", waiting]

    newer = CATALOG.model_copy(update={"scoring": CATALOG.scoring.model_copy(update={"ema_alpha": 0.3})})
    monkeypatch.setattr(miner_main, "committed_catalog", lambda st, netuid: newer.digest())
    monkeypatch.setattr(miner_main, "current_catalog", lambda *args: newer)
    caplog.clear()
    for _ in range(3):
        assert publisher.follow(chain, store) is chain and publisher.interrupted
    assert caplog.messages == [f"catalog {newer.digest()[:12]} committed; adopted after the measurement it needs"]
    publisher._measure = measuring
    caplog.clear()
    assert publisher.follow(chain, store) is chain and not publisher.interrupted
    assert f"catalog {newer.digest()[:12]} committed; endpoints rebuilt" in caplog.messages
    publisher._measure = deferred  # the next wait says again why
    caplog.clear()
    assert publisher.follow(chain, store) is chain and publisher.interrupted
    assert caplog.messages == ["an engine changed; measuring again", waiting]


def test_users_are_served_while_an_endpoint_that_fell_short_is_measured_again(tmp_path, monkeypatch, caplog) -> None:
    """The measurement for a shortfall takes the path of any other: through the service, all endpoints in the same
    rounds, its requests behind the users'."""
    caplog.set_level(logging.INFO, logger="verathos.miner")
    monkeypatch.setattr(miner_main, "measure_and_store", functools.partial(measure.measure_and_store, run_s=0.2))
    monkeypatch.setattr(miner_main, "wait_ready", functools.partial(measure.wait_ready, timeout_s=0.5))
    keypair, gateway = kp(), kp()
    port = free_port()
    config = MinerConfig.model_validate({
        "netuid": 405, "wallet_name": "w", "hotkey_name": "h", "public_host": "203.0.113.5", "port": port,
        "listen_host": "127.0.0.1", "state_dir": str(tmp_path),
        "endpoints": [{"endpoint_id": e, "model_id": "kev-4b", "engine_url": "http://127.0.0.1:9", "max_concurrency": 8}
                      for e in ("gpu0", "gpu1")]})
    publisher = miner_main.Publisher(config, SimpleNamespace(hotkey=keypair), CATALOG, publish_on_chain=False,
                                     subtensor=SimpleNamespace(get_subnet_owner_hotkey=lambda netuid: "auditor"))
    engines = {"gpu0": CapacityEngine(rps=100), "gpu1": CapacityEngine(rps=100)}
    app = create_app(config=config, keypair=keypair, manifest=lambda: publisher.signed, engines=engines,
                     authorized=lambda hk: hk == gateway.ss58_address, auditors=publisher.auditors,
                     admissions=publisher.admissions, serving=lambda: publisher.serving,
                     gateways=lambda: {gateway.ss58_address})
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.05)
    try:
        assert publisher.adopt(CATALOG)
        rate = declared(publisher)["gpu0"]
        publisher.note_tests([outcome(45417, delivered=round(0.5 * rate, 2), rate=rate),
                              outcome(45418, delivered=round(0.4 * rate, 2), rate=rate)])
        calls = {e: engine.calls for e, engine in engines.items()}
        measuring = threading.Thread(target=publisher.measure_shortfalls)
        measuring.start()
        deadline = time.monotonic() + 10
        while engines["gpu0"].calls < calls["gpu0"] + 50 and time.monotonic() < deadline:
            time.sleep(0.02)
        path = "/e/gpu0/v1/systemone"
        body = json.dumps(REQUEST).encode()
        r = httpx.post(f"http://127.0.0.1:{port}{path}", content=body, timeout=10, headers={
            "Content-Type": "application/json", **sign_request(gateway, "POST", path, body, keypair.ss58_address)})
        assert measuring.is_alive()  # answered during the measurement
        assert r.status_code == 200 and SignedReceipt.model_validate(r.json()["receipt"]).signature_valid()
        measuring.join()
        assert all(engine.calls > calls[e] + 50 for e, engine in engines.items())  # both measured again
        assert declared(publisher)["gpu0"] == pytest.approx(0.5 * rate, abs=0.01)
        assert declared(publisher)["gpu1"] == pytest.approx(0.95 * 100, rel=0.2)
        assert any("fell short in tests 45417 and 45418" in r.getMessage() for r in caplog.records)
    finally:
        server.should_exit = True
