"""What a miner sees about its endpoints: the validator's status, signed, cached and limited, and the miner's log
lines."""

from __future__ import annotations

import logging
import time

from bittensor_wallet import Keypair
from fastapi import FastAPI
from fastapi.testclient import TestClient

from verathos.miner import status as miner_status
from verathos.miner.status import StatusWatcher, describe_test
from verathos.protocol.manifest import Endpoint
from verathos.protocol.status import (
    NETWORK_DOMAIN,
    NETWORK_PATH,
    STATUS_DOMAIN,
    STATUS_MAX_AGE_S,
    STATUS_PATH,
    HotkeyStatus,
    NetworkStatus,
    ThroughputOutcome,
    sign_document,
    unverified,
)
from verathos.validator import status
from verathos.validator.audit import EndpointResult
from verathos.validator.compare import RequestComparison, Tolerance
from verathos.validator.discovery import DiscoveredEndpoint
from verathos.validator.scoring import ScoreState
from verathos.validator.service import (
    STATUS_BURST,
    STATUS_EVERY_S,
    AddressLimit,
    add_network_route,
    add_status_route,
    address_group,
)
from verathos.validator.status import StatusBoard


def kp() -> Keypair:
    return Keypair.create_from_mnemonic(Keypair.generate_mnemonic())


def watching(listed: list[str], **kwargs) -> StatusWatcher:
    return StatusWatcher(hotkey="hkA", netuid=405, auditor_url=lambda: None, auditor_hotkey=lambda: None,
                         listed=lambda: listed, **kwargs)


def endpoint(endpoint_id: str, model: str = "kev-4b", hotkey: str = "hkA") -> DiscoveredEndpoint:
    return DiscoveredEndpoint(1, hotkey, "http://x", Endpoint(endpoint_id=endpoint_id, url=f"http://x/e/{endpoint_id}",
                              model_id=model, capacity_rps=50.0))


def result(ep: DiscoveredEndpoint) -> EndpointResult:
    r = EndpointResult(ep, Tolerance(min_probabilities=1), identity_ok=True, sent=10, answered=10,
                       latencies_ms=[40.0] * 10)
    r.tally.add(RequestComparison(max_abs=0.01, sum_abs=0.06, count=60, confident_flips=0))
    return r


def test_each_endpoint_gets_one_state() -> None:
    eps = {name: endpoint(name, "clef-flash" if name == "cand" else "kev-4b")
           for name in ("ok", "new", "cand", "failed", "probe", "noref")} | {"bad": endpoint("gpu0", hotkey="hkB")}
    results = {e.key: result(e) for name, e in eps.items() if name != "noref"}
    state = ScoreState(ema={"hkA/ok": 0.6}, probation={"hkA/probe": 3, "hkA/failed": 6, "hkB/gpu0": 6},
                       window={k: [[0.06, 60.0]] for k in results})
    credited = {k: 45.0 for k in results if k != "hkA/new"}
    board = status.build(epoch=13, epoch_blocks=180, endpoints=list(eps.values()), results=results,
                         scores={"hkA/ok": 45.0}, failed={"hkA/failed", "hkB/gpu0"}, invalid_hotkeys={"hkB"},
                         credited=credited, scored_models={"kev-4b"}, state=state, weights={1: 0.75})
    assert (board["hkA"].uid, board["hkA"].weight) == (1, 0.75)
    states = {e.endpoint_id: e.state for e in board["hkA"].endpoints}
    assert states == {"ok": "scored", "new": "qualifying", "cand": "candidate", "failed": "failed",
                      "probe": "probation", "noref": "not_audited"}
    assert board["hkB"].endpoints[0].state == "invalid"
    ok = next(e for e in board["hkA"].endpoints if e.endpoint_id == "ok")
    assert (ok.declared_rps, ok.credited_rps) == (50.0, 45.0)


def test_a_miner_is_told_why_its_credit_is_zero_and_what_users_did_to_its_score() -> None:
    from verathos.miner.status import describe_endpoint

    eps = {name: endpoint(name) for name in ("ok", "drifted")}
    results = {e.key: result(e) for e in eps.values()}
    state = ScoreState(window={k: [[0.06, 60.0]] for k in results})
    board = status.build(epoch=13, epoch_blocks=180, endpoints=list(eps.values()), results=results,
                         scores={"hkA/ok": 0.4, "hkA/drifted": 0.0}, failed=set(), invalid_hotkeys=set(),
                         credited={"hkA/ok": 45.0, "hkA/drifted": 0.0}, scored_models={"kev-4b"}, state=state,
                         drifted={"hkA/drifted"}, user_availability={"hkA/ok": 0.93, "hkA/drifted": 1.0},
                         latency_factors={"hkA/ok": 1.12})
    by_id = {e.endpoint_id: e for e in board["hkA"].endpoints}
    assert (by_id["drifted"].state, by_id["drifted"].drift) == ("drifted", True)
    assert (by_id["ok"].state, by_id["ok"].user_availability, by_id["ok"].latency_factor) == ("scored", 0.93, 1.12)
    assert "NO CREDIT" in describe_endpoint(by_id["drifted"]) and "drifted" in describe_endpoint(by_id["drifted"])
    assert "ready for users 93%, latency factor 1.12" in describe_endpoint(by_id["ok"])
    capped = ThroughputOutcome(endpoint_id="gpu0", epoch=7, start_block=1300, state="confirmed", outcome="passed",
                               reason="its answers to users in the test are unchecked", credited_rps=10.0)
    assert describe_test(capped).endswith("credited 10.0 (its answers to users in the test are unchecked)")


def test_miner_logs_each_epoch_once_and_each_final_test_once() -> None:
    doc = HotkeyStatus(hotkey="hkA", epoch=13, epoch_blocks=180, endpoints=[
        {"endpoint_id": "gpu0", "model_id": "kev-4b", "state": "qualifying", "sent": 10, "answered": 10,
         "verdict": "pass", "mean_abs": 0.0014, "declared_rps": 50.0},
        {"endpoint_id": "gpu1", "model_id": "kev-4b", "state": "scored", "score": 0.441, "ema": 0.302,
         "declared_rps": 50.0, "credited_rps": 45.0, "sent": 10, "answered": 10, "verdict": "pass"}],
        tests=[{"endpoint_id": "gpu1", "epoch": 12, "start_block": 2200, "state": "confirmed", "outcome": "passed",
                "declared_rps": 50.0, "tested_rps": 45.0, "delivered_rps": 46.2, "credited_rps": 45.0,
                "latency_p95_ms": 830.0},
               {"endpoint_id": "gpu0", "epoch": 13, "start_block": 2410, "state": "unconfirmed",
                "outcome": "passed"}])
    watcher = watching(["gpu0", "gpu1", "gpu2"])
    lines = watcher.lines(doc)
    assert lines[0] == "validator's epoch 13:"
    assert "qualifying, checked but not scored until it passes a throughput test" in lines[1]
    assert ("scored 44.10% of the epoch's emission, average 30.20%; proven 45.0 of 50.0 requests/s declared"
            in lines[2])
    assert "gpu2: not in that epoch" in lines[3]
    shared = doc.model_copy(update={"model_shares": {"kev-4b": 0.6, "kev-9b": 0.4}})
    fresh = watching([])
    assert fresh.lines(shared)[1] == "  emission shares: kev-4b 60.0%, kev-9b 40.0%"
    assert watching([]).lines(shared.model_copy(update={"miner_share": 0.59}))[2] == (
        "  miner share 59.00%: the scores below are shares of it; the subnet owner's UID gets the rest of the miners' "
        "emission")
    assert not any("miner share" in line for line in watching([]).lines(shared.model_copy(update={"miner_share": 1})))
    assert lines[4] == ("throughput test of epoch 12 (block 2200), gpu1: passed, delivered 46.2 of 45.0 requests/s, "
                        "p95 latency 830 ms; credited 45.0")
    assert watcher.lines(doc) == []  # nothing new; the unconfirmed test is not final yet


def test_each_machine_logs_the_endpoints_it_serves() -> None:
    """The hotkey's status lists the endpoints of all its machines; each machine logs its own."""
    doc = HotkeyStatus(hotkey="hkA", epoch=13, epoch_blocks=180, endpoints=[
        {"endpoint_id": e, "model_id": "kev-4b", "state": "qualifying"} for e in ("gpu0", "w1-gpu0")],
        tests=[{"endpoint_id": e, "epoch": 12, "start_block": 2200, "state": "confirmed", "outcome": "passed"}
               for e in ("gpu0", "w1-gpu0")])
    lines = watching(["w1-gpu0"]).lines(doc)
    assert len(lines) == 3 and lines[1].startswith("  w1-gpu0 (kev-4b): qualifying") and "w1-gpu0: passed" in lines[2]


def test_test_lines_name_the_reason() -> None:
    partial = ThroughputOutcome(endpoint_id="gpu1", epoch=7, start_block=1300, state="confirmed", outcome="partial",
                          reason="delivered 21.0 of 40.0 requests/s", tested_rps=40.0, delivered_rps=21.0,
                          credited_rps=21.0)
    assert describe_test(partial).startswith("throughput test of epoch 7 (block 1300), gpu1: BELOW the tested rate")
    void = partial.model_copy(update={"state": "void", "reason": "the selection block was not finalized"})
    assert describe_test(void).endswith("not counted (the selection block was not finalized)")


def test_after_a_start_the_miner_logs_each_endpoints_latest_test_then_every_new_one() -> None:
    def final(endpoint_id: str, epoch: int) -> dict:
        return {"endpoint_id": endpoint_id, "epoch": epoch, "start_block": epoch * 180 + 40, "state": "confirmed",
                "outcome": "passed"}

    watcher = watching(["gpu0", "gpu1"])
    doc = HotkeyStatus(hotkey="hkA", tests=[final("gpu0", e) for e in (10, 11, 12)] + [final("gpu1", 11)])
    logged = [line.split(",")[0] for line in watcher.lines(doc)]
    assert logged == ["throughput test of epoch 11 (block 2020)", "throughput test of epoch 12 (block 2200)"]
    later = doc.model_copy(update={"tests": doc.tests + [ThroughputOutcome(**final("gpu1", 13))]})
    assert [line.split(",")[0] for line in watcher.lines(later)] == ["throughput test of epoch 13 (block 2380)"]


class FakeAuditor:
    """The auditor's side of the status: its results per hotkey, and their revision."""

    def __init__(self) -> None:
        self.revision, self.calls = 0, 0

    def outcomes(self, hotkey: str) -> list[ThroughputOutcome]:
        self.calls += 1
        return [ThroughputOutcome(endpoint_id="gpu0", epoch=4, start_block=800, state="confirmed", outcome="passed")]


def status_service(keypair: Keypair, board: StatusBoard, auditor: FakeAuditor | None, clock: list[float],
                   client: tuple[str, int] = ("203.0.113.9", 50000)) -> TestClient:
    app = FastAPI()
    add_status_route(app, board, auditor, keypair, 405, clock=lambda: clock[0])
    return TestClient(app, client=client)


def test_the_status_is_signed_cached_and_built_again_when_a_test_or_the_board_changes() -> None:
    owner, board, auditor, clock = kp(), StatusBoard(), FakeAuditor(), [1000.0]
    board.publish({"hkA": HotkeyStatus(hotkey="hkA", epoch=5, epoch_blocks=180)})
    client = status_service(owner, board, auditor, clock)

    def get(hotkey: str = "hkA"):
        return client.get(STATUS_PATH.format(hotkey=hotkey))

    first = get()
    document = first.json()
    assert unverified(document, domain=STATUS_DOMAIN, auditor=owner.ss58_address, netuid=405,
                      now_ms=int(time.time() * 1000)) is None
    got = HotkeyStatus.model_validate(document)  # a reader that knows only the status's fields
    assert (got.epoch, got.tests[0].epoch, document["auditor"]) == (5, 4, owner.ss58_address)
    assert first.headers["cache-control"] == "public, max-age=30"
    assert get().content == first.content and auditor.calls == 1  # cached
    auditor.revision += 1  # a test recorded, confirmed or voided
    assert get().content != first.content and auditor.calls == 2
    board.publish({"hkA": HotkeyStatus(hotkey="hkA", epoch=6, epoch_blocks=180)})  # the next epoch
    assert HotkeyStatus.model_validate(get().json()).epoch == 6 and auditor.calls == 3
    get()
    clock[0] += 30.0  # nothing changed, but the status is built again after 30 s
    get()
    assert auditor.calls == 4
    assert HotkeyStatus.model_validate(get("hkB").json()).epoch is None  # not scored: test results only
    assert get("x" * 65).status_code == 404

    without = status_service(owner, board, None, clock).get(STATUS_PATH.format(hotkey="hkA")).json()
    assert without["tests"] == [] and unverified(without, domain=STATUS_DOMAIN, auditor=owner.ss58_address,
                                                 netuid=405, now_ms=int(time.time() * 1000)) is None


def test_status_requests_are_limited_per_address() -> None:
    owner, board, clock = kp(), StatusBoard(), [1000.0]
    client = status_service(owner, board, None, clock, client=("2001:db8:1:2::5", 50000))
    path = STATUS_PATH.format(hotkey="hkA")
    assert all(client.get(path).status_code == 200 for _ in range(STATUS_BURST))
    refused = client.get(path)
    assert refused.status_code == 429 and refused.headers["retry-after"] == f"{STATUS_EVERY_S:.0f}"
    clock[0] += STATUS_EVERY_S
    assert client.get(path).status_code == 200 and client.get(path).status_code == 429

    assert address_group("203.0.113.5") == address_group("::ffff:203.0.113.5") == "203.0.113.5"
    assert address_group("2001:db8:1:2::5") == address_group("2001:db8:1:2:ffff::1") == "2001:db8:1:2::/64"
    assert address_group("2001:db8:1:3::5") != address_group("2001:db8:1:2::5")
    limit = AddressLimit(burst=1, every_s=5.0, max_keys=3)
    for key in ("a", "b", "c", "d"):
        assert limit.check(key, 0.0) == 0.0
    assert limit.check("d", 0.0) == 5.0 and len(limit._tat) == 3  # bounded: the oldest is forgotten ...
    assert limit.check("a", 0.0) == 0.0  # ... and starts afresh


def test_a_miner_acts_only_on_a_status_the_subnet_owner_signed_and_logs_the_others_with_one_warning(
        monkeypatch, caplog) -> None:
    caplog.set_level(logging.INFO, logger="verathos.miner.status")
    owner, stranger, now = kp(), kp(), time.time()
    tests = [ThroughputOutcome(endpoint_id="gpu0", epoch=e, start_block=e * 180 + 40, state="confirmed",
                               outcome="partial", tested_rps=11.4, delivered_rps=10.5) for e in (12, 13)]

    def signed(keypair: Keypair = owner, *, hotkey: str = "hkA", at: float = now, netuid: int = 405) -> dict:
        document = HotkeyStatus(hotkey=hotkey, epoch=13, epoch_blocks=180, tests=tests).model_dump(mode="json")
        return sign_document(document, keypair, netuid=netuid, domain=STATUS_DOMAIN, now_ms=int(at * 1000))

    served: list[dict] = []
    monkeypatch.setattr(miner_status, "get_json", lambda url, **kwargs: served[-1])
    acted: list[list[ThroughputOutcome]] = []
    watcher = StatusWatcher(hotkey="hkA", netuid=405, auditor_url=lambda: "http://192.0.2.1:8091",
                            auditor_hotkey=lambda: owner.ss58_address, listed=lambda: ["gpu0"],
                            on_tests=acted.append, clock=lambda: now)

    def poll(document: dict) -> bool:
        """Whether the miner acted on ``document``."""
        before = len(acted)
        served.append(document)
        watcher.poll_once()
        return len(acted) > before

    def warnings() -> list[str]:
        return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]

    def warning(why: str) -> str:
        return (f"the auditor's status is logged but not acted on: {why}. Until a status verifies, its test results "
                "never make the miner measure again or cap a declaration")

    unsigned = HotkeyStatus(hotkey="hkA", epoch=13, epoch_blocks=180, tests=tests).model_dump(mode="json")
    assert not poll(unsigned) and not poll(unsigned)  # an older auditor: logged, never acted on
    assert any("throughput test of epoch 13" in m for m in caplog.messages)
    assert warnings() == [warning("it is not signed (the auditor runs an older release)")]  # once
    assert poll(signed()) and acted[-1] == tests
    assert "the auditor's status verifies again: the miner acts on its test results" in caplog.messages

    tampered = signed()
    tampered["tests"][1]["delivered_rps"] = 2.0
    refused = {
        "its signature does not verify against the subnet owner hotkey": tampered,
        f"it is not signed by the subnet owner hotkey {owner.ss58_address} for netuid 405": signed(stranger),
        "it is another hotkey's status": signed(hotkey="hkB"),
        f"it was not signed within {STATUS_MAX_AGE_S:.0f} s of this machine's clock": signed(at=now - 600),
        "it was signed before a status already read": signed(at=now - 60),
    }
    for why, document in refused.items():
        caplog.clear()
        assert not poll(document), why
        assert warnings() == [warning(why)]
    assert not poll(signed(netuid=96)) and poll(signed(at=now + 1))

    old_reader = HotkeyStatus.model_validate(signed())  # a miner of an older release reads a signed status
    assert old_reader.tests == tests and "signature" not in old_reader.model_dump()


def test_network_route_serves_every_hotkey_of_the_last_epoch_and_the_weights_set_signed_every_30_s() -> None:
    owner, board, clock = kp(), StatusBoard(), [1000.0]
    app = FastAPI()
    add_network_route(app, board, owner, 405, clock=lambda: clock[0])
    client = TestClient(app)
    empty = client.get(NETWORK_PATH)
    assert NetworkStatus.model_validate(empty.json()).hotkeys == []  # nothing scored yet
    board.publish({"hkB": HotkeyStatus(hotkey="hkB", uid=7, weight=0.4, epoch=5, epoch_blocks=180,
                                       model_shares={"kev-4b": 1.0}),
                   "hkA": HotkeyStatus(hotkey="hkA", uid=2, weight=0.6, epoch=5, epoch_blocks=180,
                                       model_shares={"kev-4b": 1.0})},
                  epoch=5, epoch_blocks=180, model_shares={"kev-4b": 1.0}, weights={2: 0.6, 7: 0.4, 9: 1e-9})
    response = client.get(NETWORK_PATH)
    got = NetworkStatus.model_validate(response.json())
    assert (got.epoch, got.model_shares, [h.uid for h in got.hotkeys]) == (5, {"kev-4b": 1.0}, [2, 7])
    assert got.weights == {2: 0.6, 7: 0.4, 9: 1e-9}  # every UID set, also one without a status
    assert response.headers["cache-control"] == "public, max-age=30"
    now_ms = int(time.time() * 1000)
    assert unverified(response.json(), domain=NETWORK_DOMAIN, auditor=owner.ss58_address, netuid=405,
                      now_ms=now_ms) is None
    assert unverified(response.json(), domain=STATUS_DOMAIN, auditor=owner.ss58_address, netuid=405,
                      now_ms=now_ms) is not None  # never taken for a hotkey's status
    assert client.get(NETWORK_PATH).content == response.content  # signed at most every 30 s, not per request
    assert board.network() is board.network()  # built once per epoch
    time.sleep(0.002)
    clock[0] += 30.0
    again = client.get(NETWORK_PATH).json()
    assert again["issued_ms"] > response.json()["issued_ms"]  # its signature shows it is served now
    assert {k: v for k, v in again.items() if k not in ("issued_ms", "signature")} == {
        k: v for k, v in response.json().items() if k not in ("issued_ms", "signature")}

    # Without any endpoint the epoch and the weights are still stated.
    board.publish({}, epoch=6, epoch_blocks=180, model_shares={"kev-4b": 1.0}, weights={0: 1.0})
    got = NetworkStatus.model_validate(client.get(NETWORK_PATH).json())
    assert (got.epoch, got.epoch_blocks, got.weights, got.hotkeys) == (6, 180, {0: 1.0}, [])


def test_a_restarted_validator_serves_the_last_epochs_status_until_its_next_epoch(tmp_path) -> None:
    path = tmp_path / "status.json"
    StatusBoard(path).publish({"hkA": HotkeyStatus(hotkey="hkA", uid=2, weight=1.0, epoch=5, epoch_blocks=180)},
                              weights={2: 1.0})
    restarted = StatusBoard(path)
    assert restarted.get("hkA").epoch == 5 and b'"epoch":5' in restarted.network()
    assert NetworkStatus.model_validate_json(restarted.network()).weights == {2: 1.0}
    path.write_text("not json")
    assert StatusBoard(path).get("hkA") is None  # a damaged file is ignored
