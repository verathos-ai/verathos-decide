"""What a miner sees about its endpoints: the validator's status and the miner's log lines."""

from __future__ import annotations

from verathos.miner.status import StatusWatcher, describe_test
from verathos.protocol.manifest import Endpoint
from verathos.protocol.status import HotkeyStatus, ThroughputOutcome
from verathos.validator import status
from verathos.validator.audit import EndpointResult
from verathos.validator.compare import RequestComparison, Tolerance
from verathos.validator.discovery import DiscoveredEndpoint
from verathos.validator.scoring import ScoreState


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
                         credited=credited, scored_models={"kev-4b"}, state=state)
    states = {e.endpoint_id: e.state for e in board["hkA"].endpoints}
    assert states == {"ok": "scored", "new": "qualifying", "cand": "candidate", "failed": "failed",
                      "probe": "probation", "noref": "not_audited"}
    assert board["hkB"].endpoints[0].state == "invalid"
    ok = next(e for e in board["hkA"].endpoints if e.endpoint_id == "ok")
    assert (ok.declared_rps, ok.credited_rps) == (50.0, 45.0)


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
    watcher = StatusWatcher(hotkey="hkA", auditor_url=lambda: None, listed=lambda: ["gpu0", "gpu1", "gpu2"])
    lines = watcher.lines(doc)
    assert lines[0] == "validator's epoch 13:"
    assert "qualifying, checked but not scored until it passes a throughput test" in lines[1]
    assert ("scored 44.10% of the epoch's emission, average 30.20%; proven 45.0 of 50.0 requests/s declared"
            in lines[2])
    assert "gpu2: not in that epoch" in lines[3]
    shared = doc.model_copy(update={"model_shares": {"kev-4b": 0.6, "kev-9b": 0.4}})
    fresh = StatusWatcher(hotkey="hkA", auditor_url=lambda: None, listed=lambda: [])
    assert fresh.lines(shared)[1] == "  emission shares: kev-4b 60.0%, kev-9b 40.0%"
    assert lines[4] == ("throughput test of epoch 12 (block 2200), gpu1: passed, delivered 46.2 of 45.0 requests/s, "
                        "p95 latency 830 ms; credited 45.0")
    assert watcher.lines(doc) == []  # nothing new; the unconfirmed test is not final yet


def test_test_lines_name_the_reason() -> None:
    partial = ThroughputOutcome(endpoint_id="gpu1", epoch=7, start_block=1300, state="confirmed", outcome="partial",
                          reason="delivered 21.0 of 40.0 requests/s", tested_rps=40.0, delivered_rps=21.0,
                          credited_rps=21.0)
    assert describe_test(partial).startswith("throughput test of epoch 7 (block 1300), gpu1: BELOW the tested rate")
    void = partial.model_copy(update={"state": "void", "reason": "the selection block was not finalized"})
    assert describe_test(void).endswith("not counted (the selection block was not finalized)")


def test_status_route_serves_the_board_and_tests() -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from verathos.protocol.status import STATUS_PATH
    from verathos.throughput.policy import AuditPolicy
    from verathos.validator.service import add_status_route
    from verathos.validator.status import StatusBoard

    board = StatusBoard()
    board.publish({"hkA": HotkeyStatus(hotkey="hkA", epoch=5, epoch_blocks=180)})
    app = FastAPI()

    class Follower:
        head = 5 * 180 + 3

    class Auditor:
        policy, follower = AuditPolicy(), Follower()

        def outcomes(self, hotkey, epoch):
            return [ThroughputOutcome(endpoint_id="gpu0", epoch=epoch - 1, start_block=800, state="confirmed",
                                outcome="passed")]

    add_status_route(app, board, Auditor())
    client = TestClient(app)
    got = HotkeyStatus.model_validate(client.get(STATUS_PATH.format(hotkey="hkA")).json())
    assert got.epoch == 5 and got.tests[0].epoch == 4
    assert HotkeyStatus.model_validate(client.get(STATUS_PATH.format(hotkey="hkB")).json()).epoch is None
