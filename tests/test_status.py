"""What a miner sees about its endpoints: the validator's status and the miner's log lines."""

from __future__ import annotations

from verathos.capacity.core import AuditPolicy
from verathos.miner.status import StatusWatcher, describe_audit
from verathos.protocol.manifest import Endpoint
from verathos.protocol.status import AuditOutcome, HotkeyStatus
from verathos.validator import status
from verathos.validator.audit import EndpointResult
from verathos.validator.compare import RequestComparison, Tolerance
from verathos.validator.discovery import DiscoveredEndpoint
from verathos.validator.scoring import ScoreState


def endpoint(endpoint_id: str, model: str = "kev-4b") -> DiscoveredEndpoint:
    return DiscoveredEndpoint(1, "hkA", "http://x", Endpoint(endpoint_id=endpoint_id, url=f"http://x/e/{endpoint_id}",
                              model_id=model, hardware={"kind": "cuda", "device_id": f"GPU-{endpoint_id}"},
                              max_concurrency=1))


def result(ep: DiscoveredEndpoint) -> EndpointResult:
    r = EndpointResult(ep, Tolerance(min_probabilities=1), identity_ok=True, sent=10, answered=10,
                       latencies_ms=[40.0] * 10)
    r.tally.add(RequestComparison(max_abs=0.01, sum_abs=0.06, count=60, confident_flips=0))
    return r


def test_each_endpoint_gets_one_state() -> None:
    eps = {name: endpoint(name, "clef-flash" if name == "cand" else "kev-4b")
           for name in ("ok", "new", "cand", "gated", "probe", "noref")}
    results = {e.key: result(e) for name, e in eps.items() if name != "noref"}
    state = ScoreState(ema={"hkA/ok": 0.6}, probation={"hkA/probe": 3, "hkA/gated": 6},
                       first_seen={e.key: 10 for e in eps.values()} | {"hkA/new": 12},
                       window={k: [[0.06, 60.0]] for k in results})
    board = status.build(epoch=13, epoch_blocks=180, endpoints=list(eps.values()), results=results,
                         scores={"hkA/ok": 0.98}, failed={"hkA/gated"}, gated={"hkA/gated": "timing"}, unauditable=(),
                         qualifying={"hkA/new"}, scored_models={"kev-4b"}, qualification_epochs=4, state=state)
    states = {e.endpoint_id: e.state for e in board["hkA"].endpoints}
    assert states == {"ok": "scored", "new": "qualifying", "cand": "candidate", "gated": "gated", "probe": "probation",
                      "noref": "not_audited"}
    assert next(e for e in board["hkA"].endpoints if e.endpoint_id == "new").scored_from_epoch == 16


def test_miner_logs_each_epoch_once_and_each_final_audit_once() -> None:
    doc = HotkeyStatus(hotkey="hkA", epoch=13, epoch_blocks=180, endpoints=[
        {"endpoint_id": "gpu0", "model_id": "kev-4b", "state": "qualifying", "scored_from_epoch": 16, "sent": 10,
         "answered": 10, "verdict": "pass", "mean_abs": 0.0014},
        {"endpoint_id": "gpu1", "model_id": "kev-4b", "state": "gated", "capacity_rule": "hotkey_timing",
         "probation_epochs": 5}],
        audits=[{"endpoint_id": "gpu0", "selection_block": 2400, "window": "confirmed", "outcome": "passed",
                 "final_s": 20.4},
                {"endpoint_id": "gpu1", "selection_block": 2410, "window": "pending", "outcome": "timing_pass"}])
    watcher = StatusWatcher(hotkey="hkA", auditor_url=lambda: None, head=lambda: 13 * 180 + 60,
                            listed=lambda: ["gpu0", "gpu1", "gpu2"], policy=AuditPolicy)
    lines = watcher.lines(doc, 13 * 180 + 60)
    assert lines[0] == "validator's epoch 13:"
    assert "qualifying, checked but not scored until epoch 16 (in about 96 min)" in lines[1]
    assert "GATED by the capacity audit, its timed work missed the deadline (counted across this hotkey's" in lines[2]
    assert "gpu2: not in that epoch" in lines[3]
    assert lines[4] == "capacity audit at block 2400, gpu0: passed, timed work done in 20.4 s"
    assert watcher.lines(doc, 13 * 180 + 61) == []  # nothing new; the pending audit is not final yet


def test_audit_lines_name_the_reason() -> None:
    miss = AuditOutcome(endpoint_id="gpu1", selection_block=7, window="confirmed", outcome="timing_miss", final_s=43.8)
    assert describe_audit(miss, limit_s=33) == ("capacity audit at block 7, gpu1: MISSED the deadline, final receipt "
                                                "after 43.8 s (limit 33 s)")


def test_status_route_serves_the_board_and_audits(tmp_path) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from verathos.protocol.status import STATUS_PATH
    from verathos.validator.service import add_status_route
    from verathos.validator.status import StatusBoard

    board = StatusBoard()
    board.publish({"hkA": HotkeyStatus(hotkey="hkA", epoch=5, epoch_blocks=180)})
    app = FastAPI()
    class Auditor:
        def recent_outcomes(self, hotkey):
            return [{"endpoint_id": "gpu0", "selection_block": 1, "window": "confirmed", "outcome": "passed",
                     "reason": "", "final_s": 20.0}]

    add_status_route(app, board, Auditor())
    client = TestClient(app)
    got = HotkeyStatus.model_validate(client.get(STATUS_PATH.format(hotkey="hkA")).json())
    assert got.epoch == 5 and got.audits[0].outcome == "passed"
    assert HotkeyStatus.model_validate(client.get(STATUS_PATH.format(hotkey="hkB")).json()).epoch is None
