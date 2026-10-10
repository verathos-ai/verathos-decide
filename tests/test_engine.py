"""The engine server validates SystemOne requests and runs prepared requests in passes on one device thread."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from verathos.engine.server import Batcher, create_app

REQUEST = {"model": "clef-flash", "state": "The checkout returns 500 errors.",
           "questions": {"outage": {"type": "noul", "instructions": "Is a service down?"}}}


class RecordingEngine:
    def __init__(self, release: threading.Event | None = None) -> None:
        self.batches: list[int] = []
        self.release = release

    def prepare(self, body):
        if body["state"] == "bad":
            raise ValueError("state rejected")
        return body

    def passes(self, items):
        return [list(range(len(items)))]

    def run(self, items):
        if self.release is not None:
            self.release.wait(5)
        self.batches.append(len(items))
        return [{"model": b["model"], "answers": {"outage": {"type": "noul", "noul": 0.93}},
                 "usage": {"input_tokens": 12, "output_tokens": 0}} for b in items]


def test_engine_answers_and_rejects_malformed_requests() -> None:
    served = {"id": "Cloudflare/clef-flash", "revision": "17f0b0ad", "device": "cuda", "dtype": "bfloat16"}
    client = TestClient(create_app(RecordingEngine(), [served]))
    assert client.get("/v1/models").json() == {"data": [served]}
    assert client.post("/v1/systemone", json=REQUEST).json()["answers"]["outage"]["noul"] == 0.93
    assert client.post("/v1/systemone", json={"model": "x", "questions": {}}).status_code == 422
    assert client.post("/v1/systemone", json={**REQUEST, "state": "bad"}).status_code == 422


def test_a_body_beyond_the_largest_request_a_model_may_take_is_refused_unread() -> None:
    from verathos.engine.server import MAX_BODY_BYTES

    engine = RecordingEngine()
    client = TestClient(create_app(engine))
    big = {**REQUEST, "state": "x" * MAX_BODY_BYTES}
    assert client.post("/v1/systemone", json=big).status_code == 413
    assert client.post("/v1/systemone", content=b"[" * 10).status_code == 422
    nested = b'{"model": "kev-4b", "state": ' + b"[" * 100_000 + b"]" * 100_000 + b', "questions": {}}'
    assert client.post("/v1/systemone", content=nested).status_code == 422  # deeper than the parser goes: not a 500
    assert engine.batches == []


def test_requests_waiting_on_a_running_batch_join_the_next_batch() -> None:
    release = threading.Event()
    engine = RecordingEngine(release)
    client = TestClient(create_app(engine))
    with ThreadPoolExecutor(6) as pool:
        futures = [pool.submit(client.post, "/v1/systemone", json=REQUEST) for _ in range(6)]
        threading.Timer(0.5, release.set).start()
        assert all(f.result().status_code == 200 for f in futures)
    assert sum(engine.batches) == 6 and len(engine.batches) < 6


class ItemEngine:
    """Items are strings; ``hold`` and ``slow`` wait for their event, ``bad`` and ``oom`` fail their pass."""

    def __init__(self, per_item: bool) -> None:
        self.per_item, self.runs = per_item, []
        self.hold, self.slow = threading.Event(), threading.Event()

    def passes(self, items):
        return [[i] for i in range(len(items))] if self.per_item else [list(range(len(items)))]

    def run(self, items):
        self.runs.append(list(items))
        if "hold" in items:
            self.hold.wait(5)
        if "slow" in items:
            self.slow.wait(5)
        if "bad" in items:
            raise ValueError("schema too long")
        if "oom" in items:
            raise RuntimeError("CUDA out of memory")
        return [item.upper() for item in items]


def batch_after_hold(batcher: Batcher, engine: ItemEngine, items: list[str]):
    """Submit ``items`` while the device thread is busy, so they form one batch."""
    held = batcher.submit("hold")
    while not engine.runs:
        threading.Event().wait(0.01)
    futures = [batcher.submit(item) for item in items]
    engine.hold.set()
    assert held.result(5) == "HOLD"
    return futures


def test_each_pass_is_answered_when_it_ends_not_when_the_batch_ends() -> None:
    engine = ItemEngine(per_item=True)
    batcher = Batcher(engine)
    quick, slow = batch_after_hold(batcher, engine, ["quick", "slow"])
    assert quick.result(2) == "QUICK" and not slow.done()
    engine.slow.set()
    assert slow.result(5) == "SLOW"


def test_a_failed_pass_runs_its_requests_one_at_a_time_and_only_the_failing_ones_fail() -> None:
    engine = ItemEngine(per_item=False)
    batcher = Batcher(engine)
    futures = batch_after_hold(batcher, engine, ["a", "bad", "b", "oom", "c"])
    assert [f.result(5) for f in futures[::2]] == ["A", "B", "C"]
    with pytest.raises(ValueError, match="too long"):
        futures[1].result(5)
    with pytest.raises(RuntimeError, match="out of memory"):
        futures[3].result(5)
    assert engine.runs[1] == ["a", "bad", "b", "oom", "c"] and len(engine.runs) == 2 + 5


def test_a_request_given_up_before_its_pass_is_not_run() -> None:
    engine = ItemEngine(per_item=True)
    batcher = Batcher(engine)
    held = batcher.submit("hold")
    while not engine.runs:
        threading.Event().wait(0.01)
    gone, kept = batcher.submit("gone"), batcher.submit("kept")
    assert gone.cancel()
    engine.hold.set()
    assert held.result(5) == "HOLD" and kept.result(5) == "KEPT"
    assert ["gone"] not in engine.runs


def test_the_engine_logs_no_line_per_request(monkeypatch) -> None:
    """A miner's engine answers many requests a second: a log line for each fills a log that nothing rotates (pm2)."""
    import sys
    import types

    import verathos.engine
    from verathos import stop
    from verathos.engine import __main__ as engine_main

    fake = types.ModuleType("verathos.engine.clef")
    fake.load = lambda *args, **kwargs: object()
    monkeypatch.setitem(sys.modules, "verathos.engine.clef", fake)
    monkeypatch.setattr(verathos.engine, "clef", fake, raising=False)
    monkeypatch.setattr(stop, "on_signals", lambda what, logger: None)
    monkeypatch.setattr(engine_main, "create_app", lambda engine, served: "app")
    ran: list[dict] = []
    monkeypatch.setattr(engine_main.uvicorn, "run", lambda app, **kwargs: ran.append(kwargs))
    monkeypatch.setattr(sys, "argv", ["verathos-engine", "clef", "--repo", "r/m", "--revision", "abc",
                                      "--device", "cpu"])
    engine_main.main()
    assert ran and ran[0]["access_log"] is False and ran[0]["host"] == "127.0.0.1"
