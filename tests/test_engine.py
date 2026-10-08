"""The engine server validates SystemOne requests and runs prepared requests in batches on one device thread."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient

from verathos.engine.server import create_app

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


def test_requests_waiting_on_a_running_batch_join_the_next_batch() -> None:
    release = threading.Event()
    engine = RecordingEngine(release)
    client = TestClient(create_app(engine))
    with ThreadPoolExecutor(6) as pool:
        futures = [pool.submit(client.post, "/v1/systemone", json=REQUEST) for _ in range(6)]
        threading.Timer(0.5, release.set).start()
        assert all(f.result().status_code == 200 for f in futures)
    assert sum(engine.batches) == 6 and len(engine.batches) < 6
