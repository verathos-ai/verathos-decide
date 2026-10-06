"""The engine server validates SystemOne requests and serialises forward passes."""

from __future__ import annotations

from fastapi.testclient import TestClient

from verathos.engine.server import create_app

REQUEST = {"model": "clef-flash", "state": "The checkout returns 500 errors.",
           "questions": {"outage": {"type": "noul", "instructions": "Is a service down?"}}}


def test_engine_answers_and_rejects_malformed_requests() -> None:
    def runner(body):
        return {"model": body["model"], "answers": {"outage": {"type": "noul", "noul": 0.93}},
                "usage": {"input_tokens": 12, "output_tokens": 0}}

    served = {"id": "Cloudflare/clef-flash", "revision": "17f0b0ad", "device": "cuda", "dtype": "bfloat16"}
    client = TestClient(create_app(runner, [served]))
    assert client.get("/v1/models").json() == {"data": [served]}
    assert client.post("/v1/systemone", json=REQUEST).json()["answers"]["outage"]["noul"] == 0.93
    assert client.post("/v1/systemone", json={"model": "x", "questions": {}}).status_code == 422
