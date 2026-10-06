"""Miner service end to end with a mock engine: discovery, auth, answers, receipts, identity, test sessions."""

from __future__ import annotations

import json
import secrets
import time

import pytest
from bittensor_wallet import Keypair
from fastapi.testclient import TestClient

from tests.mock_engine import MockEngine
from verathos import catalog
from verathos.miner.app import IDENTITY_DOMAIN, create_app, identity_payload
from verathos.miner.config import MinerConfig
from verathos.protocol import signing
from verathos.protocol.api import distributions
from verathos.protocol.manifest import Endpoint, Manifest, SignedManifest, accept, axon_base, commitment_value
from verathos.protocol.receipts import SignedReceipt
from verathos.protocol.requests import sign_request
from verathos.throughput.policy import HDR_REQUEST_ID, Session, session_headers

REQUEST = {
    "model": "kev-4b",
    "state": "Hi, we were billed twice for March. Please refund today or we cancel.",
    "questions": {
        "team": {"type": "choice", "criteria": {"billing": "Charges", "technical": "Bugs"}},
        "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
        "churn": {"type": "noul", "instructions": "Does the customer threaten to cancel?"},
    },
}


AUDITOR = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())


@pytest.fixture()
def setup():
    miner, validator, stranger = (Keypair.create_from_mnemonic(Keypair.generate_mnemonic()) for _ in range(3))
    config = MinerConfig.model_validate({
        "netuid": 405, "mechid": 1, "wallet_name": "w", "hotkey_name": "h", "public_host": "203.0.113.5", "port": 8091,
        "endpoints": [
            {"endpoint_id": "gpu0", "model_id": "kev-4b", "engine_url": "http://127.0.0.1:1", "max_concurrency": 4},
        ],
    })
    cat_digest = catalog.load().digest()
    manifest = SignedManifest.create(Manifest(
        netuid=405, mechid=1, hotkey=miner.ss58_address, catalog_digest=cat_digest,
        endpoints=[Endpoint(endpoint_id="gpu0", url=config.endpoint_url("gpu0"), model_id="kev-4b",
                            capacity_rps=40.0)]), miner)
    app = create_app(config=config, keypair=miner, manifest=lambda: manifest, engines={"gpu0": MockEngine("kev-4b")},
                     authorized=lambda hk: hk == validator.ss58_address, auditor=lambda: AUDITOR.ss58_address)
    return TestClient(app), miner, validator, stranger, manifest


def post(client: TestClient, caller: Keypair, miner: Keypair, path: str, payload: dict):
    body = json.dumps(payload).encode()
    return client.post(path, content=body, headers={"Content-Type": "application/json",
                                                    **sign_request(caller, "POST", path, body, miner.ss58_address)})


def test_manifest_is_public_and_acceptable(setup) -> None:
    client, miner, *_ , manifest = setup
    assert accept(client.get("/v1/manifest").json(), hotkey=miner.ss58_address, netuid=405, mechid=1,
                  axon=axon_base("203.0.113.5", 8091), commitment=commitment_value(manifest.manifest.digest()),
                  catalog_digests={manifest.manifest.catalog_digest})


def test_answer_carries_binding_receipt(setup) -> None:
    client, miner, validator, *_ = setup
    r = post(client, validator, miner, "/e/gpu0/v1/systemone", REQUEST)
    assert r.status_code == 200, r.text
    body = r.json()
    receipt = SignedReceipt.model_validate(body["receipt"])
    assert receipt.signature_valid()
    assert receipt.binds(REQUEST, body["answers"])
    assert receipt.receipt.caller_hotkey == validator.ss58_address
    assert set(distributions(body["answers"])) == {"team", "urgency", "churn"}


def test_unsigned_and_unpermitted_callers_are_refused(setup) -> None:
    from verathos.miner.app import MAX_REQUEST_BYTES
    from verathos.protocol.requests import HDR_HOTKEY

    client, miner, validator, stranger, _ = setup
    assert client.post("/e/gpu0/v1/systemone", json=REQUEST).status_code == 403
    assert post(client, stranger, miner, "/e/gpu0/v1/systemone", REQUEST).status_code == 403
    forged = client.post("/e/gpu0/v1/systemone", json=REQUEST, headers={HDR_HOTKEY: validator.ss58_address})
    assert forged.status_code == 401
    big = {**REQUEST, "state": "x" * MAX_REQUEST_BYTES}
    assert post(client, validator, miner, "/e/gpu0/v1/systemone", big).status_code == 413


def test_wrong_model_and_unknown_endpoint(setup) -> None:
    client, miner, validator, *_ = setup
    assert post(client, validator, miner, "/e/gpu0/v1/systemone", {**REQUEST, "model": "kev-9b"}).status_code == 422
    assert post(client, validator, miner, "/e/gpu9/v1/systemone", REQUEST).status_code == 404


def test_identity_challenge_proves_endpoint_control(setup) -> None:
    client, miner, validator, *_ = setup
    challenge = "ab" * 16
    r = post(client, validator, miner, "/e/gpu0/v1/identity", {"challenge": challenge})
    assert r.status_code == 200, r.text
    payload = identity_payload("gpu0", challenge, validator.ss58_address)
    assert signing.verify(miner.ss58_address, IDENTITY_DOMAIN, payload, r.json()["signature"])


def session_post(client: TestClient, miner: Keypair, *, auditor: Keypair = AUDITOR, endpoint_id: str = "gpu0",
              request_id: str | None = None, expires_in_ms: int = 30_000, payload: dict = REQUEST):
    session = Session(netuid=405, auditor=auditor.ss58_address, miner_hotkey=miner.ss58_address,
                      endpoint_id=endpoint_id, test_id="t" * 64, not_after_ms=int(time.time() * 1000) + expires_in_ms)
    headers = {**session_headers(session, auditor), HDR_REQUEST_ID: request_id or secrets.token_hex(16)}
    return client.post(f"/e/{endpoint_id}/v1/systemone", json=payload, headers=headers)


def test_test_traffic_carries_the_auditors_session(setup) -> None:
    client, miner, *_ = setup
    request_id = secrets.token_hex(16)
    r = session_post(client, miner, request_id=request_id)
    assert r.status_code == 200, r.text
    receipt = SignedReceipt.model_validate(r.json()["receipt"])
    assert receipt.signature_valid() and receipt.binds(REQUEST, r.json()["answers"])
    assert (receipt.receipt.caller_hotkey, receipt.receipt.request_nonce) == (AUDITOR.ss58_address, request_id)
    assert session_post(client, miner, request_id=request_id).json()["error"] == "replayed request"


def test_test_sessions_are_checked(setup) -> None:
    client, miner, validator, *_ = setup
    assert "another auditor" in session_post(client, miner, auditor=validator).json()["error"]
    assert "expired" in session_post(client, miner, expires_in_ms=-1).json()["error"]
    assert session_post(client, miner, request_id="not-hex").json()["error"] == "malformed request id"
    forged = Session(netuid=405, auditor=AUDITOR.ss58_address, miner_hotkey=miner.ss58_address, endpoint_id="gpu0",
                     test_id="t" * 64, not_after_ms=int(time.time() * 1000) + 30_000)
    headers = {**session_headers(forged, validator), HDR_REQUEST_ID: secrets.token_hex(16)}
    r = client.post("/e/gpu0/v1/systemone", json=REQUEST, headers=headers)
    assert (r.status_code, r.json()["error"]) == (401, "bad test session signature")
