"""Miner service end to end with a mock engine: discovery, auth, answers, receipts, identity."""

from __future__ import annotations

import json

import pytest
from bittensor_wallet import Keypair
from fastapi.testclient import TestClient

from tests.mock_engine import MockEngine
from verathos import catalog
from verathos.miner.app import IDENTITY_DOMAIN, create_app, identity_payload
from verathos.miner.config import MinerConfig
from verathos.protocol import signing
from verathos.protocol.api import distributions
from verathos.protocol.manifest import Endpoint, Manifest, SignedManifest, accept, commitment_value
from verathos.protocol.receipts import SignedReceipt
from verathos.protocol.requests import sign_request

REQUEST = {
    "model": "kev-4b",
    "state": "Hi, we were billed twice for March. Please refund today or we cancel.",
    "questions": {
        "team": {"type": "choice", "criteria": {"billing": "Charges", "technical": "Bugs"}},
        "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
        "churn": {"type": "noul", "instructions": "Does the customer threaten to cancel?"},
    },
}


@pytest.fixture()
def setup():
    miner, validator, stranger = (Keypair.create_from_mnemonic(Keypair.generate_mnemonic()) for _ in range(3))
    config = MinerConfig.model_validate({
        "netuid": 405, "mechid": 1, "wallet_name": "w", "hotkey_name": "h", "public_host": "203.0.113.5", "port": 8091,
        "endpoints": [
            {"endpoint_id": "gpu0", "model_id": "kev-4b", "engine_url": "http://127.0.0.1:1",
             "hardware": {"kind": "cuda", "device_id": "GPU-a"}, "max_concurrency": 4},
        ],
    })
    cat_digest = catalog.load().digest()
    manifest = SignedManifest.create(Manifest(
        netuid=405, mechid=1, hotkey=miner.ss58_address, seq=1, issued_block=1, catalog_digest=cat_digest,
        endpoints=[Endpoint(endpoint_id="gpu0", url=config.endpoint_url("gpu0"), model_id="kev-4b",
                            hardware=config.endpoints[0].hardware, max_concurrency=4)]), miner)
    app = create_app(config=config, keypair=miner, manifest=lambda: manifest, engines={"gpu0": MockEngine("kev-4b")},
                     authorized=lambda hk: hk == validator.ss58_address)
    return TestClient(app), miner, validator, stranger, manifest


def post(client: TestClient, caller: Keypair, miner: Keypair, path: str, payload: dict):
    body = json.dumps(payload).encode()
    return client.post(path, content=body, headers={"Content-Type": "application/json",
                                                    **sign_request(caller, "POST", path, body, miner.ss58_address)})


def test_manifest_is_public_and_acceptable(setup) -> None:
    client, miner, *_ , manifest = setup
    served = SignedManifest.model_validate(client.get("/v1/manifest").json())
    assert accept(served, hotkey=miner.ss58_address, netuid=405,
                  commitment=commitment_value(manifest.manifest.digest()),
                  catalog_digest=manifest.manifest.catalog_digest)


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
    client, miner, validator, stranger, _ = setup
    assert client.post("/e/gpu0/v1/systemone", json=REQUEST).status_code == 401
    assert post(client, stranger, miner, "/e/gpu0/v1/systemone", REQUEST).status_code == 403


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
