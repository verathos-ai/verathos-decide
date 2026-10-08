"""The mechanism the network runs on comes from the catalog the subnet owner commits: miners publish their manifests
for it, validators accept each catalog's manifests by its mechanism and set weights on it, and a node's ``mechid``
overrides it."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest
from bittensor_wallet import Keypair
from fastapi.testclient import TestClient
from pydantic import ValidationError

from tests.mock_engine import MockEngine
from tests.test_shortfall import publisher_with
from verathos import catalog
from verathos.gateway.config import GatewayConfig
from verathos.miner import __main__ as miner_main
from verathos.miner.config import MinerConfig
from verathos.protocol.canonical import digest
from verathos.protocol.receipts import Receipt, SignedReceipt, issued_receipt
from verathos.protocol.requests import sign_request
from verathos.validator.__main__ import manifest_catalogs
from verathos.validator.config import ValidatorConfig

CATALOG = catalog.load()
MOVED = catalog.Catalog.parse({**CATALOG.document(), "mechanism": 1})
REQUEST = {"model": "kev-4b", "state": "Hi, we were billed twice for March.",
           "questions": {"churn": {"type": "noul", "instructions": "Does the customer threaten to cancel?"}}}


def kp() -> Keypair:
    return Keypair.create_from_mnemonic(Keypair.generate_mnemonic())


def test_the_catalog_names_the_mechanism_and_one_when_it_names_none() -> None:
    assert CATALOG.mechanism == 0  # the mainnet catalog's; a test network's catalog names its own
    document = CATALOG.document()
    older = catalog.Catalog.parse({k: v for k, v in document.items() if k != "mechanism"})
    assert older.mechanism == catalog.DEFAULT_MECHANISM == 1
    assert MOVED.mechanism == 1 and MOVED.unknown_settings() == [] and MOVED.digest() != CATALOG.digest()
    for invalid in (-1, 1.5, None, "one"):
        with pytest.raises(ValidationError, match="mechanism"):
            catalog.Catalog.parse({**document, "mechanism": invalid})
    assert catalog.mechanism_of(CATALOG) == 0 and catalog.mechanism_of(MOVED) == 1
    assert catalog.mechanism_of(MOVED, 0) == 0  # a node's mechid overrides the catalog's


def test_nodes_take_the_catalogs_mechanism_unless_their_configuration_names_one(tmp_path) -> None:
    base = {"netuid": 96, "wallet_name": "w", "hotkey_name": "h"}
    assert ValidatorConfig.model_validate(base).mechid is None
    assert ValidatorConfig.model_validate({**base, "mechid": 1}).mechid == 1
    endpoints = [{"endpoint_id": "gpu0", "model_id": "kev-4b", "engine_url": "http://127.0.0.1:9"}]
    miner = {**base, "public_host": "203.0.113.5", "port": 8091, "endpoints": endpoints}
    assert MinerConfig.model_validate(miner).mechid is None
    assert MinerConfig.model_validate({**miner, "mechid": 0}).mechid == 0
    for config in (ValidatorConfig, MinerConfig):
        with pytest.raises(ValidationError, match="mechid"):
            config.model_validate({**miner, "mechid": -1})
    # a gateway checks a receipt by its caller, nonce and subnet: a gateway.json naming a mechanism still loads
    gateway = GatewayConfig.model_validate({**base, "database_url": "sqlite+aiosqlite:///x", "mechid": 1})
    assert "mechid" not in gateway.model_dump()


class Validators:
    """The miner's callers: one validator, and the request epoch."""

    def __init__(self, validator: str) -> None:
        self.validator = validator

    def __call__(self, hotkey: str) -> bool:
        return hotkey == self.validator

    def epoch(self, epoch_blocks: int) -> tuple[int, float]:
        return 7, 60.0


def test_a_miner_publishes_for_the_catalogs_mechanism_and_moves_with_a_new_catalog(tmp_path, caplog) -> None:
    caplog.set_level(logging.INFO, logger="verathos.miner")
    publisher, calls = publisher_with(tmp_path, {"gpu0": 12.0}, peers=["198.51.100.7:20017"])
    validator = kp()
    miner = publisher.wallet.hotkey
    client = TestClient(miner_main.build_app(publisher.config, miner, publisher, Validators(validator.ss58_address),
                                             {"gpu0": MockEngine("kev-4b")}))

    def receipt_mechanism() -> int:
        body = json.dumps(REQUEST).encode()
        reply = client.post("/e/gpu0/v1/systemone", content=body, headers={
            "Content-Type": "application/json",
            **sign_request(validator, "POST", "/e/gpu0/v1/systemone", body, miner.ss58_address)})
        assert reply.status_code == 200, reply.text
        return reply.json()["receipt"]["receipt"]["mechid"]

    assert publisher.adopt(CATALOG)
    assert (publisher.mechid, publisher.signed.manifest.mechid, publisher.local.host.mechid) == (0, 0, 0)
    assert publisher.peers.mechid == 0 and receipt_mechanism() == 0
    assert publisher.adopt(MOVED) and calls == [False, False]  # the stored measurement serves the new catalog
    assert (publisher.signed.manifest.mechid, publisher.signed.manifest.catalog_digest) == (1, MOVED.digest())
    assert publisher.local.host.mechid == 1 and publisher.peers.mechid == 1 and receipt_mechanism() == 1
    assert f"catalog {MOVED.digest()[:12]} moves the network to mechanism 1; the manifest is for it from now on" in (
        caplog.messages)

    pinned, _ = publisher_with(tmp_path / "pinned", {"gpu0": 12.0}, mechid=0)
    assert pinned.adopt(MOVED) and pinned.signed.manifest.mechid == 0  # mechid overrides the catalog's
    assert pinned.signed.manifest.catalog_digest == MOVED.digest()


def test_validators_accept_each_catalogs_manifests_for_its_own_mechanism() -> None:
    """For an epoch after a catalog moves the network, manifests built for the previous catalog are accepted for its
    mechanism, the new catalog's for the new one; a configured mechid takes every manifest for itself."""
    current = SimpleNamespace(accepted=lambda: {CATALOG.digest(): 0, MOVED.digest(): 1})
    assert manifest_catalogs(SimpleNamespace(mechid=None), MOVED, current) == {CATALOG.digest(): 0, MOVED.digest(): 1}
    assert manifest_catalogs(SimpleNamespace(mechid=None), CATALOG,
                             SimpleNamespace(accepted=lambda: {MOVED.digest(): 1})) == {
        CATALOG.digest(): 0, MOVED.digest(): 1}  # the epoch's catalog, committed at its first block
    assert manifest_catalogs(SimpleNamespace(mechid=1), MOVED, current) == {CATALOG.digest(): 1, MOVED.digest(): 1}


def test_a_receipt_names_the_miners_mechanism_and_is_checked_for_the_subnet() -> None:
    """A receipt is bound to its caller, the request's nonce and the subnet; it names the catalog and mechanism the
    miner served under, which a reader does not require to be its own, so answers given while a new catalog is
    adopted stay valid."""
    miner, caller = kp(), kp()
    answers = {"churn": {"yes": 0.1, "no": 0.9}}

    def signed(mechid: int, netuid: int = 405) -> dict:
        receipt = Receipt(netuid=netuid, mechid=mechid, miner_hotkey=miner.ss58_address, endpoint_id="gpu0",
                          model_id="kev-4b", catalog_digest=CATALOG.digest(), caller_hotkey=caller.ss58_address,
                          request_nonce="n1", request_sha256=digest(REQUEST), answers_sha256=digest(answers),
                          received_ms=1, responded_ms=2)
        return SignedReceipt.create(receipt, miner).model_dump(mode="json")

    def issued(document: dict) -> bool:
        return issued_receipt(document, request_id="n1", hotkey=miner.ss58_address, endpoint_id="gpu0",
                              model_id="kev-4b", callers={caller.ss58_address}, netuid=405) is not None

    assert issued(signed(0)) and issued(signed(1))
    assert not issued(signed(0, netuid=406))
