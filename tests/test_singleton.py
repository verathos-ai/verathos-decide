"""One auditor per hotkey: an instance of the subnet owner's validator does not start while another auditor with its
hotkey answers at the hotkey's axon, and stops at an epoch's start once another one moved the axon. Only a fresh
document signed by the hotkey, from a host presenting the certificate the catalog names for the auditor, counts: a
host that is gone, slow, stale or does not verify never keeps the auditor from running."""

from __future__ import annotations

import contextlib
import json
import logging
import re
import socket
import sys
import time
from collections.abc import Callable
from types import SimpleNamespace

import httpx
import pytest
from bittensor_wallet import Keypair
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response

from tests.tlsfake import Presenting, certificate, free_port, pin, serving
from verathos import catalog
from verathos.protocol import tls
from verathos.protocol.manifest import axon_base
from verathos.protocol.status import (
    NETWORK_DOMAIN,
    NETWORK_PATH,
    STATUS_DOMAIN,
    STATUS_MAX_AGE_S,
    STATUS_PATH,
    HotkeyStatus,
    NetworkStatus,
    sign_document,
)
from verathos.validator import singleton
from verathos.validator.config import ValidatorConfig
from verathos.validator.service import add_network_route, add_status_route
from verathos.validator.status import StatusBoard

OWN = ("203.0.113.5", 8091)  # this instance's public_host and port
OWN_URL = axon_base(*OWN)
OTHER = ("198.51.100.7", 8091)  # where another instance points the hotkey's axon
OTHER_URL = axon_base(*OTHER)
CAT = catalog.Catalog.parse({**catalog.load().document(), "auditor": {"tls": pin(OTHER[0])}})
BLOCKS = CAT.epoch_blocks


def kp() -> Keypair:
    return Keypair.create_from_mnemonic(Keypair.generate_mnemonic())


class Chain:
    """The chain as the subnet owner's validator reads it: the owner hotkey and its axon (unset until served), the
    block, the metagraph and the subnet's weights version key."""

    def __init__(self, owner: str, axon: tuple[str, int] = ("0.0.0.0", 0), block: int = 100 * BLOCKS) -> None:
        self.owner, self.axon, self.block = owner, axon, block
        self.substrate = SimpleNamespace(query=lambda module, item, params: SimpleNamespace(value=0))

    def get_subnet_owner_hotkey(self, netuid: int, block: int | None = None) -> str:
        return self.owner

    def get_neuron_for_pubkey_and_subnet(self, hotkey: str, netuid: int) -> SimpleNamespace:
        ip, port = self.axon if hotkey == self.owner else ("0.0.0.0", 0)
        return SimpleNamespace(uid=0, is_null=False, axon_info=SimpleNamespace(ip=ip, port=port))

    def get_current_block(self) -> int:
        return self.block

    def metagraph(self, netuid: int, lite: bool = True) -> SimpleNamespace:
        return SimpleNamespace(hotkeys=[self.owner], axons=[SimpleNamespace(ip=self.axon[0], port=self.axon[1])],
                               S=[0.0], validator_permit=[True])

    def get_all_commitments(self, netuid: int) -> dict:
        return {}

    def close(self) -> None:
        pass


def auditor_service(keypair: Keypair) -> FastAPI:
    """The two documents as an auditor's service serves them (``validator.service``), signed with ``keypair``."""
    app = FastAPI()
    board = StatusBoard()
    add_status_route(app, board, None, keypair, 405)
    add_network_route(app, board, keypair, 405)
    return app


def signed(document: dict, signer: Keypair, domain: str, *, netuid: int = 405, age_s: float = 0.0) -> dict:
    """``document`` signed as a validator's service signs it, ``age_s`` ago."""
    return sign_document(document, signer, netuid=netuid, domain=domain, now_ms=int((time.time() - age_s) * 1000))


def host(hotkey: str, status: object, network: object) -> FastAPI:
    """A host answering the status of ``hotkey`` and the network's with these: a JSON body, an HTTP status code, or
    text."""
    app = FastAPI()

    def answer(body: object) -> Response:
        if isinstance(body, int):
            return JSONResponse({"error": "refused"}, status_code=body)
        if isinstance(body, str):
            return Response(body, media_type="text/plain")
        return JSONResponse(body)

    app.get(STATUS_PATH.format(hotkey=hotkey))(lambda: answer(status))
    app.get(NETWORK_PATH)(lambda: answer(network))
    return app


class Reached(httpx.AsyncBaseTransport):
    """Each host in ``apps`` as over TLS, presenting its certificate (``Presenting``); ``asked`` gets the paths."""

    def __init__(self, apps: dict[str, object], certificates: dict[str, bytes] | None, asked: list[str]) -> None:
        self.presenting, self.asked = Presenting(apps, certificates), asked

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.asked.append(request.url.path)
        return await self.presenting.handle_async_request(request)


def reach(apps: dict[str, object], certificates: dict[str, bytes] | None = None,
          asked: list[str] | None = None) -> Callable[[], httpx.AsyncClient]:
    return lambda: httpx.AsyncClient(transport=Reached(apps, certificates, [] if asked is None else asked))


def test_an_auditor_with_the_hotkey_answers_with_either_document_from_the_catalogs_certificate() -> None:
    """Its status is asked first (small); the network's only when the status does not verify (rate limited, stale)."""
    owner = kp()
    chain = Chain(owner.ss58_address, OTHER)
    status_path = STATUS_PATH.format(hotkey=owner.ss58_address)
    status = HotkeyStatus(hotkey=owner.ss58_address).model_dump(mode="json")
    network = NetworkStatus().model_dump(mode="json")

    def answering(app: FastAPI, asked: list[str]) -> str | None:
        return singleton.another_auditor(chain, 405, owner.ss58_address, OWN_URL, CAT,
                                         client=reach({OTHER[0]: app}, asked=asked))

    asked: list[str] = []
    assert answering(auditor_service(owner), asked) == OTHER_URL  # an auditor's service, before its first epoch too
    assert asked == [status_path]  # its status verifies: the network's is not asked
    asked = []
    rate_limited = host(owner.ss58_address, 429, signed(network, owner, NETWORK_DOMAIN))
    assert answering(rate_limited, asked) == OTHER_URL and asked == [status_path, NETWORK_PATH]
    stale_status = host(owner.ss58_address, signed(status, owner, STATUS_DOMAIN, age_s=STATUS_MAX_AGE_S + 60),
                        signed(network, owner, NETWORK_DOMAIN))
    assert answering(stale_status, []) == OTHER_URL
    assert answering(host(owner.ss58_address, signed(status, owner, STATUS_DOMAIN), 503), []) == OTHER_URL


def test_a_host_that_is_gone_stale_or_does_not_verify_lets_the_auditor_run(caplog) -> None:
    """Only a fresh document the hotkey signed for this netuid and kind counts, from the certificate the catalog
    names for the auditor; anything else is a host that is gone, as after a planned move, and is logged."""
    caplog.set_level(logging.WARNING)
    owner, stranger = kp(), kp()
    chain = Chain(owner.ss58_address, OTHER)
    status = HotkeyStatus(hotkey=owner.ss58_address).model_dump(mode="json")
    network = NetworkStatus().model_dump(mode="json")
    old = STATUS_MAX_AGE_S + 60

    def documents(signer: Keypair, **kwargs) -> FastAPI:
        return host(owner.ss58_address, signed(status, signer, STATUS_DOMAIN, **kwargs),
                    signed(network, signer, NETWORK_DOMAIN, **kwargs))

    hosts = {
        "gone": (None, "ConnectError: no route to 198.51.100.7"),
        "stale": (documents(owner, age_s=old), f"it was not signed within {STATUS_MAX_AGE_S:.0f} s of this machine's "
                                               "clock"),
        "ahead": (documents(owner, age_s=-old), "it was not signed within"),
        "another hotkey's": (documents(stranger), f"it is not signed by the subnet owner hotkey {owner.ss58_address}"),
        "for another netuid": (documents(owner, netuid=96), "for netuid 405"),
        "forged": (host(owner.ss58_address, {**signed(status, stranger, STATUS_DOMAIN), "auditor": owner.ss58_address},
                        {**signed(network, stranger, NETWORK_DOMAIN), "auditor": owner.ss58_address}),
                   "its signature does not verify against the subnet owner hotkey"),
        "tampered": (host(owner.ss58_address, {**signed(status, owner, STATUS_DOMAIN), "weight": 1.0},
                          {**signed(network, owner, NETWORK_DOMAIN), "epoch": 7}), "its signature does not verify"),
        "of the other kind": (host(owner.ss58_address, signed(network, owner, NETWORK_DOMAIN),
                                   signed(status, owner, STATUS_DOMAIN)), "its signature does not verify"),
        "unsigned": (host(owner.ss58_address, status, network), "it is not signed (the auditor runs an older release)"),
        "failing": (host(owner.ss58_address, 500, 404), "HTTPStatusError: Server error '500 Internal Server Error'"),
        "not JSON": (host(owner.ss58_address, "<html>", "<html>"), "JSONDecodeError"),
    }
    said = (f"the subnet owner hotkey's axon points at {OTHER_URL}, not at this auditor ({OWN_URL}), and no auditor "
            "answers there: ")
    for name, (app, why) in hosts.items():
        caplog.clear()
        apps = {} if app is None else {OTHER[0]: app}
        assert singleton.another_auditor(chain, 405, owner.ss58_address, OWN_URL, CAT, client=reach(apps)) is None, name
        (line,) = caplog.messages
        assert line.startswith(said) and why in line and "\n" not in line, (name, line)

    # Documents that verify, from a certificate other than the catalog's: refused before a request is sent.
    caplog.clear()
    asked: list[str] = []
    elsewhere = reach({OTHER[0]: auditor_service(owner)}, {OTHER[0]: certificate("elsewhere")}, asked)
    assert singleton.another_auditor(chain, 405, owner.ss58_address, OWN_URL, CAT, client=elsewhere) is None
    assert f"certificate pin {pin('elsewhere')} is not the pinned {CAT.auditor_pin()}" in caplog.messages[0]
    # A catalog that names no pin for the auditor: nothing is asked.
    caplog.clear()
    nameless = reach({OTHER[0]: auditor_service(owner)}, asked=asked)
    assert singleton.another_auditor(chain, 405, owner.ss58_address, OWN_URL, catalog.load(), client=nameless) is None
    assert "names no TLS pin for the auditor" in caplog.messages[0]
    assert len(asked) == 2  # the two refused by their pin, none for the catalog without one


def test_its_own_address_or_no_axon_asks_nothing() -> None:
    """The axon at this instance's public_host:port (as the chain writes it, IPv6 in its shortest form) is its own;
    an axon not served yet is no one's."""
    owner = kp()
    ipv6 = ValidatorConfig(netuid=405, wallet_name="w", hotkey_name="h", public_host="2001:0db8:0:0::5", port=8091)

    def nothing_asked() -> httpx.AsyncClient:
        raise AssertionError("nothing is asked")

    for axon, own in ((OWN, OWN_URL), (("2001:db8::5", 8091), axon_base(ipv6.public_host, ipv6.port)),
                      (("0.0.0.0", 0), OWN_URL), (("", 0), OWN_URL), ((OWN[0], 0), OWN_URL)):
        chain = Chain(owner.ss58_address, axon)
        assert singleton.another_auditor(chain, 405, owner.ss58_address, own, CAT, client=nothing_asked) is None


def test_a_host_that_takes_the_connection_and_never_answers_delays_a_start_by_the_timeout_at_most(caplog) -> None:
    """Over real TLS: the documents are asked within the timeout in total, the second with what the first left."""
    owner = kp()
    with socket.socket() as listener:  # takes connections into its backlog and never answers
        listener.bind(("127.0.0.1", 0))
        listener.listen(8)
        chain = Chain(owner.ss58_address, ("127.0.0.1", listener.getsockname()[1]))
        began = time.monotonic()
        assert singleton.another_auditor(chain, 405, owner.ss58_address, OWN_URL, CAT, timeout_s=0.5) is None
        assert time.monotonic() - began < 1.5
    (line,) = caplog.messages
    assert re.search(rf"{STATUS_PATH.format(hotkey=owner.ss58_address)}: (no answer within|ConnectTimeout)", line)
    assert singleton.TIMEOUT_S <= 5


# ---- the validator's start and its epochs ----

class Exited(BaseException):
    """``os._exit``: the process ends there."""

    def __init__(self, code: int) -> None:
        super().__init__(code)
        self.code = code


def run(tmp_path, monkeypatch, chain: Chain, owner: Keypair, cat: catalog.Catalog, started: list, *, once: bool,
        on_epoch: Callable[[int], None] | None = None) -> None:
    """``verathos-validator`` with the subnet owner's hotkey, as its service starts it, with the chain, the catalog
    follower, the throughput audit, the service and the epochs faked: ``started`` gets the tests (``Throughput.start``),
    the axon (the service, which serves it) and each epoch run, after which ``on_epoch`` runs and the chain moves to
    the next epoch."""
    from verathos.validator import __main__ as validator_main

    def tests(**kwargs) -> SimpleNamespace:
        started.append("tests")
        return SimpleNamespace(auditor=SimpleNamespace(), auditor_hotkey=owner.ss58_address,
                               start_slot_refresh=lambda *args: None)

    def service(**kwargs) -> None:
        started.append("axon")
        chain.axon = (kwargs["public_host"], kwargs["port"])

    async def epoch(*args) -> None:
        started.append(args[5])
        if sum(isinstance(s, int) for s in started) > 3:
            pytest.fail("the auditor kept running")  # an outcome, which the validator's loop does not catch
        if on_epoch is not None:
            on_epoch(args[5])
        chain.block += BLOCKS

    def exited(code: int) -> None:
        raise Exited(code)

    monkeypatch.setattr(validator_main.bt, "Subtensor", lambda network: chain)
    monkeypatch.setattr(validator_main.bt, "Wallet", lambda name, hotkey: SimpleNamespace(hotkey=owner))
    monkeypatch.setattr(validator_main, "CatalogFollower", lambda *args, **kwargs: SimpleNamespace(
        catalog=cat, store=SimpleNamespace(put=lambda c: None), on_change=[], start=lambda: None))
    monkeypatch.setattr(validator_main.Throughput, "start", tests)
    monkeypatch.setattr(validator_main, "start_service", service)
    monkeypatch.setattr(validator_main, "watch_egress", lambda *args: None)
    monkeypatch.setattr(validator_main, "wait_for_references", lambda references: None)
    monkeypatch.setattr(validator_main, "run_epoch", epoch)
    monkeypatch.setattr(validator_main, "time", SimpleNamespace(sleep=lambda s: None, monotonic=time.monotonic,
                                                                time=time.time))
    monkeypatch.setattr(validator_main.os, "_exit", exited)
    references = {m.model_id: f"http://127.0.0.1:{8101 + i}" for i, m in enumerate(cat.models) if m.status == "active"}
    path = tmp_path / "validator.json"
    path.write_text(json.dumps({"netuid": 405, "wallet_name": "owner", "hotkey_name": "default", "public_host": OWN[0],
                                "port": OWN[1], "state_dir": str(tmp_path / "state"),
                                "catalog_dir": str(tmp_path / "catalog"), "references": references}))
    monkeypatch.setattr(sys, "argv", ["verathos-validator", "--config", str(path), *(["--once"] if once else [])])
    validator_main.main()


def address(base: str) -> tuple[str, int]:
    ip, port = base.removeprefix("https://").rsplit(":", 1)
    return ip, int(port)


def test_another_live_auditor_keeps_a_second_one_from_starting(tmp_path, monkeypatch) -> None:
    """The hotkey's axon points at another address, where an auditor with the hotkey answers over TLS with the
    catalog's certificate: the second exits before its tests, its axon and its weights, naming the address."""
    owner = kp()
    first = tls.identity(tmp_path / "first")
    cat = catalog.Catalog.parse({**catalog.load().document(), "auditor": {"tls": first.pin}})
    started: list = []
    with serving(auditor_service(owner), first) as base:
        chain = Chain(owner.ss58_address, address(base))
        with pytest.raises(SystemExit) as refused:
            run(tmp_path, monkeypatch, chain, owner, cat, started, once=True)
    assert refused.value.code == (f"another auditor with this hotkey runs at {base}, where the hotkey's axon points: "
                                  "this one does not start; stop that one first, then start this one")
    assert started == [] and chain.axon == address(base)


def test_an_auditor_whose_old_host_is_gone_starts_and_points_the_axon_at_itself(tmp_path, monkeypatch,
                                                                                 caplog) -> None:
    """The axon still points at the old host, where nothing answers: a planned move after the old one stopped, or a
    dead host. The auditor starts as before, says so, and points the axon at itself."""
    caplog.set_level(logging.WARNING)
    owner = kp()
    gone = ("127.0.0.1", free_port())  # nothing listens there any more
    cat = catalog.Catalog.parse({**catalog.load().document(), "auditor": {"tls": tls.identity(tmp_path / "state").pin}})
    chain, started = Chain(owner.ss58_address, gone), []
    run(tmp_path, monkeypatch, chain, owner, cat, started, once=True)
    assert started == ["tests", "axon", 100] and chain.axon == OWN
    assert [r.getMessage().split(": /v1/status/")[0] for r in caplog.records if r.levelno >= logging.WARNING] == [
        f"the subnet owner hotkey's axon points at {axon_base(*gone)}, not at this auditor ({OWN_URL}), and no "
        "auditor answers there"]


def test_an_auditor_whose_axon_another_one_took_over_stops_at_the_next_epoch(tmp_path, monkeypatch, caplog) -> None:
    """While it runs, another instance points the hotkey's axon at itself. While nothing answers there (it is gone
    again, or the request failed) this one keeps running; once an auditor with the hotkey answers there, this one
    exits at the next epoch's start, before that epoch's tests are scored and its weights set, and its restart is
    refused while the other one runs."""
    caplog.set_level(logging.WARNING)
    owner = kp()
    identity = tls.identity(tmp_path / "state")  # the state directory copied to the other machine: the same pin
    cat = catalog.Catalog.parse({**catalog.load().document(), "auditor": {"tls": identity.pin}})
    chain, started, port = Chain(owner.ss58_address), [], free_port()
    other = ("127.0.0.1", port)
    with contextlib.ExitStack() as stack:
        def on_epoch(epoch: int) -> None:
            if epoch == 100:  # the other instance starts, takes the axon and is gone again
                chain.axon = other
            elif epoch == 101:  # it runs there again
                stack.enter_context(serving(auditor_service(owner), identity, port))

        with pytest.raises(Exited) as exited:
            run(tmp_path, monkeypatch, chain, owner, cat, started, once=False, on_epoch=on_epoch)
        assert exited.value.code == 1 and started == ["tests", "axon", 100, 101]
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1 and warnings[0].startswith(
            f"the subnet owner hotkey's axon points at {axon_base(*other)}, not at this auditor ({OWN_URL})")
        assert [r.getMessage() for r in caplog.records if r.levelno == logging.CRITICAL] == [
            f"another auditor with this hotkey runs at {axon_base(*other)}, where it moved the hotkey's axon: this one "
            "stops (no more tests or weights); stop that one first, then start this one"]

        # The process manager starts it again: refused while the other one answers.
        started.clear()
        with pytest.raises(SystemExit, match="another auditor with this hotkey runs at"):
            run(tmp_path, monkeypatch, chain, owner, cat, started, once=False)
        assert started == []
