"""The miner's start: its service runs first, the capacity is measured through it, and the manifest declares it. Its
background loops log a lost chain connection in one line."""

from __future__ import annotations

import asyncio
import functools
import logging
import socket
import threading
import time
from types import SimpleNamespace

import pytest
import uvicorn
from async_substrate_interface.errors import MaxRetriesExceeded
from bittensor_wallet import Keypair
from websockets.exceptions import ConnectionClosedError
from websockets.frames import Close

from tests.mock_engine import MockEngine
from verathos import catalog
from verathos.miner import __main__ as miner_main
from verathos.miner import chain as miner_chain
from verathos.miner.app import create_app
from verathos.miner.config import MinerConfig
from verathos.miner.engine import EngineError
from verathos.throughput import measure


class CapacityEngine:
    """Completes at most ``rps`` requests per second."""

    def __init__(self, rps: float, concurrency: int = 4) -> None:
        self.engine, self.service_s, self.calls = MockEngine("kev-4b"), concurrency / rps, 0
        self._slots = asyncio.Semaphore(concurrency)

    async def answer(self, request):
        self.calls += 1
        async with self._slots:
            await asyncio.sleep(self.service_s)
            return await self.engine.answer(request)


class Down:
    async def answer(self, request):
        raise EngineError("engine not running")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_capacity_is_measured_through_the_service_and_a_dead_engine_is_withdrawn(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(miner_main, "measure_and_store", functools.partial(measure.measure_and_store, run_s=0.2))
    monkeypatch.setattr(miner_main, "wait_ready", functools.partial(measure.wait_ready, timeout_s=0.5))
    keypair, auditor = (Keypair.create_from_mnemonic(Keypair.generate_mnemonic()) for _ in range(2))
    port = free_port()
    config = MinerConfig.model_validate({
        "netuid": 405, "wallet_name": "w", "hotkey_name": "h", "public_host": "203.0.113.5", "port": port,
        "listen_host": "127.0.0.1", "state_dir": str(tmp_path),
        "endpoints": [{"endpoint_id": e, "model_id": "kev-4b", "engine_url": "http://127.0.0.1:9", "max_concurrency": 8}
                      for e in ("gpu0", "gpu1")]})
    cat = catalog.load()
    chain = SimpleNamespace(get_subnet_owner_hotkey=lambda netuid: auditor.ss58_address)
    publisher = miner_main.Publisher(config, SimpleNamespace(hotkey=keypair), cat, publish_on_chain=False,
                                     subtensor=chain)
    engines = {"gpu0": CapacityEngine(rps=100), "gpu1": Down()}
    app = create_app(config=config, keypair=keypair, manifest=lambda: publisher.signed, engines=engines,
                     authorized=lambda hk: False, auditors=publisher.auditors, admissions=publisher.admissions,
                     serving=lambda: publisher.serving)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.05)
    try:
        assert publisher.adopt(cat)
        listed = {e.endpoint_id: e.capacity_rps for e in publisher.signed.manifest.endpoints}
        assert set(listed) == {"gpu0"} and publisher.serving == {"gpu0"}
        assert listed["gpu0"] == pytest.approx(0.95 * 100, rel=0.2)
        assert publisher.admissions["gpu0"].limit <= 8
        calls = engines["gpu0"].calls
        assert publisher.adopt(cat)  # stored measurement reused: one readiness request only
        assert engines["gpu0"].calls == calls + 1
        assert {e.endpoint_id: e.capacity_rps for e in publisher.signed.manifest.endpoints} == listed
    finally:
        server.should_exit = True


def test_a_test_arriving_at_the_start_delays_the_first_manifest_instead_of_stopping_the_miner(tmp_path) -> None:
    config = MinerConfig.model_validate({
        "netuid": 405, "wallet_name": "w", "hotkey_name": "h", "public_host": "203.0.113.5", "port": 8091,
        "state_dir": str(tmp_path),
        "endpoints": [{"endpoint_id": "gpu0", "model_id": "kev-4b", "engine_url": "http://127.0.0.1:9"}]})
    keypair = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
    publisher = miner_main.Publisher(config, SimpleNamespace(hotkey=keypair), catalog.load(), publish_on_chain=False,
                                     subtensor=SimpleNamespace(get_subnet_owner_hotkey=lambda netuid: "auditor"))
    outcomes = iter([(False, True), (False, True), (True, False)])  # (adopted, interrupted)

    def adopt(cat, *, remeasure):
        adopted, publisher.interrupted = next(outcomes)
        return adopted

    publisher.adopt, waits = adopt, []
    miner_main.adopt_first(publisher, catalog.load(), remeasure=False, sleep=waits.append)
    assert waits == [miner_main.TEST_QUIET_S] * 2

    publisher.adopt = lambda cat, *, remeasure: False
    publisher.interrupted = False
    with pytest.raises(SystemExit):
        miner_main.adopt_first(publisher, catalog.load(), remeasure=False, sleep=waits.append)

    assert not publisher.testing() and not publisher.test_arriving()
    publisher.note_test_traffic()
    assert publisher.testing() and publisher.test_arriving()  # no measurement starts now; validators wait
    publisher.test_seen -= miner_main.TEST_ARRIVING_S
    assert publisher.testing() and not publisher.test_arriving()  # the traffic is over: users and validators alike


def test_a_lost_chain_connection_is_one_warning_line_and_a_programming_error_keeps_its_traceback(
        tmp_path, monkeypatch, caplog) -> None:
    keepalive = ConnectionClosedError(None, Close(1011, "keepalive ping timeout"), None)
    config = MinerConfig.model_validate({
        "netuid": 405, "wallet_name": "w", "hotkey_name": "h", "public_host": "203.0.113.5", "port": 8091,
        "state_dir": str(tmp_path),
        "endpoints": [{"endpoint_id": "gpu0", "model_id": "kev-4b", "engine_url": "http://127.0.0.1:9"}]})
    keypair = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
    publisher = miner_main.Publisher(config, SimpleNamespace(hotkey=keypair), catalog.load(), publish_on_chain=False,
                                     subtensor=SimpleNamespace(get_subnet_owner_hotkey=lambda netuid: "auditor"))
    failure: list[Exception] = []

    def committed(st, netuid):
        raise failure[0]

    monkeypatch.setattr(miner_main, "committed_catalog", committed)
    store = catalog.CatalogStore(tmp_path / "catalog")

    def follow_logs(exc: Exception) -> list[logging.LogRecord]:
        failure[:] = [exc]
        caplog.clear()
        assert publisher.follow(SimpleNamespace(), store) is None  # a new connection next round
        return caplog.records

    caplog.set_level(logging.INFO)
    for exc, line in ((keepalive, "ConnectionClosedError: sent 1011 (internal error) keepalive ping timeout; no close "
                                  "frame received"),
                      (MaxRetriesExceeded("Max retries reached."), "MaxRetriesExceeded: Max retries reached."),
                      (ConnectionRefusedError(111, "Connection refused"),
                       "ConnectionRefusedError: [Errno 111] Connection refused"),
                      (catalog.CatalogUnavailable("the subnet owner has not committed a catalog"),
                       "CatalogUnavailable: the subnet owner has not committed a catalog")):
        [record] = follow_logs(exc)
        assert (record.levelno, record.exc_info) == (logging.WARNING, None)
        assert record.getMessage() == f"catalog follow failed ({line}); retrying"
    [record] = follow_logs(KeyError("hotkeys"))
    assert record.levelno == logging.ERROR and record.exc_info is not None

    class Node:
        """The metagraph a validator with a permit is read from, until the connection fails."""

        fails: Exception | None = None

        def metagraph(self, netuid, lite=True):
            if self.fails is not None:
                raise self.fails
            return SimpleNamespace(hotkeys=["validator"], validator_permit=[True], S=[1e6], block=907)

    node = Node()
    monkeypatch.setattr(miner_chain.bt, "Subtensor", lambda *args, **kwargs: node)
    authorizer = miner_chain.PermitAuthorizer("test", 405, lambda: set())
    node.fails = keepalive
    caplog.clear()
    authorizer.try_refresh()
    [record] = caplog.records
    assert (record.levelno, record.exc_info) == (logging.WARNING, None)
    assert record.getMessage() == ("permit refresh failed (ConnectionClosedError: sent 1011 (internal error) keepalive "
                                   "ping timeout; no close frame received); keeping the previous set")
    assert authorizer("validator")  # the previous set stays
    node.fails = AttributeError("'NoneType' object has no attribute 'hotkeys'")
    caplog.clear()
    authorizer.try_refresh()
    [record] = caplog.records
    assert record.levelno == logging.ERROR and record.exc_info is not None
