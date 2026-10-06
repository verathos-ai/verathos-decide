"""The miner's start: its service runs first, the capacity is measured through it, and the manifest declares it."""

from __future__ import annotations

import asyncio
import functools
import socket
import threading
import time
from types import SimpleNamespace

import pytest
import uvicorn
from bittensor_wallet import Keypair

from tests.mock_engine import MockEngine
from verathos import catalog
from verathos.miner import __main__ as miner_main
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
