"""A requested stop (SIGINT from pm2 or a terminal, SIGTERM from systemd or supervisord) ends every long-running
process with one log line and exit status 0, without a traceback (``verathos.stop``)."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def stopped(tmp_path: Path, code: str, sig: signal.Signals, timeout_s: float = 60.0) -> tuple[int, str]:
    """Run ``code`` in a new interpreter until it writes the file named by READY, send it ``sig``; -> its exit status
    and what it wrote to stderr."""
    ready = tmp_path / f"ready-{sig.name}"
    env = {**os.environ, "READY": str(ready)}
    proc = subprocess.Popen([sys.executable, "-c", textwrap.dedent(code)], cwd=ROOT, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + timeout_s
        while not ready.exists():
            if proc.poll() is not None or time.monotonic() > deadline:
                raise AssertionError(f"not ready (exit {proc.poll()}): {proc.communicate()[1][-2000:]}")
            time.sleep(0.05)
        proc.send_signal(sig)
        _, err = proc.communicate(timeout=timeout_s)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()
    return proc.returncode, err


BLOCKING = """
    import logging, os, sys, threading
    from pathlib import Path
    from verathos import stop

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    stop.on_signals("the test process", logging.getLogger("verathos.test"))
    worker = threading.Thread(target=threading.Event().wait, daemon=False)  # a thread that never ends
    worker.start()
    Path(os.environ["READY"]).touch()
    worker.join()
"""


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_a_signal_stops_a_process_with_one_line_and_status_0_whatever_its_threads_do(tmp_path, sig) -> None:
    status, err = stopped(tmp_path, BLOCKING, sig)
    assert status == 0 and f"INFO verathos.test the test process stopped ({sig.name})" in err
    assert "Traceback" not in err


UVICORN = """
    import asyncio, contextlib, logging, os, socket
    from pathlib import Path

    import uvicorn
    from fastapi import FastAPI

    from verathos import stop
    from verathos.gateway.__main__ import run  # imported before logging is set up, as the entry points do

    log = logging.getLogger("verathos.test")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    @contextlib.asynccontextmanager
    async def lifespan(app):
        Path(os.environ["READY"]).touch()
        yield
        await asyncio.sleep(0.2)
        log.info("final writes done")

    stop.on_signals("the test server", log)
    listening = socket.create_server(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(FastAPI(lifespan=lifespan), log_level="warning"))
"""
SERVE = {"gateway": "run(server.serve(sockets=[listening]))\n",
         "reference front": "server.run(sockets=[listening])\n"}  # their event loops: uvloop where installed


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
@pytest.mark.parametrize("like", sorted(SERVE))
def test_a_server_in_the_main_thread_shuts_down_gracefully_before_it_stops(tmp_path, sig, like) -> None:
    """uvicorn takes the signal while it serves, runs its lifespan's shutdown, then raises the signal again into the
    stop handler: the process ends with status 0 after its final writes, where uvicorn alone would end it by the
    signal (SIGTERM) or with a KeyboardInterrupt traceback (SIGINT)."""
    status, err = stopped(tmp_path, textwrap.dedent(UVICORN) + SERVE[like], sig)
    assert status == 0 and "Traceback" not in err
    assert err.index("final writes done") < err.index(f"the test server stopped ({sig.name})")


# Each entry point, held where it would read its configuration (the engine: where it loads its model)
ENTRY_POINTS = {
    "validator": ("""
        from types import SimpleNamespace
        from verathos.validator import __main__ as m
        m.ValidatorConfig = SimpleNamespace(load=hold)
        sys.argv = ["verathos-validator", "--config", "validator.json"]
        m.main()
    """, signal.SIGINT),
    "miner": ("""
        from types import SimpleNamespace
        from verathos.miner import __main__ as m
        m.MinerConfig = SimpleNamespace(load=hold)
        sys.argv = ["verathos-miner", "--config", "miner.json"]
        m.main()
    """, signal.SIGTERM),
    "gateway": ("""
        from types import SimpleNamespace
        from verathos.gateway import __main__ as m

        async def serve(config):
            hold()
        m.GatewayConfig = SimpleNamespace(load=lambda path: None)
        m.serve = serve
        sys.argv = ["verathos-gateway", "--config", "gateway.json", "serve"]
        m.main()
    """, signal.SIGINT),
    "reference front": ("""
        from types import SimpleNamespace
        from verathos.reference import __main__ as m
        m.ReferenceConfig = SimpleNamespace(load=hold)
        sys.argv = ["verathos-reference", "--config", "reference.json"]
        m.main()
    """, signal.SIGTERM),
    "engine": ("""
        import types
        import verathos.engine
        fake = types.ModuleType("verathos.engine.clef")
        fake.load = hold
        sys.modules["verathos.engine.clef"] = verathos.engine.clef = fake
        from verathos.engine import __main__ as m
        sys.argv = ["verathos-engine", "clef", "--repo", "r/m", "--revision", "abc", "--device", "cpu"]
        m.main()
    """, signal.SIGINT),
}
HOLD = """
    import os, sys, time
    from pathlib import Path

    def hold(*args, **kwargs):
        Path(os.environ["READY"]).touch()
        time.sleep(120)
"""


@pytest.mark.parametrize("role", sorted(ENTRY_POINTS))
def test_every_long_running_process_stops_cleanly(tmp_path, role) -> None:
    code, sig = ENTRY_POINTS[role]
    status, err = stopped(tmp_path, textwrap.dedent(HOLD) + textwrap.dedent(code), sig)
    assert status == 0 and f"the {role} stopped ({sig.name})" in err and "Traceback" not in err
