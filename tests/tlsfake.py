"""TLS for in-process services: ``serving`` runs an app over real TLS on a loopback port."""

from __future__ import annotations

import contextlib
import socket
import threading
import time
from collections.abc import Iterator

import uvicorn

from verathos.protocol import tls


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def serving(app: object, identity: tls.Identity, port: int | None = None) -> Iterator[str]:
    """``app`` served over TLS with ``identity`` on a loopback port (``port``, else a free one), as a miner or
    validator serves; -> its base URL."""
    port = port or free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning",
                                           **tls.server_options(identity)))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        assert thread.is_alive(), "the server did not start"
        time.sleep(0.02)
    try:
        yield f"https://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(5)
