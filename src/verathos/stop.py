"""A requested stop: SIGINT (pm2, a terminal) or SIGTERM (systemd, supervisord) ends a process with one log line and
exit status 0, without a traceback.

``on_signals`` installs the handler for both. It ends the process where it stands, as a kill would, so no worker
thread, connection or interpreter cleanup can hold up the stop. A process serving with uvicorn in its main thread
installs it before the server starts: uvicorn takes the signals while it serves, finishes the requests in flight and
its lifespan's shutdown, then puts this handler back and raises the signal again into it.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
from types import FrameType

SIGNALS = (signal.SIGINT, signal.SIGTERM)


def on_signals(what: str, logger: logging.Logger) -> None:
    """From now on SIGINT and SIGTERM log that ``what`` stopped and end the process with status 0."""

    def stopped(signum: int, frame: FrameType | None) -> None:
        logger.info("%s stopped (%s)", what, signal.Signals(signum).name)
        logging.shutdown()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    for sig in SIGNALS:
        signal.signal(sig, stopped)
