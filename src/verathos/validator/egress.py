"""A full validator's egress addresses, kept current while it runs.

Miners admit requests by the addresses validators publish (``protocol.callers``; their firewalls, ``miner.firewall``).
The validator publishes its configured ``egress_ips`` at start. Every ``CHECK_S`` it detects the public IPv4 address
its requests leave from, as the installer does, and when that is not among the addresses it publishes, publishes it
first, with as many of the configured ones as fit, and says so. For ``GRACE_S`` after such a change, until the miners
have read it, the endpoints its requests did not reach are held, not scored (``excused``).
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Mapping

from verathos.protocol.callers import ValidatorIdentity
from verathos.validator.audit import EndpointResult

log = logging.getLogger(__name__)
CHECK_S = 300.0
GRACE_S = 1200.0  # two of the miners' refreshes
_UNREACHED = ("request failed", "refused: HTTP 403", "identity not proven at its URL")


def fitting(pin: str | None, addresses: list[str]) -> tuple[str, ...]:
    """The first of ``addresses`` that fit in a commitment next to ``pin`` (``ValidatorIdentity``)."""
    kept = list(addresses)
    while kept:
        try:
            ValidatorIdentity(pin, tuple(kept)).commitment()
            return tuple(kept)
        except ValueError:
            kept.pop()
    return ()


class Egress:
    """Detects where this validator's requests leave from and publishes it when it changed (see the module):
    ``detect`` gives the public IPv4 address or None, ``publish`` commits an identity (the new address first)."""

    def __init__(self, configured: list[str], pin: str | None, publish: Callable[[ValidatorIdentity], None],
                 detect: Callable[[], str | None], clock: Callable[[], float] = time.time) -> None:
        self.configured, self.pin, self.publish, self.detect, self.clock = list(configured), pin, publish, detect, clock
        self.published = fitting(pin, self.configured)
        self.changed_at: float | None = None

    def check(self) -> bool:
        """Detect the address and publish it if it is new; -> whether it published."""
        address = self.detect()
        if address is None or address in self.published:
            return False
        egress = fitting(self.pin, [address, *(a for a in self.configured if a != address)])
        self.publish(ValidatorIdentity(self.pin, egress))
        self.published, self.changed_at = egress, self.clock()
        return True

    def in_grace(self, since: float) -> bool:
        """Whether the published addresses changed within ``GRACE_S`` before ``since``, or after it."""
        return self.changed_at is not None and self.changed_at >= since - GRACE_S

    def start(self) -> None:
        def loop() -> None:
            while True:
                try:
                    self.check()
                except Exception as exc:
                    log.warning("egress check failed (%s); trying again", str(exc).splitlines()[0] if str(exc) else
                                type(exc).__name__)
                time.sleep(CHECK_S)

        threading.Thread(target=loop, name="egress", daemon=True).start()


def excused(results: Mapping[str, EndpointResult]) -> set[str]:
    """The endpoints this validator's requests did not reach at all: no answer, and only failed or refused requests or
    an identity it could not check."""
    return {key for key, r in results.items()
            if r.answered == 0 and r.errors and all(e.startswith(_UNREACHED) for e in r.errors)}
