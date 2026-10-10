"""Shared fixtures."""

from __future__ import annotations

import ipaddress
import signal

import bittensor  # noqa: F401  (first, as the services import it: it silences the loggers that exist when it loads)
import pytest

from verathos.protocol import manifest

# Miners, validators and gateways of a test all run on this host: they reach each other at loopback, private and
# documentation addresses, which the network itself never connects to.
LAB_NETWORKS = tuple(ipaddress.ip_network(n) for n in ("127.0.0.0/8", "10.0.0.0/8", "192.0.2.0/24", "198.51.100.0/24",
                                                       "203.0.113.0/24", "2001:db8::/32"))


@pytest.fixture()
def lab_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Treat the lab's addresses as public for the test, so services on one test host stand for separate machines.
    Only the test harness widens the rule; configuration cannot."""
    public = manifest.public_unicast
    monkeypatch.setattr(manifest, "public_unicast", lambda ip: public(ip) or any(ip in n for n in LAB_NETWORKS))


@pytest.fixture(autouse=True)
def runner_signals():
    """An entry point run in this process installs its stop handlers (``verathos.stop``), which end the process with
    status 0: the test runner gets its own back, so an interrupt or a timeout's SIGTERM still fails the run."""
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    yield
    for sig, handler in saved.items():
        signal.signal(sig, handler)
