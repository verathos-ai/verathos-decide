"""A full validator keeps the addresses it publishes current, and holds the endpoints it could not reach while miners
learn a new one."""

from __future__ import annotations

from types import SimpleNamespace

from verathos.protocol.callers import ValidatorIdentity
from verathos.validator import egress
from verathos.validator.audit import EndpointResult
from verathos.validator.compare import Tolerance

PIN = "0f" * 32


def test_a_validator_publishes_where_its_requests_leave_from_and_its_unreached_endpoints_are_held_meanwhile() -> None:
    published, detected, clock = [], ["8.8.8.8"], [100.0]
    watch = egress.Egress(["8.8.8.8", "8.8.4.4"], PIN, published.append, lambda: detected[0], lambda: clock[0])
    assert not watch.check() and published == []
    detected[0] = "1.2.3.4"
    clock[0] = 5000.0
    assert watch.check() and published == [ValidatorIdentity(PIN, ("1.2.3.4", "8.8.8.8", "8.8.4.4"))]
    assert not watch.check() and len(published) == 1
    assert watch.in_grace(5000.0 + 600) and watch.in_grace(5000.0 - 600) and not watch.in_grace(5000.0 + 1201)
    longest = "2606:4700:4700:1111:2606:4700:4700:1112"
    assert egress.fitting(PIN, [longest, "198.211.103.255", "198.211.103.254"]) == (longest, "198.211.103.255")
    detected[0] = None  # no detection: nothing changes
    assert not watch.check()

    def result(answered: int, *errors: str) -> EndpointResult:
        return EndpointResult(endpoint=SimpleNamespace(), tolerance=Tolerance(), answered=answered,
                              errors=list(errors))

    results = {"a": result(0, "request failed: ConnectTimeout", "identity not proven at its URL"),
               "b": result(0, "refused: HTTP 403"), "c": result(2, "request failed: ReadTimeout"),
               "d": result(0, "malformed reply: KeyError"), "e": result(0)}
    assert egress.excused(results) == {"a", "b"}
