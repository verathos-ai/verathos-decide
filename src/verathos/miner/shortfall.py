"""An endpoint that falls short in the throughput tests corrects its declaration.

The auditor's tests measure what an endpoint serves under real conditions; the miner's own measurement can
overestimate it, or the hardware slow down after it. When an endpoint's last two counted tests (confirmed, not void)
both came out ``partial`` with less delivered than the rate tested, at the rate the endpoint declares now, the miner
measures again and from then on declares at most the higher rate the endpoint delivered in those two tests: its cap.
An endpoint is measured again for shortfalls at most once per ``COOLDOWN_EPOCHS``, never twice for the same test.

A cap holds until the miner measures for another reason (an engine, an endpoint's configuration or the request mix
changed, or ``--remeasure``); measuring again for a shortfall keeps every cap. A declaration so falls only on the
auditor's evidence and rises only on a measurement the tests did not cause, so it cannot oscillate.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from verathos.protocol.status import ThroughputOutcome

log = logging.getLogger("verathos.miner")
COOLDOWN_EPOCHS = 3
SAME_RATE = 0.005  # declared rates are rounded to 0.01


@dataclass(frozen=True)
class Shortfall:
    """An endpoint's last two counted tests, older first, both short of the rate tested."""

    endpoint_id: str
    tests: tuple[ThroughputOutcome, ThroughputOutcome]

    @property
    def cap_rps(self) -> float:
        return max(t.delivered_rps for t in self.tests)

    def line(self) -> str:
        """The log line when it triggers a measurement."""
        a, b = self.tests
        if f"{a.tested_rps:.1f}" == f"{b.tested_rps:.1f}":
            rates = f"delivered {a.delivered_rps:.1f} and {b.delivered_rps:.1f} of {a.tested_rps:.1f}"
        else:
            rates = (f"delivered {a.delivered_rps:.1f} of {a.tested_rps:.1f} and {b.delivered_rps:.1f} of "
                     f"{b.tested_rps:.1f}")
        return f"{self.endpoint_id} fell short in tests {a.epoch} and {b.epoch} ({rates}): measuring again"


@dataclass(frozen=True)
class Cap:
    """The tests (epochs) that last made an endpoint measure again, and the cap they set: None once a measurement for
    another reason released it (the tests still count for the cooldown)."""

    epochs: tuple[int, ...]
    rps: float | None

    def text(self) -> str:
        return f"at most the {self.rps:.1f} it delivered in tests {' and '.join(map(str, self.epochs))}"


def fell_short(t: ThroughputOutcome, declared_rps: float) -> bool:
    """Whether a counted test found the endpoint short at the rate it declares now: confirmed, ``partial``, and less
    delivered than the rate tested (a partial with the tested rate delivered came from users sending more, which no
    declaration changes)."""
    return (t.state == "confirmed" and t.outcome == "partial" and t.delivered_rps < t.tested_rps
            and abs(t.declared_rps - declared_rps) <= SAME_RATE)


class Shortfalls:
    """Each endpoint's ``Cap``, kept across restarts in ``path``."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser()
        self._caps: dict[str, Cap] = {}
        try:
            raw = json.loads(self.path.read_text())
            self._caps = {k: Cap(tuple(int(e) for e in v["epochs"]), None if v["rps"] is None else float(v["rps"]))
                          for k, v in raw.items()}
        except FileNotFoundError:
            pass
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            log.warning("%s unreadable (%s); no declaration is capped", self.path, exc)

    def cap(self, endpoint_id: str) -> Cap | None:
        """The endpoint's cap, if one holds."""
        c = self._caps.get(endpoint_id)
        return c if c is not None and c.rps is not None else None

    def due(self, tests: Iterable[ThroughputOutcome], declared: dict[str, float]) -> list[Shortfall]:
        """The endpoints of ``declared`` (endpoint id -> the rate it declares now) to measure again."""
        tests = list(tests)
        out = []
        for endpoint_id, rate in declared.items():
            counted = sorted((t for t in tests if t.endpoint_id == endpoint_id and t.state != "void"
                              and t.outcome != "void"), key=lambda t: t.epoch)[-2:]
            if len(counted) < 2 or not all(fell_short(t, rate) for t in counted):
                continue
            last = self._caps.get(endpoint_id)
            if last is not None and (counted[0].epoch <= last.epochs[-1]
                                     or counted[1].epoch < last.epochs[-1] + COOLDOWN_EPOCHS):
                continue
            out.append(Shortfall(endpoint_id, (counted[0], counted[1])))
        return out

    def record(self, shortfalls: Sequence[Shortfall]) -> None:
        """Measured again for ``shortfalls``: their tests set the endpoints' caps; the other caps hold."""
        for s in shortfalls:
            self._caps[s.endpoint_id] = Cap(tuple(t.epoch for t in s.tests), s.cap_rps)
        self._save()

    def release(self, endpoint_ids: Collection[str]) -> None:
        """Measured for another reason: the endpoints' caps no longer hold."""
        released = [k for k in endpoint_ids if self.cap(k) is not None]
        for k in released:
            self._caps[k] = Cap(self._caps[k].epochs, None)
        if released:
            self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({k: {"epochs": list(c.epochs), "rps": c.rps} for k, c in self._caps.items()},
                                  indent=1, sort_keys=True))
        tmp.replace(self.path)
