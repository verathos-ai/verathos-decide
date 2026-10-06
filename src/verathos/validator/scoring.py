"""Epoch scores, EMA, probation and weights.

Per endpoint and epoch:
- zero if the identity challenge failed or verification failed. A hard failure counts in the epoch it happens; the
  statistical bound is evaluated over the endpoint's last ``statistical_window_epochs`` audited epochs, because one
  epoch of canaries is too few probabilities to separate a lower-precision substitute from honest drift. A
  verification failure also starts probation: the endpoint scores zero for ``probation_epochs`` epochs, the failing
  one included, regardless of later results. An endpoint a hotkey adds while one of its endpoints is on probation
  inherits the remaining probation, so renaming an endpoint does not escape it;
- a wrong answer the throughput auditor proved (an answer the miner signed in a test) fails every endpoint of the
  hotkey: probation, and their averages reset to zero;
- zero until the endpoint has proven capacity in a throughput test;
- otherwise score per request/s of the model x proven capacity (requests/s) x availability (answered / sent) x latency
  factor, where the latency factor compares the endpoint's median latency with the median over the passing endpoints
  serving the same model, capped to [0.7, 1.3]. Candidate models score zero.
Per UID: the sum of its endpoints' EMAs. Weights: UID values normalised to 1.
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Collection
from dataclasses import dataclass, field, fields
from pathlib import Path

from verathos.validator.audit import EndpointResult
from verathos.validator.compare import Verdict

LATENCY_CAP = (0.7, 1.3)
PRUNE_BELOW = 1e-4  # EMA under which an endpoint that is gone is forgotten


@dataclass
class ScoreState:
    ema: dict[str, float] = field(default_factory=dict)
    probation: dict[str, int] = field(default_factory=dict)
    uid_of: dict[str, int] = field(default_factory=dict)
    window: dict[str, list[list[float]]] = field(default_factory=dict)  # key -> [[sum_abs, count], ...] newest last
    evidence_seen: dict[str, int] = field(default_factory=dict)  # evidence id -> epoch it was applied

    @classmethod
    def load(cls, path: Path) -> ScoreState:
        if not path.exists():
            return cls()
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in json.loads(path.read_text()).items() if k in known})

    def save(self, path: Path) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.__dict__, indent=1, sort_keys=True))
        tmp.replace(path)


def new_evidence(state: ScoreState, ids: Collection[str], epoch: int, keep_epochs: int) -> set[str]:
    """The evidence ids not applied before; records them. Ids are kept for ``keep_epochs`` epochs, as long as the
    auditor keeps reporting them."""
    fresh = {i for i in ids if i not in state.evidence_seen}
    state.evidence_seen = {i: e for i, e in state.evidence_seen.items() if e > epoch - keep_epochs}
    state.evidence_seen.update(dict.fromkeys(fresh, epoch))
    return fresh


def record_window(state: ScoreState, results: dict[str, EndpointResult], window_epochs: int) -> None:
    """Append this epoch's comparison totals and keep the last ``window_epochs`` audited epochs per endpoint."""
    for key, r in results.items():
        entries = state.window.get(key, []) + [[r.tally.sum_abs, float(r.tally.count)]]
        state.window[key] = entries[-window_epochs:]


def windowed_verdict(state: ScoreState, key: str, result: EndpointResult) -> Verdict:
    """Hard failures count in the epoch they happen; the statistical bound uses the rolling window."""
    if result.tally.hard_failures:
        return Verdict.FAIL_HARD
    entries = state.window.get(key, [])
    count = sum(c for _, c in entries)
    tol = result.tolerance
    if count < tol.min_probabilities:
        return Verdict.INSUFFICIENT
    return Verdict.FAIL_STATISTICAL if sum(s for s, _ in entries) / count > tol.mean_abs else Verdict.PASS


def epoch_scores(results: dict[str, EndpointResult], *, score_per_rps: dict[str, float], credited: dict[str, float],
                 state: ScoreState, failed_hotkeys: Collection[str] = ()) -> tuple[dict[str, float], set[str]]:
    """-> (score per endpoint key, keys that failed verification this epoch). Call after ``record_window``.

    ``credited`` holds each endpoint's proven capacity in requests/s (missing: none proven yet); ``failed_hotkeys``
    the hotkeys with a wrong answer proven by the throughput auditor.
    """
    scores: dict[str, float] = {}
    failed: set[str] = set()
    eligible: dict[str, EndpointResult] = {}
    for key, r in results.items():
        if r.endpoint.hotkey in failed_hotkeys or windowed_verdict(state, key, r) in (Verdict.FAIL_HARD,
                                                                                      Verdict.FAIL_STATISTICAL):
            failed.add(key)
            scores[key] = 0.0
        elif not r.identity_ok or r.answered == 0 or r.sent == 0 or credited.get(key, 0.0) <= 0:
            scores[key] = 0.0
        else:
            eligible[key] = r

    medians_by_model: dict[str, list[float]] = {}
    for r in eligible.values():
        medians_by_model.setdefault(r.endpoint.endpoint.model_id, []).append(statistics.median(r.latencies_ms))
    model_median = {m: statistics.median(v) for m, v in medians_by_model.items()}
    for key, r in eligible.items():
        model = r.endpoint.endpoint.model_id
        own = statistics.median(r.latencies_ms)
        latency = math.sqrt(model_median[model] / own) if own > 0 else LATENCY_CAP[1]
        latency = min(max(latency, LATENCY_CAP[0]), LATENCY_CAP[1])
        scores[key] = score_per_rps.get(model, 0.0) * credited[key] * (r.answered / r.sent) * latency
    return scores, failed


def update(state: ScoreState, results: dict[str, EndpointResult], scores: dict[str, float], failed: set[str],
           *, alpha: float = 0.2, probation_epochs: int = 6, held: Collection[str] = (),
           reset_hotkeys: Collection[str] = ()) -> None:
    """Fold one epoch into the state. ``held`` endpoints were discovered but could not be audited this epoch (no
    usable reference answers); their EMA is kept. Other endpoints absent this epoch decay toward zero and are
    forgotten once their EMA is negligible. Every endpoint of a hotkey in ``reset_hotkeys`` starts probation with
    its average reset to zero."""
    for key in failed:
        state.probation[key] = probation_epochs
    for key in set(state.ema) | set(results) | set(held):
        if key.split("/", 1)[0] in reset_hotkeys:
            state.probation[key] = probation_epochs
            state.ema[key] = 0.0
    on_probation: dict[str, int] = {}
    for key, remaining in state.probation.items():
        hotkey = key.split("/", 1)[0]
        on_probation[hotkey] = max(on_probation.get(hotkey, 0), remaining)
    for key, r in results.items():
        if key not in state.ema and key not in state.probation and on_probation.get(r.endpoint.hotkey):
            state.probation[key] = on_probation[r.endpoint.hotkey]
        state.uid_of[key] = r.endpoint.uid
    for key in set(state.ema) | set(scores) | set(state.probation):
        if key in held and key not in scores and key not in state.probation:
            continue
        current = scores.get(key, 0.0)
        if state.probation.get(key, 0) > 0:
            current = 0.0
            state.probation[key] -= 1
        state.ema[key] = alpha * current + (1 - alpha) * state.ema.get(key, 0.0)
    state.probation = {k: v for k, v in state.probation.items() if v > 0}
    for key in [k for k, v in state.ema.items()
                if v < PRUNE_BELOW and k not in results and k not in state.probation and k not in held]:
        del state.ema[key]
        state.uid_of.pop(key, None)
        state.window.pop(key, None)


def weights(state: ScoreState, hotkey_of_uid: dict[int, str]) -> dict[int, float]:
    """Normalised weight per UID from the sum of its endpoints' EMAs. Endpoints whose UID now belongs to another
    hotkey are ignored."""
    values: dict[int, float] = {}
    for key, value in state.ema.items():
        uid = state.uid_of.get(key)
        hotkey = key.split("/", 1)[0]
        if uid is None or hotkey_of_uid.get(uid) != hotkey or value <= 0:
            continue
        values[uid] = values.get(uid, 0.0) + value
    total = sum(values.values())
    return {uid: v / total for uid, v in values.items()} if total > 0 else {}
