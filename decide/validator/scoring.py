"""Epoch scores, EMA, probation and weights.

Per endpoint and epoch:
- zero if the identity challenge failed or verification failed (hard or statistical); a verification failure also
  starts probation, during which the endpoint scores zero regardless of later results;
- otherwise model weight x availability (answered / sent) x latency factor, where the latency factor compares the
  endpoint's median latency with the median of all endpoints serving the same model, capped to [0.7, 1.3].
Per UID: sum of its endpoints' EMAs. Weights: UID sums normalised to 1.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from decide.validator.audit import EndpointResult
from decide.verify.compare import Verdict

LATENCY_CAP = (0.7, 1.3)


@dataclass
class ScoreState:
    ema: dict[str, float] = field(default_factory=dict)
    probation: dict[str, int] = field(default_factory=dict)
    uid_of: dict[str, int] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> ScoreState:
        return cls(**json.loads(path.read_text())) if path.exists() else cls()

    def save(self, path: Path) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.__dict__, indent=1, sort_keys=True))
        tmp.replace(path)


def epoch_scores(results: dict[str, EndpointResult], model_weight: dict[str, float]) -> tuple[dict[str, float], set[str]]:
    """-> (score per endpoint key, keys that failed verification this epoch)."""
    medians_by_model: dict[str, list[float]] = {}
    for key, r in results.items():
        if r.latencies_ms:
            medians_by_model.setdefault(r.endpoint.endpoint.model_id, []).append(statistics.median(r.latencies_ms))
    model_median = {m: statistics.median(v) for m, v in medians_by_model.items()}

    scores, failed = {}, set()
    for key, r in results.items():
        verdict = r.tally.verdict()
        if verdict in (Verdict.FAIL_HARD, Verdict.FAIL_STATISTICAL):
            failed.add(key)
            scores[key] = 0.0
            continue
        if not r.identity_ok or r.answered == 0 or r.sent == 0:
            scores[key] = 0.0
            continue
        model = r.endpoint.endpoint.model_id
        own = statistics.median(r.latencies_ms)
        latency = math.sqrt(model_median[model] / own) if own > 0 else LATENCY_CAP[1]
        latency = min(max(latency, LATENCY_CAP[0]), LATENCY_CAP[1])
        scores[key] = model_weight.get(model, 1.0) * (r.answered / r.sent) * latency
    return scores, failed


def update(state: ScoreState, results: dict[str, EndpointResult], scores: dict[str, float], failed: set[str],
           *, alpha: float = 0.2, probation_epochs: int = 6) -> None:
    """Fold one epoch into the state. Endpoints absent this epoch decay toward zero."""
    for key in failed:
        state.probation[key] = probation_epochs
    for key, r in results.items():
        state.uid_of[key] = r.endpoint.uid
    for key in set(state.ema) | set(scores):
        current = scores.get(key, 0.0)
        if state.probation.get(key, 0) > 0:
            current = 0.0
            if key not in failed:
                state.probation[key] -= 1
        state.ema[key] = alpha * current + (1 - alpha) * state.ema.get(key, 0.0)
    state.probation = {k: v for k, v in state.probation.items() if v > 0}


def weights(state: ScoreState, hotkey_of_uid: dict[int, str]) -> dict[int, float]:
    """Normalised weight per UID. Endpoints whose UID now belongs to another hotkey are ignored."""
    per_uid: dict[int, float] = {}
    for key, value in state.ema.items():
        uid = state.uid_of.get(key)
        hotkey = key.split("/", 1)[0]
        if uid is None or hotkey_of_uid.get(uid) != hotkey or value <= 0:
            continue
        per_uid[uid] = per_uid.get(uid, 0.0) + value
    total = sum(per_uid.values())
    return {uid: v / total for uid, v in per_uid.items()} if total > 0 else {}
