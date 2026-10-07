"""Epoch scores, EMA, probation and weights.

Per endpoint and epoch:
- zero if the identity challenge failed or verification failed. A hard failure counts in the epoch it happens; the
  statistical bound is evaluated over the endpoint's last ``statistical_window_epochs`` audited epochs, because one
  epoch of canaries is too few probabilities to separate a lower-precision substitute from honest drift;
- a verification failure, or a wrong answer the throughput auditor proved (an answer the miner signed in a test or
  to the subnet's gateway), fails every endpoint of the hotkey: their averages reset to zero and they score zero for
  ``probation_epochs`` epochs, the failing one included, regardless of later results. An endpoint the hotkey adds
  while on probation inherits the remaining probation, so moving hardware to another endpoint does not escape it;
- zero until the endpoint has proven capacity in a throughput test;
- otherwise its model's share of the epoch's emission (``model_shares``), split among the model's scored endpoints in
  proportion to proven capacity (requests/s) x availability (answered / sent) x availability to users x latency
  factor. Availability to users is the share of the times the gateways considered the endpoint for a user request
  that it was ready (not busy, failing, or answering slower than twice the latency bound), less the requests it
  refused, once it was considered ``user_refusals_min_offers`` times: capacity shown in tests but kept from users
  does not count, and neither does keeping users waiting. The latency
  factor compares the endpoint's median latency with the median over the scored endpoints serving the same model,
  each weighted by its proven capacity (so splitting capacity over more endpoints does not move it), capped to
  [0.7, 1.3]: the latency of its answers to
  users once its hotkey's endpoints serving the model gave ``user_latency_min_answers`` of them, compared among
  endpoints measured the same way, else its latency in its throughput test, compared among those, and then at most 1
  while users measure others of the model. Every validator reads the same figures from the auditor's
  verdicts. Candidate models score zero.
An endpoint that could not be audited (no usable reference answers) keeps its average for at most ``MAX_HELD_EPOCHS``
epochs in a row. Per UID: the sum of its endpoints' EMAs. Weights: UID values normalised to 1.
"""

from __future__ import annotations

import json
import math
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path

from verathos.catalog import Catalog
from verathos.validator.audit import EndpointResult
from verathos.validator.compare import Verdict

LATENCY_CAP = (0.7, 1.3)
SCORE_SCALE = 2  # scores are shares of the emission (1: proven requests/s); averages of another scale start over
PRUNE_BELOW = 1e-4  # EMA under which an endpoint that is gone is forgotten
MAX_HELD_EPOCHS = 3


@dataclass
class ScoreState:
    ema: dict[str, float] = field(default_factory=dict)
    probation: dict[str, int] = field(default_factory=dict)
    uid_of: dict[str, int] = field(default_factory=dict)
    window: dict[str, list[list[float]]] = field(default_factory=dict)  # key -> [[sum_abs, count], ...] newest last
    evidence_seen: dict[str, int] = field(default_factory=dict)  # evidence id -> epoch it was applied
    held: dict[str, int] = field(default_factory=dict)  # key -> epochs in a row its average was held
    last_epoch: int | None = None  # the last epoch scored, so a restart does not score it again
    scale: int = SCORE_SCALE

    @classmethod
    def load(cls, path: Path) -> ScoreState:
        if not path.exists():
            return cls()
        known = {f.name for f in fields(cls)}
        data = {k: v for k, v in json.loads(path.read_text()).items() if k in known}
        if data.get("scale") != SCORE_SCALE:  # averages on another scale would outweigh the new ones for hours
            data.pop("ema", None)
            data.pop("held", None)
            data["scale"] = SCORE_SCALE
        return cls(**data)

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


def weighted_median(values: list[tuple[float, float]]) -> float:
    """The median of (value, weight) pairs: the value at which the cumulative weight reaches half the total (between
    two values when it reaches exactly half, so equal weights give the ordinary median)."""
    ordered = sorted(values)
    half, total = sum(w for _, w in ordered) / 2, 0.0
    for i, (value, weight) in enumerate(ordered):
        total += weight
        if math.isclose(total, half) and i + 1 < len(ordered):
            return (value + ordered[i + 1][0]) / 2
        if total >= half:
            return value
    return ordered[-1][0]


def model_shares(demand: Mapping[str, float], cat: Catalog) -> dict[str, float]:
    """Each active model's share of the epoch's emission, summing to 1.

    ``demand`` is each model's user demand in input tokens/s through the gateway, as the auditor publishes it. A
    model's share follows what its users pay (demand x ``price``), blended toward the catalog's anchor shares while
    the total demand is below ``scoring.demand_full_tokens_s``; every active model keeps at least
    ``scoring.min_model_share``."""
    anchors = cat.anchor_shares()
    if not anchors:
        return {}
    price = {m.model_id: m.price for m in cat.models}
    tokens = {m: max(float(demand.get(m, 0.0)), 0.0) for m in anchors}
    revenue = {m: tokens[m] * price[m] for m in anchors}
    paid = sum(revenue.values())
    confidence = min(1.0, sum(tokens.values()) / cat.scoring.demand_full_tokens_s)
    demanded = {m: revenue[m] / paid if paid > 0 else anchors[m] for m in anchors}
    shares = {m: (1 - confidence) * anchors[m] + confidence * demanded[m] for m in anchors}
    return _with_floor(shares, min(cat.scoring.min_model_share, 1.0 / len(anchors)))


def _with_floor(shares: dict[str, float], floor: float) -> dict[str, float]:
    """Shares summing to 1 with none below ``floor`` (at most an equal split): those below are raised to it, the
    others scaled down in proportion, until none falls below."""
    pinned: set[str] = set()
    while True:
        free = {k: v for k, v in shares.items() if k not in pinned}
        scale = (1.0 - floor * len(pinned)) / sum(free.values()) if sum(free.values()) > 0 else 0.0
        low = {k for k, v in free.items() if v * scale < floor}
        if not low:
            return {k: floor if k in pinned else v * scale for k, v in shares.items()}
        pinned |= low


def epoch_scores(results: dict[str, EndpointResult], *, shares: Mapping[str, float], credited: dict[str, float],
                 latency_ms: dict[str, float], state: ScoreState, failed_hotkeys: Collection[str] = (),
                 user_latency_ms: Mapping[str, float] | None = None,
                 user_availability: Mapping[str, float] | None = None) -> tuple[dict[str, float], set[str]]:
    """-> (score per endpoint key, keys that failed verification this epoch). Call after ``record_window``.

    ``shares`` holds each model's share of the emission (``model_shares``; missing: none); ``credited`` each
    endpoint's proven capacity in requests/s (missing: none proven yet), ``latency_ms`` its median latency in that
    test (missing: no latency factor); ``failed_hotkeys`` the hotkeys with a wrong answer proven by the throughput
    auditor; ``user_latency_ms`` the median latency of its answers to users, where there were enough (it takes the
    place of the test's); ``user_availability`` the share of the times it was considered for users that it was
    ready (missing: 1).
    """
    user_latency_ms, user_availability = user_latency_ms or {}, user_availability or {}
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

    def latency(key: str) -> tuple[str, float]:
        """-> (how it was measured, the endpoint's median latency; 0: unknown)"""
        if user_latency_ms.get(key, 0.0) > 0:
            return "users", user_latency_ms[key]
        return "test", latency_ms.get(key, 0.0)

    latencies: dict[tuple[str, str], list[tuple[float, float]]] = {}  # (model, measured by) -> [(ms, capacity)]
    for key, r in eligible.items():
        source, ms = latency(key)
        if ms > 0:
            latencies.setdefault((r.endpoint.endpoint.model_id, source), []).append((ms, credited[key]))
    median = {group: weighted_median(v) for group, v in latencies.items()}
    by_users = {model for model, source in latencies if source == "users"}
    weight: dict[str, float] = {}
    for key, r in eligible.items():
        (source, own), factor, model = latency(key), 1.0, r.endpoint.endpoint.model_id
        if own > 0:
            factor = min(max(math.sqrt(median[(model, source)] / own), LATENCY_CAP[0]), LATENCY_CAP[1])
            if source == "test" and model in by_users:
                factor = min(factor, 1.0)  # staying out of users' sight earns no bonus
        availability = (r.answered / r.sent) * min(max(user_availability.get(key, 1.0), 0.0), 1.0)
        weight[key] = credited[key] * availability * factor
    model_total: dict[str, float] = {}
    for key, w in weight.items():
        model = eligible[key].endpoint.endpoint.model_id
        model_total[model] = model_total.get(model, 0.0) + w
    for key, w in weight.items():
        model = eligible[key].endpoint.endpoint.model_id
        scores[key] = shares.get(model, 0.0) * w / model_total[model] if model_total[model] > 0 else 0.0
    return scores, failed


def update(state: ScoreState, results: dict[str, EndpointResult], scores: dict[str, float], failed: set[str],
           *, alpha: float = 0.2, probation_epochs: int = 6, held: Collection[str] = (),
           reset_hotkeys: Collection[str] = ()) -> None:
    """Fold one epoch into the state. ``held`` endpoints were discovered but could not be audited this epoch (no
    usable reference answers); their EMA is kept. Other endpoints absent this epoch decay toward zero and are
    forgotten once their EMA is negligible. Every endpoint of a hotkey with a failed endpoint or in ``reset_hotkeys``
    starts probation with its average reset to zero and its statistical window cleared."""
    reset = set(reset_hotkeys) | {key.split("/", 1)[0] for key in failed}
    for key in set(state.ema) | set(results) | set(held) | set(state.window):
        if key.split("/", 1)[0] in reset:
            state.probation[key] = probation_epochs
            state.ema[key] = 0.0
            state.window.pop(key, None)
    state.held = {k: state.held.get(k, 0) + 1 for k in held}
    held = {k for k in held if state.held[k] <= MAX_HELD_EPOCHS}
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
