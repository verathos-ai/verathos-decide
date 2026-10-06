"""Compare a miner's answers with a reference computation of the same request.

One forward pass of a decision model is nearly deterministic across GPUs, Apple Silicon and CPU, so the full
probability vector is a fingerprint of the exact weights. Two checks apply:

- Per request (hard): a structural mismatch, any probability off by more than ``hard_abs``, or a changed top
  answer where the reference was confident (margin of at least ``flip_margin``) fails the request outright.
- Per endpoint (statistical): the mean absolute difference over all compared probabilities must stay below
  ``mean_abs`` once at least ``min_probabilities`` have been compared. This catches lower-precision
  substitutes that stay under the hard bound on single requests.

Default bounds come from measurements on Kev-4B/9B (RTX 4090, RTX 5090, M1 Max MLX, CPU fp32; 420
probabilities per run): honest pairs differed by at most 0.017 per probability with a mean of 0.0014-0.0023;
a different model (Kev-9B answering for Kev-4B) had a mean of 0.084 and exceeded 0.10 on every request.
Bounds are per model and are set in the catalog.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, Field

from verathos.protocol.api import Distribution


class Tolerance(BaseModel):
    hard_abs: float = Field(0.10, gt=0, le=1)
    flip_margin: float = Field(0.20, ge=0, le=1)
    mean_abs: float = Field(0.005, gt=0, le=1)
    min_probabilities: int = Field(100, ge=1)


@dataclass(frozen=True)
class RequestComparison:
    """Result of comparing one response with its reference."""

    max_abs: float
    sum_abs: float
    count: int
    confident_flips: int
    structural_error: str | None = None

    def hard_fail(self, tolerance: Tolerance) -> bool:
        return self.structural_error is not None or self.max_abs > tolerance.hard_abs or self.confident_flips > 0


def _top(dist: Distribution) -> tuple[str, float]:
    """Most likely key and its margin over the runner-up."""
    ranked = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
    margin = ranked[0][1] - (ranked[1][1] if len(ranked) > 1 else 0.0)
    return ranked[0][0], margin


def compare_request(
    reference: dict[str, Distribution],
    candidate: dict[str, Distribution],
    tolerance: Tolerance,
) -> RequestComparison:
    """Compare the per-question distributions of one request."""
    if set(reference) != set(candidate):
        return RequestComparison(1.0, 0.0, 0, 0, "question ids differ")
    max_abs, sum_abs, count, flips = 0.0, 0.0, 0, 0
    for qid, ref in reference.items():
        got = candidate[qid]
        if set(ref) != set(got):
            return RequestComparison(1.0, sum_abs, count, flips, f"option keys differ for {qid!r}")
        for key, p in ref.items():
            d = abs(p - got[key])
            max_abs = max(max_abs, d)
            sum_abs += d
            count += 1
        ref_top, ref_margin = _top(ref)
        got_top, _ = _top(got)
        if got_top != ref_top and ref_margin >= tolerance.flip_margin:
            flips += 1
    return RequestComparison(max_abs, sum_abs, count, flips)


class Verdict(str, Enum):
    PASS = "pass"
    INSUFFICIENT = "insufficient"
    FAIL_HARD = "fail_hard"
    FAIL_STATISTICAL = "fail_statistical"


@dataclass
class EndpointTally:
    """Accumulates comparisons for one endpoint over an epoch."""

    tolerance: Tolerance
    sum_abs: float = 0.0
    count: int = 0
    requests: int = 0
    hard_failures: list[RequestComparison] = field(default_factory=list)

    def add(self, result: RequestComparison) -> None:
        self.requests += 1
        self.sum_abs += result.sum_abs
        self.count += result.count
        if result.hard_fail(self.tolerance):
            self.hard_failures.append(result)

    @property
    def mean_abs(self) -> float:
        return self.sum_abs / self.count if self.count else 0.0

    def verdict(self) -> Verdict:
        if self.hard_failures:
            return Verdict.FAIL_HARD
        if self.count < self.tolerance.min_probabilities:
            return Verdict.INSUFFICIENT
        if self.mean_abs > self.tolerance.mean_abs:
            return Verdict.FAIL_STATISTICAL
        return Verdict.PASS
