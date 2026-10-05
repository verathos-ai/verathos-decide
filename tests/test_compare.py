"""Verification bounds against measured outputs: the same 30 requests answered by Kev-4B on four machines and
by Kev-9B (a substituted model)."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

from decide.protocol.api import distributions
from decide.verify.compare import EndpointTally, Tolerance, Verdict, compare_request

FIXTURES = Path(__file__).parent / "fixtures" / "drift"
HONEST = ["4090", "5090", "M1Max", "CPUfp32"]


def load(name: str) -> list[dict]:
    return [distributions(r["answers"]) for r in json.loads((FIXTURES / f"drift-{name}.json").read_text())]


def tally(reference: list[dict], candidate: list[dict], tolerance: Tolerance) -> EndpointTally:
    t = EndpointTally(tolerance)
    for ref, got in zip(reference, candidate):
        t.add(compare_request(ref, got, tolerance))
    return t


@pytest.mark.parametrize("a,b", list(itertools.combinations(HONEST, 2)))
def test_honest_hardware_passes(a: str, b: str) -> None:
    t = tally(load(f"kev-4b-{a}"), load(f"kev-4b-{b}"), Tolerance())
    assert t.verdict() is Verdict.PASS
    assert t.mean_abs < 0.0025


@pytest.mark.parametrize("reference", ["kev-4b-CPUfp32", "kev-4b-5090"])
def test_substituted_model_fails_every_request(reference: str) -> None:
    tol = Tolerance()
    ref, sub = load(reference), load("kev-9b-5090")
    results = [compare_request(r, s, tol) for r, s in zip(ref, sub)]
    assert all(r.hard_fail(tol) for r in results)
    assert tally(ref, sub, tol).verdict() is Verdict.FAIL_HARD


def test_statistical_bound_catches_small_systematic_drift() -> None:
    tol = Tolerance()
    ref = load("kev-4b-CPUfp32")
    shifted = [{q: _shift(d, 0.02) for q, d in req.items()} for req in ref]
    t = tally(ref, shifted, tol)
    assert not t.hard_failures
    assert t.verdict() is Verdict.FAIL_STATISTICAL


def test_structural_mismatch_fails() -> None:
    tol = Tolerance()
    ref = load("kev-4b-CPUfp32")[0]
    missing = {q: d for q, d in ref.items() if q != "churn"}
    assert compare_request(ref, missing, tol).structural_error
    renamed = {q: ({("x" + k): p for k, p in d.items()} if q == "department" else d) for q, d in ref.items()}
    assert compare_request(ref, renamed, tol).structural_error


def test_too_few_samples_is_insufficient() -> None:
    tol = Tolerance(min_probabilities=10_000)
    ref = load("kev-4b-CPUfp32")
    assert tally(ref, load("kev-4b-5090"), tol).verdict() is Verdict.INSUFFICIENT


def _shift(dist: dict[str, float], delta: float) -> dict[str, float]:
    """Move ``delta`` of mass from the top option to the second, as a lower-precision model would on average."""
    ranked = sorted(dist, key=dist.get, reverse=True)
    out = dict(dist)
    if len(ranked) > 1:
        moved = min(delta, out[ranked[0]])
        out[ranked[0]] -= moved
        out[ranked[1]] += moved
    return out
