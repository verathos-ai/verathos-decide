"""The Clef engine's batching, and (on a CUDA GPU with the pinned weights downloaded) its answers against the release's
reference code."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from verathos import catalog
from verathos.engine.clef import FastEngine, Prepared


def prepared(length: int) -> Prepared:
    return Prepared(body={}, encoded=SimpleNamespace(input_ids=(0,) * length))


def test_passes_group_similar_lengths_within_the_token_budget() -> None:
    engine = object.__new__(FastEngine)
    engine.length_step, engine.max_rows, engine.max_batch_tokens = 32, 4, 2048
    items = [prepared(n) for n in (500, 40, 510, 600, 45, 2000, 30)]
    groups = engine._batches(items)
    assert sorted(i for g in groups for i in g) == list(range(len(items)))
    for group in groups:
        longest = -(-max(len(items[i].encoded.input_ids) for i in group) // 32) * 32
        assert len(group) <= 4 and (len(group) == 1 or len(group) * longest <= 2048)
    assert [len(items[i].encoded.input_ids) for i in groups[0]] == [30, 40, 45, 500]  # shortest first
    assert [5] in groups  # 2,000 tokens: alone


def test_graph_batches_cover_small_passes_only() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from verathos.engine.graphs import GraphedForward

    graphs = GraphedForward(lambda ids: ids, 8, torch.bfloat16, torch.device("cuda"), max_tokens=2048)
    assert graphs.graphed_batch(3, 512) == 3
    assert graphs.graphed_batch(5, 512) is None  # 2,560 tokens: eager
    assert graphs.graphed_batch(9, 64) is None  # more rows than any graph
    assert graphs.graphed_batch(1, 500) is None  # not a bucket length


REQUESTS = [
    {"model": "clef-flash", "state": "We were billed twice for March and the export button fails with a 500 error.",
     "questions": {"department": {"type": "choice", "instructions": "Which team should handle this?",
                                  "criteria": {"billing": "Charges and refunds", "technical": "Bugs or outages"}},
                   "urgency": {"type": "score", "instructions": "How urgent is this?",
                               "criteria": ["Can wait", "This week", "Today"]},
                   "refund": {"type": "noul", "instructions": "Is a refund requested?"}}},
    {"model": "clef-flash", "state": "Please add dark mode to the dashboard. No rush, thanks!",
     "questions": {"kind": {"type": "choice", "instructions": "What kind of request is this?",
                            "criteria": {"bug": None, "feature": None, "question": None}},
                   "angry": {"type": "noul", "instructions": "Is the customer angry?"}}},
]


def test_fast_engine_matches_the_release_reference() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or os.environ.get("VERATHOS_GPU_TESTS") != "1":
        pytest.skip("set VERATHOS_GPU_TESTS=1 on a CUDA GPU with the pinned clef-flash weights downloaded")
    from verathos.engine import clef
    from verathos.protocol.api import distributions

    model = catalog.load().get("clef-flash")
    reference = clef.load(model.hf_repo, model.hf_revision, device="cuda", dtype="bfloat16", fast=False)
    expected = [distributions(r["answers"]) for r in reference.run([reference.prepare(b) for b in REQUESTS])]
    del reference
    torch.cuda.empty_cache()
    fast = clef.load(model.hf_repo, model.hf_revision, device="cuda", dtype="bfloat16")
    got = [distributions(r["answers"]) for r in fast.run([fast.prepare(b) for b in REQUESTS])]
    for g, e in zip(got, expected):
        for qid, dist in e.items():
            assert max(abs(dist[k] - g[qid][k]) for k in dist) < 0.03
            assert max(dist, key=dist.get) == max(g[qid], key=g[qid].get) or max(dist.values()) < 0.6
