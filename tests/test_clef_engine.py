"""The Clef engine: request checks, batching, CUDA graph buckets, the batched head and the fast path's plumbing against
the release's per-record head (CPU, where the pinned release code is in the Hugging Face cache), and on a CUDA GPU
with the pinned weights the fast path's answers against stored CPU float32 reference answers."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from verathos import catalog
from verathos.engine.clef import FastEngine, Prepared, ReferenceEngine, _validate
from verathos.protocol.api import distributions
from verathos.validator.compare import compare_request

FIXTURE = Path(__file__).parent / "fixtures" / "drift" / "drift-clef-flash-CPUfp32.json"


def prepared(length: int) -> Prepared:
    return Prepared(body={}, encoded=SimpleNamespace(input_ids=(0,) * length))


def test_passes_group_similar_lengths_within_the_token_budget() -> None:
    engine = object.__new__(FastEngine)
    engine.length_step, engine.max_rows, engine.max_batch_tokens = 32, 4, 2048
    items = [prepared(n) for n in (500, 40, 510, 600, 45, 2000, 30)]
    groups = engine.passes(items)
    assert sorted(i for g in groups for i in g) == list(range(len(items)))
    for group in groups:
        longest = -(-max(len(items[i].encoded.input_ids) for i in group) // 32) * 32
        assert len(group) <= 4 and (len(group) == 1 or len(group) * longest <= 2048)
    assert [len(items[i].encoded.input_ids) for i in groups[0]] == [30, 40, 45, 500]  # shortest first
    assert [5] in groups  # 2,000 tokens: alone


def test_the_reference_answers_each_request_in_its_own_pass() -> None:
    assert object.__new__(ReferenceEngine).passes([prepared(10)] * 3) == [[0], [1], [2]]


def test_requests_outside_the_protocol_are_refused_before_a_pass() -> None:
    body = {"model": "clef-flash", "state": "s", "questions": {"q": {"type": "noul", "instructions": "?"}}}
    _validate(body)
    for media in ({"images": ["https://example.com/a.png"]}, {"videos": ["v.mp4"]}):
        with pytest.raises(ValueError, match="images and videos"):
            _validate({**body, **media})
    with pytest.raises(ValueError, match="criteria"):
        _validate({**body, "questions": {"q": {"type": "choice", "instructions": "?"}}})


def test_graphs_cover_small_passes_and_a_failed_capture_leaves_every_pass_eager(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    from verathos.engine.graphs import GraphedForward

    graphs = GraphedForward(lambda ids: ids, 8, torch.float32, torch.device("cpu"), max_tokens=2048)
    shapes = graphs.shapes()
    assert (1, 2048) in shapes and (8, 256) in shapes and (3, 672) in shapes and (3, 704) not in shapes
    assert all(b * n <= 2048 and n % 32 == 0 for b, n in shapes)
    graphs.graphs = dict.fromkeys(shapes)
    assert graphs.graphed_batch(3, 512) == 3
    assert graphs.graphed_batch(5, 512) is None  # 2,560 tokens: eager
    assert graphs.graphed_batch(9, 64) is None  # more rows than any graph
    assert graphs.graphed_batch(1, 500) is None  # not a bucket length

    captured = []

    def capture(batch: int, length: int) -> None:
        if len(captured) == 3:
            raise RuntimeError("operation not permitted when stream is capturing")
        captured.append((batch, length))
        graphs.graphs[(batch, length)] = object()

    graphs.graphs = {}
    monkeypatch.setattr(graphs, "_capture", capture)
    monkeypatch.setattr(torch.cuda, "graph_pool_handle", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    assert graphs.capture_all() is False
    assert graphs.graphs == {} and graphs.graphed_batch(1, 32) is None
    ids = torch.ones(2, 32, dtype=torch.long)
    assert graphs(ids) is ids  # eager: the forward itself


# ---- the batched head and the fast path's plumbing, against the release's own head (CPU, float32) ----

def release_code() -> Path | None:
    model = catalog.load().get("clef-flash")
    hub = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
               / "hub")
    path = (hub / f"models--{model.hf_repo.replace('/', '--')}" / "snapshots" / model.hf_revision
            / "joint_schema_model.py")
    return path if path.exists() else None


@pytest.fixture(scope="module")
def release():
    pytest.importorskip("torch")
    path = release_code()
    if path is None:
        pytest.skip("needs the pinned clef-flash release code (joint_schema_model.py) in the Hugging Face cache")
    from verathos.engine.clef import _release_module

    return _release_module(path.parent)


VOCAB, HIDDEN = 64, 16


def tiny_head(release):
    import torch

    torch.manual_seed(0)
    head = release.JointSchemaHead(hidden_size=HIDDEN, width=8, routing_layers=2, layers=2, heads=2, feedforward=16)
    with torch.no_grad():
        for p in (head.prior_logit_scale, head.joint_logit_scale, head.residual_gate):
            p.fill_(0.7)
    return head.eval()


def causal_backbone():
    """Hidden states where each token depends on itself and every token before it, as in the real backbone."""
    import torch

    embedding = torch.randn(VOCAB, HIDDEN, generator=torch.Generator().manual_seed(1))

    def forward(ids):
        x = embedding[ids]
        steps = torch.arange(1, ids.shape[1] + 1, dtype=x.dtype)[None, :, None]
        return x.cumsum(1) / steps + torch.sin(x)

    return forward


def record_and_body(release, rng: random.Random, index: int):
    """An encoded record with 1-3 questions of every type, its spans at random places, and its request body."""
    kinds = [rng.choice(["noul", "choice", "score"]) for _ in range(rng.randint(1, 3))]
    state = rng.randint(3, 60)
    input_ids, questions, body = [rng.randrange(1, VOCAB) for _ in range(state)], [], {}
    for q, kind in enumerate(kinds):
        options = ("true", "false") if kind == "noul" else tuple(
            sorted(f"o{i}" for i in range(rng.randint(2, 5)))) if kind == "choice" else tuple(
            str(i) for i in range(rng.randint(2, 4)))
        start = len(input_ids)
        input_ids += [rng.randrange(1, VOCAB) for _ in range(rng.randint(1, 4))]
        question_span, spans = (start, len(input_ids)), []
        for _ in options:
            begin = len(input_ids) + 1
            input_ids += [rng.randrange(1, VOCAB) for _ in range(rng.randint(2, 5))]
            spans.append((begin, len(input_ids)))
        input_ids += [rng.randrange(1, VOCAB) for _ in range(rng.randint(0, 3))]
        qid = f"q{q}"
        questions.append(release.EncodedQuestion(qid, release.QUESTION_TYPES[kind], question_span, tuple(spans),
                                                 options))
        body[qid] = {"type": kind} if kind == "noul" else {"type": kind, "criteria": {o: o for o in options}} \
            if kind == "choice" else {"type": kind, "criteria": [f"level {o}" for o in options]}
    record = release.EncodedRecord(tuple(input_ids), tuple(questions), f"r{index}")
    return record, {"model": "clef-flash", "state": "s", "questions": body}


def release_logits(release, head, backbone, output_embedding, record):
    """The release's head on one unpadded record, as ``systemone`` runs it."""
    import torch

    ids = torch.tensor([record.input_ids])
    with torch.inference_mode():
        return head(backbone(ids), ids, torch.ones_like(ids), [record], output_embedding)[0]


def test_the_batched_head_on_a_padded_pass_matches_the_release_head_per_record(release) -> None:
    import torch

    from verathos.engine.clef_head import batched_head

    head, backbone = tiny_head(release), causal_backbone()
    output_embedding = torch.randn(VOCAB, HIDDEN, generator=torch.Generator().manual_seed(2))
    rng = random.Random(3)
    records = [record_and_body(release, rng, i)[0] for i in range(6)]
    length = -(-max(len(r.input_ids) for r in records) // 8) * 8
    ids = torch.zeros((len(records) + 2, length), dtype=torch.long)  # two padding rows, as a graph bucket adds
    for row, r in enumerate(records):
        ids[row, : len(r.input_ids)] = torch.tensor(r.input_ids)
    lengths = torch.tensor([len(r.input_ids) for r in records])
    with torch.inference_mode():
        got = batched_head(head, backbone(ids), ids, lengths, records, output_embedding)
    for record, record_logits in zip(records, got):
        expected = release_logits(release, head, backbone, output_embedding, record)
        assert len(record_logits) == len(expected)
        for g, e in zip(record_logits, expected):
            torch.testing.assert_close(g, e, rtol=1e-4, atol=1e-5)


def test_the_fast_path_answers_each_request_as_it_would_alone_and_as_the_release_does(release) -> None:
    import torch

    from verathos.engine.graphs import GraphedForward

    head, backbone = tiny_head(release), causal_backbone()
    engine = object.__new__(FastEngine)
    engine.torch, engine.module, engine.head, engine.pad_id = torch, release, head, 0
    engine.output_embedding = torch.randn(VOCAB, HIDDEN, generator=torch.Generator().manual_seed(2))
    engine.device, engine.max_length = torch.device("cpu"), 16384
    engine.max_batch_tokens, engine.max_rows, engine.length_step = 256, 4, 8
    engine.forward = GraphedForward(backbone, HIDDEN, torch.float32, engine.device, max_tokens=0)
    rng = random.Random(4)
    items = [Prepared(body, record) for record, body in (record_and_body(release, rng, i) for i in range(11))]
    rng.shuffle(items)
    assert len(engine.passes(items)) > 2  # several passes, some with several requests

    together = [distributions(r["answers"]) for r in engine.run(items)]
    for item, got in zip(items, together):
        alone = distributions(engine.run([item])[0]["answers"])
        logits = release_logits(release, head, backbone, engine.output_embedding, item.encoded)
        expected = {q.question_id: dict(zip(q.option_ids, ql.float().softmax(-1).tolist()))
                    for q, ql in zip(item.encoded.questions, logits)}
        for qid, dist in expected.items():
            for key, p in dist.items():
                assert abs(got[qid][key] - p) <= 1.5e-4 and abs(alone[qid][key] - got[qid][key]) <= 1.5e-4


# ---- on a CUDA GPU with the pinned weights ----

def test_fast_engine_answers_stay_within_the_catalog_tolerances_of_the_float32_reference() -> None:
    """The 30 tickets ``scripts/probe_miner.py`` sends, against the release's own code in float32 on CPU without
    flash-linear-attention (as validators run it), answered together and one at a time."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or os.environ.get("VERATHOS_GPU_TESTS") != "1":
        pytest.skip("set VERATHOS_GPU_TESTS=1 on a CUDA GPU with the pinned clef-flash weights downloaded")
    from verathos.engine import clef

    model = catalog.load().get("clef-flash")
    reference = json.loads(FIXTURE.read_text())
    assert all(list(r["answers"]) == list(model.workload.questions) for r in reference)
    engine = clef.load(model.hf_repo, model.hf_revision, device="cuda", dtype="bfloat16")
    items = [engine.prepare({"model": model.model_id, "state": r["state"], "questions": model.workload.questions})
             for r in reference]
    together = [distributions(r["answers"]) for r in engine.run(items)]
    alone = [distributions(engine.run([item])[0]["answers"]) for item in items]
    tolerance = model.tolerance
    for answers in (together, alone):
        comparisons = [compare_request(distributions(r["answers"]), got, tolerance)
                       for r, got in zip(reference, answers)]
        assert not [c for c in comparisons if c.hard_fail(tolerance)]
        assert sum(c.sum_abs for c in comparisons) / sum(c.count for c in comparisons) <= tolerance.mean_abs
    assert max(compare_request(a, t, tolerance).max_abs for a, t in zip(alone, together)) < 0.02
