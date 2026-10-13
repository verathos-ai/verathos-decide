"""The code-readout engine: request checks, passes within the token budget, every question's checks, prompt and tokens
as each release's own code makes them (CPU, where the pinned d3-mini and pplx-decider release code and tokenizers are in
the Hugging Face cache), and on a CUDA GPU with the pinned d3-mini weights the fast path's answers against stored A100
float32 reference answers, together and one at a time."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from verathos.engine.readout import (
    D3Family,
    D3Reference,
    FastEngine,
    PplxFamily,
    PplxReference,
    Prepared,
    decision_config,
    prepare,
    validate,
)
from verathos.protocol.api import PROMPT_TEMPLATE_TOKENS, distributions, most_input_tokens
from verathos.protocol.canonical import canonical_json

REPO, REVISION = "vllm-sr/d3-mini", "0e78bbdbe3cf480ed1461c185c9e0c4fa9258011"
PPLX, PPLX_REVISION = "perplexity-ai/pplx-decider-v1.1-27b", "14c9d90137962d300f3de95633f7b6535e80c5ce"
FIXTURE = Path(__file__).parent / "fixtures" / "drift" / "drift-d3-mini-A100fp32.json"  # kev-4b mix, A100 float32


def release_path(repo: str, revision: str, code: str) -> Path | None:
    hub = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
               / "hub")
    path = hub / f"models--{repo.replace('/', '--')}" / "snapshots" / revision
    return path if (path / code).exists() and (path / "tokenizer.json").exists() else None


def item(*lengths: int) -> Prepared:
    keys = [f"q{i}" for i in range(len(lengths))]
    return Prepared("d3-mini", "s", keys, {}, {k: 2 for k in keys}, {k: [0] * n for k, n in zip(keys, lengths)})


def batcher() -> FastEngine:
    engine = object.__new__(FastEngine)
    engine.length_step, engine.max_batch_tokens, engine.max_rows = 32, 8192, 64
    return engine


def test_passes_keep_every_group_within_the_token_budget_and_the_rows_of_a_pass() -> None:
    engine = batcher()
    items = [item(700, 690, 710, 705, 702, 699), item(300, 310), item(5000, 5000), item(40), item(*[100] * 70),
             item(800, 800, 800, 800, 800, 800), item(120)]
    groups = engine.passes(items)
    assert sorted(i for g in groups for i in g) == list(range(len(items)))
    for group in groups:
        tokens = sum(engine._cost(items[i]) for i in group)
        rows = sum(len(items[i].keys) for i in group)
        assert len(group) == 1 or (tokens <= engine.max_batch_tokens and rows <= engine.max_rows)
    assert [2] in groups and [4] in groups  # each over the budget or the rows on its own


@pytest.mark.parametrize("body", [
    {"model": "m", "state": "s", "questions": {"q": {"type": "noul"}}, "images": ["data:image/png;base64,AA=="]},
    {"model": "m", "state": "s", "questions": {"q": {"type": "noul"}}, "videos": []},
    {"model": "m", "state": "s", "questions": {"q": {"type": "noul"}}, "media_kwargs": {}},
    {"state": "s", "questions": {"q": {"type": "noul"}}},
    {"model": "m", "state": float("nan"), "questions": {"q": {"type": "noul"}}},
    {"model": "m", "state": "s", "questions": {}},
])
def test_requests_the_engine_does_not_serve_are_refused_before_a_pass(body) -> None:
    with pytest.raises(ValueError):
        validate(body)


def test_a_checkpoint_of_another_format_is_refused(tmp_path) -> None:
    config = {"format_version": 1, "attention_mode": "causal", "pooling": "last", "codes": ["A"] * 255,
              "token_ids": [1] * 255}
    (tmp_path / "decision_config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="attention_mode"):
        decision_config(tmp_path)


@pytest.fixture(scope="module")
def d3():
    pytest.importorskip("transformers")
    path = release_path(REPO, REVISION, "d3_runtime.py")
    if path is None:
        pytest.skip("needs the pinned d3-mini release code and tokenizer in the Hugging Face cache")
    return D3Family(path)


@pytest.fixture(scope="module")
def pplx():
    pytest.importorskip("transformers")
    path = release_path(PPLX, PPLX_REVISION, "source/src/autojev/model.py")
    if path is None:
        pytest.skip("needs the pinned pplx-decider release code and tokenizer in the Hugging Face cache")
    return PplxFamily(path)


BODIES = [
    {"model": "d3-mini", "state": "The order arrived damaged. The customer has a receipt and wants a replacement.",
     "questions": {
         "route": {"type": "choice", "instructions": "Which team should handle this?",
                   "criteria": {"returns": "Refunds and damaged deliveries", "billing": None, "technical": "Faults"}},
         "receipt": {"type": "noul", "instructions": "Does the customer have a receipt?"},
         "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["Routine", "Soon", "Today"]}}},
    {"model": "d3-mini", "state": {"ticket": 7, "lines": ["refund please", "card charged twice"], "vip": True},
     "questions": {
         "refund": {"type": "noul", "criteria": {"true": "A refund is requested", "false": "No refund"}},
         "levels": {"type": "score", "criteria": [{"level": 0}, "one", 2]},
         "only": {"type": "choice", "criteria": {"yes": None}}}},
    {"model": "d3-mini", "state": "", "questions": {"q": {"type": "noul"}}},
]


def test_every_d3_question_gets_the_prompt_and_tokens_the_release_gives_it(d3) -> None:
    runtime = d3.runtime
    own = SimpleNamespace(tokenizer=d3.tokenizer, prompt="d3", codes=d3.codes, max_length=None,
                          permutation_average=False)
    own.text = lambda state, question: runtime.D3.text(own, state, question)
    for body in BODIES:
        ours = prepare(body, d3)
        theirs = runtime.D3.prepare(own, body["state"], body["questions"])
        assert not theirs.errors
        assert ours.keys == theirs.keys == list(body["questions"])
        assert ours.sequences == theirs.sequences
        assert {k: q.keys for k, q in ours.questions.items()} == {k: q.keys for k, q in theirs.questions.items()}


@pytest.mark.parametrize("question", [
    {"type": "score", "criteria": [str(i) for i in range(11)]},
    {"type": "noul", "criteria": {"yes": "y"}},
    {"type": "choice", "criteria": {}},
])
def test_a_question_the_release_refuses_is_refused_with_its_reason(d3, pplx, question) -> None:
    for prompts in (d3, pplx):
        with pytest.raises(ValueError, match="^q: "):
            prepare({"model": "m", "state": "s", "questions": {"q": question}}, prompts)


def test_a_prompt_over_the_length_limit_is_refused_and_never_cut(d3) -> None:
    body = {"model": "d3-mini", "state": "word " * 400, "questions": {"q": {"type": "noul"}}}
    with pytest.raises(ValueError, match="more than 100"):
        prepare(body, d3, max_length=100)


def test_every_pplx_question_gets_the_prompt_and_tokens_its_server_gives_it(pplx) -> None:
    """Through the release's own server path: its request model, the questions as it hands them to the model
    (``model_dump(exclude_none=True)``) and ``DecisionModel.prepare``, whose batches are left-padded."""
    own = SimpleNamespace(processor=pplx.processor, codes=pplx.codes, device_name="cpu")
    accepted = 0
    for body in BODIES:
        try:
            request = pplx.server.EvaluationRequest.model_validate({**body, "model": "jev-latest"})
        except ValueError:
            with pytest.raises(ValueError):  # what its server refuses, the engine refuses
                prepare(body, pplx)
            continue
        accepted += 1
        ours = prepare(body, pplx)
        questions = {k: q.model_dump(exclude_none=True) for k, q in request.questions.items()}
        batch = pplx.model.DecisionModel.prepare(own, [{"state": request.state, "question": q}
                                                       for q in questions.values()])
        rows = [ids[mask.bool()].tolist() for ids, mask in zip(batch.inputs["input_ids"],
                                                              batch.inputs["attention_mask"], strict=True)]
        assert ours.keys == list(questions)
        assert [ours.sequences[k] for k in ours.keys] == rows
        assert [ours.options[k] for k in ours.keys] == list(batch.counts)
    assert accepted >= 2


def test_both_paths_take_and_refuse_the_same_requests_with_the_same_reasons(d3, pplx) -> None:
    """What miners run (``prepare``) and what validators' references run (the release's own runtime) take the same
    requests and refuse the same ones with the same reasons: among them an empty question id (d3's runtime refuses it,
    pplx's server takes it) and a prompt over the family's length limit, which d3's runtime alone would answer."""
    runtime = d3.runtime
    own = SimpleNamespace(tokenizer=d3.tokenizer, prompt="d3", codes=d3.codes, max_length=None,
                          permutation_average=False)
    own.text = lambda state, question: runtime.D3.text(own, state, question)
    own.prepare = lambda state, questions: runtime.D3.prepare(own, state, questions)
    d3_reference, pplx_reference = object.__new__(D3Reference), object.__new__(PplxReference)
    d3_reference.prompts, d3_reference.d3, pplx_reference.prompts = d3, own, pplx
    bodies = [*BODIES,
              {"model": "m", "state": "s", "questions": {"": {"type": "noul", "instructions": "Is it?"}}},
              {"model": "m", "state": "word " * 17000, "questions": {"q": {"type": "noul"}}},
              {"model": "m", "state": "s",
               "questions": {"q": {"type": "noul"}, "r": {"type": "choice", "criteria": {}}}}]

    def verdict(run, *args) -> str:
        try:
            run(*args)
            return "answered"
        except ValueError as exc:
            return str(exc)

    for prompts, reference in ((d3, d3_reference), (pplx, pplx_reference)):
        fast = [verdict(prepare, body, prompts) for body in bodies]
        assert fast == [verdict(reference.prepare, body) for body in bodies]
        assert "more than" in fast[-2] and fast[-1].startswith("r: ")
    assert verdict(prepare, bodies[-3], d3) == "question IDs must be nonempty strings"
    assert verdict(prepare, bodies[-3], pplx) == "answered"


def test_every_prompt_fits_the_most_tokens_a_request_is_billed(d3, pplx) -> None:
    """A model that reads one prompt per question is billed at most ``protocol.api.most_input_tokens`` of a request:
    every question's prompt as each release writes it fits, also where one token per byte of the request would not (a
    tiny request, a choice of 255 options, a state of dense numbers)."""
    bodies = [*BODIES,
              {"model": "m", "state": "", "questions": {"q": {"type": "choice",
                                                               "criteria": {f"o{i}": None for i in range(255)}}}},
              {"model": "m", "state": list(range(1000)), "questions": {"q": {"type": "noul"}}},
              {"model": "m", "state": [0] * 2000, "questions": {"q": {"type": "noul"}, "r": {"type": "noul"}}}]
    for prompts in (d3, pplx):
        beyond_bytes = 0
        for body in bodies:
            try:
                prepared = prepare(body, prompts)
            except ValueError:
                assert prompts is pplx and body in BODIES  # the score levels pplx's server refuses
                continue
            request_bytes = len(canonical_json(body))
            tokens = [len(s) for s in prepared.sequences.values()]
            assert max(tokens) <= 2 * request_bytes + PROMPT_TEMPLATE_TOKENS
            assert sum(tokens) <= most_input_tokens(request_bytes, len(body["questions"]), True)
            beyond_bytes += sum(tokens) > request_bytes
        assert beyond_bytes >= 2


# ---- on a CUDA GPU with the pinned weights ----

def test_fast_engine_answers_stay_within_the_drift_bounds_of_the_float32_reference() -> None:
    """30 requests of the kev-4b mix (six questions each) against the release's own runtime in float32 on an A100
    without flash-linear-attention (as validators run it), answered together and one at a time: every probability
    within 0.05, the mean within 0.005, and no top answer changed where the reference's led by 0.2 or more."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or os.environ.get("VERATHOS_GPU_TESTS") != "1":
        pytest.skip("set VERATHOS_GPU_TESTS=1 on a CUDA GPU with the pinned d3-mini weights downloaded")
    from verathos.engine import readout

    cases = json.loads(FIXTURE.read_text())
    engine = readout.load(REPO, REVISION, device="cuda", dtype="bfloat16")
    items = [engine.prepare({"model": "d3-mini", "state": c["state"], "questions": c["questions"]}) for c in cases]
    together = engine.run(items)
    alone = [engine.run([one])[0] for one in items]
    for answers in (together, alone):
        diffs = []
        for case, got in zip(cases, answers, strict=True):
            ref, ours = distributions(case["answers"]), distributions(got["answers"])
            for qid, r in ref.items():
                diffs += [abs(r[k] - ours[qid][k]) for k in r]
                ranked = sorted(r.values(), reverse=True)
                if ranked[0] - ranked[1] >= 0.2:
                    assert max(ours[qid], key=ours[qid].get) == max(r, key=r.get)
        assert max(diffs) <= 0.05 and sum(diffs) / len(diffs) <= 0.005


def test_both_families_answers_are_well_formed_systemone_answers(d3, pplx) -> None:
    """What miners sign: one distribution per question over its own options whose visible fields agree with it
    (``protocol.api.well_formed``), built by each family's own answer code from any option probabilities."""
    import random

    from verathos.protocol.api import SystemOneRequest, well_formed

    rng = random.Random(0)
    request = SystemOneRequest.model_validate(BODIES[0])
    for prompts in (d3, pplx):
        item = prepare(BODIES[0], prompts)
        for _ in range(20):
            answers = {}
            for qid in item.keys:
                raw = [rng.random() + 1e-9 for _ in range(item.options[qid])]
                answers[qid] = prompts.answer(item.questions[qid], [v / sum(raw) for v in raw])
            assert well_formed(request, answers)
