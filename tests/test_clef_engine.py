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

from tests.images import canonical
from verathos import catalog
from verathos.engine.clef import FastEngine, Prepared, ReferenceEngine, record
from verathos.protocol.api import distributions
from verathos.validator.compare import compare_request

BOUNDS = {"token_px": 32, "min_pixels": 65_536, "max_pixels": 16_777_216}  # the pinned release's processor's

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
    assert record({**body, "id": "r1", "extra": 1}, None) == body  # the release reads nothing else of a request
    for refused, why in (({"videos": ["v.mp4"]}, "videos are not supported"),
                         ({"media_kwargs": {"max_pixels": 1 << 30}}, "media_kwargs is not accepted"),
                         ({"images": [canonical(256, 256)]}, "images are not served by this engine")):
        with pytest.raises(ValueError, match=why):
            record({**body, **refused}, None)
    with pytest.raises(ValueError, match="images are not served"):
        object.__new__(FastEngine).prepare({**body, "images": [canonical(256, 256)]})
    with pytest.raises(ValueError, match=r"images\.0: not \{"):
        record({**body, "images": ["https://example.com/a.png"]}, BOUNDS)
    with pytest.raises(ValueError, match="criteria"):
        record({**body, "questions": {"q": {"type": "choice", "instructions": "?"}}}, None)


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
    for encoded, record_logits in zip(records, got):
        expected = release_logits(release, head, backbone, output_embedding, encoded)
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


# ---- images on the reference path: a tiny random model with the release's code, tokenizer and processor ----

REQUEST = {"model": "clef-flash", "state": "What does the image show?",
           "questions": {"corner": {"type": "noul", "instructions": "Is there a green corner?"},
                         "kind": {"type": "choice", "criteria": {"photo": "A photo", "drawing": "A drawing"}}}}


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """The reference engine on a random Qwen3.5 model as the release's config describes it, shrunk (one vision block,
    two text layers, the real vocabulary), with the pinned release's code, tokenizer and processor config: the
    release's encoding and model on images, on a CPU in seconds."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("PIL")
    path = release_code()
    if path is None:
        pytest.skip("needs the pinned clef-flash release (code, tokenizer and configs) in the Hugging Face cache")
    import shutil

    from safetensors.torch import save_file

    snapshot, out = path.parent, tmp_path_factory.mktemp("tiny-clef")
    for name in ("joint_schema_model.py", "tokenizer.json", "tokenizer_config.json", "processor_config.json",
                 "chat_template.jinja"):
        shutil.copy(snapshot / name, out / name)
    real = json.loads((snapshot / "config.json").read_text())
    rope = {**real["text_config"]["rope_parameters"], "partial_rotary_factor": 1.0, "mrope_section": [3, 3, 2]}
    text = {**real["text_config"], "hidden_size": HIDDEN, "intermediate_size": 32, "num_hidden_layers": 2,
            "layer_types": ["linear_attention", "full_attention"], "num_attention_heads": 2,
            "num_key_value_heads": 1, "head_dim": 16, "partial_rotary_factor": 1.0, "rope_parameters": rope,
            "linear_key_head_dim": 8, "linear_value_head_dim": 8, "linear_num_key_heads": 2,
            "linear_num_value_heads": 4}
    vision = {**real["vision_config"], "depth": 1, "hidden_size": 32, "intermediate_size": 64, "num_heads": 2,
              "out_hidden_size": HIDDEN}
    settings = {k: v for k, v in real.items() if k not in ("architectures", "transformers_version")}
    torch.manual_seed(0)
    model = transformers.Qwen3_5ForConditionalGeneration(
        transformers.Qwen3_5Config(**{**settings, "text_config": text, "vision_config": vision}))
    model.save_pretrained(out)
    from verathos.engine.clef import _release_module

    head_config = {"hidden_size": HIDDEN, "width": 8, "routing_layers": 1, "layers": 1, "heads": 2,
                   "feedforward": 16}
    (out / "joint_head_config.json").write_text(json.dumps(head_config))
    save_file(_release_module(out).JointSchemaHead(**head_config).state_dict(), out / "joint_head.safetensors")
    return ReferenceEngine(out, "cpu", "float32")


def answer(engine: ReferenceEngine, body: dict) -> dict:
    return engine.run([engine.prepare(body)])[0]


def logits(engine: ReferenceEngine, body: dict):
    """The release's option logits for a request, as ``systemone`` computes them."""
    import torch

    module, release = engine.module, engine.prepare(body).body
    encoded = module.encode_record(engine.processor.tokenizer, release, processor=engine.processor)
    batch = module.collate_records([encoded], engine.processor.tokenizer.pad_token_id, torch.device("cpu"))
    with torch.inference_mode():
        return torch.cat(engine.model(batch)[0])


def test_the_reference_reads_canonical_images_and_counts_their_tokens_as_the_release_does(tiny) -> None:
    import io

    import torch
    from PIL import Image

    from tests.images import item
    from verathos.protocol.api import SystemOneRequest, well_formed
    from verathos.protocol.media import ImageLimits, canonical_image, request_image_tokens

    assert tiny.image_bounds == BOUNDS
    alone = answer(tiny, REQUEST)
    images = [canonical(256, 320), canonical(512, 512, "JPEG")]
    seen = answer(tiny, {**REQUEST, "images": images})
    expected = request_image_tokens([(256, 320), (512, 512)], 32)
    assert expected == 82 + 258 + 1
    assert seen["usage"]["input_tokens"] == alone["usage"]["input_tokens"] + expected  # the gateway's exact count
    assert well_formed(SystemOneRequest.model_validate(REQUEST), seen["answers"])
    out = io.BytesIO()
    Image.new("RGB", (256, 320), (250, 240, 10)).save(out, "PNG")
    yellow = canonical_image(out.getvalue(), "image/png", ImageLimits())
    other = {**REQUEST, "images": [item(yellow.data, yellow.content_type), images[1]]}
    assert not torch.allclose(logits(tiny, {**REQUEST, "images": images}), logits(tiny, other))  # pixels are read


def test_the_engine_server_answers_canonical_images_and_refuses_others(tiny) -> None:
    from fastapi.testclient import TestClient

    from tests.images import data_url, encoded
    from verathos.engine.server import create_app

    client = TestClient(create_app(tiny))
    good = client.post("/v1/systemone", json={**REQUEST, "images": [canonical(256, 256)]})
    assert good.status_code == 200 and good.json()["usage"]["input_tokens"] > 66
    for images in (["http://169.254.169.254/latest/meta-data/"], ["/etc/passwd"],
                   [data_url(encoded(256, 256), "image/png")], ["QUJD"]):
        r = client.post("/v1/systemone", json={**REQUEST, "images": images})
        assert r.status_code == 422 and "images.0" in r.json()["error"]


def test_no_string_and_no_processor_argument_of_a_request_reaches_the_processor(tiny, monkeypatch) -> None:
    """Whatever a request carries, the processor receives decoded pixels and the engine's own arguments, and
    transformers' loaders, which fetch URLs and open paths, are never called."""
    import transformers.image_processing_backends as backends
    import transformers.image_processing_base as base
    import transformers.image_utils as image_utils
    from PIL import Image

    from tests.images import b64, data_url, encoded

    def never(*args, **kwargs):
        raise AssertionError("a loader of transformers was called")

    for module, name in ((image_utils, "load_image"), (image_utils, "load_image_as_tensor"), (base, "load_image"),
                         (backends, "load_image_as_tensor")):
        monkeypatch.setattr(module, name, never)
    calls = []
    processor = tiny.processor

    class Watched:
        tokenizer = processor.tokenizer
        image_processor = processor.image_processor

        def __call__(self, *args, **kwargs):
            calls.append(kwargs)
            return processor(*args, **kwargs)

    monkeypatch.setattr(tiny, "processor", Watched())
    hostile = [["http://169.254.169.254/latest/meta-data/"], ["https://www.youtube.com/watch?v=dQw4w9WgXcQ"],
               ["/etc/passwd"], [b64(encoded(256, 256))], [data_url(encoded(256, 256), "image/png")],
               [{"content_type": "image/png", "base64": "/etc/passwd"}],
               [{"content_type": "image/png", "base64": "http://169.254.169.254/x"}]]
    for images in hostile:
        with pytest.raises(ValueError, match=r"images\.0"):
            tiny.prepare({**REQUEST, "images": images})
    for field in ({"media_kwargs": {"do_resize": True, "max_pixels": 1 << 30}}, {"videos": ["/etc/passwd"]}):
        with pytest.raises(ValueError, match="not accepted|not supported"):
            tiny.prepare({**REQUEST, **field, "images": [canonical(256, 256)]})
    assert calls == []
    answer(tiny, {**REQUEST, "images": [canonical(256, 256), canonical(512, 256, "JPEG")]})
    (call,) = calls
    assert set(call) == {"text", "images", "videos", "return_tensors", "do_resize"} and call["do_resize"] is False
    assert all(isinstance(image, Image.Image) and image.mode == "RGB" for image in call["images"])
    assert call["videos"] is None and all(isinstance(t, str) for t in call["text"])


def test_canonical_images_need_no_resizing_and_some_could_not_take_it(tiny) -> None:
    """The processor leaves a canonical image's pixels as they are with resizing on or off; and a canonical image whose
    aspect ratio rounds past 200:1 is refused by the processor's resizing, so engines pass ``do_resize=False``."""
    import torch

    from tests.images import encoded
    from verathos.protocol.media import ImageLimits, canonical_image

    text = ["<|vision_start|><|image_pad|><|vision_end|>\n"]
    for size in ((256, 320), (1024, 768), (800, 608)):
        image = tiny.prepare({**REQUEST, "images": [canonical(*size)]}).body["images"]
        resized = tiny.processor(text=text, images=image, return_tensors="pt")
        kept = tiny.processor(text=text, images=image, return_tensors="pt", do_resize=False)
        assert torch.equal(resized["pixel_values"], kept["pixel_values"])
        assert torch.equal(resized["image_grid_thw"], kept["image_grid_thw"])
    made = canonical_image(encoded(6417, 33), "image/png", ImageLimits())  # 194:1, made 6432 x 32: 201:1
    assert (made.width, made.height) == (6432, 32)
    from tests.images import item

    image = tiny.prepare({**REQUEST, "images": [item(made.data, made.content_type)]}).body["images"]
    with pytest.raises(ValueError, match="aspect ratio"):
        tiny.processor(text=text, images=image, return_tensors="pt")
    assert answer(tiny, {**REQUEST, "images": [item(made.data, made.content_type)]})["usage"]["input_tokens"] > 201


def test_the_catalog_restates_the_clef_processors_image_tokens() -> None:
    from verathos.protocol.media import ENGINE_IMAGES, ImageLimits

    path = release_code()
    if path is None:
        pytest.skip("needs the pinned clef-flash release config in the Hugging Face cache")
    images = json.loads((path.parent / "processor_config.json").read_text())["image_processor"]
    clef = ENGINE_IMAGES["clef"]
    assert images["patch_size"] * images["merge_size"] == clef.token_px == ImageLimits().token_px
    assert images["size"] == {"shortest_edge": clef.min_pixels, "longest_edge": clef.max_pixels}
    assert ImageLimits().min_pixels == clef.min_pixels and ImageLimits().max_input_pixels == clef.max_pixels
    assert BOUNDS == {"token_px": clef.token_px, "min_pixels": clef.min_pixels, "max_pixels": clef.max_pixels}


def test_a_float32_reference_on_a_gpu_computes_without_tf32(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    from verathos.engine import clef

    processor = SimpleNamespace(image_processor=SimpleNamespace(patch_size=16, merge_size=2, size={
        "shortest_edge": 65_536, "longest_edge": 16_777_216}))
    release = SimpleNamespace(load_release_model=lambda *a, **k: (SimpleNamespace(eval=lambda: None), processor))
    monkeypatch.setattr(clef, "_release_module", lambda path: release)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    clef.ReferenceEngine(Path("."), "cuda:0", "float32")
    assert not torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32


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
