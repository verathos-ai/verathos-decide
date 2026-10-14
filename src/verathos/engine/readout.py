"""Code-readout decision models: a Qwen3.5 backbone whose full-attention layers see the whole prompt (``attention_mode``
``noncausal_full_attention``), one prompt per question, and a readout over 255 answer codes at the prompt's last token
(``pooling`` ``last``), as ``decision_config.json`` and ``readout.safetensors`` describe them. pplx-decider, d3 and
others share this format; they differ in their prompts and answers, which each release ships as code.

A prompt family is that code, loaded from the pinned revision, so every question's checks, its prompt, its tokens and
the answer built from its option probabilities are exactly the release's; only the forward pass is ours. Both paths
below check a request with the same code (``prepare``): they refuse the same requests with the same reasons, a prompt
over the family's ``max_length`` among them.
- ``d3`` (vLLM Semantic Router): ``d3_format.py`` and ``d3_runtime.py``.
- ``pplx`` (Perplexity): ``source/src/autojev``: ``model.py`` (prompt, options, answer) and ``server.py`` (its
  server's question checks).

Two paths:
- ``reference``: the release's own runtime, one request at a time. Validators run it in float32 (the checkpoints store
  bfloat16 weights, so the cast is exact) without flash-linear-attention, so transformers' PyTorch reference computes
  the linear layers; on a GPU, float32 is computed without TF32.
- ``fast`` (CUDA, bfloat16): every question of every request of a pass is one row; rows sorted by length run
  right-padded under a token budget through ``verathos.engine.qwen35.Backbone`` in its noncausal mode (CUDA graphs for
  small passes), and the readout runs on each row's last token in float32, as validators' references compute it (a
  bfloat16 readout's weights cast exactly): in bfloat16 its logits round in steps of up to 0.125, which moves
  probabilities by a few hundredths between two implementations of the same pass.

A question's prompt carries the whole state, so a request costs as many prompts as it has questions: nothing can be
shared between them, because the full-attention layers make every position depend on the question that follows.
Requests with images or videos are refused (text only).
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verathos.protocol.api import REFUSED_FIELDS

log = logging.getLogger(__name__)

CODES = 255


def decision_config(path: Path) -> dict[str, Any]:
    """The release's ``decision_config.json``, checked to describe the format this engine runs."""
    config = json.loads((path / "decision_config.json").read_text())
    expected = {"format_version": 1, "attention_mode": "noncausal_full_attention", "pooling": "last"}
    wrong = {k: config.get(k) for k, v in expected.items() if config.get(k) != v}
    if wrong or len(config.get("codes", ())) != CODES or len(config.get("token_ids", ())) != CODES:
        raise ValueError(f"not a code-readout checkpoint this engine runs: {wrong or 'codes'}")
    return config


def validate(body: dict[str, Any]) -> None:
    """The fields no request may carry, and what every release needs, raised before a request joins a batch."""
    for field, why in REFUSED_FIELDS.items():
        if field in body:
            raise ValueError(why)
    if body.get("images"):
        raise ValueError("images are not supported by this engine")
    if not isinstance(body.get("model"), str) or "state" not in body:
        raise ValueError("model and state are required")
    if not isinstance(body.get("questions"), dict) or not body["questions"]:
        raise ValueError("at least one question is required")
    json.dumps(body["state"], allow_nan=False)  # ValueError for what strict JSON cannot carry


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # d3_runtime imports d3_format by this name
    spec.loader.exec_module(module)
    return module


class D3Family:
    """d3's prompts and answers: the release's ``d3_format.py`` and ``d3_runtime.py``."""

    name = "d3"
    max_length = 16384  # no limit in the release; ours, so one question's prompt cannot take the GPU's memory

    def __init__(self, path: Path) -> None:
        from transformers import AutoTokenizer

        self.format, self.runtime = _module(path, "d3_format"), _module(path, "d3_runtime")
        self.tokenizer = AutoTokenizer.from_pretrained(str(path))
        self.codes, _ = self.format.answer_codes(self.tokenizer)

    def check(self, questions: dict[str, Any]) -> None:
        """The release's check of a request's questions as a whole (``D3.prepare``): ids are nonempty."""
        if any(not qid for qid in questions):
            raise ValueError("question IDs must be nonempty strings")

    def question(self, question: dict[str, Any]) -> Any:
        return self.runtime.normalize_question(question)

    def options(self, question: Any) -> list[str]:
        return question.keys

    def text(self, state: Any, question: Any) -> str:
        return self.format.render(self.tokenizer, state, question.rendered, self.codes)

    def tokenize(self, texts: list[str]) -> list[list[int]]:
        return self.tokenizer(texts, add_special_tokens=False)["input_ids"]

    def answer(self, question: Any, probabilities: list[float]) -> dict[str, Any]:
        return self.runtime.product_answer(question, probabilities)


class PplxFamily:
    """pplx-decider's prompts and answers: the release's ``autojev`` package (``source/src``)."""

    name = "pplx"
    max_length = 8192  # the release's limit on one question's prompt

    def __init__(self, path: Path) -> None:
        from transformers import AutoProcessor

        sys.path.insert(0, str(path / "source" / "src"))
        self.model = importlib.import_module("autojev.model")
        self.server = importlib.import_module("autojev.server")
        self.processor = AutoProcessor.from_pretrained(str(path))
        self.tokenizer = self.processor.tokenizer
        self.codes = decision_config(path)["codes"]

    def check(self, questions: dict[str, Any]) -> None:
        """Its server checks the questions one by one only (any id, an empty one too)."""

    def question(self, question: dict[str, Any]) -> dict[str, Any]:
        """The question as the release's server passes it to the model (its own checks, then ``exclude_none``)."""
        kinds = {"choice": self.server.Choice, "score": self.server.Score, "noul": self.server.Noul}
        if question.get("type") not in kinds:
            raise ValueError("type must be choice, score or noul")
        try:
            checked = kinds[question["type"]].model_validate(question)
        except ValueError as exc:  # pydantic's ValidationError: each failing field and why
            raise ValueError("; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())) from None
        return checked.model_dump(exclude_none=True)

    def options(self, question: dict[str, Any]) -> list[str]:
        return self.model.options(question)[0]

    def text(self, state: Any, question: dict[str, Any]) -> str:
        messages = self.model.decision_messages({"state": state, "question": question}, self.codes)
        return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=False)

    def tokenize(self, texts: list[str]) -> list[list[int]]:
        return self.processor(text=texts, images=None)["input_ids"]

    def answer(self, question: dict[str, Any], probabilities: list[float]) -> dict[str, Any]:
        return self.model.answer(question, probabilities)


def family(path: Path) -> D3Family | PplxFamily:
    """The prompt family a release ships."""
    if (path / "d3_runtime.py").exists() and (path / "d3_format.py").exists():
        return D3Family(path)
    if (path / "source" / "src" / "autojev" / "model.py").exists():
        return PplxFamily(path)
    raise ValueError("the release ships no prompt code this engine knows (d3_runtime.py, source/src/autojev)")


@dataclass(frozen=True)
class Prepared:
    model: str
    state: Any
    keys: list[str]  # question ids in request order
    questions: dict[str, Any]  # the release's checked question per id
    options: dict[str, int]  # each question's number of options
    sequences: dict[str, list[int]]  # each question's prompt tokens


def prepare(body: dict[str, Any], prompts: D3Family | PplxFamily, max_length: int | None = None) -> Prepared:
    """A request's questions checked and rendered by the release, each prompt tokenized as the release tokenizes it.
    Raises ValueError for a request the release refuses."""
    validate(body)
    prompts.check(body["questions"])
    questions, texts = {}, []
    for qid, question in body["questions"].items():
        try:
            questions[qid] = prompts.question(question)
        except ValueError as exc:
            raise ValueError(f"{qid}: {exc}") from None
        texts.append(prompts.text(body["state"], questions[qid]))
    sequences = dict(zip(questions, prompts.tokenize(texts), strict=True))
    limit = max_length or prompts.max_length
    for qid, sequence in sequences.items():
        if len(sequence) > limit:
            raise ValueError(f"{qid}: the question's prompt has {len(sequence)} tokens, more than {limit}")
    return Prepared(body["model"], body["state"], list(questions), questions,
                    {qid: len(prompts.options(q)) for qid, q in questions.items()}, sequences)


class FastEngine:
    """Batched CUDA serving of a code-readout release: the backbone in bfloat16, every question of a pass's requests
    one row."""

    def __init__(
        self,
        path: Path,
        device: str,
        *,
        max_batch_tokens: int = 8192,
        max_rows: int = 64,
        length_step: int = 32,
        graph_tokens: int = 2048,
    ) -> None:
        import torch
        from safetensors.torch import load_file
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

        from verathos.engine.graphs import GraphedForward
        from verathos.engine.qwen35 import Backbone

        self.torch = torch
        config = decision_config(path)
        self.prompts = family(path)
        for code, token in zip(config["codes"], config["token_ids"], strict=True):
            if self.prompts.tokenizer.encode(code, add_special_tokens=False) != [token]:
                raise ValueError(f"answer code {code!r} is not the single token {token} decision_config.json names")
        self.temperature = float(config.get("temperature", 1.0))
        self.pad_id = self.prompts.tokenizer.pad_token_id
        model = Qwen3_5Model.from_pretrained(str(path), dtype=torch.bfloat16, attn_implementation="sdpa",
                                             device_map={"": device})
        model.eval()
        self.backbone = Backbone(model.language_model, noncausal=True)
        del model.visual  # text only: the vision tower's memory is freed
        self.readout = load_file(str(path / "readout.safetensors"))["weight"].to(device, torch.float32)
        torch.cuda.empty_cache()
        self.device = torch.device(device)
        self.max_batch_tokens = max_batch_tokens
        self.max_rows = max_rows
        self.length_step = length_step
        self.forward = GraphedForward(
            lambda ids, lengths: self.backbone.forward(ids, lengths=lengths),
            self.backbone.hidden_size,
            torch.bfloat16,
            self.device,
            length_step=length_step,
            max_tokens=graph_tokens,
            with_lengths=True,
        )
        self.forward.capture_all()

    def prepare(self, body: dict[str, Any]) -> Prepared:
        return prepare(body, self.prompts)

    def _cost(self, item: Prepared) -> int:
        from verathos.engine.graphs import bucket_length

        return sum(bucket_length(len(s), self.length_step) for s in item.sequences.values())

    def passes(self, items: list[Prepared]) -> list[list[int]]:
        """Indexes grouped by their prompts' padded tokens within the token budget and their questions within the rows
        of a pass, smallest first (a request over the budget on its own forms its own group)."""
        order = sorted(range(len(items)), key=lambda i: self._cost(items[i]))
        groups: list[list[int]] = []
        tokens = rows = 0
        for i in order:
            cost, n = self._cost(items[i]), len(items[i].keys)
            if groups and tokens + cost <= self.max_batch_tokens and rows + n <= self.max_rows:
                groups[-1].append(i)
                tokens, rows = tokens + cost, rows + n
            else:
                groups.append([i])
                tokens, rows = cost, n
        return groups

    def _pass(self, rows: list[tuple[list[int], int]]) -> list[list[float]]:
        """Option probabilities of one pass's prompts (token ids, options), each cut to its options."""
        from verathos.engine.graphs import bucket_length

        torch = self.torch
        length = bucket_length(max(len(ids) for ids, _ in rows), self.length_step)
        batch = self.forward.graphed_batch(len(rows), length) or len(rows)
        ids = torch.full((batch, length), self.pad_id, dtype=torch.long)
        lengths = torch.ones(batch, dtype=torch.long)  # rows that only pad a graph's batch read their first token
        for row, (sequence, _) in enumerate(rows):
            ids[row, : len(sequence)] = torch.tensor(sequence)
            lengths[row] = len(sequence)
        lengths = lengths.to(self.device, non_blocking=True)
        hidden = self.forward(ids.to(self.device, non_blocking=True), lengths)
        last = hidden[torch.arange(len(rows), device=self.device), lengths[: len(rows)] - 1]
        logits = last.float() @ self.readout.T
        counts = torch.tensor([n for _, n in rows], device=self.device)[:, None]
        invalid = torch.arange(logits.shape[-1], device=self.device)[None] >= counts
        probabilities = torch.softmax(logits.masked_fill(invalid, float("-inf")) / self.temperature, dim=-1)
        return [p[:n] for p, (_, n) in zip(probabilities.cpu().tolist(), rows, strict=True)]

    def run(self, items: list[Prepared]) -> list[dict[str, Any]]:
        from verathos.engine.graphs import bucket_length

        rows = [(i, qid) for i, item in enumerate(items) for qid in item.keys]
        order = sorted(range(len(rows)), key=lambda r: len(items[rows[r][0]].sequences[rows[r][1]]))
        chunks: list[list[int]] = []
        for r in order:
            i, qid = rows[r]
            length = bucket_length(len(items[i].sequences[qid]), self.length_step)
            if chunks and len(chunks[-1]) < self.max_rows and (len(chunks[-1]) + 1) * length <= self.max_batch_tokens:
                chunks[-1].append(r)
            else:
                chunks.append([r])
        probabilities: dict[tuple[int, str], list[float]] = {}
        with self.torch.inference_mode():
            for chunk in chunks:
                prompts = [(items[i].sequences[qid], items[i].options[qid]) for i, qid in (rows[r] for r in chunk)]
                for r, p in zip(chunk, self._pass(prompts), strict=True):
                    probabilities[rows[r]] = p
        return [{
            "model": item.model,
            "answers": {qid: self.prompts.answer(item.questions[qid], probabilities[(i, qid)]) for qid in item.keys},
            "usage": {"input_tokens": sum(len(s) for s in item.sequences.values()), "output_tokens": 0},
        } for i, item in enumerate(items)]


class D3Reference:
    """d3's own runtime (``d3_runtime.D3``), one request per pass, in the dtype asked for."""

    def __init__(self, path: Path, device: str, dtype: str) -> None:
        import torch

        self.prompts = D3Family(path)
        self.d3 = self.prompts.runtime.D3.from_pretrained(str(path), device=device, verify="none")
        if dtype == "float32":  # the release loads bfloat16; its weights are bfloat16, so the cast is exact
            self.d3.backbone.to(torch.float32)
            self.d3.fast = None

    def prepare(self, body: dict[str, Any]) -> tuple[str, Any]:
        prepare(body, self.prompts)  # the fast path's checks: both refuse the same requests with the same reasons
        prepared = self.d3.prepare(body["state"], body["questions"])
        if prepared.errors:  # the release refusing what those checks passed
            qid, error = next(iter(prepared.errors.items()))
            raise ValueError(f"{qid}: {error['message']}")
        return body["model"], prepared

    def passes(self, items: list[Any]) -> list[list[int]]:
        return [[i] for i in range(len(items))]

    def run(self, items: list[tuple[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for model, prepared in items:
            probabilities, tokens = self.d3.run(prepared)
            out.append({**self.d3.respond(prepared, probabilities, tokens), "model": model})
        return out


class PplxReference:
    """pplx-decider's own ``DecisionModel``, one request per pass in batches of eight questions as its server runs
    them, in the dtype asked for (the release computes float32 on CPU and bfloat16 on a GPU)."""

    def __init__(self, path: Path, device: str, dtype: str) -> None:
        import torch

        self.prompts = PplxFamily(path)
        self.torch = torch
        self.model = self.prompts.model.DecisionModel(path, device="cpu" if dtype == "float32" else device)
        if dtype == "float32" and device != "cpu":
            self.model.to(device)
            self.model.device_name = device

    def prepare(self, body: dict[str, Any]) -> Prepared:
        return prepare(body, self.prompts)

    def passes(self, items: list[Prepared]) -> list[list[int]]:
        return [[i] for i in range(len(items))]

    def run(self, items: list[Prepared]) -> list[dict[str, Any]]:
        out = []
        with self.torch.inference_mode():
            for item in items:
                rows = [{"state": item.state, "question": item.questions[qid]} for qid in item.keys]
                answers, tokens = {}, 0
                for start in range(0, len(rows), 8):
                    keys = item.keys[start : start + 8]
                    batch = self.model.prepare(rows[start : start + 8])
                    values = (self.model(batch) / self.model.temperature).softmax(-1).cpu().tolist()
                    for qid, v, n in zip(keys, values, batch.counts, strict=True):
                        answers[qid] = self.prompts.answer(item.questions[qid], v[:n])
                    tokens += batch.input_tokens
                out.append({"model": item.model, "answers": answers,
                            "usage": {"input_tokens": tokens, "output_tokens": 0}})
        return out


def device_index(device: str) -> str:
    """``device`` with the CUDA index this process uses when it names none: the releases' own runtimes call
    ``torch.cuda.set_device``, which refuses a bare ``cuda``."""
    if device != "cuda":
        return device
    import torch

    return f"cuda:{torch.cuda.current_device()}"


def reference(path: Path, device: str, dtype: str):
    """The release's own runtime for its prompt family."""
    device = device_index(device)
    if device.startswith("cuda") and dtype == "float32":
        import torch

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    decision_config(path)
    return D3Reference(path, device, dtype) if family(path).name == "d3" else PplxReference(path, device, dtype)


def load(repo: str, revision: str, *, device: str, dtype: str, fast: bool | None = None):
    """The engine for a pinned code-readout release: the fast path on CUDA in bfloat16 unless ``fast`` is False."""
    from huggingface_hub import snapshot_download

    if fast is None:
        fast = device.startswith("cuda") and dtype == "bfloat16"
    if fast and importlib.util.find_spec("fla") is None:
        raise RuntimeError("the bfloat16 GPU path needs flash-linear-attention at the version that "
                           "verathos/engine/qwen35.py pins (FLA_VERSION); --dtype float32 runs the release's own code")
    if not fast and dtype == "float32" and importlib.util.find_spec("fla") is not None:
        log.warning("flash-linear-attention is installed here, so transformers runs the release's linear attention on "
                    "its kernels: these float32 answers differ from validators' references, which run without it")
    path = Path(snapshot_download(repo, revision=revision))
    engine = FastEngine(path, device) if fast else reference(path, device, dtype)
    log.info("loaded %s@%s on %s (%s, %s)", repo, revision[:12], device, dtype, "fast" if fast else "reference")
    return engine
