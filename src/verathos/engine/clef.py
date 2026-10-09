"""Clef models (Cloudflare): a Qwen3.5 backbone with a joint schema head, answering every question of a request in one
pass over one sequence (state, then every question with its options).

The model repository ships its inference code (``joint_schema_model.py``); it is loaded from the pinned revision and
defines the encoding (token ids and the question and option spans) and the head, so both are exactly the release's.

Two paths:
- ``reference``: the release's own ``systemone`` on one request at a time, any device and dtype. Validators run it in
  float32 on CPU as the reference; on a GPU, float32 is computed without TF32.
- ``fast`` (CUDA, bfloat16): the backbone runs as ``verathos.engine.qwen35.Backbone`` on right-padded batches replayed
  from CUDA graphs (``verathos.engine.graphs``); the head runs for the whole pass at once
  (``verathos.engine.clef_head``). Requests are tokenized as they arrive and batched by length under a token budget.

Both give the release a record built from the request (``Prepared.body``): its model, state and questions, never
another field of it. The reference also takes images, which must be canonical (``protocol.media``): they are decoded
to RGB here and given to the release's processor as pixels with the engine's own processor arguments (no resizing: a
canonical image has the size the processor would choose, and its aspect ratio may round past the processor's limit),
so the release never fetches, opens or resizes anything. ``videos`` and ``media_kwargs`` are refused, and the fast path
refuses images.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verathos.protocol.api import REFUSED_FIELDS
from verathos.protocol.media import canonical_pixels

log = logging.getLogger(__name__)

QUESTION_TYPES = ("noul", "choice", "score")
MEDIA_KWARGS = {"do_resize": False}  # the processor arguments of a record with images: canonical images keep their size


@dataclass(frozen=True)
class Prepared:
    body: dict[str, Any]  # the record the release reads (``record``)
    encoded: Any  # the release's EncodedRecord


def _release_module(path: Path):
    spec = importlib.util.spec_from_file_location("joint_schema_model", path / "joint_schema_model.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


def _validate(body: dict[str, Any]) -> None:
    """The release's request checks (``systemone``), and the fields no request may carry, raised before a request
    joins a batch."""
    for field, why in REFUSED_FIELDS.items():
        if field in body:
            raise ValueError(why)
    questions = body.get("questions")
    if not isinstance(body.get("model"), str) or "state" not in body:
        raise ValueError("model and state are required")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("at least one question is required")
    for question_id, question in questions.items():
        if question.get("type") not in QUESTION_TYPES:
            raise ValueError(f"{question_id}: type must be noul, choice, or score")
        if question["type"] != "noul" and not question.get("criteria"):
            raise ValueError(f"{question_id}: criteria must not be empty")


def record(body: dict[str, Any], bounds: dict[str, int] | None) -> dict[str, Any]:
    """The record the release reads for a request: its model, state and questions, and with ``bounds`` (the images
    the processor takes as they are: ``image_bounds``) its canonical images decoded to RGB, with ``MEDIA_KWARGS``.
    Raises ValueError for a request the release refuses, images where they are not served, or an image that is not
    canonical."""
    _validate(body)
    out = {"model": body["model"], "state": body["state"], "questions": body["questions"]}
    if body.get("images"):
        if bounds is None:
            raise ValueError("images are not served by this engine")
        out["images"] = canonical_pixels(body["images"], **bounds)
        out["media_kwargs"] = dict(MEDIA_KWARGS)
    return out


def image_bounds(processor: Any) -> dict[str, int]:
    """The images a release's processor takes as they are: sides multiples of its patch size times its merge size,
    pixels within its size range."""
    images = processor.image_processor
    return {"token_px": images.patch_size * images.merge_size, "min_pixels": images.size["shortest_edge"],
            "max_pixels": images.size["longest_edge"]}


class ReferenceEngine:
    """The release's ``systemone``, one request per pass, so each is answered as soon as it is computed."""

    def __init__(self, path: Path, device: str, dtype: str) -> None:
        import torch

        if device.startswith("cuda") and dtype == "float32":
            # float32 throughout: matmuls are by default; convolutions (the vision tower's patch embedding) are not
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        self.module = _release_module(path)
        self.model, self.processor = self.module.load_release_model(
            str(path), device=device, dtype=getattr(torch, dtype)
        )
        self.model.eval()
        self.image_bounds = image_bounds(self.processor)

    def prepare(self, body: dict[str, Any]) -> Prepared:
        return Prepared(record(body, self.image_bounds), None)

    def passes(self, items: list[Prepared]) -> list[list[int]]:
        return [[i] for i in range(len(items))]

    def run(self, items: list[Prepared]) -> list[dict[str, Any]]:
        return [self.module.systemone(self.model, self.processor, item.body) for item in items]


class FastEngine:
    """Batched CUDA bfloat16 serving of a Clef release."""

    def __init__(
        self,
        path: Path,
        device: str,
        *,
        max_length: int = 16384,
        max_batch_tokens: int = 8192,
        max_rows: int = 64,
        length_step: int = 32,
        graph_tokens: int = 2048,
    ) -> None:
        import torch

        from verathos.engine.graphs import GraphedForward
        from verathos.engine.qwen35 import Backbone

        self.torch = torch
        self.module = _release_module(path)
        model, processor = self.module.load_release_model(str(path), device=device, dtype=torch.bfloat16)
        model.eval()
        base = model.language_model
        self.tokenizer = processor.tokenizer
        self.pad_id = self.tokenizer.pad_token_id
        self.head = model.head
        self.output_embedding = base.get_output_embeddings().weight
        if hasattr(base.model, "visual"):  # text requests only: the vision tower is not used
            del base.model.visual
        self.backbone = Backbone(base.model.language_model)
        torch.cuda.empty_cache()
        self.device = torch.device(device)
        self.max_length = max_length
        self.max_batch_tokens = max_batch_tokens
        self.max_rows = max_rows
        self.length_step = length_step
        self.forward = GraphedForward(
            lambda ids: self.backbone.forward(ids),
            self.backbone.hidden_size,
            torch.bfloat16,
            self.device,
            length_step=length_step,
            max_tokens=graph_tokens,
        )
        self.forward.capture_all()

    def prepare(self, body: dict[str, Any]) -> Prepared:
        release = record(body, None)
        return Prepared(release, self.module.encode_record(self.tokenizer, release, max_length=self.max_length))

    def passes(self, items: list[Prepared]) -> list[list[int]]:
        """Indexes grouped by length: each group's padded size stays within the token budget (a single longer request
        forms its own group)."""
        from verathos.engine.graphs import bucket_length

        order = sorted(range(len(items)), key=lambda i: len(items[i].encoded.input_ids))
        groups: list[list[int]] = []
        for i in order:
            length = bucket_length(len(items[i].encoded.input_ids), self.length_step)
            group = groups[-1] if groups else None
            if group and len(group) < self.max_rows and (len(group) + 1) * length <= self.max_batch_tokens:
                group.append(i)
            else:
                groups.append([i])
        return groups

    def _pass(self, records: list[Any]) -> list[list[Any]]:
        from verathos.engine.clef_head import batched_head
        from verathos.engine.graphs import bucket_length

        torch = self.torch
        length = bucket_length(max(len(r.input_ids) for r in records), self.length_step)
        batch = self.forward.graphed_batch(len(records), length) or len(records)
        ids = torch.full((batch, length), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(records), length), dtype=torch.long)
        for row, record in enumerate(records):
            ids[row, : len(record.input_ids)] = torch.tensor(record.input_ids)
            mask[row, : len(record.input_ids)] = 1
        ids = ids.to(self.device, non_blocking=True)
        hidden = self.forward(ids)
        lengths = mask.sum(1).to(self.device, non_blocking=True)
        return batched_head(self.head, hidden, ids, lengths, records, self.output_embedding)

    def run(self, items: list[Prepared]) -> list[dict[str, Any]]:
        out: list[dict[str, Any] | None] = [None] * len(items)
        with self.torch.inference_mode():
            for group in self.passes(items):
                logits = self._pass([items[i].encoded for i in group])
                for i, record_logits in zip(group, logits):
                    out[i] = self._answer(items[i], record_logits)
        return out  # type: ignore[return-value]

    def _answer(self, item: Prepared, logits: list[Any]) -> dict[str, Any]:
        questions = item.body["questions"]
        answers = {
            question.question_id: self.module.systemone_answer(
                questions[question.question_id],
                dict(zip(question.option_ids, question_logits.float().softmax(-1).tolist())),
            )
            for question, question_logits in zip(item.encoded.questions, logits)
        }
        return {
            "model": item.body["model"],
            "answers": answers,
            "usage": {"input_tokens": len(item.encoded.input_ids), "output_tokens": 0},
        }


def load(repo: str, revision: str, *, device: str, dtype: str, fast: bool | None = None):
    """The engine for a pinned Clef release: the fast path on CUDA in bfloat16 unless ``fast`` is False."""
    from huggingface_hub import snapshot_download

    if fast is None:
        fast = device.startswith("cuda") and dtype == "bfloat16"
    if fast and importlib.util.find_spec("fla") is None:
        raise RuntimeError("the bfloat16 GPU path needs flash-linear-attention at the version that "
                           "verathos/engine/qwen35.py pins (FLA_VERSION; install.sh miner installs it); "
                           "--dtype float32 runs the release's own code instead")
    if not fast and dtype == "float32" and importlib.util.find_spec("fla") is not None:
        log.warning("flash-linear-attention is installed here, so transformers runs the release's linear attention on "
                    "its kernels: these float32 answers differ from validators' references, which run without it")
    path = Path(snapshot_download(repo, revision=revision))
    engine = FastEngine(path, device) if fast else ReferenceEngine(path, device, dtype)
    log.info("loaded %s@%s on %s (%s, %s)", repo, revision[:12], device, dtype, "fast" if fast else "reference")
    return engine
