"""Clef models (Cloudflare): a Qwen3.5 backbone with a joint schema head, answering every question of a request in one
pass over one sequence (state, then every question with its options).

The model repository ships its inference code (``joint_schema_model.py``); it is loaded from the pinned revision and
defines the encoding (token ids and the question and option spans) and the head, so both are exactly the release's.

Two paths:
- ``reference``: the release's own ``systemone`` on one request at a time, any device and dtype. Validators run it in
  float32 on CPU as the reference.
- ``fast`` (CUDA, bfloat16): the backbone runs as ``verathos.engine.qwen35.Backbone`` on right-padded batches replayed
  from CUDA graphs (``verathos.engine.graphs``); the head runs per request on its rows. Requests are tokenized as they
  arrive and batched by length under a token budget.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

QUESTION_TYPES = ("noul", "choice", "score")


@dataclass(frozen=True)
class Prepared:
    body: dict[str, Any]
    encoded: Any  # the release's EncodedRecord


def _release_module(path: Path):
    spec = importlib.util.spec_from_file_location("joint_schema_model", path / "joint_schema_model.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


def _validate(body: dict[str, Any]) -> None:
    """The release's request checks (``systemone``), raised before a request joins a batch."""
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


class ReferenceEngine:
    """The release's ``systemone``, one request at a time."""

    def __init__(self, path: Path, device: str, dtype: str) -> None:
        import torch

        self.module = _release_module(path)
        self.model, self.processor = self.module.load_release_model(
            str(path), device=device, dtype=getattr(torch, dtype)
        )
        self.model.eval()

    def prepare(self, body: dict[str, Any]) -> Prepared:
        _validate(body)
        return Prepared(body, None)

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
        batch_sizes: tuple[int, ...] = (1, 2, 4, 8),
        length_step: int = 128,
        graph_max_length: int = 2048,
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
        self.batch_sizes = batch_sizes
        self.length_step = length_step
        self.forward = GraphedForward(
            lambda ids: self.backbone.forward(ids),
            self.backbone.hidden_size,
            torch.bfloat16,
            self.device,
            batch_sizes=batch_sizes,
            length_step=length_step,
            max_length=graph_max_length,
            max_tokens=max_batch_tokens,
        )

    def prepare(self, body: dict[str, Any]) -> Prepared:
        _validate(body)
        return Prepared(body, self.module.encode_record(self.tokenizer, body, max_length=self.max_length))

    def _batches(self, items: list[Prepared]) -> list[list[int]]:
        """Indexes grouped by length: each group's padded size stays within the token budget (a single longer request
        forms its own group)."""
        from verathos.engine.graphs import bucket_length

        order = sorted(range(len(items)), key=lambda i: len(items[i].encoded.input_ids))
        groups: list[list[int]] = []
        for i in order:
            length = bucket_length(len(items[i].encoded.input_ids), self.length_step)
            group = groups[-1] if groups else None
            if group and len(group) < self.batch_sizes[-1] and (len(group) + 1) * length <= self.max_batch_tokens:
                group.append(i)
            else:
                groups.append([i])
        return groups

    def _pass(self, records: list[Any]) -> list[list[Any]]:
        from verathos.engine.graphs import bucket_batch, bucket_length

        torch = self.torch
        length = bucket_length(max(len(r.input_ids) for r in records), self.length_step)
        batch = bucket_batch(len(records), self.batch_sizes) or len(records)
        ids = torch.full((batch, length), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(records), length), dtype=torch.long)
        for row, record in enumerate(records):
            ids[row, : len(record.input_ids)] = torch.tensor(record.input_ids)
            mask[row, : len(record.input_ids)] = 1
        ids = ids.to(self.device, non_blocking=True)
        hidden = self.forward(ids)
        return self.head(hidden, ids, mask.to(self.device), records, self.output_embedding)

    def run(self, items: list[Prepared]) -> list[dict[str, Any]]:
        out: list[dict[str, Any] | None] = [None] * len(items)
        with self.torch.inference_mode():
            for group in self._batches(items):
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

    path = Path(snapshot_download(repo, revision=revision))
    if fast is None:
        fast = device.startswith("cuda") and dtype == "bfloat16"
    engine = FastEngine(path, device) if fast else ReferenceEngine(path, device, dtype)
    log.info("loaded %s@%s on %s (%s, %s)", repo, revision[:12], device, dtype, "fast" if fast else "reference")
    return engine
