"""Clef models (Cloudflare): a Qwen3.5 backbone with a joint schema head, answering every question of a request in one
pass over one sequence (state, then every question with its options).

The model repository ships its inference code (``joint_schema_model.py``); it is loaded from the pinned revision and
defines the encoding (token ids and the question and option spans) and the head, so both are exactly the release's.

Two paths:
- ``reference``: the release's own ``systemone`` on one request at a time, any device and dtype. Validators run it in
  float32 on CPU as the reference; on a GPU, float32 is computed without TF32.
- ``fast`` (CUDA, bfloat16): the backbone runs as ``verathos.engine.qwen35.Backbone`` on right-padded batches; the head
  runs for the whole pass at once (``verathos.engine.clef_head``). Requests are encoded as they arrive and batched by
  kind and length under a token budget that counts image tokens. Text passes replay CUDA graphs
  (``verathos.engine.graphs``); passes of requests with images run eagerly as the release's forward computes them: the
  images through the release's vision tower, its features in place of the image tokens' embeddings, and the release's
  3-axis positions. The vision tower runs in float16 (``Float16Tower``), the rest in bfloat16.

Both give the release a record built from the request (``Prepared.body``): its model, state and questions, never
another field of it, and its images, which must be canonical (``protocol.media``): they are decoded to RGB here and
given to the release's processor as pixels with the engine's own processor arguments (no resizing: a canonical image
has the size the processor would choose, and its aspect ratio may round past the processor's limit), so the release
never fetches, opens or resizes anything. ``videos`` and ``media_kwargs`` are refused.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verathos.gateway.logs import Throttle
from verathos.protocol.api import REFUSED_FIELDS
from verathos.protocol.media import canonical_pixels

log = logging.getLogger(__name__)
_throttled = Throttle()

QUESTION_TYPES = ("noul", "choice", "score")
MEDIA_KWARGS = {"do_resize": False}  # the processor arguments of a record with images: canonical images keep their size


@dataclass(frozen=True)
class Media:
    """The images of a request as the fast path's passes take them, made when the request is prepared."""

    pixel_values: Any  # [patches, values per patch] in the dtype the vision tower reads (float16)
    processed: Any  # the same in float32, as the release's processor made them (its record holds them)
    grid: Any  # [images, 3]: each image's patches in time, height and width (the release's image_grid_thw)
    positions: Any  # [3, tokens]: the release's 3-axis positions of the request's tokens


@dataclass(frozen=True)
class Prepared:
    body: dict[str, Any]  # the record the release reads (``record``)
    encoded: Any  # the release's EncodedRecord
    media: Media | None = None  # the fast path's images of a request with images


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


def record(body: dict[str, Any], bounds: dict[str, int]) -> dict[str, Any]:
    """The record the release reads for a request: its model, state and questions, and its canonical images decoded to
    RGB, with ``MEDIA_KWARGS``; ``bounds`` are the images the processor takes as they are (``image_bounds``). Raises
    ValueError for a request the release refuses or an image that is not canonical."""
    _validate(body)
    out = {"model": body["model"], "state": body["state"], "questions": body["questions"]}
    if body.get("images"):
        out["images"] = canonical_pixels(body["images"], **bounds)
        out["media_kwargs"] = dict(MEDIA_KWARGS)
    return out


def image_bounds(processor: Any) -> dict[str, int]:
    """The images a release's processor takes as they are: sides multiples of its patch size times its merge size,
    pixels within its size range."""
    images = processor.image_processor
    return {"token_px": images.patch_size * images.merge_size, "min_pixels": images.size["shortest_edge"],
            "max_pixels": images.size["longest_edge"]}


class Float16Tower:
    """A release's vision tower with its weights in float16, three more bits of precision than bfloat16 and a narrower
    range; its buffers (the rotary frequencies, float32) stay as loaded. With the tower in float16, answers to requests
    with images drift from the float32 reference about as far as text answers do; in bfloat16, as released, about
    twice as far. ``released`` runs the tower with the release's weights in their dtype, for a pass whose float16
    features are not all finite (an activation beyond 65,504): the weights float16 does not hold (beyond or below its
    range) are kept apart for it."""

    def __init__(self, tower: Any) -> None:
        import torch

        self.tower, self.torch = tower, torch
        self._kept: dict[str, tuple[Any, Any, Any]] = {}  # name -> (released dtype, flat indexes, released values)
        with torch.no_grad():
            for name, parameter in tower.named_parameters():
                half = parameter.data.half()
                lost = (half.to(parameter.dtype) != parameter.data).flatten().nonzero().squeeze(1)
                self._kept[name] = (parameter.dtype, lost, parameter.data.flatten()[lost])
                parameter.data = half

    @contextmanager
    def released(self) -> Iterator[None]:
        """The tower with the release's weights in their dtype while the context lasts, in float16 again after."""
        torch = self.torch
        try:
            with torch.inference_mode(False), torch.no_grad():
                for name, parameter in self.tower.named_parameters():
                    dtype, lost, values = self._kept[name]
                    weights = parameter.data.to(dtype)
                    weights.view(-1)[lost] = values
                    parameter.data = weights
            yield
        finally:
            with torch.inference_mode(False), torch.no_grad():
                for parameter in self.tower.parameters():
                    parameter.data = parameter.data.half()


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
    """Batched CUDA serving of a Clef release: the backbone and the head in bfloat16, the vision tower in float16."""

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
        model, self.processor = self.module.load_release_model(str(path), device=device, dtype=torch.bfloat16)
        model.eval()
        base = model.language_model
        self.tokenizer = self.processor.tokenizer
        self.pad_id = self.tokenizer.pad_token_id
        self.image_bounds = image_bounds(self.processor)
        self.head = model.head
        self.output_embedding = base.get_output_embeddings().weight
        self.vision = base.model  # the release's vision tower (get_image_features) and positions (get_rope_index)
        self.tower = Float16Tower(self.vision.visual)
        self.pixel_dtype = torch.float16  # the tower's input, cast as a request is prepared
        self.image_token_id = base.config.image_token_id
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
        release = record(body, self.image_bounds)
        encoded = self.module.encode_record(self.tokenizer, release, max_length=self.max_length,
                                            processor=self.processor)
        return Prepared(release, encoded, None if encoded.media is None else self._media(encoded))

    def _media(self, encoded: Any) -> Media:
        """A request's images as a pass takes them, checked as the release's forward checks them: its image tokens
        must be as many as its images' features (its text holding the image token would add some)."""
        torch = self.torch
        media, ids = encoded.media, torch.tensor([encoded.input_ids])
        grid = media["image_grid_thw"]
        if int((ids == self.image_token_id).sum()) != int(grid.prod(-1).sum()) // self.vision.visual.spatial_merge_unit:
            raise ValueError("a request with images may not hold <|image_pad|> in its text")
        types = torch.zeros_like(ids)  # the release's mm_token_type_ids: which tokens are image tokens
        offset, values = media["token_offset"], media["mm_token_type_ids"]
        types[0, offset : offset + len(values)] = torch.tensor(values)
        positions, _ = self.vision.get_rope_index(ids, types, image_grid_thw=grid)
        # cast here, off the device's thread, so a pass sends the GPU half the bytes
        return Media(media["pixel_values"].to(self.pixel_dtype), media["pixel_values"], grid, positions[:, 0])

    def passes(self, items: list[Prepared]) -> list[list[int]]:
        """Indexes grouped by kind and length: requests with images and text requests in passes of their own, each
        group's padded size within the token budget, which counts image tokens (a single longer request forms its own
        group)."""
        from verathos.engine.graphs import bucket_length

        order = sorted(range(len(items)), key=lambda i: (items[i].media is not None, len(items[i].encoded.input_ids)))
        groups: list[list[int]] = []
        for i in order:
            length = bucket_length(len(items[i].encoded.input_ids), self.length_step)
            group = groups[-1] if groups else None
            if (group and (items[group[0]].media is None) == (items[i].media is None) and len(group) < self.max_rows
                    and (len(group) + 1) * length <= self.max_batch_tokens):
                group.append(i)
            else:
                groups.append([i])
        return groups

    def _pass(self, items: list[Prepared]) -> list[list[Any]]:
        """Option logits of one pass: of text requests only, or of requests with images only (``passes``)."""
        from verathos.engine.clef_head import batched_head
        from verathos.engine.graphs import bucket_length

        torch = self.torch
        records = [item.encoded for item in items]
        media = [item.media for item in items if item.media is not None]
        length = bucket_length(max(len(r.input_ids) for r in records), self.length_step)
        batch = len(records) if media else self.forward.graphed_batch(len(records), length) or len(records)
        ids = torch.full((batch, length), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(records), length), dtype=torch.long)
        for row, record in enumerate(records):
            ids[row, : len(record.input_ids)] = torch.tensor(record.input_ids)
            mask[row, : len(record.input_ids)] = 1
        ids = ids.to(self.device, non_blocking=True)
        hidden = self._media_forward(ids, media) if media else self.forward(ids)
        lengths = mask.sum(1).to(self.device, non_blocking=True)
        return batched_head(self.head, hidden, ids, lengths, records, self.output_embedding)

    def _features(self, media: list[Media]) -> Any:
        """The merged features of a pass's images from the float16 vision tower (attention within each image), or,
        where any is not finite, of all of them from the release's tower on the processor's pixel values
        (``Float16Tower``)."""
        torch = self.torch
        grid = torch.cat([m.grid for m in media]).to(self.device, non_blocking=True)
        pixels = torch.cat([m.pixel_values for m in media]).to(self.device, non_blocking=True)
        features = torch.cat(self.vision.get_image_features(pixels, grid, return_dict=True).pooler_output)
        if bool(torch.isfinite(features).all()):
            return features
        if times := _throttled("float16 tower"):
            log.warning("vision features not finite in float16 in %d passes since last logged, the last of %d "
                        "requests with images: their images ran through the vision tower again in bfloat16, as "
                        "released", times, len(media))
        pixels = torch.cat([m.processed for m in media]).to(self.device, non_blocking=True)
        with self.tower.released():
            return torch.cat(self.vision.get_image_features(pixels, grid, return_dict=True).pooler_output)

    def _media_forward(self, ids: Any, media: list[Media]) -> Any:
        """Last hidden states of a pass of requests with images, as the release's forward computes them: the images'
        features (``_features``) in place of the image tokens' embeddings, in order, and its 3-axis positions (padding
        after a request keeps position 0: it never reaches the request)."""
        torch = self.torch
        features = self._features(media)
        embeds = torch.nn.functional.embedding(ids, self.backbone.embed)
        embeds = embeds.masked_scatter((ids == self.image_token_id)[..., None], features.to(embeds.dtype))
        positions = torch.zeros((3, *ids.shape), dtype=torch.long)
        for row, m in enumerate(media):
            positions[:, row, : m.positions.shape[1]] = m.positions
        positions = positions.to(self.device, non_blocking=True)
        return self.backbone.forward(inputs_embeds=embeds, position_ids=positions)

    def run(self, items: list[Prepared]) -> list[dict[str, Any]]:
        out: list[dict[str, Any] | None] = [None] * len(items)
        with self.torch.inference_mode():
            for group in self.passes(items):
                logits = self._pass([items[i] for i in group])
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
