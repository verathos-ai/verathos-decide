"""The Clef joint schema head for a whole batch at once.

The release's ``JointSchemaHead.forward`` walks every record, question and option in Python, which costs a few
milliseconds of small kernels per request. ``batched_head`` computes the same function with the same modules for all
records of a pass together:
- span means (questions, options, and the option tokens' output embeddings) as gathered sums divided by the span
  lengths, accumulated in fp32 like the release's ``mean``;
- option queries attend to their record's memory, and fields to their record's fields and memory, as padded
  [records, items, width] batches whose key padding masks hide other records' padding;
- the per-question option routing softmax and the option logits over flat [options] tensors indexed by question.
Rounding differs from the release's per-record loop where a sum is ordered differently, and in the option routing
(scores, softmax and weighted sum), which runs in float32 where the release rounds each step to the model's dtype.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class Layout:
    """Flat indexes of a pass's questions and options (built on the CPU, moved once)."""

    q_record: torch.Tensor  # [Q]
    q_slot: torch.Tensor  # [Q] position among its record's questions
    q_type: torch.Tensor  # [Q]
    o_record: torch.Tensor  # [O]
    o_question: torch.Tensor  # [O] flat question index
    o_slot: torch.Tensor  # [O] position among its record's options
    span_record: torch.Tensor  # [N] record of each span token
    span_position: torch.Tensor  # [N] token position
    span_segment: torch.Tensor  # [N] segment: questions 0..Q-1, then options Q..Q+O-1
    counts: torch.Tensor  # [Q + O] tokens per segment
    max_questions: int
    max_options: int
    options_per_question: list[int]
    questions_per_record: list[int]


def layout(records: list, device: torch.device) -> Layout:
    q_record, q_slot, q_type, o_record, o_question, o_slot = [], [], [], [], [], []
    span_record, span_position, span_segment, counts = [], [], [], []
    question_spans, option_spans = [], []
    options_per_question, questions_per_record = [], []
    max_options = 0
    for r, record in enumerate(records):
        questions_per_record.append(len(record.questions))
        slot = 0
        for qs, question in enumerate(record.questions):
            q_index = len(q_record)
            q_record.append(r)
            q_slot.append(qs)
            q_type.append(question.question_type)
            question_spans.append((r, question.question_span))
            options_per_question.append(len(question.option_spans))
            for span in question.option_spans:
                o_record.append(r)
                o_question.append(q_index)
                o_slot.append(slot)
                option_spans.append((r, span))
                slot += 1
        max_options = max(max_options, slot)
    for segment, (r, (start, end)) in enumerate(question_spans + option_spans):
        span_record += [r] * (end - start)
        span_position += range(start, end)
        span_segment += [segment] * (end - start)
        counts.append(end - start)

    def tensor(values):
        return torch.tensor(values, dtype=torch.long).to(device, non_blocking=True)

    return Layout(
        q_record=tensor(q_record), q_slot=tensor(q_slot), q_type=tensor(q_type),
        o_record=tensor(o_record), o_question=tensor(o_question), o_slot=tensor(o_slot),
        span_record=tensor(span_record), span_position=tensor(span_position), span_segment=tensor(span_segment),
        counts=tensor(counts), max_questions=max(questions_per_record), max_options=max_options,
        options_per_question=options_per_question, questions_per_record=questions_per_record,
    )


def _segment_mean(values: torch.Tensor, segment: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    sums = torch.zeros(len(counts), values.shape[-1], dtype=torch.float32, device=values.device)
    sums.index_add_(0, segment, values.float())
    return (sums / counts[:, None]).to(values.dtype)


def _padded(flat: torch.Tensor, record: torch.Tensor, slot: torch.Tensor, records: int, width: int):
    out = flat.new_zeros(records, width, flat.shape[-1])
    out[record, slot] = flat
    mask = torch.ones(records, width, dtype=torch.bool, device=flat.device)
    mask[record, slot] = False
    return out, mask


def _routing(layer, queries: torch.Tensor, memory: torch.Tensor, memory_mask: torch.Tensor) -> torch.Tensor:
    """EvidenceRoutingLayer.forward with the memory's padding masked."""
    normalized = layer.memory_norm(memory)
    routed, _ = layer.attention(layer.query_norm(queries), normalized, normalized,
                                key_padding_mask=memory_mask, need_weights=False)
    queries = queries + layer.attention_dropout(routed)
    return queries + layer.feedforward(layer.feedforward_norm(queries))


def batched_head(head, hidden: torch.Tensor, input_ids: torch.Tensor, lengths: torch.Tensor, records: list,
                 output_embedding: torch.Tensor) -> list[list[torch.Tensor]]:
    """Option logits per question per record, as the release's head computes them record by record.

    hidden [B, T, H] and input_ids [B, T] may carry padding rows after ``len(records)``; lengths [len(records)]."""
    n = len(records)
    device = hidden.device
    lay = layout(records, device)
    T = hidden.shape[1]
    normalized = head.hidden_norm(hidden[:n])
    memory = head.memory_projection(normalized)
    memory_mask = torch.arange(T, device=device)[None, :] >= lengths[:, None]
    global_vector = normalized[torch.arange(n, device=device), lengths - 1]

    span_hidden = normalized[lay.span_record, lay.span_position]
    means = _segment_mean(span_hidden, lay.span_segment, lay.counts)
    Q = len(lay.q_record)
    question_vectors, context = means[:Q], means[Q:]
    option_tokens = input_ids[lay.span_record, lay.span_position]
    lexical_tokens = output_embedding[option_tokens]
    option_mask = lay.span_segment >= Q
    lexical = _segment_mean(lexical_tokens[option_mask], lay.span_segment[option_mask] - Q, lay.counts[Q:])

    queries = (head.option_context_projection(context) + head.option_lexical_projection(lexical)
               + head.option_question_projection(question_vectors)[lay.o_question])
    padded, _ = _padded(queries, lay.o_record, lay.o_slot, n, lay.max_options)
    for layer in head.evidence_layers:
        padded = _routing(layer, padded, memory, memory_mask)
    routed = padded[lay.o_record, lay.o_slot]

    base_fields = head.question_projection(question_vectors)
    scores = (routed.float() * base_fields[lay.o_question].float()).sum(-1) / math.sqrt(routed.shape[-1])
    peak = torch.full((Q,), -math.inf, device=device).scatter_reduce(0, lay.o_question, scores, "amax")
    weights = torch.exp(scores - peak[lay.o_question])
    weights = weights / torch.zeros(Q, device=device).index_add_(0, lay.o_question, weights)[lay.o_question]
    summaries = torch.zeros(Q, routed.shape[-1], dtype=torch.float32, device=device)
    summaries.index_add_(0, lay.o_question, weights[:, None] * routed.float())
    fields = (base_fields + head.option_summary_norm(summaries.to(routed.dtype))
              + head.global_projection(global_vector)[lay.q_record] + head.type_embedding(lay.q_type))

    fields, field_mask = _padded(fields, lay.q_record, lay.q_slot, n, lay.max_questions)
    for layer in head.layers:
        fields = layer(fields, memory, tgt_key_padding_mask=field_mask, memory_key_padding_mask=memory_mask)
    fields = head.field_norm(fields[lay.q_record, lay.q_slot])

    anchor = F.normalize(question_vectors + global_vector[lay.q_record], dim=-1)
    prior_scale = head.prior_logit_scale.clamp(max=math.log(100.0)).exp()
    prior = prior_scale * torch.einsum("oh,oh->o", F.normalize(lexical, dim=-1), anchor[lay.o_question])
    options = head.option_norm(routed)
    field = fields[lay.o_question]
    cosine = F.cosine_similarity(field, options, dim=-1)
    features = torch.cat([field, options, field * options, torch.abs(field - options)], dim=-1)
    residual = head.residual_scorer(features).squeeze(-1)
    joint = head.joint_logit_scale.clamp(max=math.log(100.0)).exp() * cosine + residual
    logits = prior + torch.sigmoid(head.residual_gate) * joint

    per_question = list(torch.split(logits, lay.options_per_question))
    out, start = [], 0
    for count in lay.questions_per_record:
        out.append(per_question[start : start + count])
        start += count
    return out
