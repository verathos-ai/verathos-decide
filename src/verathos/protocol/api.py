"""SystemOne-compatible request and response shapes (``POST /v1/systemone``).

The same wire format is served by TypeSafe Jev, Cloudflare Clef and the open Kev models, so existing clients
switch by changing their base URL.

Request: a state (text or JSON) and named typed questions; to a model whose catalog entry lists ``image``, also images,
placed before the state in their order (Clef's ``images``, see ``protocol.media``).
  choice: ``criteria`` maps option name -> description; answer is a distribution over the names.
  score:  ``criteria`` is an ordered list of levels; answer is a distribution over level indices.
  noul:   yes/no; answer is the probability of yes.

Every answer reduces to one probability distribution per question, keyed by ``question_keys``. Verification
compares those distributions; the fields users read (noul, choice, score) are functions of it (``derived``), and an
answer whose fields disagree with its own distribution is malformed. An answer carries only its type's fields
(``ANSWER_FIELDS``); confidence and legend are the engine's own and only checked for their form.
"""

from __future__ import annotations

import math
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from verathos.protocol.canonical import canonical_json

JSONContent = str | dict | list | int | float | bool | None
MAX_OPTIONS = 255
MAX_RESPONSE_BYTES = 64 << 10  # a response (answers and receipt) larger than this is not a valid answer
MAX_REQUEST_BYTES = 1 << 20  # a request without its images (its canonical JSON), at most
STATE_CHARS_PER_TOKEN = 4  # a generous bound: tokens of these models average fewer characters
# Fields a request is refused for: no model is served videos, and engines set their own processor arguments.
REFUSED_FIELDS = {"videos": "videos are not supported",
                  "media_kwargs": "media_kwargs is not accepted: engines set their own processor arguments"}
ANSWER_FIELDS = {"noul": {"type", "noul", "probabilities", "confidence"},
                 "choice": {"type", "choice", "probabilities", "confidence"},
                 "score": {"type", "score", "probabilities", "confidence", "legend"}}
ROUNDING = 5e-5  # engines report probabilities, and the values derived from them, to 4 decimals
DERIVED_TOLERANCE = 1e-3  # a derived value may differ by this much from the one its distribution gives


class Noul(BaseModel):
    type: Literal["noul"]
    instructions: JSONContent = None
    criteria: dict[str, JSONContent] | None = None


class Choice(BaseModel):
    type: Literal["choice"]
    instructions: JSONContent = None
    criteria: dict[str, JSONContent]

    @model_validator(mode="after")
    def _check_options(self) -> Choice:
        if not 1 <= len(self.criteria) <= MAX_OPTIONS:
            raise ValueError(f"criteria must have 1..{MAX_OPTIONS} options")
        return self


class Score(BaseModel):
    type: Literal["score"]
    instructions: JSONContent = None
    criteria: list[JSONContent] = Field(min_length=1, max_length=MAX_OPTIONS)


Question = Annotated[Noul | Choice | Score, Field(discriminator="type")]


class ImageObject(BaseModel):
    """An image as its media type and its bytes in base64."""

    content_type: str
    base64: str


# An image a request embeds: a base64 data URL (data:image/png;base64,...) or an ImageObject (``protocol.media``).
ImageInput = str | ImageObject


class SystemOneRequest(BaseModel):
    model: str
    state: JSONContent
    questions: dict[str, Question] = Field(min_length=1)
    images: list[ImageInput] | None = None

    @model_validator(mode="before")
    @classmethod
    def _refused_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for field, why in REFUSED_FIELDS.items():
                if field in data:
                    raise ValueError(why)
        return data


def text(value: JSONContent) -> str:
    """A request's content as the characters it counts: text as it is, anything else as canonical JSON."""
    return value if isinstance(value, str) else "" if value is None else canonical_json(value).decode()


def prompt_chars(request: SystemOneRequest) -> int:
    """The characters of what a model reads of a request: its state, and each question's id, instructions and
    criteria. A model's ``max_state_tokens`` bounds them all, at ``STATE_CHARS_PER_TOKEN``."""
    return len(text(request.state)) + sum(len(qid) + len(text(q.instructions)) + len(text(q.criteria))
                                          for qid, q in request.questions.items())


def question_keys(question: Noul | Choice | Score) -> list[str]:
    """Keys the question's probabilities are reported under, in option order."""
    if isinstance(question, Choice):
        return list(question.criteria)
    if isinstance(question, Noul):
        return ["false", "true"]
    return [str(i) for i in range(len(question.criteria))]


class Usage(BaseModel):
    model_config = ConfigDict(extra="allow")
    input_tokens: int = 0
    output_tokens: int = 0


class SystemOneResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str | None = None
    answers: dict[str, dict[str, Any]]
    usage: Usage | None = None


Distribution = dict[str, float]


def number(value: Any) -> float:
    """A finite JSON number; raises ValueError for anything else (strings, booleans, NaN, infinities)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("not a number")
    try:
        x = float(value)
    except OverflowError:
        raise ValueError("number out of range") from None
    if not math.isfinite(x):
        raise ValueError("not a finite number")
    return x


def probability(value: Any) -> float:
    """A JSON number in [0, 1]; raises ValueError for anything else."""
    p = number(value)
    if not 0.0 <= p <= 1.0:
        raise ValueError("probability must be within [0, 1]")
    return p


def answer_distribution(answer: Any) -> Distribution:
    """One answer -> its probability distribution keyed like ``question_keys``. Raises ValueError if malformed.

    noul answers carry only p(yes); the distribution is completed as {"false": 1 - p, "true": p}.
    """
    if not isinstance(answer, dict):
        raise ValueError("answer must be an object")
    if isinstance(answer.get("probabilities"), dict):
        return {str(k): probability(v) for k, v in answer["probabilities"].items()}
    if "noul" in answer:
        p = probability(answer["noul"])
        return {"false": 1.0 - p, "true": p}
    raise ValueError("answer carries neither probabilities nor noul")


def distributions(answers: Any) -> dict[str, Distribution]:
    """All answers of a response -> {question id: distribution}. Raises ValueError if malformed."""
    if not isinstance(answers, dict):
        raise ValueError("answers must be an object")
    return {qid: answer_distribution(a) for qid, a in answers.items()}


def derived(question: Noul | Choice | Score, distribution: Distribution) -> dict[str, Any]:
    """The field users read, as the engines derive it from the distribution: noul p(true), choice the first most
    likely option, score the expected level."""
    if isinstance(question, Noul):
        return {"noul": distribution["true"]}
    keys = question_keys(question)
    if isinstance(question, Choice):
        return {"choice": max(keys, key=distribution.__getitem__)}
    return {"score": sum(i * distribution[k] for i, k in enumerate(keys))}


def consistent(question: Noul | Choice | Score, answer: dict[str, Any], distribution: Distribution) -> bool:
    """Whether an answer carries only its type's fields, in their form, and the field users read agrees with its
    distribution (``derived``) within the engines' rounding: a chosen option within ``DERIVED_TOLERANCE`` of the most
    likely one, a noul within it of p(true), a score within it plus the rounding of the probabilities it sums."""
    kind = question.type
    if not set(answer) <= ANSWER_FIELDS[kind] or answer.get("type", kind) != kind:
        return False
    [(field, expected)] = derived(question, distribution).items()
    legend = answer.get("legend", distribution)
    if field not in answer or not (isinstance(legend, dict) and set(legend) == set(distribution)):
        return False
    try:
        if "confidence" in answer:
            probability(answer["confidence"])
        if kind == "choice":
            chosen = answer["choice"]
            return chosen in distribution and distribution[chosen] >= distribution[expected] - DERIVED_TOLERANCE
        slack = DERIVED_TOLERANCE + (ROUNDING * len(distribution) * (len(distribution) - 1) / 2 if kind == "score"
                                     else 0.0)
        return abs(number(answer[field]) - expected) <= slack
    except (ValueError, TypeError):
        return False


def well_formed(request: SystemOneRequest, answers: Any) -> bool:
    """Whether the answers give one valid distribution per question, over the question's own options, and agree with
    it (``consistent``). A miner signs nothing else: a signed malformed answer counts as a wrong answer."""
    try:
        got = distributions(answers)
    except Exception:
        return False
    return set(got) == set(request.questions) and all(
        set(got[qid]) == set(question_keys(q)) and consistent(q, answers[qid], got[qid])
        for qid, q in request.questions.items())


def known_fields(request: SystemOneRequest, answers: Any) -> Any:
    """The answers with each answer object reduced to its question type's fields (``ANSWER_FIELDS``); anything else
    as it is, for ``well_formed`` to judge."""
    if not isinstance(answers, dict):
        return answers
    out = {}
    for qid, answer in answers.items():
        q = request.questions.get(qid)
        out[qid] = ({k: v for k, v in answer.items() if k in ANSWER_FIELDS[q.type]}
                    if q is not None and isinstance(answer, dict) else answer)
    return out
