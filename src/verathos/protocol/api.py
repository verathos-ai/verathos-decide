"""SystemOne-compatible request and response shapes (``POST /v1/systemone``).

The same wire format is served by TypeSafe Jev, Cloudflare Clef and the open Kev models, so existing clients
switch by changing their base URL.

Request: a state (text or JSON) and named typed questions.
  choice: ``criteria`` maps option name -> description; answer is a distribution over the names.
  score:  ``criteria`` is an ordered list of levels; answer is a distribution over level indices.
  noul:   yes/no; answer is the probability of yes.

Every answer reduces to one probability distribution per question, keyed by ``question_keys``. Verification
compares those distributions, so the derived fields (choice, score, confidence) are not trusted.
"""

from __future__ import annotations

import math
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

JSONContent = str | dict | list | int | float | bool | None
MAX_OPTIONS = 255
MAX_RESPONSE_BYTES = 64 << 10  # a response (answers and receipt) larger than this is not a valid answer


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


class SystemOneRequest(BaseModel):
    model: str
    state: JSONContent
    questions: dict[str, Question] = Field(min_length=1)


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


def probability(value: Any) -> float:
    """A JSON number in [0, 1]; raises ValueError for anything else (strings, booleans, NaN, infinities)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("probability must be a number")
    try:
        p = float(value)
    except OverflowError:
        raise ValueError("probability out of range") from None
    if not (math.isfinite(p) and 0.0 <= p <= 1.0):
        raise ValueError("probability must be finite and within [0, 1]")
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


def well_formed(request: SystemOneRequest, answers: Any) -> bool:
    """Whether the answers give one valid distribution per question, over the question's own options. A miner signs
    nothing else: a signed malformed answer counts as a wrong answer."""
    try:
        got = distributions(answers)
    except Exception:
        return False
    return set(got) == set(request.questions) and all(set(got[qid]) == set(question_keys(q))
                                                       for qid, q in request.questions.items())
