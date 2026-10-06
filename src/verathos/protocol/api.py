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

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

JSONContent = str | dict | list | int | float | bool | None
MAX_OPTIONS = 255


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


def answer_distribution(answer: dict[str, Any]) -> Distribution:
    """One answer -> its probability distribution keyed like ``question_keys``.

    noul answers carry only p(yes); the distribution is completed as {"false": 1 - p, "true": p}.
    """
    if "probabilities" in answer and isinstance(answer["probabilities"], dict):
        return {str(k): float(v) for k, v in answer["probabilities"].items()}
    if "noul" in answer:
        p = float(answer["noul"])
        return {"false": 1.0 - p, "true": p}
    raise ValueError("answer carries neither probabilities nor noul")


def distributions(answers: dict[str, dict[str, Any]]) -> dict[str, Distribution]:
    """All answers of a response -> {question id: distribution}."""
    return {qid: answer_distribution(a) for qid, a in answers.items()}
