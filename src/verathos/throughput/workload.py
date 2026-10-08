"""The standard request mix a model's capacity is declared and tested with.

Every request is unique (a random ticket id and random words), so nothing can be served from a cache or shared
between endpoints. The catalog sets each model's mix: the state length in words and the questions. The miner's
self-measurement and the auditor's load generator build requests the same way from the same vocabulary.
"""

from __future__ import annotations

import random
import secrets
from typing import Any

from pydantic import BaseModel, Field, model_validator

from verathos.protocol.api import SystemOneRequest
from verathos.protocol.canonical import canonical_json

WORDS = tuple("""
account access address advice agent alert amount answer api app approve archive arrive attach audit backup balance
bank batch billing block bonus booking broken budget bug cancel card cart charge chat check checkout claim client
code company complaint confirm contract copy coupon crash credit customer damaged data date deadline debit delay
delete delivery deposit device discount dispute document download duplicate email error escalate estimate export
fail fee file fix form fraud friday invoice issue item key language late license limit link login loss manager
missing mobile monday month morning network notice number offer order outage package page partner password payment
pending phone plan please portal price problem product profile promise purchase quote receipt refund region renew
report request reset response restore return review route sales schedule screen security send server service
settings shipment shipping sign slow storage subscription support sync tax team ticket today token track transfer
trial update upgrade upload urgent user verify version wallet warehouse warranty week wrong
""".split())


MAX_STATE_WORDS = 100_000  # the load generator's limit


class Workload(BaseModel):
    """A model's standard request mix."""

    state_words: tuple[int, int] = Field(description="Inclusive range of words per state")
    questions: dict[str, Any] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> Workload:
        low, high = self.state_words
        if not 1 <= low <= high <= MAX_STATE_WORDS:
            raise ValueError(f"state_words must be a range [low, high] with 1 <= low <= high <= {MAX_STATE_WORDS}")
        SystemOneRequest.model_validate({"model": "m", "state": "s", "questions": self.questions})
        return self

    def max_request_bytes(self, model_id: str) -> int:
        """The size (canonical JSON) of the mix's largest request: the most words, each as long as the longest."""
        words = " ".join([max(WORDS, key=len)] * self.state_words[1])
        return len(canonical_json({"model": model_id, "state": f"Ticket {'0' * 16}. {words}.",
                                   "questions": self.questions}))


def make_request(model_id: str, workload: Workload, rng: random.Random) -> dict[str, Any]:
    """One unique request of the model's mix."""
    words = " ".join(rng.choice(WORDS) for _ in range(rng.randint(*workload.state_words)))
    return {"model": model_id, "state": f"Ticket {secrets.token_hex(8)}. {words}.", "questions": workload.questions}
