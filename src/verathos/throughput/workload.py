"""The standard request mix a model's capacity is declared and tested with.

Every request is unique (a random ticket id and random words), so nothing can be served from a cache or shared
between endpoints. The catalog sets each model's mix: the state length in words and the questions, and for a model
that reads images the share of its requests that carry images (``images``). Such a request carries images drawn from
a seed of its own (``throughput.images``), so no two requests share an image, and asks besides the mix's questions
what colour their background is, so its answers depend on its images. The miner's self-measurement and the auditor's
load generator build requests the same way from the same vocabulary, and the same images from the same seed.
"""

from __future__ import annotations

import random
import re
import secrets
from typing import Any

from pydantic import BaseModel, Field, model_validator

from verathos.protocol.api import SystemOneRequest
from verathos.protocol.canonical import canonical_json
from verathos.protocol.media import PNG
from verathos.throughput.images import BACKGROUNDS, SEED_BYTES, draw, item, png_bound

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


MAX_STATE_WORDS = 100_000  # the load generator's limits: words of a state,
MAX_IMAGES = 16  # images of a request,
MAX_IMAGE_PIXELS = 1 << 24  # and pixels of its largest image
# what a request with images asks besides the mix's questions
IMAGE_QUESTIONS = {"background": {"type": "choice", "instructions": "What colour is the background?",
                                  "criteria": {name: f"{name.capitalize()} background" for name in BACKGROUNDS}}}


def mix_state(ticket: str, words: str) -> str:
    return f"Ticket {ticket}. {words}."


class WorkloadImages(BaseModel):
    """The images of a model's standard mix: ``share`` of its requests carry from ``count[0]`` to ``count[1]``
    images, each of a width and a height drawn from ``sides_px``."""

    share: float = Field(gt=0, le=1, description="Share of the mix's requests that carry images")
    count: tuple[int, int] = Field(description="Inclusive range of images per request that carries images")
    sides_px: list[int] = Field(min_length=1, description="The sides an image's width and its height are each drawn "
                                "from, in pixels")

    @model_validator(mode="after")
    def _check(self) -> WorkloadImages:
        low, high = self.count
        if not 1 <= low <= high <= MAX_IMAGES:
            raise ValueError(f"images.count must be a range [low, high] with 1 <= low <= high <= {MAX_IMAGES}")
        if min(self.sides_px) < 1 or max(self.sides_px) ** 2 > MAX_IMAGE_PIXELS:
            raise ValueError(f"images.sides_px must be positive, and the largest squared at most {MAX_IMAGE_PIXELS}")
        return self

    def max_png_bytes(self) -> int:
        """The largest PNG an image of the mix can take (``throughput.images.png_bound``)."""
        side = max(self.sides_px)
        return png_bound(side, side)


class Workload(BaseModel):
    """A model's standard request mix."""

    state_words: tuple[int, int] = Field(description="Inclusive range of words per state")
    questions: dict[str, Any] = Field(min_length=1)
    images: WorkloadImages | None = Field(None, description="For a model that reads images: the requests of the mix "
                                          "that carry images")

    @model_validator(mode="after")
    def _check(self) -> Workload:
        low, high = self.state_words
        if not 1 <= low <= high <= MAX_STATE_WORDS:
            raise ValueError(f"state_words must be a range [low, high] with 1 <= low <= high <= {MAX_STATE_WORDS}")
        SystemOneRequest.model_validate({"model": "m", "state": "s", "questions": self.questions})
        if self.images is not None and set(self.questions) & set(IMAGE_QUESTIONS):
            raise ValueError(f"a mix with images asks {', '.join(IMAGE_QUESTIONS)} itself: no question may take its id")
        return self

    def document(self) -> dict[str, Any]:
        """The mix as the catalog writes it, without images where it has none."""
        return self.model_dump(mode="json", exclude_none=True)

    def image_questions(self) -> dict[str, Any]:
        """The questions of a request with images: the mix's, and ``IMAGE_QUESTIONS``."""
        return {**self.questions, **IMAGE_QUESTIONS}

    def max_request_bytes(self, model_id: str) -> int:
        """The size (canonical JSON) of the mix's largest request: the most words, each as long as the longest, and
        with images the most of them, each as large as its PNG can be."""
        words = " ".join([max(WORDS, key=len)] * self.state_words[1])
        request = {"model": model_id, "state": mix_state("0" * 16, words), "questions": self.questions}
        if self.images is not None:
            image = {"content_type": PNG, "base64": "A" * (4 * -(-self.images.max_png_bytes() // 3))}
            request.update(questions=self.image_questions(), images=[image] * self.images.count[1])
        return len(canonical_json(request))


def make_request(model_id: str, workload: Workload, rng: random.Random) -> dict[str, Any]:
    """One unique request of the model's mix."""
    words = " ".join(rng.choice(WORDS) for _ in range(rng.randint(*workload.state_words)))
    request = {"model": model_id, "state": mix_state(secrets.token_hex(8), words), "questions": workload.questions}
    if workload.images is not None and rng.random() < workload.images.share:
        pngs = draw(rng.randbytes(SEED_BYTES), workload.images.count, workload.images.sides_px)
        request.update(questions=workload.image_questions(), images=[item(png) for png in pngs])
    return request


_TICKET = re.compile(r"Ticket [0-9a-f]{16}")
_WORDS = frozenset(WORDS)


def in_mix_form(state: Any) -> bool:
    """Whether a state has the form of the mix's (here and in the load generator): a ticket id, then only words of
    the vocabulary. Users do not write such states, so one reaching a miner as a user's is test traffic relayed."""
    if not isinstance(state, str) or not state.endswith("."):
        return False
    ticket, _, words = state[:-1].partition(". ")
    return _TICKET.fullmatch(ticket) is not None and bool(words) and set(words.split(" ")) <= _WORDS
