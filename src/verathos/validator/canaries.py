"""Canary requests: generated per model per epoch from a validator secret.

Each epoch the validator derives a fresh pool of canaries per model, computes their reference answers once, and
gives every endpoint of that model a different random subset. Everything is derived from
(secret, epoch, model), so the pool is reproducible by the validator and unpredictable to miners.

For a model that reads images, its catalog's ``images.canary_share`` of the pool carries an image the validator draws
(``draw``): one to six shapes in named colours on a coloured background, or a small bar chart, sometimes a word or
number in Pillow's built-in font; a canonical PNG whose sides are multiples of the model's image token between 256 and
512 px, asked two or three questions about what it shows with plain answers (a shape of a colour, how many, the
largest or the tallest, the background, a word). Its state is an ordinary canary state.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import io
import math
import random
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from verathos.protocol.canonical import canonical_json
from verathos.protocol.media import PNG, ImageLimits, smart_resize

DOMAINS: dict[str, dict[str, Any]] = {
    "support": {
        "openers": ["Hi team,", "Hello,", "To whom it may concern,", "Hey support,", "Good morning,"],
        "issues": ["we were billed twice for {month}", "the export button returns a {code} error since the last update",
                   "I want to move our {seats} seats to the enterprise plan", "my order #{num} arrived broken",
                   "please delete my account and all stored data", "the app crashes when I open settings on {device}",
                   "a promised {pct}% discount never applied", "our API key stopped working at {hour}:00",
                   "can you send invoice {num} in {lang}", "the delivery is {days} days late"],
        "tones": ["", "This is unacceptable.", "Thanks a lot for your help!",
                  "If this is not fixed by Friday we will cancel.",
                  "No rush.", "This is the third time I am writing about this."],
        "questions": {
            "department": {"type": "choice", "instructions": "Which team should handle this?",
                           "criteria": {"billing": "Charges and refunds", "technical": "Bugs or outages",
                                        "sales": "Plans and pricing", "logistics": "Shipping and delivery",
                                        "privacy": "Data and account deletion", "other": "Everything else"}},
            "urgency": {"type": "score", "instructions": "How urgent is this?",
                        "criteria": ["Can wait", "This week", "Today", "Right now"]},
            "churn": {"type": "noul", "instructions": "Does the customer threaten to cancel?"},
            "refund": {"type": "noul", "instructions": "Is a refund requested?"},
            "bug": {"type": "noul", "instructions": "Is a product bug reported?"},
            "angry": {"type": "noul", "instructions": "Is the customer angry?"},
        },
    },
    "code_review": {
        "openers": ["PR summary:", "Change description:", "Commit message:", "Patch notes:"],
        "issues": ["remove the null check in {module} because callers never pass None",
                   "bump the timeout from {num} to {num2} seconds for the {module} client",
                   "store the user password in plain text in the {module} cache for speed",
                   "add retries with backoff to the {module} uploader",
                   "disable TLS certificate verification for the {module} webhook",
                   "rename variables in {module} for readability, no behaviour change",
                   "log the full request body including tokens in {module}"],
        "tones": ["", "Tests pass locally.", "No tests added.", "Urgent hotfix for production.",
                  "Reviewed by one teammate."],
        "questions": {
            "risk": {"type": "score", "instructions": "How risky is this change?",
                     "criteria": ["None", "Low", "Medium", "High"]},
            "security": {"type": "noul", "instructions": "Does the change weaken security?"},
            "behaviour": {"type": "noul", "instructions": "Does the change alter runtime behaviour?"},
            "action": {"type": "choice", "instructions": "What should the reviewer do?",
                       "criteria": {"approve": "Safe to merge", "request_changes": "Needs fixes first",
                                    "discuss": "Needs a design discussion"}},
        },
    },
}
FILL = {
    "month": ["January", "March", "June", "September", "November"], "code": ["500", "502", "403", "404"],
    "seats": ["5", "12", "40", "250"], "num": ["1182", "4411", "90210", "7", "30"], "num2": ["60", "90", "300"],
    "device": ["Android", "iOS", "Windows", "macOS"], "pct": ["10", "15", "20", "50"], "hour": ["7", "9", "13", "22"],
    "lang": ["German", "French", "Spanish", "Japanese"], "days": ["3", "8", "21"],
    "module": ["auth", "billing", "search", "export", "notifications"],
}


IMAGE_SIDES = (256, 512)  # an image canary's sides, multiples of 32 between these, then fitted to the model's limits
IMAGE_BYTES = 12 << 10  # an image canary's PNG at most (they take 1 to 7 KB): a canary drawn larger carries none
COLOURS = {"red": (220, 40, 40), "green": (40, 170, 70), "blue": (40, 90, 220), "yellow": (240, 210, 40),
           "orange": (245, 140, 30), "black": (20, 20, 20), "white": (250, 250, 250), "grey": (150, 150, 150)}
BACKGROUNDS = ("white", "black", "blue", "yellow", "grey", "green")
LIGHT = ("white", "yellow", "grey")  # backgrounds a word is written on in black, the others in white
SHAPES = ("circle", "square", "triangle", "star")
WORDS = ("SALE", "OPEN", "STOP", "EXIT", "PAID", "HELLO", "42", "1987")
NUMBERS = ("One", "Two", "Three", "Four", "Five", "Six")
GRID = 3  # shapes sit in distinct cells of a GRID x GRID grid, so none overlaps another


def seeded(secret: bytes, *parts: object) -> random.Random:
    """A random generator drawn from the validator's secret and ``parts``: reproducible by the validator,
    unpredictable without the secret."""
    seed = hmac.new(secret, "|".join(str(p) for p in parts).encode(), hashlib.sha256).digest()
    return random.Random(int.from_bytes(seed, "big"))


def _fill(template: str, choose: Callable[[list[str]], str]) -> str:
    for key, values in FILL.items():
        while "{" + key + "}" in template:
            template = template.replace("{" + key + "}", choose(values), 1)
    return template


def canary_pool(secret: bytes, epoch: int, model_id: str, size: int,
                images: ImageLimits | None = None) -> list[dict[str, Any]]:
    """``size`` SystemOne requests for ``model_id`` in ``epoch``; for a model that reads images (``images``, its
    limits), ``canary_share`` of them, at places drawn from the secret, with an image (``draw``)."""
    places: set[int] = set()
    if images is not None:
        places = set(seeded(secret, "canary images", epoch, model_id).sample(range(size),
                                                                             round(images.canary_share * size)))
    pool = []
    for i in range(size):
        rng = seeded(secret, "canary", epoch, model_id, i)
        domain = DOMAINS[rng.choice(sorted(DOMAINS))]
        issues = rng.sample(domain["issues"], 2)
        state = " ".join(x for x in [rng.choice(domain["openers"]), _fill(issues[0], rng.choice) + ".",
                                     "Also, " + _fill(issues[1], rng.choice) + ".", rng.choice(domain["tones"]),
                                     f"Ref {rng.randrange(10**5, 10**6)}."] if x)
        if images is not None and i in places:
            drawn = draw(rng, images)
            if len(drawn.png) <= IMAGE_BYTES:
                qids = rng.sample(sorted(drawn.questions), k=rng.randint(2, 3))
                pool.append({"model": model_id, "state": state, "images": [drawn.item()],
                             "questions": {q: drawn.questions[q] for q in qids}})
                continue
        qids = rng.sample(sorted(domain["questions"]), k=rng.randint(2, len(domain["questions"])))
        pool.append({"model": model_id, "state": state, "questions": {q: domain["questions"][q] for q in qids}})
    return pool


# ---- image canaries ----

@dataclass(frozen=True)
class Drawn:
    """An image canary's image and the questions it answers plainly."""

    png: bytes
    questions: dict[str, dict[str, Any]]

    def item(self) -> dict[str, str]:
        return {"content_type": PNG, "base64": base64.b64encode(self.png).decode()}


def shape_question(colour: str, shape: str) -> dict[str, Any]:
    return {"type": "noul", "instructions": f"Is there a {colour} {shape}?"}


def count_question(what: str) -> dict[str, Any]:
    return {"type": "score", "instructions": f"How many {what}s are there?",
            "criteria": [f"{n} {what}{'' if n == 'One' else 's'}" for n in NUMBERS]}


def largest_question() -> dict[str, Any]:
    return {"type": "choice", "instructions": "Which shape is the largest?",
            "criteria": {shape: f"A {shape}" for shape in SHAPES}}


def tallest_question(colours: list[str]) -> dict[str, Any]:
    return {"type": "choice", "instructions": "What colour is the tallest bar?",
            "criteria": {colour: f"The {colour} bar" for colour in colours}}


def background_question() -> dict[str, Any]:
    return {"type": "choice", "instructions": "What colour is the background?",
            "criteria": {colour: f"{colour.capitalize()} background" for colour in BACKGROUNDS}}


def word_question(word: str) -> dict[str, Any]:
    return {"type": "noul", "instructions": f"Does the image show the word {word}?"}


def _star(cx: float, cy: float, r: float) -> list[tuple[float, float]]:
    return [(cx + (r if i % 2 == 0 else 0.45 * r) * math.cos(math.pi / 2 + i * math.pi / 5),
             cy - (r if i % 2 == 0 else 0.45 * r) * math.sin(math.pi / 2 + i * math.pi / 5)) for i in range(10)]


# a shape's area as a multiple of its radius squared (half its width): the star's inner radius is 0.45 of its outer
AREA = {"circle": math.pi, "square": 4.0, "triangle": 2.0, "star": 5 * 0.45 * math.sin(math.pi / 5)}


def draw(rng: random.Random, limits: ImageLimits) -> Drawn:
    """An image canary's image, drawn from ``rng``: its sides multiples of 32 between ``IMAGE_SIDES``, fitted to the
    model's limits as its processor fits them (each a multiple of ``token_px``, the pixels within its budget). Three in
    four show one to six shapes, each in a cell of its own, the largest of at least twice the area of any other; the
    others a bar chart of three to five bars, the tallest at least 1.3 times as tall as any other; half of them a word
    or number. A question about a shape or a word asks for one shown or one not shown, as often."""
    from PIL import Image, ImageDraw, ImageFont

    height, width = smart_resize(rng.randrange(IMAGE_SIDES[0], IMAGE_SIDES[1] + 1, 32),
                                 rng.randrange(IMAGE_SIDES[0], IMAGE_SIDES[1] + 1, 32), limits.token_px,
                                 limits.min_pixels, limits.max_pixels)
    background = rng.choice(BACKGROUNDS)
    image = Image.new("RGB", (width, height), COLOURS[background])
    pen = ImageDraw.Draw(image)
    others = [c for c in COLOURS if c != background]
    cell_w, cell_h = width / GRID, height / GRID
    word = rng.choice(WORDS) if rng.random() < 0.5 else None
    questions = {"background": background_question()}
    if rng.random() < 0.75:
        cells = rng.sample(range(GRID * GRID), k=rng.randint(1, 6) + (word is not None))
        word_cell = cells.pop() if word is not None else None
        largest = AREA["star"] * (0.42 * min(cell_w, cell_h)) ** 2  # a star as wide as fits its cell
        drawn = []
        for k, cell in enumerate(cells):
            shape, colour = rng.choice(SHAPES), rng.choice(others)
            r = math.sqrt(largest * (1.0 if k == 0 else rng.uniform(0.2, 0.45)) / AREA[shape])
            cx, cy = (cell % GRID + 0.5) * cell_w, (cell // GRID + 0.5) * cell_h
            fill = COLOURS[colour]
            if shape == "circle":
                pen.ellipse((cx - r, cy - r, cx + r, cy + r), fill=fill)
            elif shape == "square":
                pen.rectangle((cx - r, cy - r, cx + r, cy + r), fill=fill)
            elif shape == "triangle":
                pen.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=fill)
            else:
                pen.polygon(_star(cx, cy, r), fill=fill)
            drawn.append((colour, shape))
        if rng.random() < 0.5:
            colour, shape = rng.choice(drawn)
        else:
            colour, shape = rng.choice([(c, s) for c in others for s in SHAPES if (c, s) not in drawn])
        questions.update(shape=shape_question(colour, shape), count=count_question("shape"),
                         largest=largest_question())
        corner = None if word_cell is None else (word_cell % GRID * cell_w, word_cell // GRID * cell_h, cell_w)
    else:
        bars = rng.randint(3, 5)
        colours = rng.sample(others, bars)
        tallest = rng.randrange(bars)
        top, bottom = 0.25 * height, 0.9 * height
        step = width / bars
        for k, colour in enumerate(colours):
            share = 1.0 if k == tallest else rng.uniform(0.3, 0.75)
            pen.rectangle(((k + 0.2) * step, bottom - share * (bottom - top), (k + 0.8) * step, bottom),
                          fill=COLOURS[colour])
        questions.update(tallest=tallest_question(colours), count=count_question("bar"))
        corner = (0.0, 0.0, float(width))  # above the bars
    if word is not None and corner is not None:
        x, y, room = corner  # where the word goes, and how wide it may be
        size = max(12, int(0.25 * min(cell_w, cell_h)))
        while size > 12 and pen.textlength(word, font=ImageFont.load_default(size)) > 0.9 * room:
            size -= 2
        pen.text((x + 0.05 * room, y + 0.05 * cell_h), word, font=ImageFont.load_default(size),
                 fill=COLOURS["black" if background in LIGHT else "white"])
    questions["word"] = word_question(word if word is not None and rng.random() < 0.5
                                      else rng.choice([w for w in WORDS if w != word]))
    out = io.BytesIO()
    image.save(out, "PNG")
    return Drawn(out.getvalue(), questions)


def max_request_bytes(model_id: str, images: ImageLimits | None = None) -> int:
    """An upper bound of a canary's size (canonical JSON): every part at its longest, with every question; for a model
    that reads images (``images``), of its image canaries too, their PNG at ``IMAGE_BYTES``."""

    def longest(values: list[str]) -> str:
        return max(values, key=len)

    sizes = []
    image = {"content_type": PNG, "base64": "A" * (4 * -(-IMAGE_BYTES // 3))}
    questions = {"background": background_question(), "count": count_question(longest(["shape", "bar"])),
                 "largest": largest_question(), "shape": shape_question(longest(list(COLOURS)), longest(list(SHAPES))),
                 "tallest": tallest_question(list(COLOURS)), "word": word_question(longest(list(WORDS)))}
    for domain in DOMAINS.values():
        issues = sorted((_fill(t, longest) for t in domain["issues"]), key=len)[-2:]
        state = " ".join([longest(domain["openers"]), issues[1] + ".", "Also, " + issues[0] + ".",
                          longest(domain["tones"]), f"Ref {10**6 - 1}."])
        sizes.append(len(canonical_json({"model": model_id, "state": state, "questions": domain["questions"]})))
        if images is not None and images.canary_share > 0:
            sizes.append(len(canonical_json({"model": model_id, "state": state, "images": [image],
                                             "questions": questions})))
    return max(sizes)


def _canary_states() -> re.Pattern[str]:
    def alternatives(options: list[str]) -> str:
        return "(?:" + "|".join(options) + ")"

    def issue(template: str) -> str:
        parts = re.split(r"\{(\w+)\}", template)  # text, key, text, ...
        return "".join(re.escape(p) if i % 2 == 0 else alternatives([re.escape(v) for v in FILL[p]])
                       for i, p in enumerate(parts))

    domains = []
    for d in DOMAINS.values():
        issues = alternatives([issue(t) for t in d["issues"]])
        tones = alternatives([re.escape(t) for t in d["tones"] if t])
        domains.append(rf"{alternatives([re.escape(o) for o in d['openers']])} {issues}\. Also, {issues}\."
                       rf"(?: {tones})? Ref \d{{6}}\.")
    return re.compile(alternatives(domains))


_CANARY_STATE = _canary_states()


def canary_like(state: Any) -> bool:
    """Whether a state has the form of a canary's. Users do not write such states, so one reaching a miner as a
    user's is a canary relayed."""
    return isinstance(state, str) and _CANARY_STATE.fullmatch(state) is not None


def assignment(secret: bytes, epoch: int, endpoint_key: str, pool_size: int, per_endpoint: int) -> list[int]:
    """Indices of the pool this endpoint receives this epoch."""
    rng = seeded(secret, "assign", epoch, endpoint_key)
    return sorted(rng.sample(range(pool_size), k=min(per_endpoint, pool_size)))
