"""Canary requests: generated per model per epoch from a validator secret.

Each epoch the validator derives a fresh pool of canaries per model, computes their reference answers once, and
gives every endpoint of that model a different random subset. Everything is derived from
(secret, epoch, model), so the pool is reproducible by the validator and unpredictable to miners.
"""

from __future__ import annotations

import hashlib
import hmac
import random
from collections.abc import Callable
from typing import Any

from verathos.protocol.canonical import canonical_json

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


def _rng(secret: bytes, *parts: object) -> random.Random:
    seed = hmac.new(secret, "|".join(str(p) for p in parts).encode(), hashlib.sha256).digest()
    return random.Random(int.from_bytes(seed, "big"))


def _fill(template: str, choose: Callable[[list[str]], str]) -> str:
    for key, values in FILL.items():
        while "{" + key + "}" in template:
            template = template.replace("{" + key + "}", choose(values), 1)
    return template


def canary_pool(secret: bytes, epoch: int, model_id: str, size: int) -> list[dict[str, Any]]:
    """``size`` SystemOne requests for ``model_id`` in ``epoch``."""
    pool = []
    for i in range(size):
        rng = _rng(secret, "canary", epoch, model_id, i)
        domain = DOMAINS[rng.choice(sorted(DOMAINS))]
        issues = rng.sample(domain["issues"], 2)
        state = " ".join(x for x in [rng.choice(domain["openers"]), _fill(issues[0], rng.choice) + ".",
                                     "Also, " + _fill(issues[1], rng.choice) + ".", rng.choice(domain["tones"]),
                                     f"Ref {rng.randrange(10**5, 10**6)}."] if x)
        qids = rng.sample(sorted(domain["questions"]), k=rng.randint(2, len(domain["questions"])))
        pool.append({"model": model_id, "state": state, "questions": {q: domain["questions"][q] for q in qids}})
    return pool


def max_request_bytes(model_id: str) -> int:
    """An upper bound of a canary's size (canonical JSON): every part at its longest, with every question."""

    def longest(values: list[str]) -> str:
        return max(values, key=len)

    sizes = []
    for domain in DOMAINS.values():
        issues = sorted((_fill(t, longest) for t in domain["issues"]), key=len)[-2:]
        state = " ".join([longest(domain["openers"]), issues[1] + ".", "Also, " + issues[0] + ".",
                          longest(domain["tones"]), f"Ref {10**6 - 1}."])
        sizes.append(len(canonical_json({"model": model_id, "state": state, "questions": domain["questions"]})))
    return max(sizes)


def assignment(secret: bytes, epoch: int, endpoint_key: str, pool_size: int, per_endpoint: int) -> list[int]:
    """Indices of the pool this endpoint receives this epoch."""
    return sorted(_rng(secret, "assign", epoch, endpoint_key).sample(range(pool_size), k=min(per_endpoint, pool_size)))
