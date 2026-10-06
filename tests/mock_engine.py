"""Deterministic in-process engine for tests."""

from __future__ import annotations

import hashlib
from typing import Any

from verathos.protocol.api import Choice, Noul, Score, SystemOneRequest, question_keys


class MockEngine:
    """Deterministic fake engine for tests: probabilities derived from a hash of the request."""

    def __init__(self, model: str, salt: str = "") -> None:
        self.model = model
        self.salt = salt

    async def answer(self, request: dict[str, Any]) -> dict[str, Any]:
        req = SystemOneRequest.model_validate(request)
        answers = {}
        for qid, q in req.questions.items():
            keys = question_keys(q)
            raw = [
                int.from_bytes(hashlib.sha256(f"{self.salt}|{req.state}|{qid}|{k}".encode()).digest()[:4], "big") + 1
                for k in keys
            ]
            probs = [round(x / sum(raw), 4) for x in raw]
            if isinstance(q, Noul):
                answers[qid] = {"type": "noul", "noul": probs[1]}
            elif isinstance(q, Choice):
                answers[qid] = {"type": "choice", "choice": keys[probs.index(max(probs))],
                                "probabilities": dict(zip(keys, probs))}
            elif isinstance(q, Score):
                answers[qid] = {"type": "score", "score": sum(i * p for i, p in enumerate(probs)),
                                "probabilities": dict(zip(keys, probs))}
        return {"model": self.model, "answers": answers,
                "usage": {"input_tokens": len(str(req.state)) // 4, "output_tokens": 0}}
