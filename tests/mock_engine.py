"""Deterministic in-process engine for tests."""

from __future__ import annotations

import hashlib
from typing import Any

from verathos.protocol.api import Noul, SystemOneRequest, derived, question_keys


class MockEngine:
    """Deterministic fake engine for tests: probabilities derived from a hash of the request."""

    def __init__(self, model: str, salt: str = "", served: dict[str, Any] | None = None) -> None:
        self.model = model
        self.salt = salt
        self.served = served  # its GET /v1/models document; None: it serves none

    async def info(self) -> Any:
        return self.served

    async def answer(self, request: dict[str, Any]) -> dict[str, Any]:
        req = SystemOneRequest.model_validate(request)
        answers = {}
        for qid, q in req.questions.items():
            keys = question_keys(q)
            raw = [
                int.from_bytes(hashlib.sha256(f"{self.salt}|{req.state}|{qid}|{k}".encode()).digest()[:4], "big") + 1
                for k in keys
            ]
            dist = dict(zip(keys, (round(x / sum(raw), 4) for x in raw)))
            answers[qid] = {"type": q.type, **derived(q, dist)}
            if not isinstance(q, Noul):
                answers[qid]["probabilities"] = dist
        return {"model": self.model, "answers": answers,
                "usage": {"input_tokens": len(str(req.state)) // 4, "output_tokens": 0}}
