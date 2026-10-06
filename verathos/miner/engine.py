"""Decision engines behind a miner endpoint.

An engine takes a SystemOne request body and returns a SystemOne response body. Production engines run as
separate processes (one per GPU or Mac) and are reached over local HTTP; the Kev server (``python -m kev.serve``)
is one such engine.
"""

from __future__ import annotations

import hashlib
from typing import Any, Protocol

import httpx

from verathos.protocol.api import Choice, Noul, Score, SystemOneRequest, question_keys


class EngineError(Exception):
    pass


class Engine(Protocol):
    async def answer(self, request: dict[str, Any]) -> dict[str, Any]: ...


class HttpEngine:
    """A local engine process speaking ``POST /v1/systemone``."""

    def __init__(self, base_url: str, api_key: str | None = None, timeout_s: float = 120.0) -> None:
        self._url = base_url.rstrip("/") + "/v1/systemone"
        self._headers = {"Authorization": f"Bearer {api_key or 'local'}"}
        self._client = httpx.AsyncClient(timeout=timeout_s)

    async def answer(self, request: dict[str, Any]) -> dict[str, Any]:
        try:
            r = await self._client.post(self._url, json=request, headers=self._headers)
        except httpx.HTTPError as e:
            raise EngineError(f"engine unreachable: {e}") from e
        if r.status_code != 200:
            raise EngineError(f"engine returned {r.status_code}: {r.text[:200]}")
        return r.json()

    async def close(self) -> None:
        await self._client.aclose()


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
            raw = [int.from_bytes(hashlib.sha256(f"{self.salt}|{req.state}|{qid}|{k}".encode()).digest()[:4], "big") + 1
                   for k in keys]
            probs = [round(x / sum(raw), 4) for x in raw]
            if isinstance(q, Noul):
                answers[qid] = {"type": "noul", "noul": probs[1]}
            elif isinstance(q, Choice):
                answers[qid] = {"type": "choice", "choice": keys[probs.index(max(probs))], "probabilities": dict(zip(keys, probs))}
            elif isinstance(q, Score):
                answers[qid] = {"type": "score", "score": sum(i * p for i, p in enumerate(probs)), "probabilities": dict(zip(keys, probs))}
        return {"model": self.model, "answers": answers, "usage": {"input_tokens": len(str(req.state)) // 4, "output_tokens": 0}}
