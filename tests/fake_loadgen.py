"""The load generator's contract (``native/loadgen``) in Python, for tests against in-process miners."""

from __future__ import annotations

import asyncio
import random
import secrets
import time
from typing import Any

import httpx

from verathos.throughput.loadgen import LoadResult
from verathos.throughput.policy import HDR_REQUEST_ID


def make_runner(transport: httpx.AsyncBaseTransport):
    async def run(plan: dict[str, Any], *, timeout_s: float) -> LoadResult:
        start = plan["start_at_ms"] / 1000
        window = (start + plan["warmup_s"], start + plan["warmup_s"] + plan["test_s"])
        rng = random.Random(plan["seed"])
        async with httpx.AsyncClient(transport=transport, timeout=plan["timeout_s"]) as client:
            await asyncio.sleep(max(0.0, start - time.time()))
            results = await asyncio.gather(*(_target(client, t, plan, start, window, rng) for t in plan["targets"]))
        return LoadResult.model_validate({"started_at_ms": int(start * 1000), "targets": results})

    return run


async def _target(client: httpx.AsyncClient, target: dict[str, Any], plan: dict[str, Any], start: float,
                  window: tuple[float, float], rng: random.Random) -> dict[str, Any]:
    completed: list[dict[str, Any]] = []
    errors: dict[str, int] = {}
    count = int((plan["warmup_s"] + plan["test_s"]) * target["rate_rps"])

    async def send(k: int) -> None:
        await asyncio.sleep(max(0.0, start + k / target["rate_rps"] - time.time()))
        words = " ".join(rng.choice(plan["words"]) for _ in range(rng.randint(*target["state_words"])))
        body = {"model": target["model"], "state": f"Ticket {secrets.token_hex(8)}. {words}.",
                "questions": target["questions"]}
        request_id = secrets.token_hex(16)
        sent = time.time()
        try:
            r = await client.post(target["url"], json=body, headers={**target["headers"], HDR_REQUEST_ID: request_id})
        except httpx.HTTPError:
            errors["transport"] = errors.get("transport", 0) + 1
            return
        done = time.time()
        if r.status_code != 200:
            errors[f"http_{r.status_code}"] = errors.get(f"http_{r.status_code}", 0) + 1
        elif window[0] <= done <= window[1]:
            completed.append({"request": body, "request_id": request_id, "status": 200, "response": r.json(),
                              "sent_at_ms": int(sent * 1000), "completed_at_ms": int(done * 1000),
                              "latency_ms": (done - sent) * 1000})

    await asyncio.gather(*(send(k) for k in range(count)))
    latencies = sorted(c.pop("latency_ms") for c in completed)
    in_window = sum(window[0] <= start + k / target["rate_rps"] < window[1] for k in range(count))
    return {"id": target["id"], "sent": in_window, "skipped": 0, "completed": len(completed),
            "delivered_rps": len(completed) / plan["test_s"],
            "latency_ms": {"p50": latencies[len(latencies) // 2] if latencies else None,
                           "p95": latencies[int(len(latencies) * 0.95)] if latencies else None,
                           "max": latencies[-1] if latencies else None},
            "errors": errors, "samples": rng.sample(completed, min(target["samples"], len(completed)))}
