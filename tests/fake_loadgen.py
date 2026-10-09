"""The load generator's contract (``native/loadgen``) in Python, for tests against in-process miners: each target
over https, accepting the certificate its pin names, its requests with images drawn as ``throughput.images`` draws
them."""

from __future__ import annotations

import asyncio
import json
import random
import secrets
import time
from typing import Any

import httpx

from verathos.protocol import tls
from verathos.throughput.images import SEED_BYTES, draw, item
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


def _json_object(text: str) -> bool:
    try:
        return isinstance(json.loads(text), dict)
    except ValueError:
        return False


async def _target(client: httpx.AsyncClient, target: dict[str, Any], plan: dict[str, Any], start: float,
                  window: tuple[float, float], rng: random.Random) -> dict[str, Any]:
    """A request counts if its reply (200, a JSON object within the size limit) arrives inside the window and it was
    scheduled no earlier than the grace before the window opened."""
    counted: list[dict[str, Any]] = []
    errors: dict[str, int] = {}
    count = int((plan["warmup_s"] + plan["test_s"]) * target["rate_rps"])
    earliest = window[0] - plan["grace_s"]

    async def send(k: int) -> None:
        scheduled = start + k / target["rate_rps"]
        await asyncio.sleep(max(0.0, scheduled - time.time()))
        words = " ".join(rng.choice(plan["words"]) for _ in range(rng.randint(*target["state_words"])))
        body = {"model": target["model"], "state": f"Ticket {secrets.token_hex(8)}. {words}.",
                "questions": target["questions"]}
        if (images := target.get("images")) is not None and rng.random() < images["share"]:
            pngs = draw(rng.randbytes(SEED_BYTES), tuple(images["count"]), images["sides_px"])
            body.update(questions=images["questions"], images=[item(png) for png in pngs])
        request_id = secrets.token_hex(16)
        sent = time.time()
        try:
            r = await asyncio.wait_for(
                client.post(target["url"], json=body, headers={**target["headers"], HDR_REQUEST_ID: request_id},
                            extensions=tls.use(target["tls"])),
                max(0.0, min(window[1], sent + plan["timeout_s"]) - sent))
        except (httpx.HTTPError, TimeoutError):
            return  # unanswered by the end of the window
        done = time.time()
        if not (earliest <= scheduled < window[1] and window[0] <= done < window[1]):
            return
        if r.status_code != 200:
            errors[f"http_{r.status_code}"] = errors.get(f"http_{r.status_code}", 0) + 1
        elif len(r.content) > plan["max_response_bytes"]:
            errors["too_large"] = errors.get("too_large", 0) + 1
        elif not _json_object(r.text):
            errors["invalid_json"] = errors.get("invalid_json", 0) + 1
        else:
            counted.append({"request": body, "request_id": request_id, "response": r.text,
                            "latency_ms": (done - sent) * 1000})

    await asyncio.gather(*(send(k) for k in range(count)))
    latencies = sorted(c.pop("latency_ms") for c in counted)
    in_window = sum(window[0] <= start + k / target["rate_rps"] < window[1] for k in range(count))
    return {"id": target["id"], "sent": in_window, "skipped": 0, "completed": len(counted),
            "latency_ms": {"p50": latencies[len(latencies) // 2] if latencies else None,
                           "p95": latencies[int(len(latencies) * 0.95)] if latencies else None,
                           "max": latencies[-1] if latencies else None},
            "errors": errors, "samples": rng.sample(counted, min(target["samples"], len(counted)))}
