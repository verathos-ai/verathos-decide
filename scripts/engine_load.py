"""Load a running engine over HTTP: throughput and latency at a given client concurrency.

    python scripts/engine_load.py --url http://127.0.0.1:8010 --records records.jsonl --concurrency 32 [--n 1000]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--n", type=int, default=1000)
    args = ap.parse_args()
    rows = [json.loads(line) for line in args.records.open()][: args.n]
    bodies = [{"model": "clef", "state": r["state"],
               "questions": {q: {k: v for k, v in x.items() if k != "label"} for q, x in r["questions"].items()}}
              for r in rows]
    client = httpx.Client(timeout=120, limits=httpx.Limits(max_connections=args.concurrency))

    def call(body: dict) -> tuple[float, int]:
        t = time.perf_counter()
        status = client.post(f"{args.url}/v1/systemone", json=body).status_code
        return 1000 * (time.perf_counter() - t), status

    with ThreadPoolExecutor(args.concurrency) as pool:
        list(pool.map(call, bodies[: 2 * args.concurrency]))  # warm-up: graph captures for common shapes
        t = time.perf_counter()
        results = list(pool.map(call, bodies))
        wall = time.perf_counter() - t
    latencies = sorted(ms for ms, _ in results)
    errors = sum(status != 200 for _, status in results)
    print(f"concurrency {args.concurrency}: {len(bodies) / wall:.1f} req/s, "
          f"latency p50 {statistics.median(latencies):.0f} ms, "
          f"p90 {latencies[int(0.9 * len(latencies))]:.0f} ms, errors {errors}", flush=True)


if __name__ == "__main__":
    main()
