"""Clef engine benchmark: the release's reference (float32 and bfloat16) against the fast engine on the same requests.

    python scripts/clef_engine_bench.py --records records.jsonl [--drift 300] [--throughput 1000]

``records.jsonl`` holds SystemOne requests, one per line (``state`` and ``questions``; labels are ignored). Reports per
path the single-request latency and the batched throughput, and for the bfloat16 paths the deviation of every option
probability from the float32 reference: maximum, 99th percentile, mean, and top answers that changed where the
reference's margin was at least 0.2.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import torch

from verathos.engine import clef
from verathos.protocol.api import distributions

REPO = "Cloudflare/clef-flash"


def requests(path: Path, n: int) -> list[dict]:
    rows = [json.loads(line) for line in path.open() if line.strip()][:n]
    return [{"model": "clef", "state": r["state"],
             "questions": {q: {k: v for k, v in x.items() if k != "label"} for q, x in r["questions"].items()}}
            for r in rows]


def drift(name: str, got: list[dict], ref: list[dict]) -> None:
    diffs, flips = [], 0
    for g, r in zip(got, ref):
        for qid, rd in r.items():
            gd = g[qid]
            diffs += [abs(rd[k] - gd[k]) for k in rd]
            ranked = sorted(rd.values(), reverse=True)
            margin = ranked[0] - (ranked[1] if len(ranked) > 1 else 0.0)
            if max(gd, key=gd.get) != max(rd, key=rd.get) and margin >= 0.2:
                flips += 1
    diffs.sort()
    print(f"  drift {name}: probabilities {len(diffs)}, max {diffs[-1]:.4f}, p99 {diffs[int(0.99 * len(diffs))]:.4f}, "
          f"mean {statistics.mean(diffs):.5f}, confident flips {flips}", flush=True)


def latency(engine, items: list) -> float:
    for item in items[:3]:
        engine.run([item])
    torch.cuda.synchronize()
    times = []
    for item in items[3:43]:
        t = time.perf_counter()
        engine.run([item])
        times.append(time.perf_counter() - t)
    return 1000 * statistics.median(times)


def throughput(engine, items: list, batch: int) -> float:
    for start in range(0, len(items), batch):  # first pass captures the CUDA graphs these shapes need
        engine.run(items[start : start + batch])
    torch.cuda.synchronize()
    t = time.perf_counter()
    for start in range(0, len(items), batch):
        engine.run(items[start : start + batch])
    torch.cuda.synchronize()
    return len(items) / (time.perf_counter() - t)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--revision", default="fde727a287004204b7518dcc983fe64379776712")
    ap.add_argument("--drift", type=int, default=300)
    ap.add_argument("--throughput", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--paths", default="reference float32,reference bfloat16,fast bfloat16")
    ap.add_argument("--reference-cache", type=Path, default=Path("clef-reference-float32.json"),
                    help="float32 reference answers, reused when present for the same requests")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    bodies = requests(args.records, max(args.drift, args.throughput))
    print(f"{torch.cuda.get_device_name(0)}, {len(bodies)} requests", flush=True)

    results = {}
    if args.reference_cache.exists():
        cached = json.loads(args.reference_cache.read_text())
        if cached["requests"] == bodies[: args.drift]:
            results["reference float32"] = cached["answers"]
    paths = (("reference float32", "float32", False), ("reference bfloat16", "bfloat16", False),
             ("fast bfloat16", "bfloat16", True))
    for name, dtype, fast in paths:
        if name not in args.paths.split(",") or name in results:
            continue
        engine = clef.load(REPO, args.revision, device="cuda", dtype=dtype, fast=fast)
        items = [engine.prepare(b) for b in bodies]
        with torch.inference_mode():
            answers = []
            for start in range(0, args.drift, args.batch):
                chunk = items[start : min(start + args.batch, args.drift)]
                answers += [distributions(r["answers"]) for r in engine.run(chunk)]
            results[name] = answers
            if dtype != "float32":  # the float32 reference only supplies the answers drift is measured against
                ms = latency(engine, items)
                rps = throughput(engine, items[: args.throughput], args.batch if fast else 1)
                print(f"{name}: latency p50 {ms:.1f} ms, throughput {rps:.1f} req/s, "
                      f"peak memory {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
        if name == "reference float32":
            args.reference_cache.write_text(json.dumps({"requests": bodies[: args.drift], "answers": answers}))
        else:
            drift(f"{name} vs float32", answers, results["reference float32"])
        del engine, items
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    main()
