"""Where the fast Clef engine spends a batch: per-kernel CUDA time and the split between backbone, head and padding.

    python scripts/clef_engine_profile.py --records records.jsonl [--batch 64]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from verathos.engine import clef

REPO = "Cloudflare/clef-flash"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--revision", default="fde727a287004204b7518dcc983fe64379776712")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()
    rows = [json.loads(line) for line in args.records.open()][: args.batch]
    bodies = [{"model": "clef", "state": r["state"],
               "questions": {q: {k: v for k, v in x.items() if k != "label"} for q, x in r["questions"].items()}}
              for r in rows]
    engine = clef.load(REPO, args.revision, device="cuda", dtype="bfloat16", fast=True)
    items = [engine.prepare(b) for b in bodies]
    real = sum(len(i.encoded.input_ids) for i in items)
    padded = sum(len(g) * -(-max(len(items[i].encoded.input_ids) for i in g) // engine.length_step) * engine.length_step
                 for g in engine._batches(items))
    print(f"{len(items)} requests, {real} real tokens, {padded} padded tokens ({padded / real - 1:.1%} padding), "
          f"{len(engine._batches(items))} passes", flush=True)
    with torch.inference_mode():
        engine.run(items)
        torch.cuda.synchronize()
        t = time.perf_counter()
        engine.run(items)
        torch.cuda.synchronize()
        total = time.perf_counter() - t
        head_time = 0.0
        for group in engine._batches(items):
            records = [items[i].encoded for i in group]
            length = -(-max(len(r.input_ids) for r in records) // engine.length_step) * engine.length_step
            ids = torch.full((len(records), length), engine.pad_id, dtype=torch.long, device="cuda")
            mask = torch.zeros_like(ids)
            for row, r in enumerate(records):
                ids[row, : len(r.input_ids)] = torch.tensor(r.input_ids, device="cuda")
                mask[row, : len(r.input_ids)] = 1
            hidden = engine.backbone.forward(ids)
            torch.cuda.synchronize()
            t = time.perf_counter()
            engine.head(hidden, ids, mask, records, engine.output_embedding)
            torch.cuda.synchronize()
            head_time += time.perf_counter() - t
        print(f"batch of {len(items)}: {1000 * total:.0f} ms total ({len(items) / total:.1f} req/s), "
              f"head {1000 * head_time:.0f} ms", flush=True)
        group = engine._batches(items)[len(engine._batches(items)) // 2]
        records = [items[i].encoded for i in group]
        length = -(-max(len(r.input_ids) for r in records) // engine.length_step) * engine.length_step
        ids = torch.full((len(records), length), engine.pad_id, dtype=torch.long, device="cuda")
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(3):
                engine.backbone.forward(ids)
            torch.cuda.synchronize()
        print(f"backbone {len(records)} x {length}:")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=22, max_name_column_width=70))


if __name__ == "__main__":
    main()
