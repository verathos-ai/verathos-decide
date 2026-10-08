"""The batched Clef head against the release's per-record head on the same backbone hidden states.

    python scripts/clef_head_parity.py --records records.jsonl [--n 64]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from verathos.engine import clef
from verathos.engine.clef_head import batched_head

REPO = "Cloudflare/clef-flash"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--revision", default="fde727a287004204b7518dcc983fe64379776712")
    ap.add_argument("--n", type=int, default=64)
    args = ap.parse_args()
    rows = [json.loads(line) for line in args.records.open()][: args.n]
    bodies = [{"model": "clef", "state": r["state"],
               "questions": {q: {k: v for k, v in x.items() if k != "label"} for q, x in r["questions"].items()}}
              for r in rows]
    engine = clef.load(REPO, args.revision, device="cuda", dtype="bfloat16", fast=True)
    items = [engine.prepare(b) for b in bodies]
    worst, flips, ref_time, new_time = 0.0, 0, 0.0, 0.0
    with torch.inference_mode():
        for group in engine._batches(items):
            records = [items[i].encoded for i in group]
            length = -(-max(len(r.input_ids) for r in records) // engine.length_step) * engine.length_step
            # one padding row after the records, as a bucketed pass has
            ids = torch.full((len(records) + 1, length), engine.pad_id, dtype=torch.long, device="cuda")
            mask = torch.zeros((len(records), length), dtype=torch.long, device="cuda")
            for row, r in enumerate(records):
                ids[row, : len(r.input_ids)] = torch.tensor(r.input_ids, device="cuda")
                mask[row, : len(r.input_ids)] = 1
            hidden = engine.backbone.forward(ids)
            torch.cuda.synchronize()
            t = time.perf_counter()
            ref = engine.head(hidden, ids, mask, records, engine.output_embedding)
            torch.cuda.synchronize()
            ref_time += time.perf_counter() - t
            t = time.perf_counter()
            new = batched_head(engine.head, hidden, ids, mask.sum(1), records, engine.output_embedding)
            torch.cuda.synchronize()
            new_time += time.perf_counter() - t
            for rq, nq in zip(ref, new):
                for a, b in zip(rq, nq):
                    pa, pb = a.float().softmax(-1), b.float().softmax(-1)
                    worst = max(worst, (pa - pb).abs().max().item())
                    flips += int(pa.argmax() != pb.argmax())
    print(f"{len(items)} requests: max |dp| {worst:.5f}, argmax changes {flips}; "
          f"release head {1000 * ref_time:.0f} ms, batched head {1000 * new_time:.0f} ms")


if __name__ == "__main__":
    main()
