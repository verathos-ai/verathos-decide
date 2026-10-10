"""Serve one decision model over ``POST /v1/systemone``.

    verathos-engine clef --repo Cloudflare/clef-flash --revision <sha> --port 8010
    verathos-engine clef --repo Cloudflare/clef-flash --revision <sha> --port 8110 --device cpu --dtype float32

Take ``--repo`` and ``--revision`` from the model's catalog entry (``hf_repo``, ``hf_revision``). Miners run their
GPU in bfloat16 (the batched fast path); validators run the release's reference code in float32.
"""

from __future__ import annotations

import argparse
import logging

import uvicorn

from verathos import stop
from verathos.engine import code_version
from verathos.engine.server import create_app
from verathos.protocol.media import keep_out_of_dumps

FAMILIES = ("clef",)
log = logging.getLogger("verathos.engine")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("family", choices=FAMILIES)
    parser.add_argument("--repo", required=True, help="Hugging Face repository (catalog hf_repo)")
    parser.add_argument("--revision", required=True, help="exact commit (catalog hf_revision)")
    parser.add_argument("--device", default="cuda", help="cuda, cuda:N, cpu, or auto (cuda when a GPU is visible)")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()
    stop.on_signals("the engine", log)
    keep_out_of_dumps()  # users' requests are in memory, their images among them

    from verathos.engine import clef

    if args.device == "auto":
        import torch

        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = clef.load(args.repo, args.revision, device=args.device, dtype=args.dtype)
    served = {"id": args.repo, "revision": args.revision, "device": args.device, "dtype": args.dtype,
              "code": code_version()}
    uvicorn.run(create_app(engine, [served]), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
