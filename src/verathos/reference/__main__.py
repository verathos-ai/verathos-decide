"""Run a reference machine's front: ``verathos-reference --config reference.json``.

Its engines run as services of their own; the front serves them to the callers (``reference.front``).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from verathos.protocol import tls
from verathos.reference import front
from verathos.reference.config import ReferenceConfig, api_key

log = logging.getLogger("verathos.reference")


def main() -> None:
    if sys.argv[1:2] in (["init"], ["uninstall"]):
        from verathos.install import common

        common.main("reference", sys.argv[1], sys.argv[2:])
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(epilog="Set up this machine as a reference machine: verathos-reference init. "
                                            "Remove it: verathos-reference uninstall.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config = ReferenceConfig.load(args.config)
    identity = tls.identity(config.state_dir)
    server, listening = front.bind(config, identity, api_key(config.state_dir))
    log.info("front on port %d, TLS certificate pin %s: %s; callers %s (other addresses are dropped)", config.port,
             identity.pin, ", ".join(f"{m} {kind}" for kind, m in config.served()) or "no engine",
             ", ".join(config.allow) or "none")
    server.run(sockets=[listening])  # until a signal stops it
    sys.stdout.flush()
    os._exit(0 if server.started else 1)


if __name__ == "__main__":
    main()
