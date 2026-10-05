"""Run a miner: ``python -m decide.miner --config miner.json``."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import bittensor as bt
import uvicorn

from decide import catalog
from decide.miner.app import create_app
from decide.miner.chain import PermitAuthorizer, build_manifest, publish
from decide.miner.config import MinerConfig
from decide.miner.engine import HttpEngine


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--no-publish", action="store_true", help="Do not touch the chain (local testing)")
    args = parser.parse_args()

    config = MinerConfig.load(args.config)
    cat = catalog.load()
    for e in config.endpoints:
        if cat.get(e.model_id) is None:
            raise SystemExit(f"endpoint {e.endpoint_id}: model {e.model_id!r} is not in the catalog")
    wallet = bt.Wallet(name=config.wallet_name, hotkey=config.hotkey_name)
    keypair = wallet.hotkey
    subtensor = bt.Subtensor(network=config.network)
    signed = build_manifest(config, keypair, cat.digest(), subtensor.get_current_block())
    if not args.no_publish:
        publish(subtensor, wallet, config, signed)
    authorizer = PermitAuthorizer(subtensor, config.netuid, config.gateway_hotkeys)
    engines = {e.endpoint_id: HttpEngine(e.engine_url, e.engine_api_key) for e in config.endpoints}
    app = create_app(config=config, keypair=keypair, manifest=lambda: signed, engines=engines,
                     authorized=authorizer, catalog_digest=cat.digest())
    uvicorn.run(app, host=config.listen_host, port=config.port, log_level="info")


if __name__ == "__main__":
    main()
