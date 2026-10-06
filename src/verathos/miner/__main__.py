"""Run a miner: ``verathos-miner --config miner.json``."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import bittensor as bt
import uvicorn

from verathos import catalog
from verathos.capacity.chain import SubtensorChainView
from verathos.capacity.miner import AuditEndpoint, AuditWorker, DrainState, bench_extension_available
from verathos.chain import committed_catalog
from verathos.miner.app import create_app
from verathos.miner.chain import PermitAuthorizer, build_manifest, publish
from verathos.miner.config import MinerConfig
from verathos.miner.engine import HttpEngine

log = logging.getLogger("verathos.miner")


def audit_endpoints(config: MinerConfig) -> list[AuditEndpoint]:
    """Open-tier CUDA endpoints with a declared GPU class and device: the ones the capacity audit covers."""
    out = []
    for e in config.endpoints:
        hw = e.hardware
        if e.tier != "open" or hw.kind != "cuda":
            continue
        if not hw.gpu_name or not hw.vram_gb or e.cuda_device is None:
            raise SystemExit(f"endpoint {e.endpoint_id}: CUDA endpoints need hardware.gpu_name, hardware.vram_gb "
                             "and cuda_device for the capacity audit")
        out.append(AuditEndpoint(e.endpoint_id, config.endpoint_url(e.endpoint_id), e.model_id, hw.gpu_name,
                                 hw.vram_gb, e.cuda_device))
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--no-publish", action="store_true", help="Do not write axon or commitment (local testing)")
    args = parser.parse_args()

    config = MinerConfig.load(args.config)
    cat = catalog.load()
    for e in config.endpoints:
        if cat.get(e.model_id) is None:
            raise SystemExit(f"endpoint {e.endpoint_id}: model {e.model_id!r} is not in the catalog")
    policy = cat.audit_policy()
    wallet = bt.Wallet(name=config.wallet_name, hotkey=config.hotkey_name)
    keypair = wallet.hotkey
    subtensor = bt.Subtensor(network=config.network)
    committed = committed_catalog(subtensor, config.netuid)
    if committed != cat.digest() and not args.no_publish:
        raise SystemExit(f"the subnet owner commits catalog {committed}; this release has {cat.digest()}. "
                         "Update verathos.")
    signed = build_manifest(config, keypair, cat.digest(), subtensor.get_current_block())
    if not args.no_publish:
        publish(subtensor, wallet, config, signed)

    drain = DrainState()
    endpoints = audit_endpoints(config)
    if endpoints:
        ok, error = bench_extension_available()
        if ok:
            chain = SubtensorChainView(config.network, config.netuid, policy.auditor_hotkey)
            AuditWorker(netuid=config.netuid, keypair=keypair, endpoints=endpoints, policy=policy, chain=chain,
                        drain=drain).start()
        elif policy.mode == "observe":
            log.warning("capacity audit bench extension unavailable (%s); audits will be missed", error[-200:])
        else:
            raise SystemExit(f"capacity audit bench extension unavailable: {error[-300:]}")

    authorizer = PermitAuthorizer(subtensor, config.netuid, config.gateway_hotkeys)
    engines = {e.endpoint_id: HttpEngine(e.engine_url, e.engine_api_key) for e in config.endpoints}
    app = create_app(config=config, keypair=keypair, manifest=lambda: signed, engines=engines, authorized=authorizer,
                     catalog_digest=cat.digest(), drained=drain.drained)
    uvicorn.run(app, host=config.listen_host, port=config.port, log_level="info")


if __name__ == "__main__":
    main()
