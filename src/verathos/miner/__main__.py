"""Run a miner: ``verathos-miner --config miner.json``."""

from __future__ import annotations

import argparse
import logging
import threading
import time
from pathlib import Path

import bittensor as bt
import uvicorn

from verathos import catalog
from verathos.capacity.chain import SubtensorChainView
from verathos.capacity.core import AuditPolicy, gpu_support_status
from verathos.capacity.miner import AuditEndpoint, AuditWorker, DrainState, bench_extension_available
from verathos.miner.app import create_app
from verathos.miner.chain import PermitAuthorizer, build_manifest, publish
from verathos.miner.config import MinerConfig
from verathos.miner.engine import HttpEngine
from verathos.subnet.chain import committed_catalog, current_catalog

log = logging.getLogger("verathos.miner")


def audit_endpoints(config: MinerConfig, policy: AuditPolicy) -> list[AuditEndpoint]:
    """Open-tier CUDA endpoints with a declared GPU class and device: the ones the capacity audit covers."""
    out = []
    for e in config.endpoints:
        hw = e.hardware
        if e.tier != "open" or hw.kind != "cuda":
            continue
        if not hw.gpu_name or not hw.vram_gb or e.cuda_device is None:
            raise SystemExit(f"endpoint {e.endpoint_id}: CUDA endpoints need hardware.gpu_name, hardware.vram_gb "
                             "and cuda_device for the capacity audit")
        if not gpu_support_status(hw.gpu_name, hw.vram_gb, policy)[0]:
            log.warning("endpoint %s: %s (%d GB) is not a calibrated GPU class; it is never audited and scores zero "
                        "once the audit gates", e.endpoint_id, hw.gpu_name, hw.vram_gb)
        out.append(AuditEndpoint(e.endpoint_id, config.endpoint_url(e.endpoint_id), e.model_id, hw.gpu_name,
                                 hw.vram_gb, e.cuda_device))
    return out


class Publisher:
    """The current signed manifest, rebuilt and re-published when the subnet owner commits a new catalog."""

    def __init__(self, config: MinerConfig, wallet: bt.Wallet, cat: catalog.Catalog, *, publish_on_chain: bool,
                 subtensor: bt.Subtensor) -> None:
        self.config, self.wallet, self.publish_on_chain = config, wallet, publish_on_chain
        self.catalog = cat
        self.signed = self._build(subtensor)
        self.on_catalog: list = []

    def _build(self, subtensor: bt.Subtensor):
        check_models(self.config, self.catalog)
        signed = build_manifest(self.config, self.wallet.hotkey, self.catalog.digest(), subtensor.get_current_block())
        if self.publish_on_chain:
            publish(subtensor, self.wallet, self.config, signed)
        return signed

    def watch(self, interval_s: float = 600.0) -> None:
        def run() -> None:
            st = bt.Subtensor(network=self.config.network)
            store = catalog.CatalogStore(self.config.catalog_dir)
            while True:
                time.sleep(interval_s)
                try:
                    if committed_catalog(st, self.config.netuid) == self.catalog.digest():
                        continue
                    self.catalog = current_catalog(st, self.config.netuid, store, self.config.catalog_mirrors)
                    self.signed = self._build(st)
                    log.info("catalog %s committed; manifest re-published", self.catalog.digest()[:12])
                    for callback in self.on_catalog:
                        callback(self.catalog)
                except (catalog.CatalogUnavailable, SystemExit, RuntimeError) as exc:
                    log.error("new catalog not applied: %s", exc)
                except Exception:
                    log.exception("catalog watch failed")

        threading.Thread(target=run, name="catalog-watch", daemon=True).start()


def check_models(config: MinerConfig, cat: catalog.Catalog) -> None:
    for e in config.endpoints:
        model = cat.get(e.model_id)
        if model is None or model.status == "retired":
            raise SystemExit(f"endpoint {e.endpoint_id}: model {e.model_id!r} is not served by the current catalog")
        if model.status != "active":
            log.warning("endpoint %s: %s is a %s model; validators check it but do not score it", e.endpoint_id,
                        e.model_id, model.status)


def resolve_catalog(subtensor: bt.Subtensor, config: MinerConfig) -> catalog.Catalog:
    store = catalog.CatalogStore(config.catalog_dir)
    while True:
        try:
            return current_catalog(subtensor, config.netuid, store, config.catalog_mirrors)
        except catalog.CatalogUnavailable as exc:
            log.error("%s; retrying in 60 s", exc)
            time.sleep(60)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--no-publish", action="store_true",
                        help="Use the shipped catalog and write nothing on chain (local testing)")
    args = parser.parse_args()

    config = MinerConfig.load(args.config)
    wallet = bt.Wallet(name=config.wallet_name, hotkey=config.hotkey_name)
    subtensor = bt.Subtensor(network=config.network)
    cat = catalog.load() if args.no_publish else resolve_catalog(subtensor, config)
    publisher = Publisher(config, wallet, cat, publish_on_chain=not args.no_publish, subtensor=subtensor)
    if not args.no_publish:
        publisher.watch()
    policy = cat.audit_policy()

    drain = DrainState()
    endpoints = audit_endpoints(config, policy)
    if endpoints:
        ok, error = bench_extension_available()
        if ok:
            chain = SubtensorChainView(config.network, config.netuid, policy.auditor_hotkey)
            worker = AuditWorker(netuid=config.netuid, keypair=wallet.hotkey, endpoints=endpoints, policy=policy,
                                 chain=chain, drain=drain)

            def adopt(cat: catalog.Catalog) -> None:
                worker.set_policy(cat.audit_policy())
                chain.set_auditor_hotkey(cat.audit_policy().auditor_hotkey)

            publisher.on_catalog.append(adopt)
            worker.start()
        elif policy.mode == "observe":
            log.warning("capacity audit bench extension unavailable (%s); audits will be missed", error[-200:])
        else:
            raise SystemExit(f"capacity audit bench extension unavailable: {error[-300:]}")

    authorizer = PermitAuthorizer(subtensor, config.netuid, config.gateway_hotkeys)
    engines = {e.endpoint_id: HttpEngine(e.engine_url, e.engine_api_key) for e in config.endpoints}
    app = create_app(config=config, keypair=wallet.hotkey, manifest=lambda: publisher.signed, engines=engines,
                     authorized=authorizer, drained=drain.drained)
    uvicorn.run(app, host=config.listen_host, port=config.port, log_level="info")


if __name__ == "__main__":
    main()
