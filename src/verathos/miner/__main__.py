"""Run a miner: ``verathos-miner --config miner.json``.

On start the miner measures the capacity of its endpoints with each model's standard request mix (all endpoints at
once), declares it in its manifest, publishes, and serves. Measurements are stored and reused until an endpoint's
engine, configuration or request mix changes, or ``--remeasure`` is given.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path

import bittensor as bt
import uvicorn

from verathos import catalog
from verathos.miner.admission import Admission
from verathos.miner.app import create_app
from verathos.miner.chain import PermitAuthorizer, build_manifest, publish
from verathos.miner.config import EndpointConfig, MinerConfig
from verathos.miner.engine import HttpEngine
from verathos.miner.status import StatusWatcher
from verathos.protocol.manifest import SignedManifest
from verathos.subnet.chain import committed_catalog, current_catalog
from verathos.throughput.chain import SubtensorChainView
from verathos.throughput.measure import (
    Measurement,
    MeasurementFailed,
    MeasurementStore,
    Target,
    current_measurements,
    fingerprint,
)
from verathos.throughput.policy import POLICY_WATCH_S

log = logging.getLogger("verathos.miner")


def servable(config: MinerConfig, cat: catalog.Catalog) -> list[EndpointConfig]:
    """The configured endpoints whose model the catalog serves; the others are withdrawn from the manifest."""
    out = []
    for e in config.endpoints:
        model = cat.get(e.model_id)
        if model is None or model.status == "retired":
            log.error("endpoint %s: model %r is not served by catalog %s; endpoint withdrawn", e.endpoint_id,
                      e.model_id, cat.digest()[:12])
            continue
        if model.status != "active":
            log.warning("endpoint %s: %s is a %s model; validators check it but do not score it", e.endpoint_id,
                        e.model_id, model.status)
        out.append(e)
    return out


async def measure(endpoints: list[EndpointConfig], cat: catalog.Catalog, store: MeasurementStore, *,
                  remeasure: bool = False) -> dict[str, Measurement]:
    engines = {e.endpoint_id: HttpEngine(e.engine_url, e.engine_api_key) for e in endpoints}
    try:
        targets, max_latency_s = [], cat.audit_policy().max_latency_s
        for e in endpoints:
            workload = cat.get(e.model_id).workload
            config = {"model_id": e.model_id, "engine_url": e.engine_url, "max_concurrency": e.max_concurrency,
                      "max_latency_s": max_latency_s}
            targets.append(Target(e.endpoint_id, e.model_id, engines[e.endpoint_id], workload, e.max_concurrency,
                                  fingerprint(config, await engines[e.endpoint_id].info(), workload)))
        return await current_measurements(targets, store, max_latency_s=max_latency_s, remeasure=remeasure)
    finally:
        for engine in engines.values():
            await engine.close()


class Publisher:
    """The current signed manifest, rebuilt and re-published when the subnet owner commits a new catalog (measuring
    again first if the catalog changed an endpoint's request mix)."""

    def __init__(self, config: MinerConfig, wallet: bt.Wallet, cat: catalog.Catalog, *, publish_on_chain: bool,
                 subtensor: bt.Subtensor, remeasure: bool) -> None:
        self.config, self.wallet, self.publish_on_chain = config, wallet, publish_on_chain
        self.store = MeasurementStore(config.state_dir / "capacity.json")
        self.admissions = {e.endpoint_id: Admission(e.max_concurrency) for e in config.endpoints}
        self.catalog, self.signed = cat, self._sign(cat, remeasure=remeasure)
        self.auditor = self._auditor(subtensor)
        if publish_on_chain:
            publish(subtensor, self.wallet, self.config, self.signed)
        self._published = True
        self.on_catalog: list[Callable[[catalog.Catalog], None]] = []

    def _sign(self, cat: catalog.Catalog, *, remeasure: bool = False) -> SignedManifest:
        endpoints = servable(self.config, cat)
        if not endpoints:
            raise SystemExit(f"no configured endpoint serves a model of catalog {cat.digest()[:12]}")
        try:
            measured = asyncio.run(measure(endpoints, cat, self.store, remeasure=remeasure))
        except MeasurementFailed as exc:
            raise SystemExit(str(exc)) from None
        margin = cat.audit_policy().capacity_margin
        for e in endpoints:
            m = measured[e.endpoint_id]
            self.admissions[e.endpoint_id].limit = m.concurrency
            log.info("endpoint %s (%s): %.1f requests/s at concurrency %d (runs %s), declares %.1f", e.endpoint_id,
                     e.model_id, m.measured_rps, m.concurrency, ", ".join(f"{r:.1f}" for r in m.runs),
                     m.declared(margin))
        return build_manifest(self.config, endpoints, {k: m.declared(margin) for k, m in measured.items()},
                              self.wallet.hotkey, cat.digest())

    def _auditor(self, st: bt.Subtensor) -> str:
        return self.catalog.audit_policy().auditor_hotkey or st.get_subnet_owner_hotkey(self.config.netuid)

    def watch(self, interval_s: float = POLICY_WATCH_S) -> None:
        """Follow the owner's catalog commitment. A new catalog is adopted (manifest, audit policy) at once; its
        on-chain publication is retried until it succeeds."""

        def run() -> None:
            st = bt.Subtensor(network=self.config.network)
            store = catalog.CatalogStore(self.config.catalog_dir)
            while True:
                time.sleep(interval_s)
                try:
                    if committed_catalog(st, self.config.netuid) != self.catalog.digest():
                        cat = current_catalog(st, self.config.netuid, store, self.config.catalog_mirrors)
                        self.catalog, self.signed, self._published = cat, self._sign(cat), False
                        log.info("catalog %s committed; manifest rebuilt", cat.digest()[:12])
                        for callback in self.on_catalog:
                            callback(cat)
                    self.auditor = self._auditor(st)
                    if self.publish_on_chain and not self._published:
                        publish(st, self.wallet, self.config, self.signed)
                        self._published = True
                        log.info("manifest published")
                except SystemExit as exc:
                    log.critical("%s; stopping", exc)
                    os._exit(1)
                except Exception:
                    log.exception("catalog follow failed; retrying")

        threading.Thread(target=run, name="catalog-watch", daemon=True).start()


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
    logging.getLogger("httpx").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--remeasure", action="store_true",
                        help="Measure every endpoint's capacity again (after changing its hardware)")
    parser.add_argument("--no-publish", action="store_true",
                        help="Use the shipped catalog and write nothing on chain (local testing)")
    args = parser.parse_args()

    config = MinerConfig.load(args.config)
    wallet = bt.Wallet(name=config.wallet_name, hotkey=config.hotkey_name)
    subtensor = bt.Subtensor(network=config.network)
    cat = catalog.load() if args.no_publish else resolve_catalog(subtensor, config)
    publisher = Publisher(config, wallet, cat, publish_on_chain=not args.no_publish, subtensor=subtensor,
                          remeasure=args.remeasure)
    if not args.no_publish:
        publisher.watch()
        chain = SubtensorChainView(config.network, config.netuid, cat.audit_policy().auditor_hotkey)
        publisher.on_catalog.append(lambda c: chain.set_auditor_hotkey(c.audit_policy().auditor_hotkey))
        StatusWatcher(hotkey=wallet.hotkey.ss58_address, auditor_url=chain.auditor_url,
                      listed=lambda: [e.endpoint_id for e in publisher.signed.manifest.endpoints]).start()

    authorizer = PermitAuthorizer(subtensor, config.netuid, config.gateway_hotkeys)
    engines = {e.endpoint_id: HttpEngine(e.engine_url, e.engine_api_key) for e in config.endpoints}
    app = create_app(config=config, keypair=wallet.hotkey, manifest=lambda: publisher.signed, engines=engines,
                     authorized=authorizer, auditor=lambda: publisher.auditor, admissions=publisher.admissions)
    log.info("serving %d endpoints on port %d; throughput tests are accepted from %s; status is logged once per "
             "epoch", len(publisher.signed.manifest.endpoints), config.port, publisher.auditor)
    uvicorn.run(app, host=config.listen_host, port=config.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
