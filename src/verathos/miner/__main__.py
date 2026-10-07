"""Run a miner: ``verathos-miner --config miner.json``.

The miner starts its service, measures the capacity of its endpoints through it with each model's standard request
mix (all endpoints at once), declares it in its manifest, and publishes. Measurements are stored and reused until an
endpoint's engine, configuration, model revision or request mix changes, or ``--remeasure`` is given. An endpoint that
cannot be measured is withdrawn; the others are served. Until the first measurement is done, the service serves the
manifest the miner last published.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import logging
import os
import resource
import threading
import time
from collections.abc import Callable
from pathlib import Path

import bittensor as bt
import httpx
import uvicorn

from verathos import catalog
from verathos.miner.admission import Admission
from verathos.miner.app import create_app
from verathos.miner.chain import PermitAuthorizer, build_manifest, publish
from verathos.miner.config import EndpointConfig, MinerConfig
from verathos.miner.engine import HttpEngine
from verathos.miner.status import StatusWatcher
from verathos.protocol.manifest import SignedManifest, axon_base
from verathos.subnet.chain import committed_catalog, current_catalog
from verathos.throughput.chain import SubtensorChainView
from verathos.throughput.measure import (
    Measurement,
    MeasurementFailed,
    MeasurementInterrupted,
    MeasurementStore,
    ServicePath,
    Target,
    fingerprint,
    measure_and_store,
    stored,
    wait_ready,
)
from verathos.throughput.policy import POLICY_WATCH_S

log = logging.getLogger("verathos.miner")
LOCAL_HOSTS = {"0.0.0.0": "127.0.0.1", "": "127.0.0.1", "::": "::1"}


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


async def engine_fingerprint(e: EndpointConfig, cat: catalog.Catalog) -> str:
    """What the endpoint's measurement depends on: its configuration, the model's pinned revisions, the engine's
    self-description and the request mix."""
    engine = HttpEngine(e.engine_url, e.engine_api_key)
    try:
        info = await engine.info()
    finally:
        await engine.close()
    model = cat.get(e.model_id)
    config = {"model_id": e.model_id, "engine_url": e.engine_url, "max_concurrency": e.max_concurrency,
              "max_latency_s": cat.audit_policy().max_latency_s, "hf_revision": model.hf_revision,
              "engine_revision": model.engine.revision}
    return fingerprint(config, info, model.workload)


def raise_open_file_limit() -> int:
    """Raise the soft open-file limit to the hard one (every connection holds a descriptor); returns it."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < hard:
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    return resource.getrlimit(resource.RLIMIT_NOFILE)[0]


class Publisher:
    """The current signed manifest: the endpoints whose capacity is measured, rebuilt when the subnet owner commits a
    new catalog or an engine changes, and published on chain."""

    def __init__(self, config: MinerConfig, wallet: bt.Wallet, cat: catalog.Catalog, *, publish_on_chain: bool,
                 subtensor: bt.Subtensor) -> None:
        self.config, self.wallet, self.publish_on_chain = config, wallet, publish_on_chain
        self.store = MeasurementStore(config.state_dir / "capacity.json")
        self.manifest_path = config.state_dir.expanduser() / "manifest.json"
        self.admissions = {e.endpoint_id: Admission(e.max_concurrency) for e in config.endpoints}
        endpoints = servable(config, cat)
        if not endpoints:
            raise SystemExit(f"no configured endpoint serves a model of catalog {cat.digest()[:12]}")
        self.catalog = cat
        # Until the first measurement: the manifest last published (so validators find the miner as committed), or
        # an unpublished one, so the service can be measured through.
        self.signed = self._last_published() or build_manifest(
            config, endpoints, dict.fromkeys((e.endpoint_id for e in endpoints), 0.0), wallet.hotkey, cat.digest())
        self.serving = {e.endpoint_id for e in endpoints}
        self.auditor = subtensor.get_subnet_owner_hotkey(config.netuid)  # the subnet owner's validator runs tests
        self.test_seen = 0.0  # monotonic time the auditor's test traffic last arrived
        self._published = False

    # ---- the auditor ----

    def auditors(self) -> set[str]:
        return {self.auditor}

    def note_test_traffic(self) -> None:
        self.test_seen = time.monotonic()

    # ---- measuring ----

    def _last_published(self) -> SignedManifest | None:
        try:
            signed = SignedManifest.model_validate_json(self.manifest_path.read_text())
        except (OSError, ValueError):
            return None
        m = signed.manifest
        same = (m.hotkey, m.netuid, m.mechid) == (self.wallet.hotkey.ss58_address, self.config.netuid,
                                                  self.config.mechid)
        return signed if same else None

    def _local_url(self, endpoint_id: str) -> str:
        host = LOCAL_HOSTS.get(self.config.listen_host, self.config.listen_host)
        return f"{axon_base(host, self.config.port)}/e/{endpoint_id}/v1/systemone"

    async def _measure(self, endpoints: list[EndpointConfig], cat: catalog.Catalog, remeasure: bool,
                       stop: Callable[[], bool]) -> tuple[dict[str, Measurement], dict[str, str]]:
        limits = httpx.Limits(max_connections=None, max_keepalive_connections=1024)
        async with httpx.AsyncClient(timeout=120.0, limits=limits) as client:
            targets = [Target(e.endpoint_id, e.model_id,
                              ServicePath(client, self._local_url(e.endpoint_id), self.wallet.hotkey,
                                          self.config.netuid, e.endpoint_id),
                              cat.get(e.model_id).workload, e.max_concurrency, "") for e in endpoints]
            failures = {}
            for t, outcome in zip(targets, await asyncio.gather(*(wait_ready(t) for t in targets),
                                                                return_exceptions=True)):
                if isinstance(outcome, MeasurementFailed):
                    failures[t.endpoint_id] = str(outcome)
                elif isinstance(outcome, BaseException):
                    raise outcome
            by_id = {e.endpoint_id: e for e in endpoints}
            targets = [dataclasses.replace(t, fingerprint=await engine_fingerprint(by_id[t.endpoint_id], cat))
                       for t in targets if t.endpoint_id not in failures]
            if not targets:
                return {}, failures
            reuse = None if remeasure else stored(targets, self.store)
            if reuse is not None:
                return reuse, failures
            measured, failed = await measure_and_store(targets, self.store,
                                                       max_latency_s=cat.audit_policy().max_latency_s, stop=stop)
            return measured, failures | failed

    def adopt(self, cat: catalog.Catalog, *, remeasure: bool = False) -> bool:
        """Measure the endpoints ``cat`` serves (or reuse their stored measurements) through the running service and
        sign a manifest of those measured. A measurement steps aside when the auditor's test traffic arrives. False if
        none could be measured or the measurement stepped aside; the current manifest stays."""
        endpoints = servable(self.config, cat)
        if not endpoints:
            log.error("no configured endpoint serves a model of catalog %s", cat.digest()[:12])
            return False
        previous_serving = set(self.serving)
        previous_limits = {k: a.limit for k, a in self.admissions.items()}
        self.serving = previous_serving | {e.endpoint_id for e in endpoints}
        for e in endpoints:
            self.admissions[e.endpoint_id].limit = e.max_concurrency
        started = time.monotonic()
        try:
            measured, failures = asyncio.run(self._measure(endpoints, cat, remeasure,
                                                           stop=lambda: self.test_seen > started))
        except MeasurementInterrupted:
            log.info("a throughput test reached the miner; measuring again later")
            measured, failures = {}, {}
        for endpoint_id, reason in sorted(failures.items()):
            log.error("endpoint %s withdrawn: %s", endpoint_id, reason)
        listed = [e for e in endpoints if e.endpoint_id in measured]
        if not listed:
            self.serving = previous_serving
            for k, limit in previous_limits.items():
                self.admissions[k].limit = limit
            return False
        margin = cat.audit_policy().capacity_margin
        for e in listed:
            m = measured[e.endpoint_id]
            self.admissions[e.endpoint_id].limit = m.concurrency
            log.info("endpoint %s (%s): %.1f requests/s at concurrency %d (runs %s), declares %.1f", e.endpoint_id,
                     e.model_id, m.measured_rps, m.concurrency, ", ".join(f"{r:.1f}" for r in m.runs),
                     m.declared(margin))
        self.catalog = cat
        self.signed = build_manifest(self.config, listed, {e.endpoint_id: measured[e.endpoint_id].declared(margin)
                                                           for e in listed}, self.wallet.hotkey, cat.digest())
        self.serving = {e.endpoint_id for e in listed}
        self._published = False
        return True

    def engines_changed(self) -> bool:
        """Whether a listed endpoint's engine or configuration no longer matches its stored measurement."""
        listed = {e.endpoint_id for e in self.signed.manifest.endpoints}
        endpoints = [e for e in servable(self.config, self.catalog) if e.endpoint_id in listed]
        stored_now = self.store.load()

        async def check() -> bool:
            for e in endpoints:
                m = stored_now.get(e.endpoint_id)
                if m is None or m.fingerprint != await engine_fingerprint(e, self.catalog):
                    return True
            return False

        return asyncio.run(check())

    # ---- publishing ----

    def publish(self, st: bt.Subtensor) -> None:
        if self.publish_on_chain and not self._published:
            publish(st, self.wallet, self.config, self.signed)
            self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
            self.manifest_path.write_text(self.signed.model_dump_json())
        self._published = True

    def watch(self, interval_s: float = POLICY_WATCH_S) -> None:
        """Follow the owner's catalog commitment and the engines. A new catalog is adopted at once (measuring again
        if it changed what a measurement depends on), so is a changed engine; on-chain publication is retried until
        it succeeds."""

        def run() -> None:
            store = catalog.CatalogStore(self.config.catalog_dir)
            st: bt.Subtensor | None = None
            while True:
                time.sleep(interval_s)
                try:
                    st = st or bt.Subtensor(network=self.config.network)
                    if committed_catalog(st, self.config.netuid) != self.catalog.digest():
                        cat = current_catalog(st, self.config.netuid, store, self.config.catalog_mirrors)
                        if self.adopt(cat):
                            log.info("catalog %s committed; manifest rebuilt", cat.digest()[:12])
                        else:
                            log.error("catalog %s committed but not adopted yet; keeping the current manifest",
                                      cat.digest()[:12])
                    elif self.engines_changed():
                        log.info("an engine changed; measuring again")
                        self.adopt(self.catalog)
                    self.auditor = st.get_subnet_owner_hotkey(self.config.netuid)
                    if not self._published:
                        self.publish(st)
                        log.info("manifest published")
                except Exception:
                    log.exception("catalog follow failed; retrying")
                    st = None

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
    publisher = Publisher(config, wallet, cat, publish_on_chain=not args.no_publish, subtensor=subtensor)

    authorizer = PermitAuthorizer(config.network, config.netuid, config.gateway_hotkeys)
    engines = {e.endpoint_id: HttpEngine(e.engine_url, e.engine_api_key, max_connections=e.max_concurrency)
               for e in config.endpoints}

    def max_state_tokens(model_id: str) -> int | None:
        model = publisher.catalog.get(model_id)
        return model.max_state_tokens if model is not None else None

    app = create_app(config=config, keypair=wallet.hotkey, manifest=lambda: publisher.signed, engines=engines,
                     authorized=authorizer, auditors=publisher.auditors, admissions=publisher.admissions,
                     serving=lambda: publisher.serving, max_state_tokens=max_state_tokens,
                     on_test_traffic=publisher.note_test_traffic)
    # Connections beyond the descriptors left are refused, so a flood of idle connections cannot starve the rest.
    limit = max(256, raise_open_file_limit() - 256)
    server = uvicorn.Server(uvicorn.Config(app, host=config.listen_host, port=config.port, log_level="warning",
                                           access_log=False, limit_concurrency=limit))
    service = threading.Thread(target=server.run, name="service", daemon=True)
    service.start()
    while not server.started:
        if not service.is_alive():
            raise SystemExit(f"the service could not start on port {config.port}")
        time.sleep(0.1)

    if not publisher.adopt(cat, remeasure=args.remeasure):
        raise SystemExit("no endpoint could be measured")
    publisher.publish(subtensor)
    if not args.no_publish:
        publisher.watch()
        chain = SubtensorChainView(config.network, config.netuid)
        StatusWatcher(hotkey=wallet.hotkey.ss58_address, auditor_url=chain.auditor_url,
                      listed=lambda: [e.endpoint_id for e in publisher.signed.manifest.endpoints]).start()
    log.info("serving %d endpoints on port %d; throughput tests are accepted from %s; status is logged once per "
             "epoch", len(publisher.signed.manifest.endpoints), config.port, ", ".join(sorted(publisher.auditors())))
    service.join()
    os._exit(1)  # the service stopped


if __name__ == "__main__":
    main()
