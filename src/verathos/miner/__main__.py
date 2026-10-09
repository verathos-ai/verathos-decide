"""Run a miner: ``verathos-miner --config miner.json``, on every machine of a hotkey.

The miner starts its service, measures the capacity of its endpoints through it with each model's standard request
mix (all endpoints at once), and declares it. Measurements are stored and reused until an endpoint's engine,
configuration, model revision or request mix changes, or ``--remeasure`` is given. An endpoint whose last two
throughput tests fell short is measured again and declares at most what it delivered in them (``shortfall``). A
measurement waits for a selected test (``timing``). An endpoint that cannot be measured is withdrawn; the others are
served. The primary (``role``) signs the hotkey's manifest, its own endpoints and its peers' (``miner.peers``), and
publishes it; a worker serves its endpoints and what it declares (``GET /v1/endpoints``) and publishes nothing. Until
the first measurement is done, a machine serves what it last served.

The service speaks TLS only, with the machine's certificate (``protocol.tls``, kept in ``state_dir``), whose pin the
manifest names; the measurement reaches the endpoints through it as tests do. Before it starts, the firewall
(``miner.firewall``) lets only the network's callers and the hotkey's machines reach its port.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import logging
import os
import sys
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import bittensor as bt
import httpx
import uvicorn
from fastapi import FastAPI

from verathos import catalog, update
from verathos.miner import firewall
from verathos.miner.admission import Admission, RequestBudget
from verathos.miner.app import caller_request_bytes, create_app
from verathos.miner.chain import PermitAuthorizer, build_host_endpoints, build_manifest, own_endpoints, publish
from verathos.miner.config import EndpointConfig, MinerConfig
from verathos.miner.engine import Engine, HttpEngine
from verathos.miner.peers import Peers, merged, pins
from verathos.miner.shortfall import Shortfall, Shortfalls
from verathos.miner.status import StatusWatcher
from verathos.miner.timing import MeasurementDeferred, MeasurementWindow
from verathos.protocol import tls
from verathos.protocol.http import raise_open_file_limit
from verathos.protocol.manifest import (
    ENDPOINTS_PATH,
    SignedHostEndpoints,
    SignedManifest,
    axon_base,
    endpoint_host,
    is_public,
)
from verathos.protocol.media import ImageLimits
from verathos.protocol.status import ThroughputOutcome
from verathos.subnet.chain import CHAIN_ERRORS, committed_catalog, current_catalog, one_line
from verathos.throughput.chain import SubtensorChainView
from verathos.throughput.follow import ScheduleFollower
from verathos.throughput.measure import (
    STORE_FILE,
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
from verathos.throughput.policy import BLOCK_TIME_S, POLICY_WATCH_S

log = logging.getLogger("verathos.miner")
LOCAL_HOSTS = {"0.0.0.0": "127.0.0.1", "": "127.0.0.1", "::": "::1"}
TEST_QUIET_S = 60.0  # a measurement starts only this long after the last test request
TEST_ARRIVING_S = 5.0  # test traffic arrives while its last request came this recently (it sends one a second or more)


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
    engine = HttpEngine(e.engine_url, e.engine_api_key, tls_pin=e.engine_tls_pin)
    try:
        info = await engine.info()
    finally:
        await engine.close()
    model = cat.get(e.model_id)
    config = {"model_id": e.model_id, "engine_url": e.engine_url, "max_concurrency": e.max_concurrency,
              "max_latency_s": cat.audit_policy().max_latency_s, "hf_revision": model.hf_revision,
              "engine_revision": model.engine.revision}
    return fingerprint(config, info, model.workload)


class Publisher:
    """What this machine serves: its endpoints whose capacity is measured, and on the primary the hotkey's manifest,
    which lists them and the peers' (``miner.peers``). Rebuilt when the subnet owner commits a new catalog, an engine
    changes, an endpoint falls short in the throughput tests (``shortfall``) or a peer's endpoints change; the manifest
    is published on chain. With a ``window`` (``timing``), a measurement waits for a selected test."""

    def __init__(self, config: MinerConfig, wallet: bt.Wallet, cat: catalog.Catalog, *, publish_on_chain: bool,
                 subtensor: bt.Subtensor, identity: tls.Identity) -> None:
        self.config, self.wallet, self.identity = config, wallet, identity
        self.publish_on_chain = publish_on_chain and config.role == "primary"
        self.store = MeasurementStore(config.state_dir / STORE_FILE)
        self.shortfalls = Shortfalls(config.state_dir / "shortfalls.json")
        self.tests: list[ThroughputOutcome] = []  # the hotkey's latest test results, from a verified status
        state = config.state_dir.expanduser()
        self.manifest_path, self.endpoints_path = state / "manifest.json", state / "endpoints.json"
        self.admissions = {e.endpoint_id: Admission(e.max_concurrency) for e in config.endpoints}
        endpoints = servable(config, cat)
        if not endpoints:
            raise SystemExit(f"no configured endpoint serves a model of catalog {cat.digest()[:12]}")
        self.catalog = cat
        # Until the first measurement: what this machine last served (so validators find the miner as committed and a
        # primary keeps a worker's endpoints), or an unpublished manifest, so the service can be measured through.
        self.local = self._last_served()
        self.peers: Peers | None = None
        self.signed: SignedManifest | None = None
        if config.role == "primary":
            self.peers = Peers(config.peers, hotkey=wallet.hotkey.ss58_address, netuid=config.netuid,
                               mechid=self.mechid)
            self.signed = self._last_published() or build_manifest(
                config, own_endpoints(config, endpoints, dict.fromkeys((e.endpoint_id for e in endpoints), 0.0)),
                wallet.hotkey, cat, {config.base_url: identity.pin})
            self.peers.seed(self.signed.manifest)
        self.serving = {e.endpoint_id for e in endpoints}
        self.auditor = subtensor.get_subnet_owner_hotkey(config.netuid)  # the subnet owner's validator runs tests
        self.test_seen = float("-inf")  # monotonic time the auditor's test traffic last arrived
        self.interrupted = False  # the last adopt stepped aside for a test, or waits for one
        self.window: MeasurementWindow | None = None
        self._deferral: int | None = None  # the epoch of the test a measurement waits for, as logged
        self._announced: set[object] = set()  # why the waiting measurement is wanted, as logged
        self._published = False
        self._left_out: list[str] = []
        self.updater: update.Updater | None = None

    @property
    def mechid(self) -> int:
        """The mechanism this machine serves for: its catalog's, or the configured ``mechid``."""
        return catalog.mechanism_of(self.catalog, self.config.mechid)

    # ---- callers ----

    def auditors(self) -> set[str]:
        return {self.auditor}

    def catalog_auditor_pin(self) -> str:
        """The pin of the auditor's certificate, as the catalog names it. Raises catalog.AuditorUnknown."""
        return self.catalog.auditor_pin()

    def gateways(self) -> set[str]:
        """Callers served without the per-caller limits: the catalog's gateways and the configured ones."""
        return {*(g.hotkey for g in self.config.gateways), *self.catalog.gateway_hotkeys()}

    def note_test_traffic(self) -> None:
        self.test_seen = time.monotonic()

    def testing(self) -> bool:
        """Whether the auditor's test traffic arrived within the last ``TEST_QUIET_S``."""
        return time.monotonic() - self.test_seen < TEST_QUIET_S

    def test_arriving(self) -> bool:
        """Whether the auditor's test traffic arrives now (validators' requests wait behind it): its last request came
        within ``TEST_ARRIVING_S``."""
        return time.monotonic() - self.test_seen < TEST_ARRIVING_S

    def note_tests(self, tests: list[ThroughputOutcome]) -> None:
        self.tests = list(tests)

    def own_ids(self) -> list[str]:
        """The endpoints this machine serves, as it lists them."""
        return [e.endpoint_id for e in self.local.host.endpoints] if self.local is not None else []

    # ---- measuring ----

    def _last_published(self) -> SignedManifest | None:
        try:
            signed = SignedManifest.model_validate_json(self.manifest_path.read_text())
        except (OSError, ValueError):
            return None
        m = signed.manifest
        same = (m.hotkey, m.netuid, m.mechid, m.pin(self.config.base_url)) == (
            self.wallet.hotkey.ss58_address, self.config.netuid, self.mechid, self.identity.pin)
        return signed if same else None

    def _last_served(self) -> SignedHostEndpoints | None:
        try:
            signed = SignedHostEndpoints.model_validate_json(self.endpoints_path.read_text())
        except (OSError, ValueError):
            return None
        h = signed.host
        same = (h.hotkey, h.netuid, h.mechid, h.url, h.tls) == (self.wallet.hotkey.ss58_address, self.config.netuid,
                                                                self.mechid, self.config.base_url, self.identity.pin)
        return signed if same else None

    def _local_url(self, endpoint_id: str) -> str:
        host = LOCAL_HOSTS.get(self.config.listen_host, self.config.listen_host)
        return f"{axon_base(host, self.config.port)}/e/{endpoint_id}/v1/systemone"

    async def _measure(self, endpoints: list[EndpointConfig], cat: catalog.Catalog, remeasure: bool,
                       stop: Callable[[], bool]) -> tuple[dict[str, Measurement], dict[str, str], bool]:
        """-> (measurements, failures, whether they were measured now rather than reused)."""
        limits = httpx.Limits(max_connections=None, max_keepalive_connections=1024)
        async with tls.client(timeout=120.0, limits=limits) as client:  # through this machine's TLS service
            targets = [Target(e.endpoint_id, e.model_id,
                              ServicePath(client, self._local_url(e.endpoint_id), self.wallet.hotkey,
                                          self.config.netuid, e.endpoint_id, tls=self.identity.pin),
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
                return {}, failures, False
            reuse = None if remeasure else stored(targets, self.store)
            if reuse is not None:
                return reuse, failures, False
            if self.testing():
                raise MeasurementInterrupted()  # a test is running: measuring now would distort both
            if self.window is not None and self.local is not None:  # endpoints of this machine may be tested
                if (deferral := await asyncio.to_thread(self.window.deferral)) is not None:
                    raise MeasurementDeferred(deferral)
            measured, failed = await measure_and_store(targets, self.store,
                                                       max_latency_s=cat.audit_policy().max_latency_s, stop=stop)
            return measured, failures | failed, True

    def adopt(self, cat: catalog.Catalog, *, remeasure: bool = False, shortfalls: Sequence[Shortfall] = ()) -> bool:
        """Measure the endpoints ``cat`` serves (or reuse their stored measurements) through the running service and
        sign what this machine serves, and on the primary the manifest. A measurement neither starts nor continues
        while the auditor's test traffic arrives, nor starts while a selected test is due (``timing``). Measured for
        ``shortfalls``, their tests cap those endpoints' declarations unless they show an outage; measured for any
        other reason, the caps are released. False if none could be measured or the measurement stepped aside or waits
        (``interrupted``); what is served stays."""
        endpoints = servable(self.config, cat)
        if not endpoints:
            log.error("no configured endpoint serves a model of catalog %s", cat.digest()[:12])
            return False
        if self.window is not None:
            self.window.set_policy(cat.audit_policy())
        previous_serving = set(self.serving)
        previous_limits = {k: a.limit for k, a in self.admissions.items()}
        self.serving = previous_serving | {e.endpoint_id for e in endpoints}
        for e in endpoints:
            self.admissions[e.endpoint_id].limit = e.max_concurrency
        started, self.interrupted = time.monotonic(), False
        try:
            measured, failures, fresh = asyncio.run(self._measure(endpoints, cat, remeasure or bool(shortfalls),
                                                                  stop=lambda: self.test_seen > started))
            self._deferral = None
        except MeasurementDeferred as exc:
            if exc.deferral.epoch != self._deferral:
                log.info(exc.deferral.line)
                self._deferral = exc.deferral.epoch
            measured, failures, fresh, self.interrupted = {}, {}, False, True
        except MeasurementInterrupted:
            log.info("a throughput test is running; measuring again after it")
            measured, failures, fresh, self.interrupted = {}, {}, False, True
        if not self.interrupted:
            self._announced.clear()
        if shortfalls and not self.interrupted:
            self.shortfalls.record(shortfalls)  # also if nothing could be measured: not again before the cooldown
        elif fresh:
            self.shortfalls.release(measured)
        for endpoint_id, reason in sorted(failures.items()):
            log.error("endpoint %s withdrawn: %s", endpoint_id, reason)
        listed = [e for e in endpoints if e.endpoint_id in measured]
        if not listed:
            self.serving = previous_serving
            for k, limit in previous_limits.items():
                self.admissions[k].limit = limit
            return False
        margin = cat.audit_policy().capacity_margin
        declared = {}
        for e in listed:
            m, cap = measured[e.endpoint_id], self.shortfalls.cap(e.endpoint_id)
            declared[e.endpoint_id] = m.declared(margin) if cap is None else min(m.declared(margin), cap.rps)
            self.admissions[e.endpoint_id].limit = m.concurrency
            log.info("endpoint %s (%s): %.1f requests/s at concurrency %d (runs %s), declares %.1f%s", e.endpoint_id,
                     e.model_id, m.measured_rps, m.concurrency, ", ".join(f"{r:.1f}" for r in m.runs),
                     declared[e.endpoint_id], "" if cap is None else ", " + cap.text())
        if catalog.mechanism_of(cat, self.config.mechid) != self.mechid:
            log.warning("catalog %s moves the network to mechanism %d; the manifest is for it from now on",
                        cat.digest()[:12], catalog.mechanism_of(cat, self.config.mechid))
        self.catalog = cat
        own = own_endpoints(self.config, listed, declared)
        self.local = build_host_endpoints(self.config, own, self.wallet.hotkey, cat, self.identity.pin)
        self.endpoints_path.parent.mkdir(parents=True, exist_ok=True)
        self.endpoints_path.write_text(self.local.model_dump_json())
        self.serving = {e.endpoint_id for e in listed}
        if self.peers is not None:
            self.peers.mechid = self.mechid
            self.merge()
        return True

    def engines_changed(self) -> bool:
        """Whether a listed endpoint's engine or configuration no longer matches its stored measurement."""
        listed = set(self.own_ids())
        endpoints = [e for e in servable(self.config, self.catalog) if e.endpoint_id in listed]
        stored_now = self.store.load()

        async def check() -> bool:
            for e in endpoints:
                m = stored_now.get(e.endpoint_id)
                if m is None or m.fingerprint != await engine_fingerprint(e, self.catalog):
                    return True
            return False

        return asyncio.run(check())

    def measure_shortfalls(self) -> bool:
        """Measure again if an endpoint of this machine fell short in its last two throughput tests at the rate it
        declares (``shortfall``); not while a test runs. True if it measured or waits to (``interrupted``)."""
        if self.local is None or self.testing():
            return False
        due = self.shortfalls.due(self.tests, {e.endpoint_id: e.capacity_rps for e in self.local.host.endpoints})
        if not due:
            return False
        for s in due:
            if self.announce(("shortfall", s.endpoint_id, s.tests[1].epoch)):
                log.warning(s.line())
        self.adopt(self.catalog, shortfalls=due)
        return True

    def announce(self, reason: object) -> bool:
        """Whether to log why a measurement is wanted: once while it waits (``interrupted``)."""
        if reason in self._announced:
            return False
        self._announced.add(reason)
        return True

    # ---- the primary: peers and publishing ----

    def poll_peers(self) -> None:
        """Fetch the peers' endpoints and rebuild the manifest (``miner.peers``); a peer out for an epoch is
        withdrawn."""

        async def run() -> None:
            async with tls.client() as client:
                await self.peers.poll(client, self.catalog.digest(), self.catalog.epoch_blocks * BLOCK_TIME_S)

        asyncio.run(run())
        self.merge()

    def merge(self) -> None:
        """Sign the manifest of this machine's measured endpoints and the peers'; it is published once it changed."""
        endpoints, left_out = merged(self.local.host.endpoints, self.peers.peers, self.catalog)
        if left_out != self._left_out:
            for line in left_out:
                log.error(line)
            self._left_out = left_out
        signed = build_manifest(self.config, endpoints, self.wallet.hotkey, self.catalog,
                                pins(self.config.base_url, self.identity.pin, endpoints, self.peers.peers))
        if signed.manifest.digest() != self.signed.manifest.digest():
            self.signed, self._published = signed, False

    def publish(self, st: bt.Subtensor) -> None:
        if self.publish_on_chain and not self._published:
            publish(st, self.wallet, self.config, self.signed)
            self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
            self.manifest_path.write_text(self.signed.model_dump_json())
        self._published = True

    def watch(self, interval_s: float = POLICY_WATCH_S) -> None:
        """Follow the owner's catalog commitment, the engines, the throughput tests and, on the primary, the peers
        (``follow``), every ``interval_s``."""

        def run() -> None:
            store = catalog.CatalogStore(self.config.catalog_dir)
            st: bt.Subtensor | None = None
            while True:
                time.sleep(interval_s)
                st = self.follow(st, store)
                if self.updater is not None and self.updater.due():
                    try:
                        if self.update_held() is None:
                            self.updater.at_safe_point()
                    except Exception:
                        log.exception("self-update failed; trying again in a minute")

        threading.Thread(target=run, name="catalog-watch", daemon=True).start()

    def update_held(self) -> str | None:
        """Why the miner does not restart for a self-update now: a throughput test's traffic arrives, or a test is
        selected and not over, or a measurement waits for one; None if it may (between rounds of ``follow``, so no
        measurement runs)."""
        if self.testing():
            return "a throughput test is running"
        if self.interrupted:
            return "a measurement waits for a throughput test"
        if self.window is not None and (deferral := self.window.deferral()) is not None:
            return deferral.line
        return None

    def follow(self, st: bt.Subtensor | None, store: catalog.CatalogStore) -> bt.Subtensor | None:
        """One round of ``watch``. A new catalog is adopted at once (measuring again if it changed what a measurement
        depends on), so is a changed engine; otherwise an endpoint that fell short in its tests is measured again.
        On-chain publication is retried until it succeeds. -> the chain connection for the next round (None after a
        failure: a new one then)."""
        try:
            st = st or bt.Subtensor(network=self.config.network)
            if committed_catalog(st, self.config.netuid) != self.catalog.digest():
                cat = current_catalog(st, self.config.netuid, store, self.config.catalog_mirrors)
                if self.adopt(cat):
                    log.info("catalog %s committed; endpoints rebuilt", cat.digest()[:12])
                elif not self.interrupted:
                    log.error("catalog %s committed but not adopted yet; keeping the current endpoints",
                              cat.digest()[:12])
                elif self.announce(("catalog", cat.digest())):
                    log.info("catalog %s committed; adopted after the measurement it needs", cat.digest()[:12])
            elif self.engines_changed():
                if self.announce("engine"):
                    log.info("an engine changed; measuring again")
                self.adopt(self.catalog)
            else:
                self.measure_shortfalls()
            if self.peers is not None and self.peers.peers:
                self.poll_peers()
            self.auditor = st.get_subnet_owner_hotkey(self.config.netuid)
            if self.publish_on_chain and not self._published:
                self.publish(st)
                log.info("manifest published")
            return st
        except (*CHAIN_ERRORS, catalog.CatalogUnavailable) as exc:
            log.warning("catalog follow failed (%s); retrying", one_line(exc))
        except Exception:
            log.exception("catalog follow failed; retrying")
        return None


def permits(config: MinerConfig, publisher: Publisher) -> PermitAuthorizer:
    """Who this machine answers: validators with a permit and the catalog's ``validator_min_stake``, the gateways and
    the auditor, whatever its stake (the subnet owner's validator runs the tests every miner needs)."""
    return PermitAuthorizer(config.network, config.netuid, lambda: publisher.gateways() | publisher.auditors(),
                            lambda: publisher.catalog.scoring.validator_min_stake)


def build_app(config: MinerConfig, keypair, publisher: Publisher, authorizer: PermitAuthorizer,
              engines: dict[str, Engine], keeper: firewall.Keeper | None = None) -> FastAPI:
    """This machine's miner service, primary or worker alike: callers are those ``authorizer`` admits (``permits``)
    and the auditor's test sessions, each within its limits; a validator within the catalog's request budget per
    endpoint and epoch. With the firewall's ``keeper``, it reports where authenticated callers come from and refuses
    the addresses the firewall does not allow."""
    budget = RequestBudget(lambda: publisher.catalog.scoring.request_budget(),
                           lambda: authorizer.epoch(publisher.catalog.epoch_blocks))

    def max_state_tokens(model_id: str) -> int | None:
        model = publisher.catalog.get(model_id)
        return model.max_state_tokens if model is not None else None

    def image_limits(model_id: str) -> ImageLimits | None:
        model = publisher.catalog.get(model_id)
        return model.images if model is not None and model.reads_images() else None

    def max_caller_request(model_id: str) -> int | None:
        model = publisher.catalog.get(model_id)
        return caller_request_bytes(model_id, model.workload, image_limits(model_id)) if model is not None else None

    return create_app(config=config, keypair=keypair, manifest=lambda: publisher.signed,
                      host_endpoints=lambda: publisher.local, engines=engines, authorized=authorizer,
                      auditors=publisher.auditors, admissions=publisher.admissions, serving=lambda: publisher.serving,
                      max_state_tokens=max_state_tokens, max_caller_request=max_caller_request,
                      image_limits=image_limits,
                      on_test_traffic=publisher.note_test_traffic, testing=publisher.test_arriving,
                      gateways=publisher.gateways, catalog_digest=lambda: publisher.catalog.digest(),
                      mechid=lambda: publisher.mechid, budget=budget,
                      **({"on_verified": keeper.observe, "allows": keeper.allows} if keeper is not None else {}))


def resolve_catalog(subtensor: bt.Subtensor, config: MinerConfig) -> catalog.Catalog:
    store = catalog.CatalogStore(config.catalog_dir)
    while True:
        try:
            return current_catalog(subtensor, config.netuid, store, config.catalog_mirrors)
        except catalog.CatalogUnavailable as exc:
            log.error("%s; retrying in 60 s", exc)
            time.sleep(60)


def adopt_first(publisher: Publisher, cat: catalog.Catalog, *, remeasure: bool,
                sleep: Callable[[float], None] = time.sleep) -> None:
    """The first measurement (or the stored one reused), after any running or selected test. Exits if no endpoint can
    be measured."""
    while not publisher.adopt(cat, remeasure=remeasure):
        if not publisher.interrupted:
            raise SystemExit("no endpoint could be measured")
        sleep(TEST_QUIET_S)


def serving_line(config: MinerConfig, publisher: Publisher) -> str:
    """What the miner logs once it serves."""
    line = f"serving {len(publisher.own_ids())} endpoints on port {config.port}"
    if config.role == "worker":
        line += f" for the hotkey's primary, which lists them from {config.base_url}{ENDPOINTS_PATH}"
    elif config.peers:
        machines = {endpoint_host(e.url, e.endpoint_id) for e in publisher.signed.manifest.endpoints}
        line += (f"; the manifest lists {len(publisher.signed.manifest.endpoints)} endpoints on {len(machines)} "
                 "machines")
    return (f"{line}; throughput tests are accepted from {', '.join(sorted(publisher.auditors()))}; status is logged "
            "once per epoch")


def main() -> None:
    if sys.argv[1:2] in (["init"], ["uninstall"]):
        from verathos.install import common

        common.main("miner", sys.argv[1], sys.argv[2:])
    if sys.argv[1:2] == ["firewall"]:
        from verathos.install import firewall as firewall_setup

        firewall_setup.main(sys.argv[2:])
        return
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(epilog="Set up this machine as a miner: verathos-miner init. Remove it: "
                                            "verathos-miner uninstall. Its firewall: verathos-miner firewall "
                                            "install|status|remove.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--remeasure", action="store_true",
                        help="Measure every endpoint's capacity again (after changing its hardware)")
    parser.add_argument("--no-publish", action="store_true",
                        help="Use the shipped catalog and write nothing on chain (local testing)")
    args = parser.parse_args()

    config = MinerConfig.load(args.config)
    if not args.no_publish and not is_public(config.public_host):
        raise SystemExit(f"public_host {config.public_host} is not a public address: validators refuse endpoints "
                         "there")
    wallet = bt.Wallet(name=config.wallet_name, hotkey=config.hotkey_name)
    subtensor = bt.Subtensor(network=config.network)
    cat = catalog.load() if args.no_publish else resolve_catalog(subtensor, config)
    identity = tls.identity(config.state_dir)
    log.info("TLS certificate %s, pin %s (the manifest names it; validators, gateways and the auditor accept this "
             "machine by it)", identity.cert_path, identity.pin)
    publisher = Publisher(config, wallet, cat, publish_on_chain=not args.no_publish, subtensor=subtensor,
                          identity=identity)
    keeper = firewall.start(config, args.config.expanduser().absolute(), wallet.hotkey.ss58_address,
                            lambda: publisher.catalog, lambda: publisher.auditor,
                            enabled=config.firewall and not args.no_publish,
                            why_off="--no-publish" if args.no_publish else "firewall is false in the configuration")
    authorizer = permits(config, publisher)
    engines = {e.endpoint_id: HttpEngine(e.engine_url, e.engine_api_key, max_connections=e.max_concurrency,
                                         tls_pin=e.engine_tls_pin)
               for e in config.endpoints}
    app = build_app(config, wallet.hotkey, publisher, authorizer, engines, keeper)
    # Connections beyond the descriptors left are refused, so a flood of idle connections cannot starve the rest.
    limit = max(256, raise_open_file_limit() - 256)
    service_config = uvicorn.Config(app, host=config.listen_host, port=config.port, log_level="warning",
                                    access_log=False, limit_concurrency=limit, **tls.server_options(identity))
    server = uvicorn.Server(service_config)
    try:
        listening = service_config.bind_socket()
    except SystemExit:
        raise SystemExit(f"the service could not start on port {config.port}") from None
    if keeper is not None:
        keeper.attach(listening)  # the firewall's filter goes with the socket
    service = threading.Thread(target=server.run, kwargs={"sockets": [listening]}, name="service", daemon=True)
    service.start()
    while not server.started:
        if not service.is_alive():
            raise SystemExit(f"the service could not start on port {config.port}")
        time.sleep(0.1)

    chain = SubtensorChainView(config.network, config.netuid)
    if not args.no_publish:
        publisher.window = MeasurementWindow(ScheduleFollower(chain, cat.audit_policy()))
    adopt_first(publisher, cat, remeasure=args.remeasure)
    if publisher.peers is not None and publisher.peers.peers:
        publisher.poll_peers()
    publisher.publish(subtensor)
    if not args.no_publish:
        publisher.updater = update.start(config.auto_update, "miner")
        publisher.watch()
        StatusWatcher(hotkey=wallet.hotkey.ss58_address, netuid=config.netuid, auditor_url=chain.auditor_url,
                      auditor_pin=publisher.catalog_auditor_pin, auditor_hotkey=lambda: publisher.auditor,
                      listed=publisher.own_ids, on_tests=publisher.note_tests).start()
    log.info(serving_line(config, publisher))
    service.join()
    os._exit(1)  # the service stopped


if __name__ == "__main__":
    main()
