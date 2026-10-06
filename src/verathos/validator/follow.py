"""The catalog the subnet owner commits, followed in the background.

The validator checks the owner's commitment every ``POLICY_WATCH_S``, like the miners, so a new throughput-audit
policy reaches the audit within a minute of the miners. Models, tolerances and the epoch length apply from the next
epoch. Manifests built for the previous catalog stay acceptable for one epoch after a change, while miners
re-publish.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

import bittensor as bt

from verathos import catalog
from verathos.subnet.chain import current_catalog
from verathos.throughput.policy import BLOCK_TIME_S, POLICY_WATCH_S

log = logging.getLogger(__name__)


class CatalogFollower:
    def __init__(self, st: bt.Subtensor, network: str, netuid: int, store: catalog.CatalogStore,
                 mirrors: list[str]) -> None:
        self.network, self.netuid, self.store, self.mirrors = network, netuid, store, mirrors
        self.catalog = self._wait(st)
        self.on_change: list[Callable[[catalog.Catalog], None]] = []
        self._previous: tuple[str, float] | None = None
        self._lock = threading.Lock()

    def _wait(self, st: bt.Subtensor) -> catalog.Catalog:
        while True:
            try:
                return current_catalog(st, self.netuid, self.store, self.mirrors)
            except catalog.CatalogUnavailable as exc:
                log.error("%s; retrying in 60 s", exc)
                time.sleep(60)

    def accepted(self) -> set[str]:
        """Catalog digests a manifest may be built for."""
        with self._lock:
            digests = {self.catalog.digest()}
            if self._previous and time.time() < self._previous[1]:
                digests.add(self._previous[0])
        return digests

    def refresh(self, st: bt.Subtensor) -> bool:
        """Re-resolve the committed catalog; True if it changed. Keeps the current one while a new one is
        unavailable."""
        try:
            new = current_catalog(st, self.netuid, self.store, self.mirrors)
        except Exception as exc:
            log.error("committed catalog unavailable, keeping %s: %s", self.catalog.digest()[:12], exc)
            return False
        if new.digest() == self.catalog.digest():
            return False
        with self._lock:
            grace_s = self.catalog.epoch_blocks * BLOCK_TIME_S
            self._previous = (self.catalog.digest(), time.time() + grace_s)
            self.catalog = new
        log.info("catalog %s committed", new.digest()[:12])
        for callback in self.on_change:
            callback(new)
        return True

    def start(self, interval_s: float = POLICY_WATCH_S) -> None:
        def run() -> None:
            st = bt.Subtensor(network=self.network)
            while True:
                time.sleep(interval_s)
                try:
                    self.refresh(st)
                except Exception:
                    log.exception("catalog follow failed")

        threading.Thread(target=run, name="catalog-follow", daemon=True).start()
