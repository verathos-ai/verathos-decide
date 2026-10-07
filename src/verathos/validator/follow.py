"""The catalog the subnet owner commits, followed in the background.

The validator checks the owner's commitment every ``POLICY_WATCH_S``, like the miners, so a new throughput-audit
policy reaches the audit within a minute of the miners. Each epoch is scored under the catalog committed at its first
block (``for_epoch``), the same for every validator. Manifests built for the previous catalog stay acceptable for
one epoch after a change, while miners re-publish; the change is kept on disk, so this holds across a restart, also
for a change made while the validator was down.
"""

from __future__ import annotations

import json
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
        self._state_path = store.directory / "followed.json"
        self._resume()

    def _grace_s(self) -> float:
        return self.catalog.epoch_blocks * BLOCK_TIME_S

    def _resume(self) -> None:
        """Take up the grace of the catalog followed before the restart: still running, or starting now if the
        committed catalog changed meanwhile."""
        try:
            saved = json.loads(self._state_path.read_text())
            followed, previous, until = saved["catalog"], saved.get("previous"), float(saved.get("until", 0))
        except FileNotFoundError:
            followed = None
        except (ValueError, KeyError, TypeError) as exc:
            log.warning("%s unreadable, starting without a previous catalog: %s", self._state_path, exc)
            followed = None
        if followed is not None and followed != self.catalog.digest():
            self._previous = (followed, time.time() + self._grace_s())
        elif followed is not None and previous and until > time.time():
            self._previous = (previous, until)
        self._save()

    def _save(self) -> None:
        previous, until = self._previous or (None, 0.0)
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"catalog": self.catalog.digest(), "previous": previous, "until": until}))
        tmp.replace(self._state_path)

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

    def for_epoch(self, st: bt.Subtensor, start_block: int) -> catalog.Catalog:
        """The catalog committed at an epoch's first block, so every validator scores the epoch under the same one,
        whenever it noticed a new commitment. Falls back to the current catalog if that one is unavailable."""
        try:
            return current_catalog(st, self.netuid, self.store, self.mirrors, block=start_block)
        except Exception as exc:
            log.error("catalog committed at block %d unavailable, using %s: %s", start_block,
                      self.catalog.digest()[:12], exc)
            return self.catalog

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
            self._previous = (self.catalog.digest(), time.time() + self._grace_s())
            self.catalog = new
            self._save()
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
