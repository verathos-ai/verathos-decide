"""The catalog the subnet owner commits, followed in the background.

The validator checks the owner's commitment every ``POLICY_WATCH_S``, like the miners, so a new throughput-audit
policy reaches the audit within a minute of the miners. Each epoch is scored under the catalog committed at its first
block (``for_epoch``), the same for every validator. Manifests built for the previous catalog (for its mechanism)
stay acceptable for one epoch after a change, while miners re-publish; the change is kept on disk, so this holds
across a restart, also for a change made while the validator was down.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import bittensor as bt

from verathos import catalog
from verathos.subnet.chain import current_catalog
from verathos.throughput.policy import BLOCK_TIME_S, POLICY_WATCH_S

log = logging.getLogger(__name__)


class CatalogFollower:
    def __init__(self, st: bt.Subtensor, network: str, netuid: int, store: catalog.CatalogStore,
                 mirrors: list[str], *, state_path: Path) -> None:
        self.network, self.netuid, self.store, self.mirrors = network, netuid, store, mirrors
        self.catalog = self._wait(st)
        self.on_change: list[Callable[[catalog.Catalog], None]] = []
        self._previous: tuple[str, int, float] | None = None  # (digest, its mechanism, accepted until)
        self._lock = threading.Lock()
        self._state_path = state_path  # this validator's own: a catalog store may be shared
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
            mechanism = self._mechanism(followed, saved.get("mechanism"))
            self._previous = (followed, mechanism, time.time() + self._grace_s())
        elif followed is not None and previous and until > time.time():
            self._previous = (previous, self._mechanism(previous, saved.get("previous_mechanism")), until)
        self._save()

    def _mechanism(self, catalog_digest: str, saved: Any) -> int:
        """The mechanism of a catalog followed before: as saved, else from the store, else the current catalog's."""
        if isinstance(saved, int) and not isinstance(saved, bool) and saved >= 0:
            return saved
        stored = self.store.get(catalog_digest)
        return (stored or self.catalog).mechanism

    def _save(self) -> None:
        """Keep the followed catalog and the grace on disk; a failure to keep them only costs the grace after a
        restart."""
        previous, mechanism, until = self._previous or (None, None, 0.0)
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"catalog": self.catalog.digest(), "mechanism": self.catalog.mechanism,
                                       "previous": previous, "previous_mechanism": mechanism, "until": until}))
            tmp.replace(self._state_path)
        except OSError as exc:
            log.warning("keeping the followed catalog in %s failed: %s", self._state_path, exc)

    def _wait(self, st: bt.Subtensor) -> catalog.Catalog:
        while True:
            try:
                return current_catalog(st, self.netuid, self.store, self.mirrors)
            except catalog.CatalogUnavailable as exc:
                log.error("%s; retrying in 60 s", exc)
                time.sleep(60)

    def accepted(self) -> dict[str, int]:
        """The digests of the catalogs a manifest may be built for, each with its mechanism."""
        with self._lock:
            catalogs = {self.catalog.digest(): self.catalog.mechanism}
            if self._previous and time.time() < self._previous[2]:
                catalogs.setdefault(self._previous[0], self._previous[1])
        return catalogs

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
            self._previous = (self.catalog.digest(), self.catalog.mechanism, time.time() + self._grace_s())
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
