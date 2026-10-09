"""The auditor's signed routes (``protocol.gateway``), polled every ``POLL_S`` over TLS with the auditor's pin.

A table is used only if the auditor signed it for this subnet, it is newer than the one in use and not issued more
than ``MAX_AHEAD_S`` ahead of this host's clock (one dated in the future would shut out every correct table until
then, across restarts), and everything that depends on it (its catalog) could be set up. The last good table is kept
on disk too, so a restarted gateway serves at once. While the auditor serves no newer table, the gateway keeps using
the last one; past ``STALE_EPOCHS`` epochs it says so in the log and in ``/ready``.

A newer table that drops every route of a model its catalog still has active (``dropped_models``) is refused, and the
current one kept: an auditor that lost its view of the miners must not take a model off the gateway. Once the current
table is stale, the auditor's newer word is taken.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

import httpx

from verathos import catalog as catalog_module
from verathos.gateway import metrics
from verathos.gateway.logs import Throttle
from verathos.protocol.gateway import ROUTES_PATH, Routes, SignedRoutes
from verathos.protocol.http import request_json
from verathos.protocol.tls import Service
from verathos.throughput.policy import BLOCK_TIME_S

log = logging.getLogger(__name__)
POLL_S = 30.0
MAX_ROUTES_BYTES = 16 << 20
MAX_AHEAD_S = 300.0
STALE_EPOCHS = 3


def stale_after_s(cat: catalog_module.Catalog) -> float:
    """How old routes may grow before they are stale: ``STALE_EPOCHS`` of the catalog's epochs."""
    return STALE_EPOCHS * cat.epoch_blocks * BLOCK_TIME_S


def dropped_models(current: Routes | None, new: Routes, cat: catalog_module.Catalog) -> list[str]:
    """The models active in ``cat`` (the new table's catalog) that the current table routes and the new one does
    not route at all."""
    if current is None:
        return []
    dropped = {r.model_id for r in current.routes} - {r.model_id for r in new.routes}
    return sorted(m for m in dropped if (model := cat.get(m)) is not None and model.status == "active")


class Refused(Exception):
    """A table that is signed and newer, but not used (raised by an ``on_change`` callback)."""


class RouteTable:
    def __init__(self, *, client: httpx.AsyncClient, auditor: Callable[[], Awaitable[Service | None]],
                 auditor_hotkey: str, netuid: int, path: Path, clock: Callable[[], float] = time.time) -> None:
        self.client, self.auditor, self.auditor_hotkey, self.netuid = client, auditor, auditor_hotkey, netuid
        self.path, self.clock = path, clock
        self.table: Routes | None = None
        self.on_change: list[Callable[[Routes], Awaitable[None] | None]] = []
        self.stale_s: float | None = None  # set with the catalog (``stale_after_s``)
        self._throttled = Throttle(every_s=600.0)

    def _accept(self, signed: SignedRoutes) -> bool:
        if not signed.verify(self.auditor_hotkey, self.netuid):
            log.error("routes rejected: not signed by the auditor %s for netuid %d", self.auditor_hotkey, self.netuid)
            return False
        ahead = signed.routes.issued_ms / 1000 - self.clock()
        if ahead > MAX_AHEAD_S:
            log.error("routes rejected: issued %.0f s ahead of this host's clock", ahead)
            return False
        return self.table is None or signed.routes.issued_ms > self.table.issued_ms

    async def _adopt(self, signed: SignedRoutes) -> bool:
        try:
            for callback in self.on_change:
                result = callback(signed.routes)
                if asyncio.iscoroutine(result):
                    await result
        except Refused as exc:
            if self._throttled(f"refused {signed.routes.issued_ms}"):
                log.error("routes of epoch %d refused: %s", signed.routes.epoch, exc)
            return False
        self.table = signed.routes
        return True

    def stale(self) -> bool:
        """Whether the routes in use are older than ``stale_s``."""
        return (self.table is not None and self.stale_s is not None
                and self.clock() - self.table.issued_ms / 1000 > self.stale_s)

    async def load(self) -> None:
        """Adopt the table kept on disk, if any and valid."""
        try:
            signed = SignedRoutes.model_validate_json(self.path.read_text())
            if self._accept(signed) and await self._adopt(signed):
                log.info("routes from %s loaded: %d endpoints", self.path, len(signed.routes.routes))
        except Exception as exc:
            log.warning("routes on disk not used: %s: %s", type(exc).__name__, exc)

    async def refresh(self) -> bool:
        """Fetch the auditor's table; True if a newer one was adopted."""
        auditor = await self.auditor()
        if auditor is None:
            return False
        document = await request_json(self.client, "GET", auditor.url.rstrip("/") + ROUTES_PATH,
                                      limit=MAX_ROUTES_BYTES, timeout=15.0, tls=auditor.pin)
        signed = SignedRoutes.model_validate(document)
        if not self._accept(signed) or not await self._adopt(signed):
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(signed.model_dump_json())
        tmp.replace(self.path)
        return True

    async def run(self, poll_s: float = POLL_S) -> None:
        while True:
            try:
                await self.refresh()
            except Exception as exc:
                log.warning("routes not refreshed (keeping %s): %s: %s",
                            "none" if self.table is None else f"{len(self.table.routes)} endpoints",
                            type(exc).__name__, exc)
            if self.table is not None:
                age = self.clock() - self.table.issued_ms / 1000
                metrics.ROUTES_AGE.set(age)
                if self.stale() and self._throttled("stale"):
                    log.warning("the routes in use are %.0f min old: the auditor serves no newer ones, so endpoints "
                                "that lost their credit since may still be used", age / 60)
            await asyncio.sleep(poll_s)
