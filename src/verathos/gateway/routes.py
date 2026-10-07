"""The auditor's signed routes (``protocol.gateway``), polled every ``POLL_S``.

A table is used only if the auditor signed it for this subnet and it is newer than the one in use. The last good
table is kept on disk too, so a restarted gateway serves at once.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

import httpx

from verathos.gateway import metrics
from verathos.protocol.gateway import ROUTES_PATH, Routes, SignedRoutes
from verathos.protocol.http import request_json

log = logging.getLogger(__name__)
POLL_S = 30.0
MAX_ROUTES_BYTES = 16 << 20


class RouteTable:
    def __init__(self, *, client: httpx.AsyncClient, auditor_url: Callable[[], Awaitable[str | None]],
                 auditor_hotkey: str, netuid: int, path: Path) -> None:
        self.client, self.auditor_url, self.auditor_hotkey, self.netuid = client, auditor_url, auditor_hotkey, netuid
        self.path = path
        self.table: Routes | None = None
        self.on_change: list[Callable[[Routes], Awaitable[None] | None]] = []

    def _accept(self, signed: SignedRoutes) -> bool:
        if not signed.verify(self.auditor_hotkey, self.netuid):
            log.error("routes rejected: not signed by the auditor %s for netuid %d", self.auditor_hotkey, self.netuid)
            return False
        return self.table is None or signed.routes.issued_ms > self.table.issued_ms

    async def _adopt(self, signed: SignedRoutes) -> None:
        self.table = signed.routes
        for callback in self.on_change:
            result = callback(signed.routes)
            if asyncio.iscoroutine(result):
                await result

    async def load(self) -> None:
        """Adopt the table kept on disk, if any and valid."""
        try:
            signed = SignedRoutes.model_validate_json(self.path.read_text())
        except (OSError, ValueError):
            return
        if self._accept(signed):
            await self._adopt(signed)
            log.info("routes from %s loaded: %d endpoints", self.path, len(signed.routes.routes))

    async def refresh(self) -> bool:
        """Fetch the auditor's table; True if a newer one was adopted."""
        url = await self.auditor_url()
        if not url:
            return False
        document = await request_json(self.client, "GET", url.rstrip("/") + ROUTES_PATH, limit=MAX_ROUTES_BYTES,
                                      timeout=15.0)
        signed = SignedRoutes.model_validate(document)
        if not self._accept(signed):
            return False
        await self._adopt(signed)
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
                metrics.ROUTES_AGE.set(time.time() - self.table.issued_ms / 1000)
            await asyncio.sleep(poll_s)
