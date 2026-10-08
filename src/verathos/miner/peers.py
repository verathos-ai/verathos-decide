"""One hotkey on several machines: the primary lists every machine's endpoints in the manifest it publishes.

Each worker serves the endpoints it measured at ``GET /v1/endpoints``, signed by the hotkey. The primary polls its
configured peers every minute and takes a peer's endpoints from a document the hotkey signed, for this netuid and
mechanism, built for the primary's catalog, whose endpoints are all at that peer's own address (a public one, as
validators require). A peer that stops answering so keeps its last endpoints until it has been out for ``hold_s`` (an
epoch), then is withdrawn. Endpoint ids are unique across the hotkey's machines: an endpoint whose id an earlier
machine lists (the primary's own first, then the peers in their configured order) is left out.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

import httpx

from verathos.catalog import Catalog
from verathos.protocol.http import request_json
from verathos.protocol.manifest import (
    ENDPOINTS_PATH,
    MAX_ENDPOINTS,
    Endpoint,
    Manifest,
    ManifestRejected,
    accept_host,
    endpoint_host,
)

log = logging.getLogger("verathos.miner")
FETCH_TIMEOUT_S = 10.0
MAX_DOCUMENT_BYTES = 256 * 1024


@dataclass
class Peer:
    url: str  # http://ip:port
    endpoints: list[Endpoint] = field(default_factory=list)
    seen: float | None = None  # when its endpoints were last confirmed (time.monotonic)
    said: str = ""  # what was last logged about it


class Peers:
    """The primary's view of its peers (see the module)."""

    def __init__(self, urls: list[str], *, hotkey: str, netuid: int, mechid: int) -> None:
        self.peers = [Peer(url) for url in urls]
        self.hotkey, self.netuid, self.mechid = hotkey, netuid, mechid

    def seed(self, manifest: Manifest) -> None:
        """After a restart: each peer's endpoints as the manifest last published lists them, held from now."""
        now = time.monotonic()
        for p in self.peers:
            p.endpoints = [e for e in manifest.endpoints if endpoint_host(e.url, e.endpoint_id) == p.url]
            p.seen = now if p.endpoints else None

    async def poll(self, client: httpx.AsyncClient, catalog_digest: str, hold_s: float) -> None:
        await asyncio.gather(*(self._poll(client, p, catalog_digest, hold_s) for p in self.peers))

    async def _poll(self, client: httpx.AsyncClient, p: Peer, catalog_digest: str, hold_s: float) -> None:
        try:
            document = await request_json(client, "GET", p.url + ENDPOINTS_PATH, limit=MAX_DOCUMENT_BYTES,
                                          timeout=FETCH_TIMEOUT_S)
            host = accept_host(document, hotkey=self.hotkey, netuid=self.netuid, mechid=self.mechid, url=p.url)
            if host.catalog_digest != catalog_digest:
                raise ManifestRejected(f"its endpoints are for catalog {host.catalog_digest[:12]}, this machine's "
                                       f"for {catalog_digest[:12]}")
        except Exception as exc:  # whatever a peer serves, it only costs that peer its endpoints
            why = str(exc) or type(exc).__name__
            if p.seen is not None and time.monotonic() - p.seen > hold_s:
                p.endpoints, p.seen = [], None
                self._say(p, logging.ERROR, f"peer {p.url}: endpoints withdrawn, none confirmed for an epoch ({why})")
            elif p.endpoints:
                self._say(p, logging.WARNING, f"peer {p.url}: {why}; its {len(p.endpoints)} endpoints stay listed "
                          "until it has been out for an epoch")
            else:
                self._say(p, logging.WARNING, f"peer {p.url}: {why}")
            return
        p.endpoints, p.seen = host.endpoints, time.monotonic()
        self._say(p, logging.INFO, f"peer {p.url}: " + (", ".join(
            f"{e.endpoint_id} ({e.model_id}, {e.capacity_rps:.1f} requests/s)" for e in p.endpoints) or "no endpoint"))

    @staticmethod
    def _say(p: Peer, level: int, text: str) -> None:
        if text != p.said:
            log.log(level, text)
            p.said = text


def merged(own: list[Endpoint], peers: list[Peer], cat: Catalog) -> tuple[list[Endpoint], list[str]]:
    """The manifest's endpoints: this machine's, then each peer's in order; -> (endpoints, why each other one is left
    out: its id already listed, a model the catalog does not serve, or beyond ``MAX_ENDPOINTS``)."""
    out, left_out = list(own), []
    listed = {e.endpoint_id: "this machine" for e in own}
    for p in peers:
        for e in p.endpoints:
            model = cat.get(e.model_id)
            if e.endpoint_id in listed:
                left_out.append(f"endpoint {e.endpoint_id} of peer {p.url} left out: {listed[e.endpoint_id]} lists "
                                f"an endpoint {e.endpoint_id}; endpoint ids must be unique across the hotkey's "
                                "machines")
            elif model is None or model.status == "retired":
                left_out.append(f"endpoint {e.endpoint_id} of peer {p.url} left out: catalog {cat.digest()[:12]} "
                                f"does not serve {e.model_id}")
            elif len(out) >= MAX_ENDPOINTS:
                left_out.append(f"endpoint {e.endpoint_id} of peer {p.url} left out: a manifest lists at most "
                                f"{MAX_ENDPOINTS} endpoints")
            else:
                out.append(e)
                listed[e.endpoint_id] = f"peer {p.url}"
    return out, left_out
