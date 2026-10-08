"""Find miners and their endpoints from chain state, and check that each endpoint is the hotkey's.

Input is a chain snapshot (hotkey, axon and commitment per UID); output is the accepted manifests. Only hotkeys
with a manifest commitment and an axon on a public address are contacted; a manifest accepted before stands in for a
while when the axon is out (``ManifestCache``). A manifest may list endpoints on any public host, so an endpoint gets
traffic only once it proved control at its URL (``Identities``).
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass

import httpx
from bittensor_wallet import Keypair

from verathos.miner.app import IDENTITY_DOMAIN, identity_payload
from verathos.protocol import signing
from verathos.protocol.http import request_json
from verathos.protocol.manifest import (
    Endpoint,
    Manifest,
    ManifestRejected,
    accept,
    axon_base,
    committed_digest,
    endpoint_host,
    is_public,
)
from verathos.protocol.requests import sign_request

log = logging.getLogger(__name__)
FETCH_TIMEOUT_S = 10.0
FETCH_CONCURRENCY = 32
MAX_MANIFEST_BYTES = 256 * 1024
MAX_IDENTITY_BYTES = 4096
IDENTITY_CONCURRENCY = 256
CLAIM_POLL_S = 0.2


@dataclass(frozen=True)
class Neuron:
    uid: int
    hotkey: str
    axon_ip: str
    axon_port: int
    commitment: str | None
    stake: float = 0.0  # the metagraph's S
    permit: bool = False  # a validator permit


@dataclass(frozen=True)
class DiscoveredEndpoint:
    uid: int
    hotkey: str
    axon: str | None  # the axon its manifest was just fetched from (``axon_base``)
    endpoint: Endpoint

    @property
    def path_prefix(self) -> str:
        return "/e/" + self.endpoint.endpoint_id

    @property
    def key(self) -> str:
        return f"{self.hotkey}/{self.endpoint.endpoint_id}"

    @property
    def url(self) -> str:
        """Where the endpoint is reached; its path is ``path_prefix``."""
        return self.endpoint.url

    @property
    def host(self) -> str:
        """The miner service that serves it (``http://ip:port``)."""
        return endpoint_host(self.endpoint.url, self.endpoint.endpoint_id) or self.endpoint.url


@dataclass(frozen=True)
class DiscoveredMiner:
    uid: int
    hotkey: str
    axon: str | None
    manifest: Manifest

    def endpoints(self) -> list[DiscoveredEndpoint]:
        return [DiscoveredEndpoint(self.uid, self.hotkey, self.axon, e) for e in self.manifest.endpoints]


class ManifestCache:
    """The last manifest each hotkey's axon served that was accepted. While the axon serves none that is (it is down,
    or restarting), the hotkey's endpoints are taken from this one, for at most ``hold_s`` after it was last fetched
    and only while the hotkey's commitment still names it: endpoints on the miner's other machines keep their
    traffic while the machine serving its manifest is out."""

    def __init__(self, hold_s: float) -> None:
        self.hold_s = hold_s
        self._lock = threading.Lock()
        self._accepted: dict[str, tuple[float, str, Manifest]] = {}  # hotkey -> (when, its digest, manifest)

    def put(self, hotkey: str, manifest_digest: str, manifest: Manifest) -> None:
        with self._lock:
            self._accepted[hotkey] = (time.monotonic(), manifest_digest, manifest)

    def get(self, n: Neuron, catalogs: Mapping[str, int]) -> tuple[float, Manifest] | None:
        """(seconds since it was fetched, the manifest), if it may stand in for the axon's (see the class): still
        committed, and built for one of ``catalogs`` (as ``discover`` takes them)."""
        with self._lock:
            held = self._accepted.get(n.hotkey)
        if (held is None or held[1] != committed_digest(n.commitment)
                or catalogs.get(held[2].catalog_digest) != held[2].mechid):
            return None
        age = time.monotonic() - held[0]
        return (age, held[2]) if age <= self.hold_s else None

    def prune(self) -> None:
        now = time.monotonic()
        with self._lock:
            self._accepted = {k: v for k, v in self._accepted.items() if now - v[0] <= self.hold_s}


async def _fetch_manifest(n: Neuron, netuid: int, client: httpx.AsyncClient, catalogs: Mapping[str, int],
                          cache: ManifestCache | None) -> DiscoveredMiner | None:
    base = axon_base(n.axon_ip, n.axon_port)
    try:
        if not is_public(n.axon_ip) or not 0 < n.axon_port < 65536:
            raise ManifestRejected("the axon is not on a public address")
        document = await request_json(client, "GET", base + "/v1/manifest", limit=MAX_MANIFEST_BYTES,
                                      timeout=FETCH_TIMEOUT_S)
        manifest = accept(document, hotkey=n.hotkey, netuid=netuid, commitment=n.commitment, catalogs=catalogs)
    except Exception as e:  # whatever a neuron serves, it only costs that neuron its discovery
        held = cache.get(n, catalogs) if cache is not None else None
        if held is None:
            log.info("uid %d manifest not accepted: %s", n.uid, str(e) or type(e).__name__)
            return None
        log.info("uid %d manifest not accepted (%s); using the one it commits, fetched %.0f min ago", n.uid,
                 str(e) or type(e).__name__, held[0] / 60)
        return DiscoveredMiner(n.uid, n.hotkey, None, held[1])
    if cache is not None:
        cache.put(n.hotkey, committed_digest(n.commitment), manifest)
    return DiscoveredMiner(n.uid, n.hotkey, base, manifest)


async def discover(neurons: list[Neuron], netuid: int, client: httpx.AsyncClient, catalogs: Mapping[str, int],
                   cache: ManifestCache | None = None) -> list[DiscoveredMiner]:
    """Fetch and accept the manifest of every neuron that has a manifest commitment, from its axon on a public
    address, or from ``cache`` while the axon serves none that is accepted.

    ``catalogs`` maps the digests of the catalogs a manifest may be built for (the current one, plus the previous one
    while miners re-publish after a change) to the mechanism a manifest built for each is for.
    """
    candidates = [n for n in neurons if committed_digest(n.commitment) is not None]
    limit = asyncio.Semaphore(FETCH_CONCURRENCY)

    async def fetch(n: Neuron) -> DiscoveredMiner | None:
        async with limit:
            return await _fetch_manifest(n, netuid, client, catalogs, cache)

    found = await asyncio.gather(*(fetch(n) for n in candidates))
    if cache is not None:
        cache.prune()
    return [m for m in found if m is not None]


async def verify_identity(endpoint: DiscoveredEndpoint, keypair: Keypair, client: httpx.AsyncClient) -> bool:
    """The endpoint must sign a fresh challenge with the hotkey that lists it, at its URL."""
    challenge = secrets.token_hex(16)
    body = (f'{{"challenge":"{challenge}"}}').encode()
    headers = {"Content-Type": "application/json",
               **sign_request(keypair, "POST", endpoint.path_prefix + "/v1/identity", body, endpoint.hotkey)}
    try:
        reply = await request_json(client, "POST", endpoint.url + "/v1/identity", content=body, headers=headers,
                                   limit=MAX_IDENTITY_BYTES, timeout=FETCH_TIMEOUT_S)
        signature = reply["signature"] if isinstance(reply, dict) else None
    except Exception:
        return False
    return signing.verify(endpoint.hotkey, IDENTITY_DOMAIN,
                          identity_payload(endpoint.endpoint.endpoint_id, challenge, keypair.ss58_address), signature)


class Identities:
    """Which endpoints proved control at their URL, by epoch: everything in a validator that sends miners traffic
    asks here first.

    A hotkey's own hosts are its axon, while its manifest was just fetched from there, and every host where one of its
    endpoints passed this epoch or the last: there its endpoints are challenged at once, and again after a failure.
    At any other host they are challenged one after another until one passes; once one fails, that host gets no
    further challenge for the hotkey in the epoch. A manifest naming someone else's server so costs that server one
    challenge per epoch from each validator, and no other traffic."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._passed: dict[tuple[str, str], int] = {}  # (endpoint key, url) -> latest epoch it passed
        self._hosts: dict[tuple[str, str], int] = {}  # (hotkey, host) -> latest epoch an endpoint passed there
        self._failed: dict[tuple[str, str], int] = {}  # (hotkey, host) -> latest epoch a challenge there failed
        self._tried: dict[tuple[str, str], int] = {}  # (endpoint key, url) -> latest epoch it was challenged
        self._busy: set[tuple[str, str]] = set()  # (hotkey, host) not the hotkey's, with a challenge under way

    def verified(self, key: str, url: str, since_epoch: int) -> bool:
        """Whether endpoint ``key`` passed at ``url`` in ``since_epoch`` or later."""
        with self._lock:
            return self._passed.get((key, url), -1) >= since_epoch

    def own(self, ep: DiscoveredEndpoint, epoch: int) -> bool:
        """Whether the endpoint's host is one of its hotkey's own (see the class)."""
        with self._lock:
            return ep.host == ep.axon or self._hosts.get((ep.hotkey, ep.host), -1) >= epoch - 1

    async def verify(self, endpoints: list[DiscoveredEndpoint], epoch: int, keypair: Keypair,
                     client: httpx.AsyncClient, *, retry: bool = True,
                     concurrency: int = IDENTITY_CONCURRENCY) -> None:
        """Challenge the endpoints that have not passed at their URL in ``epoch`` (see the class); without ``retry``
        only those not challenged in it yet."""
        slots = asyncio.Semaphore(concurrency)
        hosts: dict[tuple[str, str], list[DiscoveredEndpoint]] = {}
        for ep in endpoints:
            hosts.setdefault((ep.hotkey, ep.host), []).append(ep)

        def due(ep: DiscoveredEndpoint) -> bool:
            with self._lock:
                return (self._passed.get((ep.key, ep.url), -1) < epoch
                        and (retry or self._tried.get((ep.key, ep.url), -1) < epoch))

        async def challenge(ep: DiscoveredEndpoint) -> bool:
            with self._lock:
                self._tried[ep.key, ep.url] = epoch
            async with slots:
                ok = await verify_identity(ep, keypair, client)
            if ok:
                with self._lock:
                    self._passed[ep.key, ep.url] = self._hosts[ep.hotkey, ep.host] = epoch
            return ok

        async def at_host(key: tuple[str, str], group: list[DiscoveredEndpoint]) -> None:
            if not self.own(group[0], epoch):
                if not await self._claim(key, epoch):
                    return
                try:
                    for ep in group:
                        if self.own(ep, epoch):
                            break
                        if due(ep) and not await challenge(ep):
                            with self._lock:
                                self._failed[key] = epoch
                            return
                finally:
                    with self._lock:
                        self._busy.discard(key)
            await asyncio.gather(*(challenge(ep) for ep in group if due(ep)))

        await asyncio.gather(*(at_host(key, group) for key, group in hosts.items()))
        with self._lock:
            for table in (self._passed, self._hosts, self._failed, self._tried):
                for stale in [k for k, e in table.items() if e < epoch - 1]:
                    del table[stale]

    async def _claim(self, key: tuple[str, str], epoch: int) -> bool:
        """Take the host's challenges, after any under way elsewhere (the auditor also discovers in a thread of its
        own); False if a challenge there failed in ``epoch``."""
        while True:
            with self._lock:
                if self._failed.get(key, -1) >= epoch:
                    return False
                if key not in self._busy:
                    self._busy.add(key)
                    return True
            await asyncio.sleep(CLAIM_POLL_S)
