"""Find miners and their endpoints from chain state.

Input is a chain snapshot (hotkey, axon and commitment per UID); output is the accepted manifests. Only hotkeys
with a manifest commitment are contacted.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
from collections.abc import Collection
from dataclasses import dataclass

import httpx
from bittensor_wallet import Keypair

from verathos.miner.app import IDENTITY_DOMAIN, identity_payload
from verathos.protocol import signing
from verathos.protocol.http import request_json
from verathos.protocol.manifest import Endpoint, Manifest, accept, axon_base, committed_digest
from verathos.protocol.requests import sign_request

log = logging.getLogger(__name__)
FETCH_TIMEOUT_S = 10.0
FETCH_CONCURRENCY = 32
MAX_MANIFEST_BYTES = 256 * 1024
MAX_IDENTITY_BYTES = 4096


@dataclass(frozen=True)
class Neuron:
    uid: int
    hotkey: str
    axon_ip: str
    axon_port: int
    commitment: str | None


@dataclass(frozen=True)
class DiscoveredEndpoint:
    uid: int
    hotkey: str
    base_url: str
    endpoint: Endpoint

    @property
    def path_prefix(self) -> str:
        return "/e/" + self.endpoint.endpoint_id

    @property
    def key(self) -> str:
        return f"{self.hotkey}/{self.endpoint.endpoint_id}"


@dataclass(frozen=True)
class DiscoveredMiner:
    uid: int
    hotkey: str
    base_url: str
    manifest: Manifest

    def endpoints(self) -> list[DiscoveredEndpoint]:
        return [DiscoveredEndpoint(self.uid, self.hotkey, self.base_url, e) for e in self.manifest.endpoints]


async def _fetch_manifest(n: Neuron, netuid: int, mechid: int, client: httpx.AsyncClient,
                          catalog_digests: Collection[str]) -> DiscoveredMiner | None:
    base = axon_base(n.axon_ip, n.axon_port)
    try:
        document = await request_json(client, "GET", base + "/v1/manifest", limit=MAX_MANIFEST_BYTES,
                                      timeout=FETCH_TIMEOUT_S)
        manifest = accept(document, hotkey=n.hotkey, netuid=netuid, mechid=mechid, axon=base,
                          commitment=n.commitment, catalog_digests=catalog_digests)
    except Exception as e:  # whatever a neuron serves, it only costs that neuron its discovery
        log.info("uid %d manifest not accepted: %s", n.uid, str(e) or type(e).__name__)
        return None
    return DiscoveredMiner(n.uid, n.hotkey, base, manifest)


async def discover(neurons: list[Neuron], netuid: int, mechid: int, client: httpx.AsyncClient,
                   catalog_digests: Collection[str]) -> list[DiscoveredMiner]:
    """Fetch and accept the manifest of every neuron that has a manifest commitment and a served axon.

    ``catalog_digests`` are the catalogs a manifest may be built for: the current one, plus the previous one while
    miners re-publish after a change.
    """
    candidates = [n for n in neurons
                  if committed_digest(n.commitment) is not None and n.axon_ip not in ("", "0.0.0.0") and n.axon_port]
    limit = asyncio.Semaphore(FETCH_CONCURRENCY)

    async def fetch(n: Neuron) -> DiscoveredMiner | None:
        async with limit:
            return await _fetch_manifest(n, netuid, mechid, client, catalog_digests)

    found = await asyncio.gather(*(fetch(n) for n in candidates))
    return [m for m in found if m is not None]


async def verify_identity(endpoint: DiscoveredEndpoint, keypair: Keypair, client: httpx.AsyncClient) -> bool:
    """The endpoint must sign a fresh challenge with the hotkey that lists it."""
    challenge = secrets.token_hex(16)
    path = endpoint.path_prefix + "/v1/identity"
    body = (f'{{"challenge":"{challenge}"}}').encode()
    headers = {"Content-Type": "application/json", **sign_request(keypair, "POST", path, body, endpoint.hotkey)}
    try:
        reply = await request_json(client, "POST", endpoint.base_url + path, content=body, headers=headers,
                                   limit=MAX_IDENTITY_BYTES, timeout=FETCH_TIMEOUT_S)
        signature = reply["signature"] if isinstance(reply, dict) else None
    except Exception:
        return False
    return signing.verify(endpoint.hotkey, IDENTITY_DOMAIN,
                          identity_payload(endpoint.endpoint.endpoint_id, challenge, keypair.ss58_address), signature)
