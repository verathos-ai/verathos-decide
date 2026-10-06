"""Find miners and their endpoints from chain state.

Input is a chain snapshot (hotkey, axon and commitment per UID); output is the accepted manifests. Only hotkeys
with a manifest commitment are contacted.
"""

from __future__ import annotations

import logging
import secrets
from dataclasses import dataclass

import httpx
from bittensor_wallet import Keypair

from verathos.miner.app import IDENTITY_DOMAIN, identity_payload
from verathos.protocol import signing
from verathos.protocol.manifest import Endpoint, Manifest, ManifestRejected, SignedManifest, accept, committed_digest
from verathos.protocol.requests import sign_request

log = logging.getLogger(__name__)


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


async def discover(neurons: list[Neuron], netuid: int, client: httpx.AsyncClient) -> list[DiscoveredMiner]:
    """Fetch and accept the manifest of every neuron that has a manifest commitment and a served axon."""
    found = []
    for n in neurons:
        if committed_digest(n.commitment) is None or n.axon_ip in ("", "0.0.0.0") or not n.axon_port:
            continue
        base = f"http://{n.axon_ip}:{n.axon_port}"
        try:
            r = await client.get(base + "/v1/manifest", timeout=10)
            r.raise_for_status()
            manifest = accept(SignedManifest.model_validate_json(r.content), hotkey=n.hotkey, netuid=netuid, commitment=n.commitment)
        except (httpx.HTTPError, ManifestRejected, ValueError) as e:
            log.info("uid %d manifest not accepted: %s", n.uid, e)
            continue
        found.append(DiscoveredMiner(n.uid, n.hotkey, base, manifest))
    return found


async def verify_identity(endpoint: DiscoveredEndpoint, keypair: Keypair, client: httpx.AsyncClient) -> bool:
    """The endpoint must sign a fresh challenge with the hotkey that lists it."""
    challenge = secrets.token_hex(16)
    path = endpoint.path_prefix + "/v1/identity"
    body = ('{"challenge":"%s"}' % challenge).encode()
    headers = {"Content-Type": "application/json", **sign_request(keypair, "POST", path, body, endpoint.hotkey)}
    try:
        r = await client.post(endpoint.base_url + path, content=body, headers=headers, timeout=10)
        r.raise_for_status()
        sig = r.json()["signature"]
    except (httpx.HTTPError, KeyError, ValueError):
        return False
    return signing.verify(endpoint.hotkey, IDENTITY_DOMAIN,
                          identity_payload(endpoint.endpoint.endpoint_id, challenge, keypair.ss58_address), sig)
