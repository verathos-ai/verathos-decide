"""Miner manifest: the endpoints a hotkey serves.

Discovery uses only native chain state:
- the hotkey's axon (``serve_axon``) points at the miner's manifest server, ``GET /v1/manifest``;
- the manifest lists every endpoint with its model and declared capacity, and is signed by the hotkey;
- the hotkey's commitment (``set_commitment``) holds the manifest digest, so every validator scores the same
  version.

A fetched manifest is accepted only if its signature verifies against the hotkey, its digest equals the hotkey's
current commitment, and every endpoint URL is on the hotkey's axon. Signature and digest cover the JSON exactly as
received, so nodes running different code versions agree on them.
"""

from __future__ import annotations

import ipaddress
import re
from collections.abc import Collection
from typing import Any, Literal

from bittensor_wallet import Keypair
from pydantic import BaseModel, Field, field_validator, model_validator

from verathos.protocol import signing
from verathos.protocol.canonical import digest

MANIFEST_DOMAIN = "verathos/manifest/v1"
COMMITMENT_PREFIX = "verathos/manifest/v1 sha256:"
MAX_ENDPOINTS = 64
_ENDPOINT_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,31}$")


class Endpoint(BaseModel):
    endpoint_id: str
    url: str = Field(min_length=1, max_length=256)
    model_id: str = Field(min_length=1, max_length=128)
    tier: Literal["open", "private"] = "open"
    capacity_rps: float = Field(ge=0, le=100_000, description="Declared serving capacity in requests/s of the "
                                "model's standard request mix, measured by the miner; the throughput audit tests it")

    @field_validator("endpoint_id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        if not _ENDPOINT_ID.match(value):
            raise ValueError("endpoint_id must be 1-32 chars of a-z, 0-9 and '-'")
        return value

    @field_validator("url")
    @classmethod
    def _check_url(cls, value: str) -> str:
        if not value.startswith(("https://", "http://")):
            raise ValueError("url must be http(s)")
        return value


class Manifest(BaseModel):
    version: Literal[1] = 1
    netuid: int = Field(ge=0)
    mechid: int = Field(ge=0)
    hotkey: str
    catalog_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    endpoints: list[Endpoint] = Field(max_length=MAX_ENDPOINTS)

    @model_validator(mode="after")
    def _unique(self) -> Manifest:
        ids = [e.endpoint_id for e in self.endpoints]
        if len(ids) != len(set(ids)):
            raise ValueError("endpoint ids must be unique")
        return self

    def digest(self) -> str:
        return digest(self.model_dump(mode="json"))


class SignedManifest(BaseModel):
    manifest: Manifest
    signature: str

    @classmethod
    def create(cls, manifest: Manifest, keypair: Keypair) -> SignedManifest:
        if keypair.ss58_address != manifest.hotkey:
            raise ValueError("keypair does not match manifest.hotkey")
        return cls(manifest=manifest,
                   signature=signing.sign(keypair, MANIFEST_DOMAIN, manifest.model_dump(mode="json")))


def commitment_value(manifest_digest: str) -> str:
    """The string a miner commits on chain for a manifest digest."""
    return COMMITMENT_PREFIX + manifest_digest


def committed_digest(commitment: str | None) -> str | None:
    """The manifest digest inside a commitment string, or None if it is not a manifest commitment."""
    if not commitment or not commitment.startswith(COMMITMENT_PREFIX):
        return None
    value = commitment[len(COMMITMENT_PREFIX):]
    return value if re.fullmatch(r"[0-9a-f]{64}", value) else None


class ManifestRejected(Exception):
    pass


def axon_base(ip: str, port: int) -> str:
    """The base URL of an axon (``http://ip:port``)."""
    return f"http://[{ip}]:{port}" if ":" in ip else f"http://{ip}:{port}"


def endpoint_url(axon: str, endpoint_id: str) -> str:
    """The URL of an endpoint on its hotkey's axon; manifests must list exactly this URL."""
    return f"{axon}/e/{endpoint_id}"


def public_ip(value: str) -> str:
    """``value`` if it is an IP address literal (the chain stores axons as IPs); raises ValueError otherwise."""
    ipaddress.ip_address(value)
    return value


def accept(document: Any, *, hotkey: str, netuid: int, mechid: int, axon: str, commitment: str | None,
           catalog_digests: Collection[str]) -> Manifest:
    """Validator-side acceptance of a fetched manifest document (the parsed JSON body). Raises ManifestRejected."""
    if not isinstance(document, dict) or not isinstance(document.get("manifest"), dict):
        raise ManifestRejected("not a signed manifest")
    raw = document["manifest"]
    if not signing.verify(hotkey, MANIFEST_DOMAIN, raw, document.get("signature")):
        raise ManifestRejected("signature does not verify against the axon's hotkey")
    expected = committed_digest(commitment)
    if expected is None:
        raise ManifestRejected("hotkey has no manifest commitment")
    try:
        if digest(raw) != expected:
            raise ManifestRejected("manifest digest differs from the on-chain commitment")
        manifest = Manifest.model_validate(raw)
    except (ValueError, TypeError) as exc:
        if isinstance(exc, ManifestRejected):
            raise
        raise ManifestRejected(f"malformed manifest: {exc}") from None
    if manifest.hotkey != hotkey or manifest.netuid != netuid or manifest.mechid != mechid:
        raise ManifestRejected("manifest is for another hotkey, netuid or mechanism")
    if manifest.catalog_digest not in catalog_digests:
        raise ManifestRejected("manifest was built for another catalog")
    for e in manifest.endpoints:
        if e.url != endpoint_url(axon, e.endpoint_id):
            raise ManifestRejected(f"endpoint {e.endpoint_id} is not served at the hotkey's axon")
    return manifest
