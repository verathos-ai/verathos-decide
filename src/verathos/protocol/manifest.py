"""Miner manifest: the endpoints a hotkey serves.

Discovery uses only native chain state:
- the hotkey's axon (``serve_axon``) points at the miner's manifest server, ``GET /v1/manifest``;
- the manifest lists every endpoint (one per GPU or Mac) with its model, and is signed by the hotkey;
- the hotkey's commitment (``set_commitment``) holds the manifest digest, so every validator scores the same
  version and the commitment block records when an endpoint was first claimed.

A fetched manifest is accepted only if its signature verifies against the hotkey and its digest equals the
hotkey's current commitment.
"""

from __future__ import annotations

import re
from typing import Literal

from bittensor_wallet import Keypair
from pydantic import BaseModel, Field, field_validator, model_validator

from verathos.protocol import signing
from verathos.protocol.canonical import digest

MANIFEST_DOMAIN = "verathos/manifest/v1"
COMMITMENT_PREFIX = "verathos/manifest/v1 sha256:"
MAX_ENDPOINTS = 64
_ENDPOINT_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,31}$")


class Hardware(BaseModel):
    kind: Literal["cuda", "apple"]
    device_id: str = Field(min_length=1, max_length=128)
    gpu_name: str | None = Field(None, max_length=128,
                                 description="Claimed GPU class; selects the capacity-audit workload")
    vram_gb: int | None = Field(None, ge=1, le=4096)


class Endpoint(BaseModel):
    endpoint_id: str
    url: str = Field(min_length=1, max_length=256)
    model_id: str = Field(min_length=1, max_length=128)
    tier: Literal["open", "private"] = "open"
    hardware: Hardware
    max_concurrency: int = Field(ge=1, le=4096)

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
    seq: int = Field(ge=0)
    issued_block: int = Field(ge=0)
    catalog_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    endpoints: list[Endpoint] = Field(max_length=MAX_ENDPOINTS)

    @model_validator(mode="after")
    def _unique(self) -> Manifest:
        ids = [e.endpoint_id for e in self.endpoints]
        if len(ids) != len(set(ids)):
            raise ValueError("endpoint ids must be unique")
        devices = [(e.hardware.kind, e.hardware.device_id) for e in self.endpoints]
        if len(devices) != len(set(devices)):
            raise ValueError("one hardware device may back only one endpoint")
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

    def signature_valid(self) -> bool:
        return signing.verify(self.manifest.hotkey, MANIFEST_DOMAIN, self.manifest.model_dump(mode="json"),
                              self.signature)


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


def accept(signed: SignedManifest, *, hotkey: str, netuid: int, commitment: str | None) -> Manifest:
    """Validator-side acceptance of a fetched manifest. Raises ManifestRejected with the reason."""
    manifest = signed.manifest
    if manifest.hotkey != hotkey:
        raise ManifestRejected("manifest hotkey does not match the axon's hotkey")
    if manifest.netuid != netuid:
        raise ManifestRejected("manifest is for another netuid")
    if not signed.signature_valid():
        raise ManifestRejected("signature does not verify against the hotkey")
    expected = committed_digest(commitment)
    if expected is None:
        raise ManifestRejected("hotkey has no manifest commitment")
    if manifest.digest() != expected:
        raise ManifestRejected("manifest digest differs from the on-chain commitment")
    return manifest
