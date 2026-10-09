"""Miner manifest: the endpoints a hotkey serves.

Discovery uses only native chain state:
- the hotkey's axon (``serve_axon``) points at the miner's manifest server, ``GET /v1/manifest``;
- the manifest lists every endpoint with its model and declared capacity, and is signed by the hotkey;
- the hotkey's commitment (``set_commitment``) holds the manifest digest, so every validator scores the same
  version.

A fetched manifest is accepted only if its signature verifies against the hotkey, its digest equals the hotkey's
current commitment, it is built for a catalog the validator accepts and for that catalog's mechanism, every endpoint
URL is ``https://<ip>:<port>/e/<endpoint_id>`` on a public unicast address (the axon, or another machine of the
miner), the manifest names the TLS pin of every machine it lists (``tls``: base URL -> pin, ``protocol.tls``), the
axon's included, and the axon presented the certificate with that pin when it served the manifest. Every later
connection to a machine accepts exactly its pin. Signature and digest cover the JSON exactly as received, so nodes
running different code versions agree on them. A listed host gets traffic only once the endpoint proved control at its
URL with its pin (``validator.discovery.Identities``).

A hotkey with GPUs on several machines publishes from one of them, the primary. Each other machine serves the
endpoints it measured, with its own pin, as a ``HostEndpoints`` document signed by the hotkey (``GET /v1/endpoints``),
which the primary checks (``accept_host``, against the certificate the machine presented) and lists.
"""

from __future__ import annotations

import ipaddress
import re
from collections.abc import Mapping
from typing import Annotated, Any, Literal

from bittensor_wallet import Keypair
from pydantic import BaseModel, Field, field_validator, model_validator

from verathos.protocol import signing
from verathos.protocol.canonical import digest
from verathos.protocol.tls import PIN_PATTERN

MANIFEST_DOMAIN = "verathos/manifest/v1"
COMMITMENT_PREFIX = "verathos/manifest/v1 sha256:"
ENDPOINTS_DOMAIN = "verathos/endpoints/v1"
ENDPOINTS_PATH = "/v1/endpoints"
MAX_ENDPOINTS = 64
MAX_HOSTS = 64  # distinct hosts (ip:port) per manifest
_ENDPOINT_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,31}$")
_BASE = r"https://(?:\[([0-9a-f:]+)\]|([0-9.]+)):([0-9]{1,5})"
_ENDPOINT_URL = re.compile(rf"^{_BASE}/e/[^/?#]*$")
_BASE_URL = re.compile(rf"^{_BASE}$")
# IPv6 forms that stand for an IPv4 address (mapped, NAT64, 6to4): a host's IPv4 address is listed as itself
_TRANSLATED = tuple(ipaddress.ip_network(n) for n in ("::ffff:0:0/96", "64:ff9b::/96", "64:ff9b:1::/48", "2002::/16"))


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
        if not value.startswith("https://"):
            raise ValueError("url must be https")
        return value


class Manifest(BaseModel):
    version: Literal[1] = 1
    netuid: int = Field(ge=0)
    mechid: int = Field(ge=0)
    hotkey: str
    catalog_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    endpoints: list[Endpoint] = Field(max_length=MAX_ENDPOINTS)
    tls: dict[str, Annotated[str, Field(pattern=PIN_PATTERN)]] = Field(
        max_length=MAX_HOSTS, description="Each machine's base URL (https://ip:port) -> the pin of its certificate")

    @model_validator(mode="after")
    def _unique(self) -> Manifest:
        ids = [e.endpoint_id for e in self.endpoints]
        if len(ids) != len(set(ids)):
            raise ValueError("endpoint ids must be unique")
        return self

    def pin(self, host: str) -> str | None:
        """The pin of the machine at ``host`` (a base URL), if the manifest names one."""
        return self.tls.get(host)

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


class HostEndpoints(BaseModel):
    """The endpoints one machine of a hotkey serves (``GET /v1/endpoints`` on it), as the hotkey's manifest lists
    them, the machine's base URL (``https://ip:port``) and the pin of its certificate: the machine that publishes the
    manifest gathers them. It holds nothing the manifest does not publish."""

    version: Literal[1] = 1
    netuid: int = Field(ge=0)
    mechid: int = Field(ge=0)
    hotkey: str
    catalog_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    url: str = Field(max_length=64)
    tls: str = Field(pattern=PIN_PATTERN)
    endpoints: list[Endpoint] = Field(max_length=MAX_ENDPOINTS)

    @model_validator(mode="after")
    def _unique(self) -> HostEndpoints:
        ids = [e.endpoint_id for e in self.endpoints]
        if len(ids) != len(set(ids)):
            raise ValueError("endpoint ids must be unique")
        return self


class SignedHostEndpoints(BaseModel):
    host: HostEndpoints
    signature: str

    @classmethod
    def create(cls, host: HostEndpoints, keypair: Keypair) -> SignedHostEndpoints:
        if keypair.ss58_address != host.hotkey:
            raise ValueError("keypair does not match host.hotkey")
        return cls(host=host, signature=signing.sign(keypair, ENDPOINTS_DOMAIN, host.model_dump(mode="json")))


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
    """The base URL of a service at ``ip:port``: a miner's axon or another of its machines, a validator's axon
    (``https://ip:port``)."""
    return f"https://[{ip}]:{port}" if ":" in ip else f"https://{ip}:{port}"


def endpoint_url(host: str, endpoint_id: str) -> str:
    """The URL of an endpoint on the miner service at ``host`` (``axon_base``); manifests list exactly this URL."""
    return f"{host}/e/{endpoint_id}"


def public_ip(value: str) -> str:
    """``value`` as a normalised IP address literal (the chain stores axons as IPs, and gives IPv6 back in its
    normal form); raises ValueError for anything else."""
    return str(ipaddress.ip_address(value))


def public_unicast(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Whether validators, the auditor and gateways may connect to ``ip``: a global unicast address (not private,
    loopback, link-local, multicast, reserved or unspecified), and not an IPv6 form of an IPv4 address."""
    if ip.version == 6 and any(ip in network for network in _TRANSLATED):
        return False
    return ip.is_global and not ip.is_multicast and not ip.is_reserved


def is_public(value: str) -> bool:
    """Whether ``value`` is an IP address literal that is ``public_unicast``."""
    try:
        return public_unicast(ipaddress.ip_address(value))
    except ValueError:
        return False


def _public_base(match: re.Match[str] | None) -> str | None:
    if match is None:
        return None
    try:
        ip = ipaddress.ip_address(match[1] or match[2])
    except ValueError:
        return None
    if not public_unicast(ip) or not 0 < int(match[3]) < 65536:
        return None
    return axon_base(str(ip), int(match[3]))


def endpoint_host(url: str, endpoint_id: str) -> str | None:
    """The host (``https://ip:port``) of an endpoint URL in the one form a manifest may list: exactly
    ``https://<ip>:<port>/e/<endpoint_id>``, the IP a ``public_unicast`` literal as the chain writes it (``public_ip``)
    and the port in 1-65535; None for any other URL."""
    host = _public_base(_ENDPOINT_URL.match(url))
    return host if host is not None and url == endpoint_url(host, endpoint_id) else None


def public_base(url: str) -> str | None:
    """``url`` if it is a machine's base URL in the one form a manifest may name: exactly ``https://<ip>:<port>``,
    the IP a ``public_unicast`` literal as the chain writes it; None otherwise."""
    host = _public_base(_BASE_URL.match(url))
    return host if host == url else None


def accept(document: Any, *, hotkey: str, netuid: int, commitment: str | None, catalogs: Mapping[str, int],
           axon: str, seen: str | None) -> Manifest:
    """Validator-side acceptance of a manifest document (the parsed JSON body) fetched from ``axon`` (its base URL),
    which presented the certificate with pin ``seen``: ``catalogs`` maps the digest of each catalog a manifest may be
    built for to the mechanism a manifest built for it is for. Raises ManifestRejected."""
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
    if manifest.hotkey != hotkey or manifest.netuid != netuid:
        raise ManifestRejected("manifest is for another hotkey or netuid")
    if manifest.catalog_digest not in catalogs:
        raise ManifestRejected("manifest was built for another catalog")
    if manifest.mechid != catalogs[manifest.catalog_digest]:
        raise ManifestRejected(f"manifest is for mechanism {manifest.mechid}, not {catalogs[manifest.catalog_digest]}")
    for base in manifest.tls:
        if public_base(base) is None:
            raise ManifestRejected(f"it names a certificate for {base!r}, not https://<public IP>:<port>")
    for e in manifest.endpoints:
        host = endpoint_host(e.url, e.endpoint_id)
        if host is None:
            raise ManifestRejected(f"endpoint {e.endpoint_id}: its URL is not https://<public IP>:<port>/e/"
                                   f"{e.endpoint_id}")
        if host not in manifest.tls:
            raise ManifestRejected(f"endpoint {e.endpoint_id}: the manifest names no certificate for {host}")
    pinned = manifest.pin(axon)
    if pinned is None:
        raise ManifestRejected(f"the manifest names no certificate for its axon {axon}")
    if seen != pinned:
        raise ManifestRejected(f"the axon's TLS certificate (pin {seen or 'none'}) is not the one its manifest names "
                               f"({pinned})")
    return manifest


def accept_host(document: Any, *, hotkey: str, netuid: int, mechid: int, url: str, seen: str | None) -> HostEndpoints:
    """The publishing machine's check of what another machine of its hotkey serves at ``url`` (``GET
    /v1/endpoints``, the parsed JSON body), which presented the certificate with pin ``seen``: signed by the hotkey,
    for this netuid and mechanism, naming that certificate's pin, and every endpoint at a URL a manifest may list on
    that very machine. Raises ManifestRejected."""
    if not isinstance(document, dict) or not isinstance(document.get("host"), dict):
        raise ManifestRejected("not a signed endpoints document")
    raw = document["host"]
    if not signing.verify(hotkey, ENDPOINTS_DOMAIN, raw, document.get("signature")):
        raise ManifestRejected("not signed by this hotkey")
    try:
        host = HostEndpoints.model_validate(raw)
    except ValueError as exc:
        raise ManifestRejected(f"malformed endpoints document: {exc}") from None
    if host.hotkey != hotkey or host.netuid != netuid or host.mechid != mechid:
        raise ManifestRejected("for another hotkey, netuid or mechanism")
    if host.url != url:
        raise ManifestRejected(f"it says it is {host.url}")
    if host.tls != seen:
        raise ManifestRejected(f"its TLS certificate (pin {seen or 'none'}) is not the one it names ({host.tls})")
    for e in host.endpoints:
        if endpoint_host(e.url, e.endpoint_id) != url:
            raise ManifestRejected(f"endpoint {e.endpoint_id} is not listed at {url}/e/{e.endpoint_id} on a public "
                                   "address")
    return host
