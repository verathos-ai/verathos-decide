"""Who calls miners, and from which addresses.

Miners answer the auditor, the catalog's gateways and validators with a permit and the catalog's
``validator_min_stake`` (``miner.chain.PermitAuthorizer``). Where each sends its requests from (its egress addresses) is
published next to its TLS pin:
- the auditor's in the catalog (``auditor``: ``tls`` and ``egress``), each gateway's there too (``gateways``: ``hotkey``
  and ``egress``);
- a validator's in its chain commitment, ``verathos/validator/v1 <pin> <addresses>`` (``ValidatorIdentity``): the pin
  of its service's certificate, if it serves others (``-`` if not), as unpadded base64url of its 32 bytes, and its
  egress addresses, comma-separated (``-`` for none). A commitment holds at most 128 bytes.

Addresses are IP literals in their normal form (as the chain writes axons) on public unicast addresses, as manifest
URLs are. ``allowlist`` gives every address a permitted caller sends from, for a firewall that drops other traffic
before a TLS handshake.
"""

from __future__ import annotations

import base64
import binascii
import ipaddress
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from verathos.protocol.manifest import is_public, public_ip
from verathos.protocol.tls import check_pin, is_pin

if TYPE_CHECKING:
    from verathos.catalog import Catalog

VALIDATOR_PREFIX = "verathos/validator/v1 "
MAX_COMMITMENT_BYTES = 128  # a commitment's raw field
MAX_EGRESS = 16
NONE = "-"
_B64_PIN = re.compile(r"^[A-Za-z0-9_-]{43}$")


def egress_address(value: str) -> str:
    """``value`` if it is a public unicast IP literal in its normal form; raises ValueError otherwise."""
    try:
        normal = public_ip(value)
    except ValueError:
        raise ValueError(f"{value!r} is not an IP address") from None
    if normal != value:
        raise ValueError(f"{value!r}: write the address as {normal}")
    if not is_public(value):
        raise ValueError(f"{value} is not a public unicast address")
    return value


def egress_addresses(values: list[str]) -> list[str]:
    """``values``, each an ``egress_address``, without repeats; raises ValueError otherwise."""
    out = [egress_address(v) for v in values]
    if len(out) != len(set(out)):
        raise ValueError("egress addresses must be unique")
    if len(out) > MAX_EGRESS:
        raise ValueError(f"at most {MAX_EGRESS} egress addresses")
    return out


@dataclass(frozen=True)
class ValidatorIdentity:
    """What a validator commits: its service's TLS pin (None if it serves no one) and its egress addresses."""

    tls: str | None
    egress: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.tls is not None:
            check_pin(self.tls)
        egress_addresses(list(self.egress))

    def commitment(self) -> str:
        """The commitment string; raises ValueError if it exceeds ``MAX_COMMITMENT_BYTES``."""
        pin = base64.urlsafe_b64encode(bytes.fromhex(self.tls)).rstrip(b"=").decode() if self.tls else NONE
        value = f"{VALIDATOR_PREFIX}{pin} {','.join(self.egress) or NONE}"
        if len(value.encode()) > MAX_COMMITMENT_BYTES:
            raise ValueError(f"the validator commitment would be {len(value.encode())} bytes, more than "
                             f"{MAX_COMMITMENT_BYTES}: list fewer egress addresses")
        return value


def validator_identity(commitment: str | None) -> ValidatorIdentity | None:
    """The identity in a validator commitment, or None if ``commitment`` is not one, or not a valid one."""
    if not commitment or not commitment.startswith(VALIDATOR_PREFIX):
        return None
    parts = commitment[len(VALIDATOR_PREFIX):].split(" ")
    if len(parts) != 2:
        return None
    pin, egress = parts
    try:
        tls = None
        if pin != NONE:
            raw = base64.urlsafe_b64decode(pin + "=") if _B64_PIN.match(pin) else b""
            if base64.urlsafe_b64encode(raw).rstrip(b"=").decode() != pin or not is_pin(raw.hex()):
                return None
            tls = raw.hex()
        return ValidatorIdentity(tls, () if egress == NONE else tuple(egress.split(",")))
    except (ValueError, binascii.Error):
        return None


class Caller(Protocol):
    """A neuron as a chain snapshot gives it (``validator.discovery.Neuron``)."""

    hotkey: str
    commitment: str | None
    stake: float
    permit: bool


def allowlist(neurons: Iterable[Caller], cat: Catalog) -> list[str]:
    """The egress addresses of the callers miners answer under ``cat``: the auditor's and the gateways' (the catalog),
    and those of every validator with a permit and at least ``validator_min_stake`` that publishes its identity. IPv4
    first, then IPv6, each in order."""
    addresses = set(cat.auditor.egress if cat.auditor is not None else ())
    addresses.update(a for g in cat.gateways for a in g.egress)
    floor = cat.scoring.validator_min_stake
    for n in neurons:
        if n.permit and n.stake >= floor and (identity := validator_identity(n.commitment)) is not None:
            addresses.update(identity.egress)
    return sorted(addresses, key=lambda a: (ipaddress.ip_address(a).version, ipaddress.ip_address(a)))
