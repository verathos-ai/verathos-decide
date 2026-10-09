"""TLS on every connection between machines of the subnet.

Every serving process (a miner's service on each machine, a validator's service) holds a TLS key and a self-signed
certificate, created once in its state directory (``tls.key`` and ``tls.crt``, mode 600) and reused across restarts;
deleting both rotates them (``identity``). Peers are reached by IP address, where no public CA applies: a certificate
is accepted by its pin, the SHA-256 of its SubjectPublicKeyInfo (DER), written as 64 lowercase hex digits (``pin_of``).
Pins reach their clients through documents the chain authenticates: a miner's manifest, the catalog (the auditor's),
a validator's commitment.

Clients (``client``, ``sync_client``) speak https only. Each request names in its ``tls`` extension (``use``) what it
accepts:
- a pin: exactly the certificate with that pin, checked once the handshake completes and before any byte of the
  request is sent. Connections are pooled per pin, so a connection checked for one pin never carries a request for
  another;
- ``ANY``: any certificate, whose pin ``peer_pin`` gives; only for documents that are authenticated otherwise (a
  manifest signed by its hotkey and committed on chain, a catalog fetched by its digest), whose pin is then checked
  against the one they name;
- ``CA``: a certificate of a public CA for the URL's hostname (catalog mirrors, a remote engine without a pin).
A request without one is refused.
"""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import ipaddress
import os
import re
import secrets
import ssl
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

KEY_FILE, CERT_FILE = "tls.key", "tls.crt"
PIN = re.compile(r"^[0-9a-f]{64}$")
PIN_PATTERN = PIN.pattern
EXTENSION = "verathos_tls"
ANY = "any"  # any certificate: only for documents authenticated otherwise
CA = "ca"  # a public CA's certificate for the hostname
MAX_POOLS = 4096  # pins with a connection pool at once; the least recently used one is closed beyond
NOT_AFTER = dt.datetime(9999, 12, 31, 23, 59, 59, tzinfo=dt.UTC)  # RFC 5280: no well-defined expiration
DEFAULT_LIMITS = httpx.Limits(max_connections=None, max_keepalive_connections=256)


class PinMismatch(ssl.SSLCertVerificationError):
    """The peer's certificate is not the pinned one."""

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else "certificate pin mismatch"


def is_pin(value: object) -> bool:
    return isinstance(value, str) and PIN.match(value) is not None


def check_pin(value: str) -> str:
    """``value`` if it is a pin; raises ValueError otherwise (for validators)."""
    if not is_pin(value):
        raise ValueError("a TLS pin is the SHA-256 of the certificate's public key, 64 lowercase hex digits")
    return value


def pin_of(certificate: x509.Certificate | bytes) -> str:
    """The pin of a certificate (an object, or its DER encoding): SHA-256 of its SubjectPublicKeyInfo."""
    if isinstance(certificate, bytes):
        certificate = x509.load_der_x509_certificate(certificate)
    spki = certificate.public_key().public_bytes(serialization.Encoding.DER,
                                                 serialization.PublicFormat.SubjectPublicKeyInfo)
    return hashlib.sha256(spki).hexdigest()


def curl_pin(pin: str) -> str:
    """The pin as curl's ``--pinnedpubkey`` takes it (``sha256//`` and base64)."""
    return "sha256//" + base64.b64encode(bytes.fromhex(check_pin(pin))).decode()


@dataclass(frozen=True)
class Identity:
    """A serving process's TLS key and certificate, and the certificate's pin."""

    key_path: Path
    cert_path: Path
    pin: str


def _write_private(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` with mode 600 from the start, replacing it atomically."""
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def certificate_for(key: ec.EllipticCurvePrivateKey) -> x509.Certificate:
    """A self-signed certificate for ``key``, without expiry."""
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "verathos")])
    now = dt.datetime.now(dt.UTC)
    return (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - dt.timedelta(days=1))
            .not_valid_after(NOT_AFTER)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .sign(key, hashes.SHA256()))


def identity(directory: Path) -> Identity:
    """The TLS identity kept in ``directory``: created on first use (an ECDSA P-256 key and a self-signed certificate
    for it without expiry, both mode 600), reused after. A certificate missing, unreadable or not for the key is made
    again for the same key (same pin); a missing key starts a new identity (a new pin)."""
    directory = directory.expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    key_path, cert_path = directory / KEY_FILE, directory / CERT_FILE
    try:
        key = serialization.load_pem_private_key(key_path.read_bytes(), password=None)
    except FileNotFoundError:
        key = ec.generate_private_key(ec.SECP256R1())
        cert_path.unlink(missing_ok=True)
        _write_private(key_path, key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                                   serialization.NoEncryption()))
    if not isinstance(key, ec.EllipticCurvePrivateKey):
        raise ValueError(f"{key_path} is not an ECDSA key; delete {key_path} and {cert_path} to create a new identity")
    os.chmod(key_path, 0o600)
    try:
        cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
        if cert.public_key() != key.public_key():
            raise ValueError("certificate for another key")
    except (FileNotFoundError, ValueError):
        cert = certificate_for(key)
        _write_private(cert_path, cert.public_bytes(serialization.Encoding.PEM))
    os.chmod(cert_path, 0o600)
    return Identity(key_path, cert_path, pin_of(cert))


def stored_pin(directory: Path) -> str | None:
    """The pin of the certificate kept in ``directory``, or None if there is none (nothing is created)."""
    try:
        return pin_of(x509.load_pem_x509_certificate((directory.expanduser() / CERT_FILE).read_bytes()))
    except (FileNotFoundError, ValueError):
        return None


@dataclass(frozen=True)
class Service:
    """A service on another machine: its base URL (https) and the pin of its certificate."""

    url: str
    pin: str


def server_options(ident: Identity) -> dict[str, Any]:
    """uvicorn ``Config`` arguments that serve TLS with ``ident``."""
    return {"ssl_keyfile": str(ident.key_path), "ssl_certfile": str(ident.cert_path)}


# ---- clients ----

def _verify(der: bytes | None, pin: str) -> None:
    seen = pin_of(der) if der else None
    if seen != pin:
        raise PinMismatch(f"certificate pin {seen or 'none'} is not the pinned {pin}")


class _PinnedObject(ssl.SSLObject):
    def do_handshake(self) -> None:
        super().do_handshake()
        _verify(self.getpeercert(binary_form=True), self.context.pin)


class _PinnedSocket(ssl.SSLSocket):
    def do_handshake(self, block: bool = False) -> None:
        super().do_handshake(block)
        _verify(self.getpeercert(binary_form=True), self.context.pin)


class _PinnedContext(ssl.SSLContext):
    """Accepts exactly the certificate whose pin is ``pin``, whatever its issuer, names and dates."""

    sslobject_class = _PinnedObject
    sslsocket_class = _PinnedSocket
    pin: str


def context(tls: str) -> ssl.SSLContext | bool:
    """The client SSL context for a ``tls`` mode (a pin, ``ANY`` or ``CA``); ``True`` stands for httpx's default
    (the public CAs)."""
    if tls == CA:
        return True
    if tls != ANY and not is_pin(tls):
        raise ValueError(f"not a TLS pin, {ANY!r} or {CA!r}: {tls!r}")
    ctx = _PinnedContext(ssl.PROTOCOL_TLS_CLIENT) if tls != ANY else ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE  # the pin is the check: self-signed, any name, no dates
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    if tls != ANY:
        ctx.pin = tls
    return ctx


def use(tls: str) -> dict[str, str]:
    """The request extension that makes a request accept ``tls`` (a pin, ``ANY`` or ``CA``)."""
    return {EXTENSION: tls}


def mode_of(request: httpx.Request) -> str:
    """What ``request`` accepts. Raises httpx.UnsupportedProtocol for a plain-http URL and httpx.ConnectError for a
    request that names nothing (or something that is not a mode)."""
    if request.url.scheme != "https":
        raise httpx.UnsupportedProtocol(f"{request.url.scheme}://{request.url.netloc.decode()}: only https is used "
                                        "between machines", request=request)
    tls = request.extensions.get(EXTENSION)
    if tls != ANY and tls != CA and not is_pin(tls):
        raise httpx.ConnectError(f"no TLS pin for {request.url.netloc.decode()}", request=request)
    if tls == CA and _is_ip(request.url.host):
        raise httpx.ConnectError(f"{request.url.host}: an IP address has no CA certificate; give its pin",
                                 request=request)
    return tls


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


class PinnedTransport(httpx.AsyncBaseTransport):
    """https only; one connection pool per mode (each pin, ``ANY``, ``CA``), each with ``limits``."""

    def __init__(self, *, limits: httpx.Limits = DEFAULT_LIMITS, max_pools: int = MAX_POOLS) -> None:
        self._limits, self._max_pools = limits, max_pools
        self._pools: OrderedDict[str, httpx.AsyncHTTPTransport] = OrderedDict()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        tls = mode_of(request)
        pool = self._pools.get(tls)
        if pool is None:
            pool = self._pools[tls] = httpx.AsyncHTTPTransport(verify=context(tls), limits=self._limits)
            while len(self._pools) > self._max_pools:
                await self._pools.popitem(last=False)[1].aclose()
        self._pools.move_to_end(tls)
        return await pool.handle_async_request(request)

    async def aclose(self) -> None:
        pools, self._pools = list(self._pools.values()), OrderedDict()
        for pool in pools:
            await pool.aclose()


class SyncPinnedTransport(httpx.BaseTransport):
    """``PinnedTransport`` for synchronous clients."""

    def __init__(self, *, limits: httpx.Limits = DEFAULT_LIMITS, max_pools: int = MAX_POOLS) -> None:
        self._limits, self._max_pools = limits, max_pools
        self._pools: OrderedDict[str, httpx.HTTPTransport] = OrderedDict()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        tls = mode_of(request)
        pool = self._pools.get(tls)
        if pool is None:
            pool = self._pools[tls] = httpx.HTTPTransport(verify=context(tls), limits=self._limits)
            while len(self._pools) > self._max_pools:
                self._pools.popitem(last=False)[1].close()
        self._pools.move_to_end(tls)
        return pool.handle_request(request)

    def close(self) -> None:
        pools, self._pools = list(self._pools.values()), OrderedDict()
        for pool in pools:
            pool.close()


def client(*, limits: httpx.Limits = DEFAULT_LIMITS, **kwargs: Any) -> httpx.AsyncClient:
    """An async client for other machines (see the module). Proxies from the environment are not used: they would
    bypass the pin checks."""
    return httpx.AsyncClient(transport=PinnedTransport(limits=limits), trust_env=False, **kwargs)


def sync_client(*, limits: httpx.Limits = DEFAULT_LIMITS, **kwargs: Any) -> httpx.Client:
    """``client`` for synchronous code."""
    return httpx.Client(transport=SyncPinnedTransport(limits=limits), trust_env=False, **kwargs)


def peer_pin(response: httpx.Response) -> str | None:
    """The pin of the certificate the server of ``response`` presented, or None (no TLS connection)."""
    stream = response.extensions.get("network_stream")
    ssl_object = stream.get_extra_info("ssl_object") if stream is not None else None
    der = ssl_object.getpeercert(binary_form=True) if ssl_object is not None else None
    return pin_of(der) if der else None


def authenticated(url: str) -> str:
    """What to accept from ``url`` for a document authenticated otherwise (a catalog by its digest): a public CA's
    certificate for a hostname, any certificate at an IP address (none has a CA certificate)."""
    return ANY if _is_ip(urlsplit(url).hostname or "") else CA


def https_url(value: str, *, what: str = "url") -> str:
    """``value`` if it is an https URL with a host; raises ValueError otherwise (for validators)."""
    parts = urlsplit(value)
    if parts.scheme != "https" or not parts.hostname:
        raise ValueError(f"{what} {value!r} must be an https:// URL: every connection between machines is TLS")
    return value


def chain_network(value: str) -> str:
    """``value`` if the chain node it names is reached over TLS or on this machine: a network name the bittensor client
    maps to such a node (``finney``, ``test``, ``local``), a ``wss://`` URL, or a ``ws://`` URL on loopback; raises
    ValueError otherwise (for validators)."""
    from bittensor.core.settings import NETWORK_MAP

    url = NETWORK_MAP.get(value, value)
    parts = urlsplit(url)
    if parts.scheme == "wss" and parts.hostname or parts.scheme == "ws" and loopback_url(url):
        return value
    raise ValueError(f"network {value!r}: a chain node is reached over wss:// (finney, test or a wss:// URL), plain "
                     "ws:// only on this machine (local or ws://127.0.0.1:<port>): every connection between machines "
                     "is TLS")


def loopback_url(value: str) -> bool:
    """Whether ``value`` is a URL on this machine (localhost or a loopback address)."""
    host = urlsplit(value).hostname or ""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False
