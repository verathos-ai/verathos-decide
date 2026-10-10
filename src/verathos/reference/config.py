"""A reference machine's configuration (``reference.json``), its callers' key, and what it hands its callers."""

from __future__ import annotations

import ipaddress
import os
import secrets
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, field_validator

from verathos import configfile
from verathos.protocol.manifest import axon_base, public_ip
from verathos.protocol.tls import loopback_url

REFERENCE, SCREEN = "reference", "screen"  # a model's float32 reference and its bfloat16 screen
KINDS = (REFERENCE, SCREEN)
KEY_FILE = "api.key"
MAX_CALLERS = 256


def caller_address(value: str) -> str:
    """``value`` as an IP literal in its normal form (an IPv4-mapped IPv6 address as IPv4); raises ValueError for
    anything else, or an address no caller sends from."""
    try:
        ip = ipaddress.ip_address(value.strip())
    except ValueError:
        raise ValueError(f"{value!r} is not an IP address") from None
    if ip.version == 6 and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    if ip.is_unspecified or ip.is_multicast:
        raise ValueError(f"{ip} is not a caller's address")
    return str(ip)


class ReferenceConfig(BaseModel):
    """``reference.json``: the engines this machine runs on its loopback, and the front that serves them."""

    network: str = Field("finney", description="The chain whose subnet catalog pins the engines' models (the setup's "
                         "default on its next run)")
    netuid: int | None = Field(None, description="That subnet")
    public_host: str = Field(description="IP address the callers reach this machine at")
    port: int = Field(ge=1, le=65535, description="Port of the front (TLS)")
    listen_host: str = "0.0.0.0"
    allow: list[str] = Field(description="The callers' addresses; the front drops every other one (this machine's "
                             "loopback always passes)")
    engines: dict[str, dict[Literal["reference", "screen"], str]] = Field(description="model_id -> reference and "
                                                                          "screen -> the engine's URL on this "
                                                                          "machine's loopback (plain http)")
    state_dir: Path = Path("~/.verathos/reference")

    @field_validator("public_host")
    @classmethod
    def _ip(cls, value: str) -> str:
        return public_ip(value)

    @field_validator("allow")
    @classmethod
    def _callers(cls, value: list[str]) -> list[str]:
        out = list(dict.fromkeys(caller_address(v) for v in value))
        if len(out) > MAX_CALLERS:
            raise ValueError(f"at most {MAX_CALLERS} caller addresses")
        return out

    @field_validator("engines")
    @classmethod
    def _loopback(cls, value: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
        for kinds in value.values():
            for url in kinds.values():
                if urlsplit(url).scheme != "http" or not loopback_url(url):
                    raise ValueError(f"{url}: the engines serve plain http on this machine's loopback; the front "
                                     "serves them to other machines")
        return value

    @classmethod
    def load(cls, path: Path) -> ReferenceConfig:
        """The configuration in ``path``; raises SystemExit saying what is wrong, never with the values there."""
        return configfile.load(cls, path)

    def base_url(self) -> str:
        return axon_base(self.public_host, self.port)

    def served(self) -> list[tuple[str, str]]:
        """(kind, model) of every engine served, references first."""
        return [(kind, m) for kind in KINDS for m in sorted(self.engines) if kind in self.engines[m]]


def api_key(state_dir: Path) -> str:
    """The callers' bearer key, kept in ``state_dir`` (``api.key``, mode 600): made on first use, reused after; deleting
    the file makes a new one at the next start."""
    directory = state_dir.expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / KEY_FILE
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        pass
    else:
        with os.fdopen(fd, "w") as f:
            f.write(secrets.token_urlsafe(32) + "\n")
    os.chmod(path, 0o600)
    key = path.read_text().strip()
    if len(key) < 32:
        raise SystemExit(f"{path} holds no key; delete it to make a new one")
    return key


def engines_document(config: ReferenceConfig, pin: str, key: str) -> dict:
    """What a validator needs to use this machine's engines, as its ``validator.json`` takes them: ``references`` and
    ``audit_screens``, model -> the engine's URL on the front, the front's TLS pin and the key."""
    def entry(kind: str, model: str) -> dict[str, str]:
        return {"url": f"{config.base_url()}/{kind}/{model}", "tls_pin": pin, "api_key": key}

    return {"references": {m: entry(REFERENCE, m) for kind, m in config.served() if kind == REFERENCE},
            "audit_screens": {m: entry(SCREEN, m) for kind, m in config.served() if kind == SCREEN}}
