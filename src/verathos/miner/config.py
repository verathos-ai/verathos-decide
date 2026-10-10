"""Miner configuration."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from verathos import configfile
from verathos.catalog import Gateway
from verathos.miner.engine import engine_tls
from verathos.protocol import tls
from verathos.protocol.manifest import MAX_HOSTS, axon_base, endpoint_url, public_ip
from verathos.protocol.referral import site_origin

_PEER = re.compile(r"^(?:https://)?(?:\[([0-9A-Fa-f:.]+)\]|([0-9.]+)):([0-9]{1,5})/?$")


def peer_url(value: str) -> str:
    """A machine's miner base URL, ``https://ip:port``, from ``ip:port``, ``[ipv6]:port`` or the URL itself; raises
    ValueError for anything else."""
    match = _PEER.match(value.strip())
    if match is None or not 0 < int(match[3]) < 65536:
        raise ValueError(f"{value!r} is not <ip>:<port>")
    return axon_base(public_ip(match[1] or match[2]), int(match[3]))


class EndpointConfig(BaseModel):
    endpoint_id: str
    model_id: str
    engine_url: str = Field(description="Decision engine serving POST /v1/systemone: on this machine (e.g. "
                            "http://127.0.0.1:8008), or https with engine_tls_pin or a public CA's certificate")
    engine_api_key: str | None = None
    engine_tls_pin: str | None = Field(None, pattern=tls.PIN_PATTERN, description="Pin of a remote https engine's "
                                       "certificate; none: a public CA's certificate for its hostname")
    max_concurrency: int = Field(256, ge=1, le=4096, description="Highest concurrency the capacity measurement tries; "
                                 "the engine then never gets more than the concurrency that measured best")
    tier: Literal["open", "private"] = "open"

    @model_validator(mode="after")
    def _engine_tls(self) -> EndpointConfig:
        engine_tls(self.engine_url, self.engine_tls_pin)
        return self


class MinerConfig(BaseModel):
    network: str = "finney"
    netuid: int
    mechid: int | None = Field(None, ge=0, description="Mechanism to publish the manifest for instead of the "
                               "catalog's; none: the catalog's")
    wallet_name: str
    hotkey_name: str
    public_host: str = Field(description="Public IP address validators and gateways reach this miner at")
    port: int = Field(ge=1, le=65535)
    listen_host: str = "0.0.0.0"
    role: Literal["primary", "worker"] = Field("primary", description="primary: publishes the hotkey's manifest, its "
                                               "own endpoints and its peers'; worker: serves its endpoints for a "
                                               "primary, publishes nothing on chain")
    peers: list[str] = Field(default_factory=list, description="Primary only: the workers' miner base URLs "
                             "(ip:port), whose endpoints the manifest lists")
    gateways: list[Gateway] = Field(default_factory=list, description="The miner's own gateways besides the catalog's, "
                                    "each {hotkey, egress}: callers served like the catalog's, from the public IP "
                                    "addresses their requests come from (egress)")
    endpoints: list[EndpointConfig] = Field(min_length=1)
    catalog_mirrors: list[str] = Field(default_factory=list, description="Catalog URL templates (https) with "
                                       "{digest}, tried after the subnet owner's axon")
    catalog_dir: Path = Path("~/.verathos/catalog")
    state_dir: Path = Path("~/.verathos/miner")
    firewall: bool = Field(True, description="Drop connections to the port from anyone but the network's callers and "
                           "the hotkey's machines, in the kernel (miner.firewall); takes the helper the installer sets "
                           "up, or root")
    auto_update: bool = Field(False, description="Update the code to each new release of the repository it was "
                              "installed from (release tags), at a quiet moment, and exit for the process manager to "
                              "start it (the installer turns it on; verathos.update)")
    referral_site: str | None = Field(None, description="The web app the referral commands talk to (https://host; "
                                      "verathos-miner referral)")

    @field_validator("network")
    @classmethod
    def _network(cls, value: str) -> str:
        return tls.chain_network(value)

    @field_validator("public_host")
    @classmethod
    def _public_ip(cls, value: str) -> str:
        return public_ip(value)

    @field_validator("catalog_mirrors")
    @classmethod
    def _mirrors(cls, value: list[str]) -> list[str]:
        return [tls.https_url(v, what="catalog mirror") for v in value]

    @field_validator("referral_site")
    @classmethod
    def _site(cls, value: str | None) -> str | None:
        return None if value is None else site_origin(value)

    @field_validator("peers")
    @classmethod
    def _peers(cls, value: list[str]) -> list[str]:
        peers = [peer_url(v) for v in value]
        if len(peers) != len(set(peers)):
            raise ValueError("peers must be unique")
        if len(peers) >= MAX_HOSTS:
            raise ValueError(f"at most {MAX_HOSTS - 1} peers")
        return peers

    @field_validator("gateways")
    @classmethod
    def _gateways(cls, value: list[Gateway]) -> list[Gateway]:
        hotkeys = [g.hotkey for g in value]
        if len(hotkeys) != len(set(hotkeys)):
            raise ValueError("gateway hotkeys must be unique")
        for g in value:
            if not g.egress:
                raise ValueError(f"gateway {g.hotkey}: name the addresses its requests come from (egress)")
        return value

    @model_validator(mode="after")
    def _check(self) -> MinerConfig:
        ids = [e.endpoint_id for e in self.endpoints]
        if len(ids) != len(set(ids)):
            raise ValueError("endpoint ids must be unique")
        if self.peers and self.role == "worker":
            raise ValueError("a worker has no peers: its primary lists it")
        if self.base_url in self.peers:
            raise ValueError("peers list the other machines, not this one")
        return self

    @property
    def base_url(self) -> str:
        """Where this machine's miner service is reached (``https://public_host:port``)."""
        return axon_base(self.public_host, self.port)

    def endpoint_url(self, endpoint_id: str) -> str:
        return endpoint_url(self.base_url, endpoint_id)

    @classmethod
    def load(cls, path: Path) -> MinerConfig:
        """The configuration in ``path``; raises SystemExit saying what is wrong, never with the values there."""
        return configfile.load(cls, path)
