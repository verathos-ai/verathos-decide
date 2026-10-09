"""Validator configuration."""

from __future__ import annotations

import ipaddress
import json
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from bittensor_wallet import Keypair
from pydantic import BaseModel, Field, field_validator, model_validator

from verathos.miner.engine import HttpEngine, engine_tls
from verathos.protocol.callers import egress_addresses
from verathos.protocol.manifest import public_ip
from verathos.protocol.tls import https_url


class ReferenceEngine(BaseModel):
    """An engine serving ``POST /v1/systemone``: on this machine over plain http, or remote over https, accepting its
    certificate by ``tls_pin`` or, without one, a public CA's certificate for its hostname, with a bearer key (the Kev
    server takes one from ``KEV_API_KEY``). Plain http off this machine is refused. With a ``command`` (references
    one at a time), the validator starts it on this machine when it needs it and stops it after; without one, the
    engine is taken as running."""

    url: str = Field(pattern=r"^https?://")
    api_key: str | None = None
    tls_pin: str | None = Field(None, description="Pin of a remote engine's TLS certificate (SHA-256 of its public "
                                "key, as verathos prints it); none: a public CA's certificate for its hostname")
    command: list[str] | None = Field(None, min_length=1, description="References one at a time: the command that "
                                      "serves the engine at url (the installer writes it)")
    directory: str | None = Field(None, description="Working directory of command")
    environment: dict[str, str] = Field(default_factory=dict, description="Environment variables of command, over "
                                        "the validator's")

    def local(self) -> bool:
        host = urlsplit(self.url).hostname or ""
        try:
            return host == "localhost" or ipaddress.ip_address(host).is_loopback
        except ValueError:
            return False

    @model_validator(mode="after")
    def _remote_needs_tls(self) -> ReferenceEngine:
        engine_tls(self.url, self.tls_pin)
        if self.command and not self.local():
            raise ValueError(f"{self.url}: an engine the validator starts serves on this machine (127.0.0.1)")
        return self

    def engine(self, timeout_s: float = 120.0) -> HttpEngine:
        return HttpEngine(self.url, self.api_key, timeout_s=timeout_s, tls_pin=self.tls_pin)


def _engines(value: Any) -> Any:
    """Plain URLs stand for engines without a key."""
    if isinstance(value, dict):
        return {k: {"url": v} if isinstance(v, str) else v for k, v in value.items()}
    return value


class ValidatorConfig(BaseModel):
    network: str = "finney"
    netuid: int
    mechid: int | None = Field(None, ge=0, description="Mechanism to set weights on and accept manifests for instead "
                               "of the catalog's; none: the catalog's")
    wallet_name: str
    hotkey_name: str
    mode: Literal["full", "light"] = Field("full", description="full: check the miners and score them; light: set the "
                                           "weights a source validator publishes each epoch, without contacting "
                                           "miners (not for the subnet owner's validator, the auditor)")
    source_hotkey: str | None = Field(None, description="The validator whose published results to follow, if not "
                                      "the subnet owner's: light mode sets its weights; full mode with references "
                                      "one at a time takes its results for the models it does not check in an "
                                      "epoch. It must hold a validator permit")
    references: dict[str, ReferenceEngine] = Field(default_factory=dict, description="Full mode: model_id -> the "
                                                   "model's reference engine in fp32 (a URL, or {url, api_key, "
                                                   "command, directory, environment})")
    reference_mode: Literal["resident", "sequential"] = Field("resident", description="Full mode: resident (every "
                                                              "reference runs all the time) or sequential (one at a "
                                                              "time: the validator starts each for its model's "
                                                              "canaries and stops it, and takes the source's results "
                                                              "for the models it cannot check in an epoch)")
    audit_screens: dict[str, ReferenceEngine] = Field(default_factory=dict, description="Auditor only: model_id -> a "
                                                      "faster bf16 engine that screens sampled answers; an answer it "
                                                      "flags counts only once the fp32 reference confirms it")
    public_host: str | None = Field(None, description="Public IP of this validator's service (TLS); required for the "
                                    "subnet owner (catalog) and the throughput auditor (verdicts)")
    port: int = Field(8091, ge=1, le=65535, description="Port of this validator's service")
    egress_ips: list[str] = Field(default_factory=list, description="Full mode: the public IP addresses this "
                                  "validator's requests to miners come from, published in its chain commitment "
                                  "(protocol.callers); the auditor's go in the catalog")
    catalog_mirrors: list[str] = Field(default_factory=list, description="Catalog URL templates (https) with "
                                       "{digest}, tried after the subnet owner's axon")
    catalog_dir: Path = Path("~/.verathos/catalog")
    state_dir: Path = Path("~/.verathos/validator")
    dry_run: bool = False
    auto_update: bool = Field(False, description="Update the code to each new release of the repository it was "
                              "installed from (release tags), at a quiet moment, and exit for the process manager to "
                              "start it (the installer turns it on; verathos.update)")

    @field_validator("public_host")
    @classmethod
    def _public_ip(cls, value: str | None) -> str | None:
        return None if value is None else public_ip(value)

    @field_validator("egress_ips")
    @classmethod
    def _egress(cls, value: list[str]) -> list[str]:
        return egress_addresses(value)

    @field_validator("catalog_mirrors")
    @classmethod
    def _mirrors(cls, value: list[str]) -> list[str]:
        return [https_url(v, what="catalog mirror") for v in value]

    @field_validator("source_hotkey")
    @classmethod
    def _ss58(cls, value: str | None) -> str | None:
        if value is not None:
            try:
                Keypair(ss58_address=value)
            except ValueError:
                raise ValueError(f"{value!r} is not a hotkey (SS58 address)") from None
        return value

    @field_validator("references", "audit_screens", mode="before")
    @classmethod
    def _plain_urls(cls, value: Any) -> Any:
        return _engines(value)

    @classmethod
    def load(cls, path: Path) -> ValidatorConfig:
        return cls.model_validate(json.loads(path.read_text()))
