"""Validator configuration."""

from __future__ import annotations

import ipaddress
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, field_validator, model_validator

from verathos.miner.engine import HttpEngine
from verathos.protocol.manifest import public_ip


class ReferenceEngine(BaseModel):
    """An engine serving ``POST /v1/systemone``: on this machine, or remote behind TLS with a bearer key (the Kev
    server takes one from ``KEV_API_KEY``)."""

    url: str = Field(pattern=r"^https?://")
    api_key: str | None = None

    @model_validator(mode="after")
    def _key_needs_tls(self) -> ReferenceEngine:
        host = urlsplit(self.url).hostname or ""
        try:
            local = host == "localhost" or ipaddress.ip_address(host).is_loopback
        except ValueError:
            local = False
        if self.api_key and self.url.startswith("http://") and not local:
            raise ValueError(f"{self.url}: a key for a remote engine needs https")
        return self

    def engine(self) -> HttpEngine:
        return HttpEngine(self.url, self.api_key)


def _engines(value: Any) -> Any:
    """Plain URLs stand for engines without a key."""
    if isinstance(value, dict):
        return {k: {"url": v} if isinstance(v, str) else v for k, v in value.items()}
    return value


class ValidatorConfig(BaseModel):
    network: str = "finney"
    netuid: int
    mechid: int = 1
    wallet_name: str
    hotkey_name: str
    references: dict[str, ReferenceEngine] = Field(description="model_id -> the model's reference engine in fp32 (a "
                                                   "URL, or {url, api_key})")
    audit_screens: dict[str, ReferenceEngine] = Field(default_factory=dict, description="Auditor only: model_id -> a "
                                                      "faster bf16 engine that screens sampled answers; an answer it "
                                                      "flags counts only once the fp32 reference confirms it")
    public_host: str | None = Field(None, description="Public IP of this validator's service; required for the subnet "
                                    "owner (catalog) and the throughput auditor (verdicts)")
    port: int = Field(8091, ge=1, le=65535, description="Port of this validator's service")
    catalog_mirrors: list[str] = Field(default_factory=list, description="Catalog URL templates with {digest}, "
                                       "tried after the subnet owner's axon")
    catalog_dir: Path = Path("~/.verathos/catalog")
    state_dir: Path = Path("~/.verathos/validator")
    dry_run: bool = False

    @field_validator("public_host")
    @classmethod
    def _public_ip(cls, value: str | None) -> str | None:
        return None if value is None else public_ip(value)

    @field_validator("references", "audit_screens", mode="before")
    @classmethod
    def _plain_urls(cls, value: Any) -> Any:
        return _engines(value)

    @classmethod
    def load(cls, path: Path) -> ValidatorConfig:
        return cls.model_validate(json.loads(path.read_text()))
