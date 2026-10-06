"""Miner configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from verathos.protocol.manifest import Hardware, axon_base, endpoint_url, public_ip


class EndpointConfig(BaseModel):
    endpoint_id: str
    model_id: str
    engine_url: str = Field(description="Local decision engine serving POST /v1/systemone, e.g. http://127.0.0.1:8008")
    engine_api_key: str | None = None
    hardware: Hardware
    cuda_device: int | None = Field(None, ge=0, description="CUDA device index of this endpoint's GPU; the capacity "
                                    "audit bench runs on it")
    max_concurrency: int = Field(64, ge=1, le=4096)
    tier: Literal["open", "private"] = "open"


class MinerConfig(BaseModel):
    network: str = "finney"
    netuid: int
    mechid: int = 1
    wallet_name: str
    hotkey_name: str
    public_host: str = Field(description="Public IP address validators and gateways reach this miner at")
    port: int = Field(ge=1, le=65535)
    listen_host: str = "0.0.0.0"
    gateway_hotkeys: list[str] = Field(default_factory=list,
                                       description="Extra caller hotkeys allowed besides permitted validators")
    endpoints: list[EndpointConfig] = Field(min_length=1)
    catalog_mirrors: list[str] = Field(default_factory=list, description="Catalog URL templates with {digest}, "
                                       "tried after the subnet owner's axon")
    catalog_dir: Path = Path("~/.verathos/catalog")

    @field_validator("public_host")
    @classmethod
    def _public_ip(cls, value: str) -> str:
        return public_ip(value)

    @model_validator(mode="after")
    def _unique(self) -> MinerConfig:
        ids = [e.endpoint_id for e in self.endpoints]
        if len(ids) != len(set(ids)):
            raise ValueError("endpoint ids must be unique")
        return self

    def endpoint_url(self, endpoint_id: str) -> str:
        return endpoint_url(axon_base(self.public_host, self.port), endpoint_id)

    @classmethod
    def load(cls, path: Path) -> MinerConfig:
        return cls.model_validate(json.loads(path.read_text()))
