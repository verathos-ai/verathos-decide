"""Gateway configuration (``gateway.json``)."""

from __future__ import annotations

import ipaddress
import json
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator


class GatewayConfig(BaseModel):
    network: str = "finney"
    netuid: int
    mechid: int = 1
    wallet_name: str
    hotkey_name: str = Field(description="The gateway's hotkey: one the catalog lists under gateways")
    database_url: str = Field(description="postgresql+asyncpg://... in production; sqlite+aiosqlite:///path for "
                              "development and tests")
    listen_host: str = "127.0.0.1"
    port: int = Field(8080, ge=1, le=65535)
    metrics_port: int | None = Field(None, ge=1, le=65535, description="Prometheus metrics on 127.0.0.1")
    admin_host: str = Field("127.0.0.1", description="Address of the admin API: loopback or a private network")
    admin_port: int | None = Field(None, ge=1, le=65535, description="The admin API's port; none: no admin API")
    admin_token_file: Path | None = Field(None, description="File holding the admin token (at least 32 characters, "
                                          "readable by the gateway's user only)")
    auditor_url: str | None = Field(None, description="The auditor's service; default: the subnet owner's axon")
    instances: int = Field(1, ge=1, description="Gateway processes sharing the miners' capacity (each uses 1 / "
                           "instances of it); route each API key to one process, so its limits hold exactly")
    trusted_proxies: list[str] = Field(default_factory=lambda: ["127.0.0.1/32", "::1/128"],
                                       description="Networks whose X-Forwarded-For is believed")
    request_timeout_s: float = Field(10.0, gt=0, description="Per user request, retries included")
    attempt_timeout_s: float = Field(4.0, gt=0, description="Per attempt at one miner")
    max_attempts: int = Field(3, ge=1, le=5)
    max_in_flight: int = Field(4096, ge=1, description="User requests in flight in this process; more get 503")
    max_body_bytes: int = Field(1 << 20, ge=1024, le=1 << 20)
    sample_window_s: float = Field(600.0, ge=10, description="One exchange per endpoint and window goes to the "
                                   "auditor")
    catalog_dir: Path = Path("~/.verathos/catalog")
    state_dir: Path = Path("~/.verathos/gateway")

    @field_validator("trusted_proxies")
    @classmethod
    def _networks(cls, value: list[str]) -> list[str]:
        for network in value:
            ipaddress.ip_network(network, strict=False)
        return value

    @field_validator("admin_host")
    @classmethod
    def _not_public(cls, value: str) -> str:
        address = ipaddress.ip_address(value)
        if address.is_unspecified or not (address.is_loopback or address.is_private):
            raise ValueError("the admin API listens on loopback or one private address only")
        return value

    @model_validator(mode="after")
    def _admin_needs_token(self) -> GatewayConfig:
        if self.admin_port is not None and self.admin_token_file is None:
            raise ValueError("admin_port needs admin_token_file")
        if self.admin_port is not None and self.admin_port == self.port:
            raise ValueError("admin_port must differ from port")
        return self

    @classmethod
    def load(cls, path: Path) -> GatewayConfig:
        return cls.model_validate(json.loads(path.read_text()))
