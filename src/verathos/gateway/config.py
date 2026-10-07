"""Gateway configuration (``gateway.json``)."""

from __future__ import annotations

import ipaddress
import json
from pathlib import Path

from pydantic import BaseModel, Field, field_validator


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
    auditor_url: str | None = Field(None, description="The auditor's service; default: the subnet owner's axon")
    instances: int = Field(1, ge=1, description="Gateway processes sharing the miners' capacity (each uses 1 / "
                           "instances of it); route each API key to one process, so its limits hold exactly")
    trusted_proxies: list[str] = Field(default_factory=lambda: ["127.0.0.1/32", "::1/128"],
                                       description="Networks whose X-Forwarded-For is believed")
    request_timeout_s: float = Field(10.0, gt=0, description="Per user request, retries included")
    attempt_timeout_s: float = Field(4.0, gt=0, description="Per attempt at one miner")
    max_attempts: int = Field(3, ge=1, le=5)
    max_in_flight: int = Field(4096, ge=1, description="User requests in flight in this process; more get 503")
    max_body_bytes: int = Field(1 << 20, ge=1024)
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

    @classmethod
    def load(cls, path: Path) -> GatewayConfig:
        return cls.model_validate(json.loads(path.read_text()))
