"""Validator configuration."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from verathos.protocol.manifest import public_ip


class ValidatorConfig(BaseModel):
    network: str = "finney"
    netuid: int
    mechid: int = 1
    wallet_name: str
    hotkey_name: str
    references: dict[str, str] = Field(description="model_id -> local reference engine URL serving POST /v1/systemone")
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

    @classmethod
    def load(cls, path: Path) -> ValidatorConfig:
        return cls.model_validate(json.loads(path.read_text()))
