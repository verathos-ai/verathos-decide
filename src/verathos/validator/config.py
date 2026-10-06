"""Validator configuration."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class ValidatorConfig(BaseModel):
    network: str = "finney"
    netuid: int
    mechid: int = 1
    wallet_name: str
    hotkey_name: str
    references: dict[str, str] = Field(description="model_id -> local reference engine URL serving POST /v1/systemone")
    pool_size: int = Field(16, ge=1, description="Canaries per model per epoch (each computed once by the reference)")
    per_endpoint: int = Field(10, ge=1, description="Canaries each endpoint receives per epoch")
    window_epochs: int = Field(3, ge=1, description="Epochs over which the statistical bound is evaluated")
    audit_fraction: float = Field(0.6, gt=0, le=0.9, description="Share of the epoch over which canaries are spread")
    ema_alpha: float = Field(0.2, gt=0, le=1)
    probation_epochs: int = Field(6, ge=1)
    public_host: str | None = Field(None, description="Public IP of this validator's service; required for the subnet "
                                    "owner (catalog) and the capacity auditor (audit ingest)")
    port: int = Field(8091, ge=1, le=65535, description="Port of this validator's service")
    catalog_mirrors: list[str] = Field(default_factory=list, description="Catalog URL templates with {digest}, "
                                       "tried after the subnet owner's axon")
    catalog_dir: Path = Path("~/.verathos/catalog")
    state_dir: Path = Path("~/.verathos/validator")
    dry_run: bool = False

    @classmethod
    def load(cls, path: Path) -> ValidatorConfig:
        return cls.model_validate(json.loads(path.read_text()))
