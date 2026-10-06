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
    pool_size: int = Field(10, ge=1, description="Canaries per model per epoch (each computed once by the reference)")
    per_endpoint: int = Field(6, ge=1, description="Canaries each endpoint receives per epoch")
    audit_fraction: float = Field(0.6, gt=0, le=0.9, description="Share of the epoch over which canaries are spread")
    ema_alpha: float = Field(0.2, gt=0, le=1)
    probation_epochs: int = Field(6, ge=1)
    state_dir: Path = Path("~/.verathos-decide/validator")
    dry_run: bool = False

    @classmethod
    def load(cls, path: Path) -> ValidatorConfig:
        return cls.model_validate(json.loads(path.read_text()))
