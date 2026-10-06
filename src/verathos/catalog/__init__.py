"""Model catalog: the models the network serves and how their answers are verified.

The catalog file ships with the code. The subnet owner commits its digest on chain; validators and miners use a
catalog only when its digest equals the owner's current commitment.
"""

from __future__ import annotations

import json
import re
from importlib import resources
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from verathos.protocol.canonical import digest
from verathos.validator.compare import Tolerance

COMMITMENT_PREFIX = "verathos/catalog/v1 sha256:"


class CatalogModel(BaseModel):
    model_id: str = Field(min_length=1, max_length=128)
    status: Literal["candidate", "active", "retired"]
    engine: Literal["kev", "clef"]
    hf_repo: str
    hf_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    base_repo: str | None = None
    base_revision: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    license: str
    max_state_tokens: int = Field(ge=1)
    score_weight: float = Field(1.0, gt=0, description="Relative reward per verified answer")
    tolerance: Tolerance = Tolerance()


class Catalog(BaseModel):
    version: Literal[1] = 1
    revision: str
    models: list[CatalogModel]

    @model_validator(mode="after")
    def _unique(self) -> Catalog:
        ids = [m.model_id for m in self.models]
        if len(ids) != len(set(ids)):
            raise ValueError("model ids must be unique")
        return self

    def digest(self) -> str:
        return digest(self.model_dump(mode="json"))

    def get(self, model_id: str) -> CatalogModel | None:
        return next((m for m in self.models if m.model_id == model_id), None)


def load(path: Path | None = None) -> Catalog:
    """Load the catalog from ``path``, or the one shipped with the package."""
    text = path.read_text() if path else resources.files(__package__).joinpath("catalog.json").read_text()
    return Catalog.model_validate(json.loads(text))


def commitment_value(catalog_digest: str) -> str:
    return COMMITMENT_PREFIX + catalog_digest


def committed_digest(commitment: str | None) -> str | None:
    if not commitment or not commitment.startswith(COMMITMENT_PREFIX):
        return None
    value = commitment[len(COMMITMENT_PREFIX):]
    return value if re.fullmatch(r"[0-9a-f]{64}", value) else None
