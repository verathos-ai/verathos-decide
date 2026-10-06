"""Subnet catalog: the models the network serves, how their answers are verified, and the capacity-audit policy.

The catalog is a JSON document. The subnet owner commits its digest on chain; miners and validators use the catalog
whose digest equals the owner's current commitment, whatever its source: the copy shipped with the code, a local
cache, the owner's axon, or a mirror. A new catalog therefore takes effect without a code release.

Only ``active`` models are scored. ``candidate`` models may be served and checked on the network but earn nothing;
``retired`` models are ignored.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field, ValidationError, model_validator

from verathos.capacity.core import AuditPolicy
from verathos.protocol.canonical import digest
from verathos.validator.compare import Tolerance

log = logging.getLogger(__name__)

COMMITMENT_PREFIX = "verathos/catalog/v1 sha256:"
CATALOG_PATH = "/v1/catalog/{digest}"
DEFAULT_CACHE_DIR = Path("~/.verathos/catalog")


class Engine(BaseModel):
    """The reference engine of a model, pinned to a source revision. Validators compute reference answers with it;
    miners may serve the model with any engine whose answers stay within the model's tolerances."""

    name: str = Field(min_length=1, max_length=64)
    source: str = Field(pattern=r"^https://")
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")


class CatalogModel(BaseModel):
    model_id: str = Field(min_length=1, max_length=128)
    status: Literal["candidate", "active", "retired"]
    engine: Engine
    hf_repo: str
    hf_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    base_repo: str | None = None
    base_revision: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    license: str
    max_state_tokens: int = Field(ge=1)
    score_weight: float = Field(1.0, gt=0, description="Relative reward per verified answer")
    tolerance: Tolerance = Tolerance()


_AUDIT_FIELDS = {f.name for f in dataclasses.fields(AuditPolicy)} - {"gpu_classes"}


class Catalog(BaseModel):
    version: Literal[1] = 1
    revision: str
    models: list[CatalogModel]
    capacity_audit: dict[str, Any] = Field(default_factory=dict, description="Overrides of the capacity-audit policy")

    @model_validator(mode="after")
    def _check(self) -> Catalog:
        ids = [m.model_id for m in self.models]
        if len(ids) != len(set(ids)):
            raise ValueError("model ids must be unique")
        unknown = set(self.capacity_audit) - _AUDIT_FIELDS
        if unknown:
            raise ValueError(f"unknown capacity_audit settings: {sorted(unknown)}")
        self.audit_policy().validate()
        return self

    def audit_policy(self) -> AuditPolicy:
        """The capacity-audit policy: code defaults (live-calibrated values) with this catalog's overrides."""
        return dataclasses.replace(AuditPolicy(), **self.capacity_audit)

    def digest(self) -> str:
        return digest(self.model_dump(mode="json"))

    def get(self, model_id: str) -> CatalogModel | None:
        return next((m for m in self.models if m.model_id == model_id), None)

    def scored(self, model_id: str) -> bool:
        model = self.get(model_id)
        return model is not None and model.status == "active"


class CatalogUnavailable(Exception):
    pass


def load(path: Path | None = None) -> Catalog:
    """Load the catalog from ``path``, or the one shipped with the package."""
    text = path.read_text() if path else resources.files(__package__).joinpath("catalog.json").read_text()
    return Catalog.model_validate(json.loads(text))


class CatalogStore:
    """Catalogs by digest in a local directory: verified downloads, and catalogs the owner publishes."""

    def __init__(self, directory: Path = DEFAULT_CACHE_DIR) -> None:
        self.directory = directory.expanduser()

    def path(self, catalog_digest: str) -> Path:
        return self.directory / f"{catalog_digest}.json"

    def get(self, catalog_digest: str) -> Catalog | None:
        path = self.path(catalog_digest)
        if not path.exists():
            return None
        catalog = load(path)
        return catalog if catalog.digest() == catalog_digest else None

    def put(self, catalog: Catalog) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.path(catalog.digest())
        tmp = path.with_suffix(".tmp")
        tmp.write_text(catalog.model_dump_json(indent=2) + "\n")
        tmp.replace(path)
        return path


def resolve(catalog_digest: str, *, store: CatalogStore, sources: list[str]) -> Catalog:
    """The catalog with ``catalog_digest``: shipped copy, local store, then each source URL in order.

    Sources are URL templates containing ``{digest}``. Anything fetched is accepted only if its digest matches.
    """
    shipped = load()
    if shipped.digest() == catalog_digest:
        return shipped
    stored = store.get(catalog_digest)
    if stored is not None:
        return stored
    for template in sources:
        url = template.format(digest=catalog_digest)
        try:
            response = httpx.get(url, timeout=15, follow_redirects=True)
            response.raise_for_status()
            catalog = Catalog.model_validate(response.json())
        except (httpx.HTTPError, ValueError, ValidationError) as exc:
            log.info("catalog %s not available from %s: %s", catalog_digest[:12], url, exc)
            continue
        if catalog.digest() == catalog_digest:
            store.put(catalog)
            return catalog
        log.warning("catalog from %s has digest %s, not the committed %s", url, catalog.digest()[:12],
                    catalog_digest[:12])
    raise CatalogUnavailable(f"catalog {catalog_digest} not found in the shipped copy, {store.directory} or "
                             f"{len(sources)} sources")


def commitment_value(catalog_digest: str) -> str:
    return COMMITMENT_PREFIX + catalog_digest


def committed_digest(commitment: str | None) -> str | None:
    if not commitment or not commitment.startswith(COMMITMENT_PREFIX):
        return None
    value = commitment[len(COMMITMENT_PREFIX):]
    return value if re.fullmatch(r"[0-9a-f]{64}", value) else None
