"""Subnet catalog: the models the network serves, how their answers are verified and scored, and the audit policy.

The catalog is a JSON document. The subnet owner commits its digest on chain; miners and validators use the catalog
whose digest equals the owner's current commitment, whatever its source: the copy shipped with the code, a local
cache, the owner's axon, or a mirror. A new catalog therefore takes effect without a code release. The digest covers
the JSON document as published, so nodes whose code knows fewer fields still agree on it.

Only ``active`` models are scored. ``candidate`` models may be served and checked on the network but are not scored;
``retired`` models are ignored.
"""

from __future__ import annotations

import json
import logging
import re
from importlib import resources
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, PrivateAttr, model_validator

from verathos.protocol.canonical import digest
from verathos.protocol.http import get_json
from verathos.throughput.policy import AuditPolicy, policy_from
from verathos.throughput.workload import Workload
from verathos.validator.compare import Tolerance

log = logging.getLogger(__name__)

COMMITMENT_PREFIX = "verathos/catalog/v1 sha256:"
CATALOG_PATH = "/v1/catalog/{digest}"
DEFAULT_CACHE_DIR = Path("~/.verathos/catalog")
MAX_CATALOG_BYTES = 1 << 20


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
    workload: Workload = Field(description="Standard request mix capacity is declared and tested with")
    score_per_rps: float = Field(gt=0, description="Score per request/s of proven capacity, so models of different "
                                 "cost per request score fairly")
    first_test_rps: float = Field(gt=0, description="Highest rate an endpoint is tested at before its first test; "
                                  "later tests grow from what it delivered")
    tolerance: Tolerance = Tolerance()


class Scoring(BaseModel):
    """Scoring settings every validator applies."""

    canary_pool_size: int = Field(16, ge=1, description="Canaries per model per epoch, each computed once")
    canaries_per_endpoint: int = Field(10, ge=1)
    canary_spread_fraction: float = Field(0.6, gt=0, le=0.9, description="Share of the epoch canaries spread over")
    statistical_window_epochs: int = Field(3, ge=1)
    ema_alpha: float = Field(0.2, gt=0, le=1)
    probation_epochs: int = Field(6, ge=1)


class Catalog(BaseModel):
    version: Literal[1] = 1
    revision: str
    epoch_blocks: int = Field(180, ge=50, description="Length of the subnet's scoring epoch in blocks; validators "
                              "score, set weights and evaluate capacity evidence once per epoch")
    models: list[CatalogModel]
    throughput_audit: dict[str, Any] = Field(default_factory=dict, description="Throughput-audit settings over the "
                                             "code defaults")
    scoring: Scoring = Scoring()
    _document: Any = PrivateAttr(None)

    @classmethod
    def parse(cls, document: Any) -> Catalog:
        """Validate a catalog document (parsed JSON), keeping the document for the digest. Raises ValueError."""
        catalog = cls.model_validate(document)
        digest(document)  # raises ValueError for documents without a canonical encoding (NaN)
        catalog._document = document
        return catalog

    def model_copy(self, *, update: dict[str, Any] | None = None, deep: bool = False) -> Catalog:
        """A copy; a changed copy no longer carries the parsed document, so its digest covers its own fields."""
        copy = super().model_copy(update=update, deep=deep)
        if update:
            copy._document = None
        return copy

    def document(self) -> Any:
        """The JSON document this catalog was parsed from (or its own fields if built in code)."""
        return self._document if self._document is not None else self.model_dump(mode="json")

    @model_validator(mode="after")
    def _check(self) -> Catalog:
        ids = [m.model_id for m in self.models]
        if len(ids) != len(set(ids)):
            raise ValueError("model ids must be unique")
        self.audit_policy()
        return self

    def audit_policy(self, *, strict: bool = False) -> AuditPolicy:
        """The throughput-audit policy: code defaults with this catalog's settings (``strict``: settings this code
        does not know are an error, as when publishing a catalog)."""
        return policy_from(self.throughput_audit, self.epoch_blocks, strict=strict)

    def digest(self) -> str:
        return digest(self.document())

    def get(self, model_id: str) -> CatalogModel | None:
        return next((m for m in self.models if m.model_id == model_id), None)


class CatalogUnavailable(Exception):
    pass


def load(path: Path | None = None) -> Catalog:
    """Load the catalog from ``path``, or the one shipped with the package."""
    text = path.read_text() if path else resources.files(__package__).joinpath("catalog.json").read_text()
    return Catalog.parse(json.loads(text))


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
        tmp.write_text(json.dumps(catalog.document(), indent=2, ensure_ascii=False) + "\n")
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
            document = get_json(url, limit=MAX_CATALOG_BYTES, timeout=15)
            if digest(document) != catalog_digest:
                log.warning("catalog from %s is not the committed %s", url, catalog_digest[:12])
                continue
            catalog = Catalog.parse(document)
        except Exception as exc:  # whatever a source serves, it only costs that source
            log.info("catalog %s not available from %s: %s", catalog_digest[:12], url, exc)
            continue
        store.put(catalog)
        return catalog
    raise CatalogUnavailable(f"catalog {catalog_digest} not found in the shipped copy, {store.directory} or "
                             f"{len(sources)} sources")


def commitment_value(catalog_digest: str) -> str:
    return COMMITMENT_PREFIX + catalog_digest


def committed_digest(commitment: str | None) -> str | None:
    if not commitment or not commitment.startswith(COMMITMENT_PREFIX):
        return None
    value = commitment[len(COMMITMENT_PREFIX):]
    return value if re.fullmatch(r"[0-9a-f]{64}", value) else None
