"""Subnet catalog: the models the network serves, how their answers are verified and scored, and the audit policy.

The catalog is a JSON document. The subnet owner commits its digest on chain; miners and validators use the catalog
whose digest equals the owner's current commitment, whatever its source: the copy shipped with the code, a local
cache, the owner's axon, or a mirror. A new catalog therefore takes effect without a code release. The digest covers
the JSON document as published, so nodes whose code knows fewer fields still agree on it. The catalog also names the
subnet mechanism the network runs on: miners publish their manifests for it and validators set weights on it.

It names the auditor's TLS pin (``auditor.tls``), by which miners, validators, light validators and gateways accept the
auditor's service, and the addresses the auditor and each gateway send their requests from (``egress``,
``protocol.callers``).

Only ``active`` models are scored. ``candidate`` models may be served and checked on the network but are not scored;
``retired`` models are ignored.
"""

from __future__ import annotations

import json
import logging
import re
from importlib import resources
from pathlib import Path
from typing import Annotated, Any, Literal

from bittensor_wallet import Keypair
from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_validator

from verathos.protocol.callers import egress_addresses
from verathos.protocol.canonical import digest
from verathos.protocol.http import get_json
from verathos.protocol.tls import PIN_PATTERN, authenticated
from verathos.throughput.policy import AuditPolicy, policy_from
from verathos.throughput.workload import Workload
from verathos.validator.compare import Tolerance

log = logging.getLogger(__name__)

COMMITMENT_PREFIX = "verathos/catalog/v1 sha256:"
CATALOG_PATH = "/v1/catalog/{digest}"
DEFAULT_CACHE_DIR = Path("~/.verathos/catalog")
MAX_CATALOG_BYTES = 1 << 20
SHARE_DIGITS = 6  # decimals of the miners' and the usage pool's shares an epoch's weights are formed with
DEFAULT_MECHANISM = 1  # of a catalog that names none
MAX_PERIOD_EPOCHS = 400


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
    price: float = Field(gt=0, description="What users pay, in USD per million input tokens; the model's user "
                         "demand is weighed by it")
    anchor_share: float = Field(0.0, ge=0, le=1, description="The model's share of the miners' emission while there "
                                "is no user demand; the active models' anchors sum to 1")
    first_test_rps: float = Field(gt=0, description="Highest rate an endpoint is tested at before its first test; "
                                  "later tests grow from what it delivered")
    relative_speed: float = Field(1.0, gt=0, description="Requests/s of its standard mix a GPU serves of this model "
                                  "relative to the other models, measured on one reference GPU; miners' setups "
                                  "estimate what each model pays on a GPU from it")
    tolerance: Tolerance = Tolerance()


class MinerShareChange(BaseModel):
    """The miners' share moving linearly from ``from`` to ``to`` over ``epochs`` epochs from ``start_epoch`` on:
    ``from`` until then, and throughout while ``start_epoch`` is null."""

    from_: float = Field(alias="from", gt=0, le=1)
    to: float = Field(gt=0, le=1)
    start_epoch: int | None = Field(None, ge=0, description="The epoch (block // epoch_blocks) the change starts at; "
                                    "null: not started")
    epochs: int = Field(ge=1, description="Epochs the change takes")

    def at(self, epoch: int) -> float:
        if self.start_epoch is None or epoch <= self.start_epoch:
            return self.from_
        return self.from_ + (self.to - self.from_) * min(epoch - self.start_epoch, self.epochs) / self.epochs


class UsagePool(BaseModel):
    """Weight on top of the scores' for the hotkeys whose endpoints answered paying users, in proportion to the paid
    input tokens they answered in a period, sized from that period and set over each epoch of the next. ``burned``:
    the alpha ``buyback_coldkey`` burned on the subnet in the period; ``revenue``: the period's paid usage in USD at
    the models' prices, converted to alpha. Either times ``fraction``, as a share of the mechanism's miner emission
    over a period, at most what the miner share leaves."""

    mode: Literal["burned", "revenue"]
    fraction: float = Field(gt=0, le=1, description="Share of the period's burned alpha (burned) or of its revenue in "
                            "alpha (revenue) the pool pays")
    period_epochs: int = Field(40, ge=1, le=MAX_PERIOD_EPOCHS, description="Epochs per period: each period's pool is "
                               "paid over the epochs of the next")
    buyback_coldkey: str | None = Field(None, description="The coldkey whose AlphaBurned events on the subnet count "
                                        "(burned mode)")
    excluded: list[str] = Field(default_factory=list, max_length=256, description="Hotkeys and coldkeys whose "
                                "endpoints get no usage pay; their part stays on the subnet owner's UID")

    @model_validator(mode="after")
    def _check(self) -> UsagePool:
        if self.mode == "burned" and not self.buyback_coldkey:
            raise ValueError("usage_pool in burned mode names its buyback_coldkey")
        for key in [k for k in (self.buyback_coldkey, *self.excluded) if k is not None]:
            if not is_ss58(key):
                raise ValueError(f"usage_pool: {key!r} is not an ss58 address")
        return self

    def period(self, epoch: int) -> int:
        """The period ``epoch`` falls in: epochs ``period x period_epochs`` up to the next period's first."""
        return epoch // self.period_epochs


class Scoring(BaseModel):
    """Scoring settings every validator applies."""

    canary_pool_size: int = Field(16, ge=1, description="Canaries per model per epoch, each computed once")
    canaries_per_endpoint: int = Field(10, ge=1)
    canary_spread_fraction: float = Field(0.6, gt=0, le=0.9, description="Share of the epoch canaries spread over")
    statistical_window_epochs: int = Field(3, ge=1)
    ema_alpha: float = Field(0.2, gt=0, le=1)
    probation_epochs: int = Field(6, ge=1)
    demand_ema_alpha: float = Field(0.1, gt=0, le=1, description="Weight of the latest epoch in each model's "
                                    "user demand")
    demand_full_tokens_s: float = Field(10_000.0, gt=0, description="Total user demand (input tokens/s) from which "
                                        "the models' emission shares follow demand alone; below it they lean toward "
                                        "the anchor shares")
    min_model_share: float = Field(0.1, ge=0, le=1, description="Emission share every active model keeps, whatever "
                                   "its demand")
    max_share_step: float = Field(0.05, gt=0, le=1, description="Most a model's emission share moves per epoch "
                                  "toward what demand sets")
    miner_share: Annotated[float, Field(gt=0, le=1)] | MinerShareChange = Field(
        1.0, description="Share of the mechanism's miner emission the scores pay the miners; the rest is weight on "
        "the subnet owner's UID, whose incentive the chain does not pay out. A number, or a linear change "
        "{from, to, start_epoch, epochs}")
    usage_pool: UsagePool | None = Field(None, description="Usage weight for the hotkeys that served paying users, "
                                         "out of what the miner share leaves; absent: none")
    revision_transition_epochs: int = Field(3, ge=0, description="Epochs after a model's pinned weights or engine "
                                            "change in place during which its answers are not checked, while miners "
                                            "and references update")
    user_offers_prior: int = Field(20, ge=1, description="Ready offers every hotkey's endpoints of a model start an "
                                   "epoch with, so a few offers move its availability to users little")
    user_latency_min_answers: int = Field(50, ge=1, description="Answers to users an endpoint needs in an epoch for "
                                          "its latency factor to come from them; with fewer, from its test")
    validator_min_stake: float = Field(0.0, ge=0, description="Stake a validator needs for miners to answer it, as "
                                       "the metagraph's S (alpha plus weighted TAO), the stake validator permits "
                                       "follow; 0: every validator with a permit")
    validator_requests_per_epoch: int | None = Field(None, ge=1, description="Requests a validator may send one "
                                                     "endpoint per epoch (canaries and identity challenges); beyond "
                                                     "them miners answer 429 until the next epoch. Default: "
                                                     "``request_budget``")

    def miner_share_at(self, epoch: int) -> float:
        """``miner_share`` in ``epoch``, rounded to ``SHARE_DIGITS`` decimals, so every validator sets the same."""
        share = self.miner_share.at(epoch) if isinstance(self.miner_share, MinerShareChange) else self.miner_share
        return round(share, SHARE_DIGITS)

    def request_budget(self) -> int:
        """``validator_requests_per_epoch``, by default 4 x ``canary_pool_size``. In one pass of an epoch a validator
        sends an endpoint at most ``canaries_per_endpoint`` (at most ``canary_pool_size``) requests in the canaries'
        slots, each a canary or a retried identity challenge, and three more challenges (at the epoch's start, one
        followed by a canary in the same slot, one from the auditor's discovery): the default holds more than three
        passes, so an epoch run again after a validator failure stays within it."""
        return self.validator_requests_per_epoch or 4 * self.canary_pool_size


class AuditorIdentity(BaseModel):
    """The auditor's service (the subnet owner's validator) and where its requests come from."""

    tls: str = Field(pattern=PIN_PATTERN, description="Pin of the TLS certificate of the auditor's service "
                     "(SHA-256 of its public key): miners, validators and gateways accept the auditor by it")
    egress: list[str] = Field(default_factory=list, description="Public IP addresses the auditor sends its requests "
                              "from (identity challenges, canaries, throughput tests)")

    @field_validator("egress")
    @classmethod
    def _egress(cls, value: list[str]) -> list[str]:
        return egress_addresses(value)


class Gateway(BaseModel):
    """A gateway of the subnet: miners serve it like a validator without the per-caller limits, and the auditor
    checks a sample of its answers."""

    hotkey: str
    egress: list[str] = Field(default_factory=list, description="Public IP addresses its requests to miners come "
                              "from")

    @field_validator("hotkey")
    @classmethod
    def _ss58(cls, value: str) -> str:
        if not is_ss58(value):
            raise ValueError(f"gateway {value!r} is not an ss58 address")
        return value

    @field_validator("egress")
    @classmethod
    def _egress(cls, value: list[str]) -> list[str]:
        return egress_addresses(value)


class Catalog(BaseModel):
    version: Literal[1] = 1
    revision: str
    mechanism: int = Field(DEFAULT_MECHANISM, ge=0, description="The subnet mechanism the network runs on: miners "
                           "publish their manifests for it, validators set weights on it")
    epoch_blocks: int = Field(180, ge=50, description="Length of the subnet's scoring epoch in blocks; validators "
                              "score, set weights and evaluate capacity evidence once per epoch")
    models: list[CatalogModel]
    throughput_audit: dict[str, Any] = Field(default_factory=dict, description="Throughput-audit settings over the "
                                             "code defaults")
    scoring: Scoring = Scoring()
    gateways: list[Gateway] = Field(default_factory=list, max_length=16, description="The subnet's gateways, each "
                                    "{hotkey, egress}")
    auditor: AuditorIdentity | None = Field(None, description="The auditor's TLS pin and egress addresses; nodes reach "
                                            "the auditor only once the catalog names its pin")
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
        return self._document if self._document is not None else self.model_dump(mode="json", by_alias=True)

    @model_validator(mode="after")
    def _check(self) -> Catalog:
        ids = [m.model_id for m in self.models]
        if len(ids) != len(set(ids)):
            raise ValueError("model ids must be unique")
        hotkeys = self.gateway_hotkeys()
        if len(hotkeys) != len(set(hotkeys)):
            raise ValueError("gateway hotkeys must be unique")
        self.audit_policy()
        return self

    def gateway_hotkeys(self) -> list[str]:
        """The gateways' hotkeys."""
        return [g.hotkey for g in self.gateways]

    def auditor_pin(self) -> str:
        """The pin of the auditor's TLS certificate. Raises AuditorUnknown if the catalog names none."""
        if self.auditor is None:
            raise AuditorUnknown(f"catalog {self.digest()[:12]} names no TLS pin for the auditor (auditor.tls): the "
                                 "auditor's service is not reached until the subnet owner commits one")
        return self.auditor.tls

    def identity_problems(self) -> list[str]:
        """What a catalog must name before it is published: the auditor's TLS pin and egress addresses, and each
        gateway's egress addresses."""
        problems = []
        if self.auditor is None:
            problems.append("auditor: the auditor's TLS pin (auditor.tls) and egress addresses (auditor.egress)")
        elif not self.auditor.egress:
            problems.append("auditor.egress: the addresses the auditor sends its requests from")
        problems += [f"gateways: the egress addresses of {g.hotkey}" for g in self.gateways if not g.egress]
        return problems

    def anchor_shares(self, *, strict: bool = False) -> dict[str, float]:
        """The active models' anchor shares, summing to 1 (equal if none is set). ``strict``: they must sum to 1 as
        written, as when publishing a catalog."""
        active = [m for m in self.models if m.status == "active"]
        total = sum(m.anchor_share for m in active)
        if strict and abs(total - 1.0) > 1e-6:
            raise ValueError(f"the active models' anchor_share must sum to 1, not {total:g}")
        if total <= 0:
            return {m.model_id: 1.0 / len(active) for m in active}
        return {m.model_id: m.anchor_share / total for m in active}

    def unknown_settings(self) -> list[str]:
        """Keys of the document this code does not know (the catalog's own, its scoring's, its models'): nodes ignore
        them, so a publisher must not mistype one."""
        doc = self._document or {}
        unknown = [k for k in doc if k not in type(self).model_fields]
        scoring = doc.get("scoring") or {}
        unknown += [f"scoring.{k}" for k in scoring if k not in Scoring.model_fields]
        if isinstance(scoring.get("miner_share"), dict):
            known = {f.alias or name for name, f in MinerShareChange.model_fields.items()}
            unknown += [f"scoring.miner_share.{k}" for k in scoring["miner_share"] if k not in known]
        if isinstance(scoring.get("usage_pool"), dict):
            unknown += [f"scoring.usage_pool.{k}" for k in scoring["usage_pool"] if k not in UsagePool.model_fields]
        for m in doc.get("models") or []:
            unknown += [f"models.{m.get('model_id', '?')}.{k}" for k in m if k not in CatalogModel.model_fields]
        if isinstance(doc.get("auditor"), dict):
            unknown += [f"auditor.{k}" for k in doc["auditor"] if k not in AuditorIdentity.model_fields]
        for g in doc.get("gateways") or []:
            if isinstance(g, dict):
                unknown += [f"gateways.{g.get('hotkey', '?')}.{k}" for k in g if k not in Gateway.model_fields]
        return unknown

    def audit_policy(self, *, strict: bool = False) -> AuditPolicy:
        """The throughput-audit policy: code defaults with this catalog's settings (``strict``: settings this code
        does not know are an error, as when publishing a catalog)."""
        return policy_from(self.throughput_audit, self.epoch_blocks, strict=strict)

    def digest(self) -> str:
        return digest(self.document())

    def get(self, model_id: str) -> CatalogModel | None:
        return next((m for m in self.models if m.model_id == model_id), None)


def mechanism_of(cat: Catalog, override: int | None = None) -> int:
    """The mechanism a node works on under ``cat``: ``override`` (its configuration's ``mechid``) if set, else the
    catalog's."""
    return cat.mechanism if override is None else override


def is_ss58(value: str) -> bool:
    try:
        Keypair(ss58_address=value)
    except Exception:
        return False
    return True


class CatalogUnavailable(Exception):
    pass


class AuditorUnknown(Exception):
    """The catalog names no TLS pin for the auditor."""


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
        """The stored catalog, or None if there is none, or none this code can read (a catalog of an older format
        stays on disk while a newer release runs)."""
        path = self.path(catalog_digest)
        if not path.exists():
            return None
        try:
            catalog = load(path)
        except Exception as exc:
            log.warning("stored catalog %s not readable by this release: %s", catalog_digest[:12],
                        str(exc).splitlines()[0])
            return None
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

    Sources are https URL templates containing ``{digest}``: a public CA's certificate is required for a hostname,
    any certificate is accepted at an IP address (the owner's axon), since anything fetched is accepted only if its
    digest matches. The shipped copy, when committed, is kept in the store too, so a later release still finds it
    without a source.
    """
    shipped = load()
    if shipped.digest() == catalog_digest:
        if not store.path(catalog_digest).exists():
            store.put(shipped)
        return shipped
    stored = store.get(catalog_digest)
    if stored is not None:
        return stored
    for template in sources:
        url = template.format(digest=catalog_digest)
        try:
            document = get_json(url, limit=MAX_CATALOG_BYTES, timeout=15, tls=authenticated(url))
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
