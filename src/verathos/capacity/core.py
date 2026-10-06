"""Capacity audit: deterministic protocol core.

The audit is non-interactive once current-head chain data is available. The owner validator and every miner
independently derive audit windows, cohorts, seeds and proof challenges from public chain data plus the endpoints
listed in committed miner manifests.

Security model: an elimination sieve over repeated windows. Random beacon windows draft cohorts of endpoint slots
that must each run a calibrated GPU workload simultaneously, beside their resident model, inside a timing window,
followed by a sampled proof of the work. Slots sharing hardware cannot pass together; their misses accumulate to the
gate, and a gated endpoint scores zero and goes on probation (``validator.scoring``), which new endpoints of the
same hotkey inherit.

A slot is one endpoint of one hotkey: slot_id = H(netuid, hotkey, endpoint_id). Artifacts are signed with the
hotkey (sr25519). The workload rows per GPU class are part of the policy and are published in the subnet catalog.
"""

from __future__ import annotations

import dataclasses
import hashlib
import ipaddress
import json
import math
import re
import struct
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from bittensor_wallet import Keypair

from verathos.protocol import signing

PROTOCOL_VERSION = "verathos-capacity-audit-v1"
DEFAULT_MAX_PROOF_PAYLOAD_BYTES = 32 * 1024 * 1024
DEFAULT_BLOCK_TIME_S = 12.0
SELECTION_COOLDOWN_MARGIN_BLOCKS = 2
ACTIVE_STATE_MARGIN_S = 30.0
POLICY_WATCH_S = 60.0  # miners and validators check the committed catalog (and so the policy) this often

# Auditor ingest routes (on the auditor's axon)
RECEIPT_PATH = "/capacity/v1/receipt"
PROOF_PATH = "/capacity/v1/proof"
HDR_AUDIT_ID = "X-Verathos-Audit-Id"
HDR_SLOT_ID = "X-Verathos-Slot-Id"

# Slot, lease and audit identity
SLOT_PREFIX = b"VERATHOS_CAPACITY_SLOT_V1"
LEASE_PREFIX = b"VERATHOS_CAPACITY_LEASE_V1"
AUDIT_ID_PREFIX = b"VERATHOS_CAPACITY_AUDIT_ID_V1"
ARTIFACT_DOMAIN = "verathos/capacity-artifact/v1"
# Beacon and proof derivations
AUDIT_PREFIX = b"VERATHOS_AUDIT_V1"
WINDOW_TRIGGER_PREFIX = b"VERATHOS_CAPACITY_WINDOW_TRIGGER_V1"
PROOF_SEED_PREFIX = b"VERATHOS_HOT_CAPACITY_SEED_V1"
PROOF_PASS_PREFIX = b"VERATHOS_HOT_CAPACITY_PROOF_PASS_V1"
PROOF_CHALLENGE_PREFIX = b"VERATHOS_HOT_CAPACITY_CHALLENGE_V1"
SELECTION_SCORE_PREFIX = b"VERATHOS_CAPACITY_SELECTION_SCORE_V1"

MARKETED_VRAM_SIZES_GB = (12, 16, 20, 24, 32, 40, 48, 80, 96, 128, 141, 192, 288)


def normalize_vram_gb(raw_gb: float) -> int:
    """Map a usable or provider-reported VRAM value to its marketed memory class."""
    value = float(raw_gb)
    if value <= 0:
        return round(value)
    for marketed in MARKETED_VRAM_SIZES_GB:
        if value * 0.95 <= marketed <= value * 1.25:
            return marketed
    return round(value)


@dataclass(frozen=True)
class GpuClass:
    """Calibrated workload row for one GPU class: pass counts sized so the timed work takes about two thirds of the
    audit deadline on an unshared GPU of the class."""

    gpu_name: str
    vram_gb: int
    capacity_passes: int
    matrix_dim: int = 8960
    capacity_rounds: int = 1
    capacity_warmup_passes: int = 1
    capacity_block_size: int = 64
    transition_mix_rounds: int = 0
    transition_fanout: int = 1
    capacity_tail_passes: int = 0
    capacity_tail_rounds: int = 1
    capacity_tail_warmup_passes: int = 1
    capacity_tail_transition_mix_rounds: int = 1
    capacity_tail_transition_fanout: int = 8
    fp64_matrix_dim: int = 4096
    fp64_passes: int = 0
    fp64_rounds: int = 1
    fp64_warmup_passes: int = 1
    fp64_block_size: int = 64
    spot_checks: int = 1

    @classmethod
    def from_dict(cls, row: Any) -> GpuClass:
        if not isinstance(row, Mapping):
            raise ValueError("a GPU class must be an object")
        unknown = set(row) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"unknown GPU class settings: {sorted(unknown)}")
        try:
            return cls(**row)
        except TypeError as exc:
            raise ValueError(f"GPU class {row.get('gpu_name')!r}: {exc}") from None


@dataclass(frozen=True)
class AuditPolicy:
    """Published audit policy. Only the auditor runs and evaluates audits: ``auditor_hotkey`` if set, otherwise the
    subnet owner's hotkey read from chain."""

    auditor_hotkey: str = ""
    mode: str = "observe"  # observe | score_gate
    cohort_fraction: float = 0.3
    windows_per_epoch: int = 5
    max_drain_fraction: float = 0.1
    group_stress_fraction: float = 0.35
    lead_blocks: int = 5
    proof_challenge_delay_blocks: int = 4
    drain_seconds: float = 30.0
    deadline_s: float = 30.0
    transport_grace_s: float = 3.0
    payload_deadline_s: float = 90.0
    max_proof_payload_bytes: int = DEFAULT_MAX_PROOF_PAYLOAD_BYTES
    repeat_window_epochs: int = 10
    timing_misses_for_zero_score: int = 2
    hard_proof_misses_for_zero_score: int = 2
    invalid_proof_misses_for_zero_score: int = 1
    gpu_classes: tuple[GpuClass, ...] = ()

    def validate(self, block_time_s: float = DEFAULT_BLOCK_TIME_S) -> None:
        if not 0.0 <= self.group_stress_fraction <= 0.5:
            raise ValueError("group_stress_fraction must be between 0.0 and 0.5")
        proof_delay_s = max(1, self.proof_challenge_delay_blocks) * block_time_s
        if self.deadline_s + self.transport_grace_s >= proof_delay_s:
            raise ValueError(
                "deadline_s + transport_grace_s must be smaller than proof_challenge_delay_blocks * block_time"
            )
        if self.max_proof_payload_bytes <= 0:
            raise ValueError("max_proof_payload_bytes must be positive")
        if self.mode not in ("observe", "score_gate"):
            raise ValueError("mode must be observe or score_gate")
        names = [_canonical_gpu_name(r.gpu_name) for r in self.gpu_classes]
        if len(names) != len(set(names)):
            raise ValueError("GPU class names must be unique")
        if any(gpu_pass_count(r) <= 0 for r in self.gpu_classes):
            raise ValueError("every GPU class needs a positive pass count")


def active_hold_seconds(policy: AuditPolicy, block_time_s: float = DEFAULT_BLOCK_TIME_S) -> float:
    """The miner's complete busy hold from B_select."""
    pre_challenge_blocks = max(0, policy.lead_blocks) + max(1, policy.proof_challenge_delay_blocks)
    evidence_s = policy.deadline_s + policy.transport_grace_s + policy.payload_deadline_s + ACTIVE_STATE_MARGIN_S
    return max(policy.drain_seconds, pre_challenge_blocks * block_time_s + evidence_s, 60.0)


def payload_cooldown_blocks(policy: AuditPolicy, block_time_s: float = DEFAULT_BLOCK_TIME_S) -> int:
    """Post-B_proof cooldown before a slot may be selected again."""
    pre_challenge_blocks = max(0, policy.lead_blocks) + max(1, policy.proof_challenge_delay_blocks)
    post_s = max(0.0, active_hold_seconds(policy, block_time_s) - pre_challenge_blocks * block_time_s)
    return int(math.ceil(post_s / block_time_s)) + SELECTION_COOLDOWN_MARGIN_BLOCKS


@dataclass(frozen=True)
class Slot:
    """One endpoint of one hotkey, as listed in its committed manifest."""

    netuid: int
    hotkey: str
    endpoint_id: str
    endpoint_url: str
    model_id: str
    gpu_name: str = ""
    vram_gb: int = 0

    @property
    def slot_id(self) -> str:
        return _hash_hex(SLOT_PREFIX, _u64(self.netuid), self.hotkey.encode(), b"\0", self.endpoint_id.encode())


def _bytes_from_hash(value: bytes | str) -> bytes:
    if isinstance(value, bytes):
        return value
    raw = str(value).strip()
    if raw.startswith("0x"):
        raw = raw[2:]
    try:
        return bytes.fromhex(raw)
    except ValueError:
        return raw.encode("utf-8")


def _hash_hex(*parts: bytes) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part)
    return h.hexdigest()


def _u64(value: int) -> bytes:
    return struct.pack(">Q", int(value))


def lease_id(slot: Slot, audit_epoch: int) -> str:
    """The bench lease of a slot in an epoch. Binds only identity, so a refreshed endpoint URL or model on either
    side never changes the work."""
    return _hash_hex(LEASE_PREFIX, _u64(slot.netuid), slot.hotkey.encode(), b"\0", slot.endpoint_id.encode(),
                     _u64(audit_epoch))


def derive_audit_seed(selection_block_hash: bytes | str, subnet_epoch: int) -> str:
    return _hash_hex(AUDIT_PREFIX, _bytes_from_hash(selection_block_hash), _u64(subnet_epoch))


def derive_audit_id(*, netuid: int, epoch_number: int, selection_block: int, audit_block: int, cohort_seed: str) -> str:
    return _hash_hex(AUDIT_ID_PREFIX, _u64(netuid), _u64(epoch_number), _u64(selection_block), _u64(audit_block),
                     _bytes_from_hash(cohort_seed))


def derive_proof_seed(audit_block_hash: bytes | str, slot_identifier: str, gpu_index: int) -> str:
    return _hash_hex(PROOF_SEED_PREFIX, _bytes_from_hash(audit_block_hash), _bytes_from_hash(slot_identifier),
                     _u64(gpu_index))


def derive_proof_challenge_seed(transcript_root_hex: bytes | str, challenge_block_hash: bytes | str,
                                lease_identifier: str, slot_identifier: str, gpu_index: int) -> str:
    """Public post-commit challenge seed: the timed final receipt commits the transcript root before the challenge
    block hash exists, and the proof sample is drawn from this seed."""
    return _hash_hex(PROOF_CHALLENGE_PREFIX, _bytes_from_hash(transcript_root_hex),
                     _bytes_from_hash(challenge_block_hash), lease_identifier.encode("utf-8"),
                     _bytes_from_hash(slot_identifier), _u64(gpu_index))


def canonical_json(data: Mapping[str, Any], *, exclude_signature: bool = False) -> str:
    """Canonical JSON used inside proof transcripts (ASCII, sorted keys)."""
    payload = dict(data)
    if exclude_signature:
        payload.pop("miner_signature", None)
        payload.pop("signature", None)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def root_words_digest(root_words: Any) -> str:
    """Digest of one compact CUDA root vector exactly as miner artifacts encode it."""
    return hashlib.sha256(
        json.dumps(root_words, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def derive_sampled_pass_index(root_hex: str, lease_identifier: str, gpu_index: int, passes: int) -> int:
    if passes <= 0:
        return 0
    digest = hashlib.sha256(PROOF_PASS_PREFIX + _bytes_from_hash(root_hex) + lease_identifier.encode("utf-8")
                            + _u64(gpu_index)).digest()
    return int.from_bytes(digest[:8], "big") % passes


def window_triggered(block_number: int, block_hash: bytes | str, epoch_blocks: int, policy: AuditPolicy) -> bool:
    """Whether this current-head block opens a surprise audit window (rate windows_per_epoch / epoch_blocks)."""
    epoch_blocks = max(1, int(epoch_blocks))
    windows = max(0, policy.windows_per_epoch)
    if windows <= 0:
        return False
    if windows >= epoch_blocks:
        return True
    digest = hashlib.sha256(WINDOW_TRIGGER_PREFIX + _bytes_from_hash(block_hash) + _u64(block_number)
                            + _u64(epoch_blocks) + _u64(windows)).digest()
    return int.from_bytes(digest[:16], "big") < ((1 << 128) * windows) // epoch_blocks


def window_fits_epoch(selection_block: int, epoch_blocks: int, policy: AuditPolicy) -> bool:
    """All audit blocks must stay in one tempo; late selections are dropped, never carried into the next epoch."""
    epoch_blocks = max(1, int(epoch_blocks))
    epoch_end_exclusive = ((selection_block // epoch_blocks) + 1) * epoch_blocks
    audit_block = selection_block + policy.lead_blocks
    proof_block = audit_block + max(1, policy.proof_challenge_delay_blocks)
    return audit_block < epoch_end_exclusive and proof_block < epoch_end_exclusive


def _token(value: str) -> str:
    return str(value or "").strip().lower().replace("|", "%7c")


def group_keys(slot: Slot) -> tuple[str, ...]:
    """Public grouping tokens from committed manifest data: the hotkey and the axon IP (endpoint URLs are on the
    hotkey's axon), each alone and with the model."""
    host = (urlparse(slot.endpoint_url).hostname or "").lower()
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None
    ip_literal = str(ip) if ip else ""
    ip24 = ".".join(ip_literal.split(".")[:3]) if ip and ip.version == 4 else ""
    hk, model = _token(slot.hotkey), _token(slot.model_id)
    tokens = [f"hk:{hk}" if hk else "", f"ip:{ip_literal}" if ip_literal else "", f"ip24:{ip24}" if ip24 else ""]
    if model:
        tokens += [f"{t.split(':', 1)[0]}_model:{t.split(':', 1)[1]}::{model}" for t in tokens if t]
    return tuple(sorted(t for t in tokens if t))


def _selection_probability(policy: AuditPolicy) -> float:
    target = max(0.0, policy.cohort_fraction) / max(1, policy.windows_per_epoch)
    if policy.max_drain_fraction > 0:
        target = min(target, policy.max_drain_fraction)
    return min(1.0, max(0.0, target))


def _selection_hit(audit_seed_hex: str, namespace: str, key: str, probability: float) -> bool:
    p = min(1.0, max(0.0, probability))
    if p <= 0.0:
        return False
    if p >= 1.0:
        return True
    h = hashlib.sha256()
    h.update(SELECTION_SCORE_PREFIX)
    for part in (audit_seed_hex, namespace, key):
        h.update(str(part).encode("utf-8"))
        h.update(b"\0")
    return int(h.hexdigest(), 16) < int(p * (1 << 256))


def slot_selected(slot: Slot, audit_seed_hex: str, policy: AuditPolicy) -> bool:
    """Per-slot public predicate: base draw on the slot id, plus group-stress draws on its grouping tokens."""
    target = _selection_probability(policy)
    if target <= 0.0:
        return False
    group_fraction = min(1.0, max(0.0, policy.group_stress_fraction))
    if _selection_hit(audit_seed_hex, "base", slot.slot_id, target * (1.0 - group_fraction)):
        return True
    keys = group_keys(slot)
    if not keys or group_fraction <= 0.0:
        return False
    per_key = min(1.0, target * group_fraction)
    return any(_selection_hit(audit_seed_hex, "group", k, per_key) for k in keys)


def select_slots(slots: Iterable[Slot], audit_seed_hex: str, policy: AuditPolicy) -> list[Slot]:
    """Deterministic cohort from the public seed and committed manifests; needs no global miner state."""
    unique = {s.slot_id: s for s in slots}
    return sorted((s for s in unique.values() if slot_selected(s, audit_seed_hex, policy)),
                  key=lambda s: (s.hotkey, s.endpoint_id))


@dataclass(frozen=True)
class Window:
    """An audit window opened by ``selection_block``."""

    audit_id: str
    epoch: int
    seed: str
    selection_block: int
    start_block: int
    proof_block: int


def open_window(netuid: int, block_number: int, block_hash: bytes, epoch_blocks: int,
                policy: AuditPolicy) -> Window | None:
    """The window this current-head block opens, if any."""
    if not window_triggered(block_number, block_hash, epoch_blocks, policy) \
            or not window_fits_epoch(block_number, epoch_blocks, policy):
        return None
    epoch = block_number // epoch_blocks
    seed = derive_audit_seed(block_hash, epoch)
    start_block = block_number + policy.lead_blocks
    proof_block = start_block + max(1, policy.proof_challenge_delay_blocks)
    audit_id = derive_audit_id(netuid=netuid, epoch_number=epoch, selection_block=block_number,
                               audit_block=start_block, cohort_seed=seed)
    return Window(audit_id, epoch, seed, block_number, start_block, proof_block)


def draft(window: Window, slots: Iterable[Slot], policy: AuditPolicy,
          busy: Collection[str] = ()) -> list[tuple[Slot, GpuClass]]:
    """Slots the window drafts, with their workload rows. ``busy`` holds slot ids still in an earlier window's
    cooldown; slots without an auditable GPU class are never drafted. Miners, the auditor and other validators
    all derive the same list."""
    drafted = []
    for slot in select_slots([s for s in slots if s.slot_id not in busy], window.seed, policy):
        ok, _reason, row = gpu_support_status(slot.gpu_name, slot.vram_gb, policy)
        if ok and row is not None and gpu_pass_count(row) > 0:
            drafted.append((slot, row))
    return drafted


def _canonical_gpu_name(gpu_name: str) -> str:
    normalized = " ".join(str(gpu_name or "").strip().lower().split())
    normalized = normalized.replace("pci-e", "pcie").replace("pci e", "pcie")
    return re.sub(r"\b(\d{2,3})\s*g(?:b)?\b", r"\1gb", normalized)


def match_gpu_class(gpu_name: str, vram_gb: int, policy: AuditPolicy) -> GpuClass | None:
    normalized = _canonical_gpu_name(gpu_name)
    for row in policy.gpu_classes:
        if _canonical_gpu_name(row.gpu_name) != normalized:
            continue
        if row.vram_gb and vram_gb:
            if abs(vram_gb - row.vram_gb) > 2 and abs(normalize_vram_gb(vram_gb) - normalize_vram_gb(row.vram_gb)) > 2:
                continue
        return row
    return None


def gpu_support_status(gpu_name: str, vram_gb: int, policy: AuditPolicy) -> tuple[bool, str, GpuClass | None]:
    row = match_gpu_class(gpu_name, vram_gb, policy)
    if row is None:
        return False, "unsupported_gpu_class", None
    return True, "supported", row


def gpu_pass_count(row: GpuClass) -> int:
    return row.capacity_passes + row.capacity_tail_passes + row.fp64_passes


def gpu_workload_spec(row: GpuClass) -> dict[str, Any]:
    """Workload spec handed to the bench for a calibrated GPU row."""
    return {
        "workload_version": "hot_capacity_combined",
        "capacity_matrix_dim": row.matrix_dim,
        "capacity_passes": row.capacity_passes,
        "capacity_rounds": row.capacity_rounds,
        "capacity_warmup_passes": row.capacity_warmup_passes,
        "capacity_block_size": row.capacity_block_size,
        "transition_mix_rounds": row.transition_mix_rounds,
        "transition_fanout": row.transition_fanout,
        "capacity_tail_passes": row.capacity_tail_passes,
        "capacity_tail_rounds": row.capacity_tail_rounds,
        "capacity_tail_warmup_passes": row.capacity_tail_warmup_passes,
        "capacity_tail_transition_mix_rounds": row.capacity_tail_transition_mix_rounds,
        "capacity_tail_transition_fanout": row.capacity_tail_transition_fanout,
        "fp64_matrix_dim": row.fp64_matrix_dim,
        "fp64_passes": row.fp64_passes,
        "fp64_rounds": row.fp64_rounds,
        "fp64_warmup_passes": row.fp64_warmup_passes,
        "fp64_block_size": row.fp64_block_size,
        "spot_checks": row.spot_checks,
        "pass_count": gpu_pass_count(row),
    }


def _unsigned(data: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in data.items() if k != "miner_signature"}


def sign_artifact(data: Mapping[str, Any], keypair: Keypair) -> str:
    """sr25519 signature of an audit artifact by the slot's hotkey."""
    return signing.sign(keypair, ARTIFACT_DOMAIN, _unsigned(data))


def verify_artifact_signature(data: Mapping[str, Any], expected_hotkey: str) -> bool:
    sig = str(data.get("miner_signature") or "")
    return bool(sig) and signing.verify(expected_hotkey, ARTIFACT_DOMAIN, _unsigned(data), sig)
