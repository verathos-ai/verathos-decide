"""The validator's view of the capacity audit: which discovered endpoints it drafts from and which it cannot cover."""

from __future__ import annotations

from verathos.capacity.core import AuditPolicy, Slot, gpu_support_status
from verathos.validator.discovery import DiscoveredMiner


def capacity_slots(miners: list[DiscoveredMiner], netuid: int) -> list[Slot]:
    """Every open CUDA endpoint with declared hardware: the slots the capacity audit drafts from."""
    return [Slot(netuid, m.hotkey, e.endpoint_id, e.url, e.model_id, e.hardware.gpu_name, e.hardware.vram_gb)
            for m in miners for e in m.manifest.endpoints
            if e.tier == "open" and e.hardware.kind == "cuda" and e.hardware.gpu_name and e.hardware.vram_gb]


def unauditable_endpoints(miners: list[DiscoveredMiner], policy: AuditPolicy) -> set[str]:
    """Open endpoints the capacity audit cannot draft: not CUDA, or no calibrated GPU class declared."""
    return {f"{m.hotkey}/{e.endpoint_id}" for m in miners for e in m.manifest.endpoints if e.tier == "open"
            and (e.hardware.kind != "cuda"
                 or not gpu_support_status(e.hardware.gpu_name or "", e.hardware.vram_gb or 0, policy)[0])}
