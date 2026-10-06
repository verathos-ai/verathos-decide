"""Chain access for the capacity audit (bittensor SDK 10.x)."""

from __future__ import annotations

import logging
import threading
import time

import bittensor as bt

log = logging.getLogger(__name__)


def _hash_bytes(raw: object) -> bytes | None:
    text = str(raw or "").strip().removeprefix("0x")
    if len(text) != 64:
        return None
    try:
        return bytes.fromhex(text)
    except ValueError:
        return None


class SubtensorChainView:
    """Head blocks, block hashes, tempo and the auditor's axon, read through one dedicated connection."""

    def __init__(self, network: str, netuid: int, auditor_hotkey: str = "", *, cache_s: float = 600.0) -> None:
        self._network, self._netuid, self._auditor_override = network, netuid, auditor_hotkey
        self._cache_s = cache_s
        self._lock = threading.Lock()
        self._subtensor: bt.Subtensor | None = None
        self._tempo: tuple[float, int] | None = None
        self._auditor: tuple[float, str] | None = None

    def set_auditor_hotkey(self, auditor_hotkey: str) -> None:
        """Follow a newly committed auditor (empty: the subnet owner)."""
        if auditor_hotkey != self._auditor_override:
            self._auditor_override, self._auditor = auditor_hotkey, None

    def _st(self) -> bt.Subtensor:
        if self._subtensor is None:
            self._subtensor = bt.Subtensor(network=self._network)
        return self._subtensor

    def _rpc(self, method: str, params: list) -> object:
        with self._lock:
            try:
                response = self._st().substrate.rpc_request(method, params)
            except Exception:
                self._reset()
                raise
        return response.get("result") if isinstance(response, dict) else response

    def _reset(self) -> None:
        if self._subtensor is not None:
            try:
                self._subtensor.close()
            except Exception:
                pass
        self._subtensor = None

    def head(self) -> tuple[int, bytes | None]:
        header = self._rpc("chain_getHeader", [])
        number = header.get("number") if isinstance(header, dict) else None
        if number is None:
            return 0, None
        block = int(number, 0) if isinstance(number, str) else int(number)
        return block, self.block_hash(block)

    def block_hash(self, block_number: int) -> bytes | None:
        return _hash_bytes(self._rpc("chain_getBlockHash", [int(block_number)]))

    def finalized_block(self) -> int:
        header = self._rpc("chain_getHeader", [self._rpc("chain_getFinalizedHead", [])])
        number = header.get("number") if isinstance(header, dict) else None
        return int(number, 0) if isinstance(number, str) else int(number or 0)

    def epoch_blocks(self) -> int:
        now = time.monotonic()
        if self._tempo is None or now - self._tempo[0] > self._cache_s:
            with self._lock:
                tempo = int(self._st().tempo(self._netuid) or 360)
            self._tempo = (now, tempo)
        return self._tempo[1]

    def auditor_url(self) -> str | None:
        """The auditor's axon URL, cached once resolved; an unresolved auditor is looked up again on each call."""
        now = time.monotonic()
        if self._auditor is None or now - self._auditor[0] > self._cache_s:
            url = self._resolve_auditor_url()
            if url is None:
                return None
            self._auditor = (now, url)
        return self._auditor[1]

    def _resolve_auditor_url(self) -> str | None:
        with self._lock:
            st = self._st()
            hotkey = self._auditor_override or st.get_subnet_owner_hotkey(self._netuid)
            if not hotkey:
                return None
            neuron = st.get_neuron_for_pubkey_and_subnet(hotkey, self._netuid)
        if neuron is None or neuron.is_null:
            log.warning("auditor hotkey %s is not registered on netuid %d", hotkey, self._netuid)
            return None
        axon = neuron.axon_info
        if axon.ip in ("", "0.0.0.0") or not axon.port:
            log.warning("auditor hotkey %s has no axon served", hotkey)
            return None
        return f"http://{axon.ip}:{axon.port}"
