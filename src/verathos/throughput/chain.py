"""Chain access for the throughput audit (bittensor SDK 10.x): head blocks, block hashes, finality and the
auditor's axon; and for the usage pool, the subnet's burns, its miners' emission, its alpha price and hotkeys' owners
at a block (``throughput.usage``)."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterable
from typing import Any

import bittensor as bt

from verathos.protocol.manifest import axon_base
from verathos.throughput import usage

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
    """Head blocks, block hashes and the auditor's axon (the subnet owner's validator), read through one dedicated
    connection."""

    def __init__(self, network: str, netuid: int, *, cache_s: float = 600.0) -> None:
        self._network, self._netuid = network, netuid
        self._cache_s = cache_s
        self._lock = threading.Lock()
        self._subtensor: bt.Subtensor | None = None
        self._auditor: tuple[float, str] | None = None

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
        """The best block's number and hash, both read from the same header (a load-balanced node cannot pair the
        number of one block with the hash of another)."""
        block_hash = _hash_bytes(self._rpc("chain_getBlockHash", []))
        if block_hash is None:
            return 0, None
        header = self._rpc("chain_getHeader", ["0x" + block_hash.hex()])
        number = header.get("number") if isinstance(header, dict) else None
        if number is None:
            return 0, None
        return (int(number, 0) if isinstance(number, str) else int(number)), block_hash

    def _read(self, read: Callable[..., Any], *args: Any) -> Any:
        with self._lock:
            try:
                return read(self._st(), *args)
            except Exception:
                self._reset()
                raise

    def burns(self, block: int, netuid: int) -> list[usage.Burn]:
        """The burns on ``netuid`` in ``block``. Raises if the node no longer holds the block's state."""
        return self._read(usage.burns_at, block, netuid)

    def emission(self, block: int, netuid: int, mechanism: int) -> usage.Emission:
        return self._read(usage.emission_at, block, netuid, mechanism)

    def alpha_price(self, block: int, netuid: int) -> float:
        return self._read(usage.alpha_price_at, block, netuid)

    def coldkeys(self, hotkeys: Iterable[str], block: int) -> dict[str, str | None]:
        return self._read(usage.coldkeys_at, list(hotkeys), block)

    def block_hash(self, block_number: int) -> bytes | None:
        return _hash_bytes(self._rpc("chain_getBlockHash", [int(block_number)]))

    def finalized_block(self) -> int:
        header = self._rpc("chain_getHeader", [self._rpc("chain_getFinalizedHead", [])])
        number = header.get("number") if isinstance(header, dict) else None
        return int(number, 0) if isinstance(number, str) else int(number or 0)

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
            hotkey = st.get_subnet_owner_hotkey(self._netuid)
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
        return axon_base(axon.ip, axon.port)
