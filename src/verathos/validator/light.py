"""Light mode: a validator that sets the weights a source validator publishes, without checking miners itself.

Each epoch the source (the subnet owner's validator, the auditor, or the validator ``source_hotkey`` names, which holds
a permit and runs a service) scores the network, publishes its results with the weights it sets at ``GET /v1/network``,
signed with its hotkey, and sets them. A light validator fetches that document from the source's axon over TLS,
accepting the certificate whose pin the catalog names for the auditor, or the source validator's chain commitment names
(``protocol.callers``), and sets exactly those weights, on the mechanism of the catalog it follows (or the configured
``mechid``), through the code full validators use (version key, commit-reveal, retries, dry run), only from a document
that verifies: signed by the source hotkey under ``verathos/network-status/v1`` for this netuid, issued within
``STATUS_MAX_AGE_S`` of its clock, for the current or the previous epoch of the catalog's length, and for a later epoch
than the weights it set last. It sends nothing to miners, runs no reference engine and needs no stake beyond the
validator permit that setting weights takes.

The source scores an epoch once its canaries are out, after the first ``canary_spread_fraction`` of the epoch, and
publishes the results then. A light validator asks for epoch ``e``'s results from that point of epoch ``e`` every
``POLL_S`` until it has set them (or a later epoch's), and at most until the same point of epoch ``e + 1``, the
earliest the results of ``e + 1`` can come; results of an epoch before ``e`` are not set any more. Without a document
that verifies by then it sets nothing for ``e`` (the weights it set before stay on chain) and logs why, once.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import bittensor as bt
import httpx
from pydantic import ValidationError

from verathos.catalog import AuditorUnknown, Catalog, mechanism_of
from verathos.protocol.callers import validator_identity
from verathos.protocol.http import ResponseTooLarge, get_json
from verathos.protocol.manifest import axon_base
from verathos.protocol.status import MAX_NETWORK_BYTES, NETWORK_DOMAIN, NETWORK_PATH, NetworkStatus, unverified
from verathos.protocol.tls import Service
from verathos.subnet.chain import CHAIN_ERRORS, one_line
from verathos.throughput.policy import BLOCK_TIME_S
from verathos.validator.chain import ReleaseTooOld, weights_version_key
from verathos.validator.config import ValidatorConfig

log = logging.getLogger("verathos.validator")
POLL_S = 30.0  # how often the source's document is asked for while due; the source signs it again as often
FETCH_TIMEOUT_S = 30.0
TOP_WEIGHTS = 5  # weights named in an epoch's log line
OWNER_IS_FULL = ("this is the subnet owner's hotkey: its validator is the auditor, whose results light validators set, "
                 "and runs in full mode; set mode to full (docs/VALIDATING.md)")


class Refused(ValueError):
    """Why no weights are set from the source's document now."""


class NotTaken(Refused):
    """The source's document is verified, but its weights cannot be set on chain now."""


def check(document: Any, *, source: str, netuid: int, epoch: int, epoch_blocks: int, done_block: int | None,
          now_ms: int) -> NetworkStatus:
    """``document`` (as received) as a network status, if its weights may be set now: in ``epoch`` of
    ``epoch_blocks`` blocks, the weights of the epoch starting at ``done_block`` set last. Raises Refused."""
    if why := unverified(document, domain=NETWORK_DOMAIN, auditor=source, netuid=netuid, now_ms=now_ms,
                         signer="the source hotkey"):
        raise Refused(why)
    try:
        network = NetworkStatus.model_validate(document)
    except ValidationError as exc:
        raise Refused(f"it is not a network status ({exc.error_count()} errors)") from None
    if network.weights is None:
        raise Refused("it lists no weights (the source runs an older release)")
    if network.epoch is None or network.epoch_blocks != epoch_blocks:
        raise Refused(f"it is not for an epoch of {epoch_blocks} blocks, the catalog's")
    if network.epoch not in (epoch, epoch - 1):
        raise Refused(f"it is for epoch {network.epoch}, neither the current epoch {epoch} nor the previous one")
    if done_block is not None and network.epoch * epoch_blocks <= done_block:
        raise Refused(f"the source's latest results are for epoch {network.epoch}, whose weights are set already")
    if not all(uid >= 0 and math.isfinite(w) and w >= 0 for uid, w in network.weights.items()):
        raise Refused("its weights are not all non-negative numbers for UIDs")
    return network


def find_source(st: bt.Subtensor, netuid: int, owner: str | None, source_hotkey: str | None,
                cat: Catalog) -> tuple[str, Service]:
    """The source hotkey (``source_hotkey``, else the subnet owner's, ``owner``) and its service, from the chain:
    registered, with a validator permit unless it is the owner's, serving an axon; its pin is the auditor's in ``cat``
    for the owner, else the one its validator commitment names. Raises Refused."""
    source = source_hotkey or owner
    if not source:
        raise Refused("the subnet owner hotkey is not known")
    neuron = st.get_neuron_for_pubkey_and_subnet(source, netuid)
    if neuron is None or neuron.is_null:
        raise Refused(f"the source hotkey {source} is not registered on netuid {netuid}")
    if source != owner and not neuron.validator_permit:
        raise Refused(f"the source hotkey {source} holds no validator permit")
    ip, port = neuron.axon_info.ip, neuron.axon_info.port
    if ip in ("", "0.0.0.0") or not port:
        raise Refused(f"the source hotkey {source} serves no axon")
    if source == owner:
        try:
            pin = cat.auditor_pin()
        except AuditorUnknown as exc:
            raise Refused(str(exc)) from None
    else:
        identity = validator_identity(st.get_commitment(netuid, neuron.uid))
        if identity is None or identity.tls is None:
            raise Refused(f"the source hotkey {source} commits no TLS pin for its service (a verathos/validator/v1 "
                          "commitment, which a full validator with public_host publishes)")
        pin = identity.tls
    return source, Service(axon_base(ip, port), pin)


def describe(weights: dict[int, float]) -> str:
    """The number of UIDs and the highest weights."""
    top = sorted(weights.items(), key=lambda item: (-item[1], item[0]))[:TOP_WEIGHTS]
    return f"{len(weights)} UIDs" + (", top " + ", ".join(f"{uid}: {w:.3f}" for uid, w in top) if top else "")


class LightValidator:
    """Sets each epoch's weights from the source's published results (see the module). ``current.catalog``: the
    catalog followed; ``set_weights``: the full validators' way of setting weights on a mechanism, True once they are
    set."""

    def __init__(self, cfg: ValidatorConfig, st: bt.Subtensor, hotkey: str, current: Any, *,
                 set_weights: Callable[[dict[int, float], int], bool], state_path: Path,
                 fetch: Callable[..., Any] = get_json, clock: Callable[[], float] = time.time) -> None:
        self.cfg, self.st, self.hotkey, self.current = cfg, st, hotkey, current
        self.set_weights, self.state_path, self.fetch, self.clock = set_weights, state_path, fetch, clock
        self.done_block = self._load()  # the first block of the epoch whose weights were set last
        self.why: str | None = None  # why the current window's results are not set yet
        self.verified = False  # whether they are verified, so that the chain or this hotkey is why
        self._window: int | None = None

    def _load(self) -> int | None:
        try:
            saved = json.loads(self.state_path.read_text())
            return int(saved["epoch"]) * int(saved["epoch_blocks"])
        except FileNotFoundError:
            return None
        except (ValueError, KeyError, TypeError) as exc:
            log.warning("%s unreadable, starting without the last epoch set: %s", self.state_path, exc)
            return None

    def _save(self, epoch: int, epoch_blocks: int) -> None:
        self.done_block = epoch * epoch_blocks
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"epoch": epoch, "epoch_blocks": epoch_blocks}))
            tmp.replace(self.state_path)
        except OSError as exc:
            log.warning("keeping the last epoch set in %s failed: %s", self.state_path, exc)

    @staticmethod
    def offset(cat: Catalog) -> int:
        """Blocks into an epoch before which the source has not scored it: its canaries' share of the epoch."""
        return int(cat.epoch_blocks * cat.scoring.canary_spread_fraction)

    def source(self) -> tuple[str, Service]:
        """The source hotkey and its service, from the chain. Raises Refused, or SystemExit once this hotkey is the
        subnet owner's."""
        owner = self.st.get_subnet_owner_hotkey(self.cfg.netuid)
        if owner == self.hotkey:
            raise SystemExit(OWNER_IS_FULL)
        return find_source(self.st, self.cfg.netuid, owner, self.cfg.source_hotkey, self.current.catalog)

    def permit(self) -> str | None:
        """Why this validator cannot set weights; None if it can."""
        neuron = self.st.get_neuron_for_pubkey_and_subnet(self.hotkey, self.cfg.netuid)
        if neuron is None or neuron.is_null:
            return f"this hotkey is not registered on netuid {self.cfg.netuid}"
        if not neuron.validator_permit:
            return "this hotkey holds no validator permit, which setting weights takes"
        try:
            weights_version_key(self.st, self.cfg.netuid)
        except ReleaseTooOld as exc:
            return str(exc)
        return None

    def due(self, block: int) -> int:
        """The epoch whose results are due at ``block``: the source publishes them from the offset into it on."""
        cat = self.current.catalog
        return (block - self.offset(cat)) // cat.epoch_blocks

    def step(self) -> float:
        """Set the source's latest results if they are due and verify; the seconds until the next step."""
        block = self.st.get_current_block()
        cat = self.current.catalog
        epoch_blocks, offset, window = cat.epoch_blocks, self.offset(cat), self.due(block)
        if self._window is not None and window > self._window and not self.covers(self._window, epoch_blocks):
            deadline = (self._window + 1) * epoch_blocks + offset
            if self.verified:
                log.warning("epoch %d: no weights set by block %d: the source's results are verified, but %s",
                            self._window, deadline, self.why)
            else:
                log.warning("epoch %d: no weights set: no results of the source verified by block %d (%s)",
                            self._window, deadline, self.why or "not asked")
        if window != self._window:
            self._window, self.why, self.verified = window, None, False
        if self.covers(window, epoch_blocks):
            return max((window + 1) * epoch_blocks + offset - block, 1) * BLOCK_TIME_S
        try:
            self.attempt(block)
        except Refused as exc:
            self.why, self.verified = str(exc), isinstance(exc, NotTaken)
            return POLL_S
        return BLOCK_TIME_S

    def covers(self, epoch: int, epoch_blocks: int) -> bool:
        """Whether weights of ``epoch`` or a later one are set."""
        return self.done_block is not None and self.done_block >= epoch * epoch_blocks

    def attempt(self, block: int) -> None:
        """Fetch the source's document and set its weights, at ``block``. Raises Refused."""
        epoch_blocks, window = self.current.catalog.epoch_blocks, self.due(block)
        source, service = self.source()
        try:
            document = self.fetch(service.url + NETWORK_PATH, limit=MAX_NETWORK_BYTES, timeout=FETCH_TIMEOUT_S,
                                  tls=service.pin)
        except (httpx.HTTPError, ResponseTooLarge, ValueError) as exc:
            raise Refused(f"the source's axon {service.url} did not serve it: {one_line(exc)}") from None
        network = check(document, source=source, netuid=self.cfg.netuid, epoch=block // epoch_blocks,
                        epoch_blocks=epoch_blocks, done_block=self.done_block, now_ms=int(self.clock() * 1000))
        if network.epoch < window:  # an earlier epoch's, not set before: the due ones follow within the window
            raise Refused(f"the source has not published the results of epoch {window} yet (its latest are for "
                          f"epoch {network.epoch})")
        if why := self.permit():
            raise NotTaken(why)
        if not self.set_weights(network.weights, mechanism_of(self.current.catalog, self.cfg.mechid)):
            raise NotTaken("setting them on chain failed")
        self._save(network.epoch, epoch_blocks)
        if network.weights:
            log.info("epoch %d: set the weights of %s: %s", network.epoch, self.label(source),
                     describe(network.weights))
        else:
            log.info("epoch %d: %s set no weights, and neither does this validator", network.epoch,
                     self.label(source))

    def label(self, source: str) -> str:
        return f"the validator {source}" if self.cfg.source_hotkey else f"the subnet owner's validator {source}"


def run(cfg: ValidatorConfig, st: bt.Subtensor, wallet: bt.Wallet, current: Any, *,
        set_weights: Callable[[dict[int, float], int], bool], once: bool = False,
        between: Callable[[], None] | None = None) -> None:
    """Run a light validator; with ``once``, try to set the source's latest results once and return. ``between`` is
    called between steps, where no weights are being set (the self-update)."""
    hotkey = wallet.hotkey.ss58_address
    if hotkey == st.get_subnet_owner_hotkey(cfg.netuid):
        raise SystemExit(OWNER_IS_FULL)
    if cfg.source_hotkey == hotkey:
        raise SystemExit("source_hotkey is this validator's own hotkey: name the validator whose weights to set")
    light = LightValidator(cfg, st, hotkey, current, set_weights=set_weights,
                           state_path=cfg.state_dir.expanduser() / "light.json")
    if why := light.permit():
        log.critical("%s; this validator sets no weights until that changes", why)
    log.info("light mode: setting the weights %s publishes each epoch, once its document verifies; no requests to "
             "miners", f"the validator {cfg.source_hotkey}" if cfg.source_hotkey else "the subnet owner's validator")
    if once:
        try:
            light.attempt(st.get_current_block())
        except Refused as exc:
            log.warning("no weights set: %s", exc)
        return
    while True:
        try:
            wait = light.step()
        except CHAIN_ERRORS as exc:
            log.warning("light mode: chain unavailable (%s); retrying in one block", one_line(exc))
            wait = BLOCK_TIME_S
        except Exception:
            log.exception("light mode step failed; retrying in one block")
            wait = BLOCK_TIME_S
        if between is not None:
            between()
        time.sleep(wait)
