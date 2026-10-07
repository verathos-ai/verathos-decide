"""Run a validator: ``verathos-validator --config validator.json``.

Per epoch (``epoch_blocks`` from the catalog, independent of the subnet tempo): snapshot chain, discover miners, audit
every endpoint with the epoch's canaries spread over the first part of the epoch, score, and set weights on the
configured mechanism (dry run until that mechanism exists).

Every validator takes part in the throughput audit (see ``validator.throughput``): the auditor runs the tests, the
others apply its signed verdicts; endpoints score in proportion to the capacity they proved.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import secrets
import time
from pathlib import Path
from typing import NamedTuple

import bittensor as bt
import httpx

from verathos import catalog
from verathos.throughput.feed import Verdicts
from verathos.throughput.policy import BLOCK_TIME_S
from verathos.validator import chain, status
from verathos.validator.audit import audit_epoch
from verathos.validator.config import ValidatorConfig
from verathos.validator.discovery import DiscoveredEndpoint, discover
from verathos.validator.follow import CatalogFollower
from verathos.validator.scoring import (
    ScoreState,
    epoch_scores,
    model_shares,
    new_evidence,
    record_window,
    update,
    weights,
)
from verathos.validator.service import start_service
from verathos.validator.status import StatusBoard
from verathos.validator.throughput import Throughput

log = logging.getLogger("verathos.validator")
WEIGHT_ATTEMPTS = 10


def load_secret(state_dir: Path) -> bytes:
    """This validator's canary secret, created once (owner-only permissions from the start)."""
    path = state_dir / "canary.secret"
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        pass
    else:
        with os.fdopen(fd, "wb") as f:
            f.write(secrets.token_bytes(32))
    secret = path.read_bytes()
    if len(secret) != 32:
        raise SystemExit(f"{path} must hold 32 random bytes; delete it to create a new one")
    return secret


class Proven(NamedTuple):
    """Per endpoint key, from the auditor's verdicts."""

    credited: dict[str, float]  # proven capacity, requests/s
    latency_ms: dict[str, float]  # median latency in its throughput test
    user_latency_ms: dict[str, float]  # median latency of its answers to users, where there were enough of them
    user_availability: dict[str, float]  # share of the user requests offered to it it did not refuse, where enough


def credited_capacity(endpoints: list[DiscoveredEndpoint], verdicts: Verdicts | None, cat: catalog.Catalog,
                      mode: str) -> Proven:
    """What the verdicts prove, for the model each endpoint serves now and at most what it declares now. In observe
    mode an endpoint's declaration counts, up to its model's ``first_test_rps``, and nothing else does."""
    if mode == "observe":
        first = {m.model_id: m.first_test_rps for m in cat.models}
        return Proven({e.key: min(e.endpoint.capacity_rps, first.get(e.endpoint.model_id, 0.0)) for e in endpoints},
                      {}, {}, {})
    credits = {(c.hotkey, c.endpoint_id, c.model_id): c for c in (verdicts.credits if verdicts else []) if c.qualified}
    sc, proven = cat.scoring, Proven({}, {}, {}, {})
    for e in endpoints:
        c = credits.get((e.hotkey, e.endpoint.endpoint_id, e.endpoint.model_id))
        if c is None:
            continue
        proven.credited[e.key], proven.latency_ms[e.key] = min(c.credited_rps, e.endpoint.capacity_rps), c.latency_ms
        if c.user_answers >= sc.user_latency_min_answers and c.user_latency_ms > 0:
            proven.user_latency_ms[e.key] = c.user_latency_ms
        if c.users_offered >= sc.user_refusals_min_offers:
            proven.user_availability[e.key] = max(0.0, 1.0 - c.users_refused / c.users_offered)
    return proven


def missing_references(cfg: ValidatorConfig, cat: catalog.Catalog) -> list[str]:
    return [m.model_id for m in cat.models if m.status == "active" and m.model_id not in cfg.references]


async def run_epoch(cfg: ValidatorConfig, st: bt.Subtensor, wallet: bt.Wallet, secret: bytes, state: ScoreState,
                    epoch: int, blocks_left: int, throughput: Throughput, current: CatalogFollower,
                    board: StatusBoard) -> None:
    cat = current.for_epoch(st, epoch * current.catalog.epoch_blocks)
    sc, policy = cat.scoring, cat.audit_policy()
    neurons, hotkey_of_uid = chain.snapshot(st, cfg.netuid)
    references = {m: r.engine() for m, r in cfg.references.items()
                  if cat.get(m) is not None and cat.get(m).status != "retired"}
    tolerances = {m.model_id: m.tolerance for m in cat.models}
    # No pool limit: a miner's hanging connections must not delay requests to the others.
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=256)
    async with httpx.AsyncClient(limits=limits) as client:
        miners = await discover(neurons, cfg.netuid, cfg.mechid, client, {cat.digest(), *current.accepted()})
        throughput.update_slots(miners, neurons)
        endpoints = [e for m in miners for e in m.endpoints() if e.endpoint.tier == "open"
                     and cat.get(e.endpoint.model_id) is not None and cat.get(e.endpoint.model_id).status != "retired"]
        log.info("epoch %d: %d miners, %d open endpoints", epoch, len(miners), len(endpoints))
        spread_s = blocks_left * BLOCK_TIME_S * sc.canary_spread_fraction / sc.canaries_per_endpoint
        results = await audit_epoch(endpoints, epoch=epoch, secret=secret, keypair=wallet.hotkey,
                                    references=references, tolerances=tolerances, client=client, netuid=cfg.netuid,
                                    mechid=cfg.mechid, pool_size=sc.canary_pool_size,
                                    per_endpoint=sc.canaries_per_endpoint, spread_s=spread_s,
                                    holding=throughput.holding)
        verdicts = None if policy.mode == "observe" else await throughput.verdicts(client, epoch)
        evidence = []
        if verdicts is not None:
            fresh = new_evidence(state, [f"{e.test_id}/{e.hotkey}/{e.endpoint_id}" for e in verdicts.invalid],
                                 epoch, policy.credit_ttl_epochs + 1)
            evidence = await throughput.proven_invalid(
                [e for e in verdicts.invalid if f"{e.test_id}/{e.hotkey}/{e.endpoint_id}" in fresh], references, cat)
    for e in references.values():
        await e.close()
    if policy.mode != "observe" and verdicts is None:
        log.warning("  no throughput verdicts: no endpoint has proven capacity this epoch")
    for e in evidence:
        where = "gateway traffic" if e.from_gateway else f"throughput test {e.test_id[:12]}"
        log.info("  %s/%s gave a wrong answer in %s: %s", e.hotkey, e.endpoint_id, where, e.reason)
    invalid_hotkeys = {e.hotkey for e in evidence}
    proven = credited_capacity(endpoints, verdicts, cat, policy.mode)
    credited = proven.credited
    record_window(state, results, sc.statistical_window_epochs)
    shares = model_shares(verdicts.demand if verdicts is not None else {}, cat)
    log.info("  emission shares: %s", ", ".join(f"{m} {s:.1%}" for m, s in sorted(shares.items())))
    scores, failed = epoch_scores(results, shares=shares, credited=credited, latency_ms=proven.latency_ms, state=state,
                                  failed_hotkeys=invalid_hotkeys, user_latency_ms=proven.user_latency_ms,
                                  user_availability=proven.user_availability)
    held = {e.key for e in endpoints} - set(results)
    if held:
        log.info("  %d endpoints not audited this epoch (no usable reference answers); scores held", len(held))
    update(state, results, scores, failed, alpha=sc.ema_alpha, probation_epochs=sc.probation_epochs, held=held,
           reset_hotkeys=invalid_hotkeys)
    for key, r in sorted(results.items()):
        log.info("  %s epoch_verdict=%s answered=%d/%d identity=%s mean_abs=%.4f proven_rps=%.1f users=%.2f "
                 "score=%.3f%s",
                 key, r.tally.verdict().value, r.answered, r.sent, r.identity_ok, r.tally.mean_abs,
                 credited.get(key, 0.0), proven.user_availability.get(key, 1.0), scores.get(key, 0.0),
                 f" errors={sorted(set(r.errors))}" if r.errors else "")
    board.publish(status.build(
        epoch=epoch, epoch_blocks=cat.epoch_blocks, endpoints=endpoints, results=results, scores=scores, failed=failed,
        invalid_hotkeys=invalid_hotkeys, credited=credited,
        scored_models={m.model_id for m in cat.models if m.status == "active"}, state=state, model_shares=shares))
    # The epoch is scored; a failure from here on must not run it again.
    try:
        set_weights(cfg, st, wallet, weights(state, hotkey_of_uid), hotkey_of_uid)
    except Exception:
        log.exception("setting weights failed this epoch")


def set_weights(cfg: ValidatorConfig, st: bt.Subtensor, wallet: bt.Wallet, w: dict[int, float],
                hotkey_of_uid: dict[int, str]) -> None:
    if not w:
        owner = chain.owner_uid(st, cfg.netuid, hotkey_of_uid)
        if owner is None:
            log.warning("no miner has a positive score and the subnet owner hotkey holds no UID; no weights set")
        w = {owner: 1.0} if owner is not None else {}
    if cfg.dry_run or not chain.mechanism_ready(st, cfg.netuid, cfg.mechid):
        log.info("dry run (mechanism %d %s): weights %s", cfg.mechid,
                 "not enabled on chain" if not cfg.dry_run else "configured dry", w)
        return
    for attempt in range(1, WEIGHT_ATTEMPTS + 1):
        if not w or chain.submit_weights(st, wallet, cfg.netuid, cfg.mechid, w):
            return
        log.warning("setting weights failed (attempt %d of %d); retrying in one block", attempt, WEIGHT_ATTEMPTS)
        time.sleep(BLOCK_TIME_S)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--once", action="store_true", help="Run one epoch immediately and exit")
    args = parser.parse_args()

    cfg = ValidatorConfig.load(args.config)
    state_dir = cfg.state_dir.expanduser()
    state_dir.mkdir(parents=True, exist_ok=True)
    secret = load_secret(state_dir)
    state_path = state_dir / "scores.json"
    state = ScoreState.load(state_path)
    wallet = bt.Wallet(name=cfg.wallet_name, hotkey=cfg.hotkey_name)
    st = bt.Subtensor(network=cfg.network)
    current = CatalogFollower(st, cfg.network, cfg.netuid, catalog.CatalogStore(cfg.catalog_dir), cfg.catalog_mirrors)
    if missing := missing_references(cfg, current.catalog):
        raise SystemExit(f"no reference engine configured for the active models {missing}")
    throughput = Throughput.start(network=cfg.network, netuid=cfg.netuid, mechid=cfg.mechid,
                                  current=lambda: current.catalog, st=st, wallet=wallet, references=cfg.references,
                                  screens=cfg.audit_screens,
                                  state_dir=state_dir)
    board = StatusBoard()
    is_owner = wallet.hotkey.ss58_address == st.get_subnet_owner_hotkey(cfg.netuid)
    if is_owner or throughput.auditor is not None:
        if not cfg.public_host:
            raise SystemExit("this validator serves the catalog or the throughput verdicts: set public_host")
        if is_owner:
            current.store.put(current.catalog)
        start_service(st=st, wallet=wallet, netuid=cfg.netuid, public_host=cfg.public_host, port=cfg.port,
                      store=current.store if is_owner else None, auditor=throughput.auditor, board=board,
                      probation=lambda: [key for key, left in list(state.probation.items()) if left > 0],
                      gateways=lambda: current.catalog.gateways)
    def adopt(cat: catalog.Catalog) -> None:
        if missing := missing_references(cfg, cat):
            log.critical("catalog %s activates %s, which this validator has no reference engine for; stopping",
                         cat.digest()[:12], missing)
            os._exit(1)
        throughput.set_policy(cat.audit_policy())

    current.on_change.append(adopt)
    current.start()
    throughput.start_slot_refresh(cfg.network, current.accepted)

    done = state.last_epoch
    while True:
        try:
            block = st.get_current_block()
            epoch_blocks = current.catalog.epoch_blocks
            epoch = block // epoch_blocks
            next_start = (epoch + 1) * epoch_blocks
            if epoch != done:
                blocks_left = max(next_start - block, 1)
                asyncio.run(run_epoch(cfg, st, wallet, secret, state, epoch, 1 if args.once else blocks_left,
                                      throughput, current, board))
                done = state.last_epoch = epoch
                state.save(state_path)
                if args.once:
                    break
            time.sleep(max((next_start - st.get_current_block()) * BLOCK_TIME_S, 0) + BLOCK_TIME_S)
        except Exception:
            if args.once:
                raise
            log.exception("epoch step failed; retrying in one block")
            time.sleep(BLOCK_TIME_S)
    st.close()


if __name__ == "__main__":
    main()
    os._exit(0)  # the SDK leaves non-daemon threads behind
