"""Run a validator: ``verathos-validator --config validator.json``.

Per epoch: snapshot chain, discover miners, audit every endpoint with the epoch's canaries spread over the first part
of the epoch, score, and set weights on the configured mechanism (dry run until that mechanism exists).

Every validator takes part in the capacity audit (see ``validator.capacity``): the auditor runs it, the others apply
its signed verdicts; gated endpoints score zero.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import secrets
import time
from pathlib import Path

import bittensor as bt
import httpx

from verathos import catalog
from verathos.miner.engine import HttpEngine
from verathos.validator import chain
from verathos.validator.audit import audit_epoch
from verathos.validator.capacity import Capacity, unauditable_endpoints
from verathos.validator.config import ValidatorConfig
from verathos.validator.discovery import discover
from verathos.validator.follow import CatalogFollower
from verathos.validator.scoring import ScoreState, epoch_scores, record_window, update, weights
from verathos.validator.service import start_service

log = logging.getLogger("verathos.validator")
BLOCK_S = 12.0


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


async def run_epoch(cfg: ValidatorConfig, st: bt.Subtensor, wallet: bt.Wallet, secret: bytes, state: ScoreState,
                    epoch: int, blocks_left: int, capacity: Capacity, current: CatalogFollower) -> None:
    cat = current.catalog
    capacity_epoch = capacity.tempo_epoch()
    neurons, hotkey_of_uid = chain.snapshot(st, cfg.netuid)
    references = {m: HttpEngine(url) for m, url in cfg.references.items()
                  if cat.get(m) is not None and cat.get(m).status != "retired"}
    tolerances = {m.model_id: m.tolerance for m in cat.models}
    async with httpx.AsyncClient() as client:
        miners = await discover(neurons, cfg.netuid, cfg.mechid, client, current.accepted())
        capacity.update_slots(miners)
        endpoints = [e for m in miners for e in m.endpoints() if e.endpoint.tier == "open"
                     and cat.get(e.endpoint.model_id) is not None and cat.get(e.endpoint.model_id).status != "retired"]
        log.info("epoch %d: %d miners, %d open endpoints", epoch, len(miners), len(endpoints))
        spread_s = blocks_left * BLOCK_S * cfg.audit_fraction / max(cfg.per_endpoint, 1)
        results = await audit_epoch(endpoints, epoch=epoch, secret=secret, keypair=wallet.hotkey,
                                    references=references, tolerances=tolerances, client=client, netuid=cfg.netuid,
                                    mechid=cfg.mechid, pool_size=cfg.pool_size, per_endpoint=cfg.per_endpoint,
                                    spread_s=spread_s, drained=capacity.drained)
        gated = await capacity.gated(client, capacity_epoch)
    for e in references.values():
        await e.close()
    for (hotkey, endpoint_id), rule in sorted(gated.items()):
        log.info("  %s/%s capacity gated: %s", hotkey, endpoint_id, rule)
    gating = capacity.policy.mode == "score_gate"
    unauditable = unauditable_endpoints(miners, capacity.policy) if gating else set()
    record_window(state, results, cfg.window_epochs)
    model_weight = {m.model_id: m.score_weight if m.status == "active" else 0.0 for m in cat.models}
    scores, failed = epoch_scores(results, model_weight, state,
                                  {f"{hk}/{ep}" for hk, ep in gated}, unauditable)
    held = {e.key for e in endpoints} - set(results)
    if held:
        log.info("  %d endpoints not audited this epoch (no usable reference answers); scores held", len(held))
    update(state, results, scores, failed, alpha=cfg.ema_alpha, probation_epochs=cfg.probation_epochs, held=held)
    for key, r in sorted(results.items()):
        log.info("  %s epoch_verdict=%s answered=%d/%d identity=%s mean_abs=%.4f score=%.3f%s",
                 key, r.tally.verdict().value, r.answered, r.sent, r.identity_ok, r.tally.mean_abs,
                 scores.get(key, 0.0), f" errors={sorted(set(r.errors))}" if r.errors else "")
    w = weights(state, hotkey_of_uid, per_uid="sum" if gating else "max")
    if not w:
        owner = chain.owner_uid(st, cfg.netuid, hotkey_of_uid)
        if owner is None:
            log.warning("no miner earned weight and the subnet owner hotkey holds no UID; no weights set")
        w = {owner: 1.0} if owner is not None else {}
    if cfg.dry_run or not chain.mechanism_ready(st, cfg.netuid, cfg.mechid):
        log.info("dry run (mechanism %d %s): weights %s", cfg.mechid,
                 "not enabled on chain" if not cfg.dry_run else "configured dry", w)
    elif w:
        chain.submit_weights(st, wallet, cfg.netuid, cfg.mechid, w)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
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
    current = CatalogFollower(st, cfg.network, cfg.netuid, catalog.CatalogStore(cfg.catalog_dir), cfg.catalog_mirrors,
                              grace_s=(st.tempo(cfg.netuid) + 1) * BLOCK_S)
    capacity = Capacity.start(network=cfg.network, netuid=cfg.netuid, policy=current.catalog.audit_policy(), st=st,
                              wallet=wallet, state_dir=state_dir)
    is_owner = wallet.hotkey.ss58_address == st.get_subnet_owner_hotkey(cfg.netuid)
    if is_owner or capacity.auditor is not None:
        if not cfg.public_host:
            raise SystemExit("this validator serves the catalog or the capacity audit: set public_host")
        if is_owner:
            current.store.put(current.catalog)
        start_service(st=st, wallet=wallet, netuid=cfg.netuid, public_host=cfg.public_host, port=cfg.port,
                      store=current.store if is_owner else None, auditor=capacity.auditor)
    current.on_change.append(lambda cat: capacity.set_policy(cat.audit_policy()))
    current.start()
    capacity.start_slot_refresh(cfg.network, cfg.mechid, current.accepted)

    done: int | None = None
    while True:
        try:
            block = st.get_current_block()
            next_start = st.get_next_epoch_start_block(cfg.netuid) or block + 360
            epoch = next_start  # an epoch is identified by the block it ends at
            if epoch != done:
                blocks_left = max(next_start - block, 1)
                asyncio.run(run_epoch(cfg, st, wallet, secret, state, epoch, 1 if args.once else blocks_left,
                                      capacity, current))
                state.save(state_path)
                done = epoch
                if args.once:
                    break
            time.sleep(max((next_start - st.get_current_block()) * BLOCK_S, 0) + BLOCK_S)
        except Exception:
            if args.once:
                raise
            log.exception("epoch step failed; retrying in one block")
            time.sleep(BLOCK_S)
    st.close()


if __name__ == "__main__":
    main()
    os._exit(0)  # the SDK leaves non-daemon threads behind
