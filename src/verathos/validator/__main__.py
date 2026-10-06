"""Run a validator: ``verathos-validator --config validator.json``.

Per epoch: snapshot chain, discover miners, audit every endpoint with the epoch's canaries spread over the first part
of the epoch, score, and set weights on the configured mechanism (dry run until that mechanism exists).

The validator whose hotkey is the policy's auditor (by default the subnet owner) also runs the capacity audit: it
serves the artifact ingest on its axon, drafts audit windows, and its gated endpoints score zero.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import secrets
import threading
import time
from pathlib import Path

import bittensor as bt
import httpx
import uvicorn

from verathos import catalog
from verathos.capacity.auditor import Auditor
from verathos.capacity.chain import SubtensorChainView
from verathos.capacity.ingest import create_ingest_app
from verathos.chain import serve_axon
from verathos.miner.engine import HttpEngine
from verathos.validator import chain
from verathos.validator.audit import audit_epoch
from verathos.validator.capacity import capacity_slots, unauditable_endpoints
from verathos.validator.config import ValidatorConfig
from verathos.validator.discovery import discover
from verathos.validator.scoring import ScoreState, epoch_scores, record_window, update, weights

log = logging.getLogger("verathos.validator")
BLOCK_S = 12.0


def load_secret(state_dir: Path) -> bytes:
    path = state_dir / "canary.secret"
    if not path.exists():
        path.write_bytes(secrets.token_bytes(32))
        os.chmod(path, 0o600)
    return path.read_bytes()


def start_auditor(cfg: ValidatorConfig, st: bt.Subtensor, wallet: bt.Wallet, state_dir: Path) -> Auditor | None:
    """Run the capacity audit if this validator is the auditor: block follower, ingest server and axon."""
    policy = catalog.load().audit_policy()
    auditor_hotkey = policy.auditor_hotkey or st.get_subnet_owner_hotkey(cfg.netuid)
    if wallet.hotkey.ss58_address != auditor_hotkey:
        log.info("capacity audits are evaluated by %s", auditor_hotkey)
        return None
    if not cfg.capacity_public_host:
        raise SystemExit("this validator is the capacity auditor: set capacity_public_host")
    auditor = Auditor(netuid=cfg.netuid, policy=policy, db_path=state_dir / "capacity.sqlite",
                      chain=SubtensorChainView(cfg.network, cfg.netuid, policy.auditor_hotkey))
    auditor.start()
    server = uvicorn.Server(uvicorn.Config(create_ingest_app(auditor), host="0.0.0.0", port=cfg.capacity_port,
                                           log_level="warning"))
    threading.Thread(target=server.run, name="capacity-ingest", daemon=True).start()
    serve_axon(st, wallet, cfg.netuid, cfg.capacity_public_host, cfg.capacity_port)
    log.info("capacity auditor running (%s mode), ingest on port %d", policy.mode, cfg.capacity_port)
    return auditor


async def run_epoch(cfg: ValidatorConfig, st: bt.Subtensor, wallet: bt.Wallet, secret: bytes, state: ScoreState,
                    epoch: int, blocks_left: int, auditor: Auditor | None) -> None:
    cat = catalog.load()
    neurons, hotkey_of_uid = chain.snapshot(st, cfg.netuid)
    references = {m: HttpEngine(url) for m, url in cfg.references.items() if cat.get(m) is not None}
    tolerances = {m.model_id: m.tolerance for m in cat.models}
    async with httpx.AsyncClient() as client:
        miners = await discover(neurons, cfg.netuid, client)
        if auditor is not None:
            auditor.update_slots(capacity_slots(miners, cfg.netuid))
        endpoints = [e for m in miners for e in m.endpoints() if cat.get(e.endpoint.model_id) is not None
                     and e.endpoint.tier == "open"]
        log.info("epoch %d: %d miners, %d open endpoints", epoch, len(miners), len(endpoints))
        spread_s = blocks_left * BLOCK_S * cfg.audit_fraction / max(cfg.per_endpoint, 1)
        results = await audit_epoch(endpoints, epoch=epoch, secret=secret, keypair=wallet.hotkey,
                                    references=references, tolerances=tolerances, client=client,
                                    pool_size=cfg.pool_size, per_endpoint=cfg.per_endpoint, spread_s=spread_s,
                                    drained=auditor.drained if auditor is not None else lambda hotkey, ep: False)
    for e in references.values():
        await e.close()
    gated: set[str] = set()
    unauditable: set[str] = set()
    if auditor is not None and auditor.policy.mode == "score_gate":
        unauditable = unauditable_endpoints(miners, auditor.policy)
    if auditor is not None:
        tempo_epoch = st.get_current_block() // auditor.chain.epoch_blocks()
        for (hotkey, endpoint_id), rule in sorted(auditor.gated(tempo_epoch).items()):
            log.info("  %s/%s capacity gated: %s", hotkey, endpoint_id, rule)
            gated.add(f"{hotkey}/{endpoint_id}")
        auditor.prune(tempo_epoch)
    record_window(state, results, cfg.window_epochs)
    scores, failed = epoch_scores(results, {m.model_id: m.score_weight for m in cat.models}, state, gated,
                                  unauditable)
    update(state, results, scores, failed, alpha=cfg.ema_alpha, probation_epochs=cfg.probation_epochs)
    for key, r in sorted(results.items()):
        log.info("  %s epoch_verdict=%s answered=%d/%d identity=%s mean_abs=%.4f score=%.3f",
                 key, r.tally.verdict().value, r.answered, r.sent, r.identity_ok, r.tally.mean_abs,
                 scores.get(key, 0.0))
    w = weights(state, hotkey_of_uid)
    if not w:
        owner = chain.owner_uid(st, cfg.netuid, hotkey_of_uid)
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
    auditor = start_auditor(cfg, st, wallet, state_dir)

    done: int | None = None
    while True:
        block = st.get_current_block()
        next_start = st.get_next_epoch_start_block(cfg.netuid) or block + 360
        epoch = next_start  # an epoch is identified by the block it ends at
        if epoch != done:
            blocks_left = max(next_start - block, 1)
            asyncio.run(run_epoch(cfg, st, wallet, secret, state, epoch, 1 if args.once else blocks_left, auditor))
            state.save(state_path)
            done = epoch
            if args.once:
                break
        time.sleep(max((next_start - st.get_current_block()) * BLOCK_S, 0) + BLOCK_S)
    st.close()


if __name__ == "__main__":
    main()
    os._exit(0)  # the SDK leaves non-daemon threads behind
