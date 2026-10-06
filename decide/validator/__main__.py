"""Run a validator: ``python -m decide.validator --config validator.json``.

Per epoch: snapshot chain, discover miners, audit every endpoint with the epoch's canaries spread over the first part
of the epoch, score, and set weights on the configured mechanism (dry run until that mechanism exists).
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

from decide import catalog
from decide.miner.engine import HttpEngine
from decide.validator import chain
from decide.validator.audit import audit_epoch
from decide.validator.config import ValidatorConfig
from decide.validator.discovery import discover
from decide.validator.scoring import ScoreState, epoch_scores, update, weights

log = logging.getLogger("decide.validator")
BLOCK_S = 12.0


def load_secret(state_dir: Path) -> bytes:
    path = state_dir / "canary.secret"
    if not path.exists():
        path.write_bytes(secrets.token_bytes(32))
        os.chmod(path, 0o600)
    return path.read_bytes()


async def run_epoch(cfg: ValidatorConfig, st: bt.Subtensor, wallet: bt.Wallet, secret: bytes, state: ScoreState,
                    epoch: int, blocks_left: int) -> None:
    cat = catalog.load()
    neurons, hotkey_of_uid = chain.snapshot(st, cfg.netuid)
    references = {m: HttpEngine(url) for m, url in cfg.references.items() if cat.get(m) is not None}
    tolerances = {m.model_id: m.tolerance for m in cat.models}
    async with httpx.AsyncClient() as client:
        miners = await discover(neurons, cfg.netuid, client)
        endpoints = [e for m in miners for e in m.endpoints() if cat.get(e.endpoint.model_id) is not None
                     and e.endpoint.tier == "open"]
        log.info("epoch %d: %d miners, %d open endpoints", epoch, len(miners), len(endpoints))
        spread_s = blocks_left * BLOCK_S * cfg.audit_fraction / max(cfg.per_endpoint, 1)
        results = await audit_epoch(endpoints, epoch=epoch, secret=secret, keypair=wallet.hotkey,
                                    references=references, tolerances=tolerances, client=client,
                                    pool_size=cfg.pool_size, per_endpoint=cfg.per_endpoint, spread_s=spread_s)
    for e in references.values():
        await e.close()
    scores, failed = epoch_scores(results, {m.model_id: m.score_weight for m in cat.models})
    update(state, results, scores, failed, alpha=cfg.ema_alpha, probation_epochs=cfg.probation_epochs)
    for key, r in sorted(results.items()):
        log.info("  %s verdict=%s answered=%d/%d identity=%s mean_abs=%.4f score=%.3f", key, r.tally.verdict().value,
                 r.answered, r.sent, r.identity_ok, r.tally.mean_abs, scores.get(key, 0.0))
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

    done: int | None = None
    while True:
        block = st.get_current_block()
        next_start = st.get_next_epoch_start_block(cfg.netuid) or block + 360
        epoch = next_start  # an epoch is identified by the block it ends at
        if epoch != done:
            blocks_left = max(next_start - block, 1)
            asyncio.run(run_epoch(cfg, st, wallet, secret, state, epoch, 1 if args.once else blocks_left))
            state.save(state_path)
            done = epoch
            if args.once:
                break
        time.sleep(max((next_start - st.get_current_block()) * BLOCK_S, 0) + BLOCK_S)
    st.close()


if __name__ == "__main__":
    main()
    os._exit(0)  # the SDK leaves non-daemon threads behind
