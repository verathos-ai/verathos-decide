"""Run a validator: ``verathos-validator --config validator.json``.

Per epoch (``epoch_blocks`` from the catalog, independent of the subnet tempo): snapshot chain, discover miners, audit
every endpoint with the epoch's canaries spread over the first part of the epoch, score, publish the status with the
weights, and set those weights on the configured mechanism (dry run until that mechanism exists).

Every full validator takes part in the throughput audit (see ``validator.throughput``): the auditor runs the tests,
the others apply its signed verdicts; endpoints score in proportion to the capacity they proved. A light validator
(``mode: light``, ``validator.light``) does none of this: it sets the weights its source validator publishes.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import logging
import os
import secrets
import sys
import time
from pathlib import Path
from typing import NamedTuple

import bittensor as bt
import httpx

from verathos import catalog
from verathos.throughput.feed import Credit, Verdicts, published_shares
from verathos.throughput.policy import BLOCK_TIME_S
from verathos.validator import chain, light, status
from verathos.validator.audit import audit_epoch
from verathos.validator.config import ReferenceEngine, ValidatorConfig
from verathos.validator.discovery import DiscoveredEndpoint, Neuron, discover
from verathos.validator.follow import CatalogFollower
from verathos.validator.references import transitioning, wrong_revision
from verathos.validator.scoring import (
    ScoreState,
    epoch_scores,
    model_shares,
    new_evidence,
    record_window,
    update,
    weighted_median,
    weights,
)
from verathos.validator.service import start_service
from verathos.validator.status import StatusBoard
from verathos.validator.throughput import Throughput

log = logging.getLogger("verathos.validator")
WEIGHT_ATTEMPTS = 10
REFERENCE_WAIT_S = 900.0  # the most the first epoch waits for reference engines still loading their models
EVIDENCE_SHARE = 0.1  # of the epoch, the most rechecking the auditor's evidence may take


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
    user_latency_ms: dict[str, float]  # median latency of its hotkey's answers to users of the model, if enough
    user_availability: dict[str, float]  # share of the times its hotkey's endpoints of the model were ready
    drifted: set[str]  # credited nothing: its answers drifted beyond the model's statistical tolerance


def credited_capacity(endpoints: list[DiscoveredEndpoint], verdicts: Verdicts | None, cat: catalog.Catalog,
                      mode: str) -> Proven:
    """What the verdicts prove, for the model each endpoint serves now and at most what it declares now. In observe
    mode an endpoint's declaration counts, up to its model's ``first_test_rps``, and nothing else does."""
    if mode == "observe":
        first = {m.model_id: m.first_test_rps for m in cat.models}
        return Proven({e.key: min(e.endpoint.capacity_rps, first.get(e.endpoint.model_id, 0.0)) for e in endpoints},
                      {}, {}, {}, set())
    credits = {(c.hotkey, c.endpoint_id, c.model_id): c for c in (verdicts.credits if verdicts else []) if c.qualified}
    sc, proven = cat.scoring, Proven({}, {}, {}, {}, set())
    groups: dict[tuple[str, str], list[tuple[str, Credit]]] = {}
    for e in endpoints:
        c = credits.get((e.hotkey, e.endpoint.endpoint_id, e.endpoint.model_id))
        if c is None:
            continue
        proven.credited[e.key], proven.latency_ms[e.key] = min(c.credited_rps, e.endpoint.capacity_rps), c.latency_ms
        if c.drift:
            proven.drifted.add(e.key)
        groups.setdefault((e.hotkey, e.endpoint.model_id), []).append((e.key, c))
    # What users got counts per hotkey and model: splitting capacity over endpoints hides none of it.
    for members in groups.values():
        offers = sum(c.user_offers for _, c in members)
        ready = max(offers - sum(c.user_refusals for _, c in members), 0)
        answers = sum(c.user_answers for _, c in members)
        latencies = [(c.user_latency_ms, c.user_answers) for _, c in members if c.user_answers and c.user_latency_ms]
        for key, _ in members:
            proven.user_availability[key] = (ready + sc.user_offers_prior) / (offers + sc.user_offers_prior)
            if answers >= sc.user_latency_min_answers and latencies:
                proven.user_latency_ms[key] = weighted_median(latencies)
    return proven


def epoch_shares(verdicts: Verdicts | None, cat: catalog.Catalog) -> dict[str, float]:
    """The active models' emission shares: as the auditor publishes them (bounded per epoch, from paid demand once
    the gateways tell it apart), or, from an auditor that does not or for other models, from its demand."""
    published = published_shares(verdicts) if verdicts is not None else None
    if published is not None and set(published) == set(cat.anchor_shares()):
        return published
    return model_shares(verdicts.demand if verdicts is not None else {}, cat)


def auditor_moved(st: bt.Subtensor, netuid: int, auditor_hotkey: str) -> str | None:
    """The subnet owner's hotkey, if it is no longer ``auditor_hotkey`` (the auditor this validator is or follows)."""
    owner = st.get_subnet_owner_hotkey(netuid)
    return owner if owner and owner != auditor_hotkey else None


def refused_by_miners(neurons: list[Neuron], hotkey: str, auditor: str, cat: catalog.Catalog) -> str | None:
    """Why miners refuse this validator's requests, by the rule they apply: no validator permit, or less stake (the
    metagraph's S) than the catalog's ``validator_min_stake``; the auditor is always answered. None if they answer
    it."""
    if hotkey == auditor:
        return None
    own = next((n for n in neurons if n.hotkey == hotkey), None)
    if own is None or not own.permit:
        return "this hotkey holds no validator permit"
    floor = cat.scoring.validator_min_stake
    if own.stake < floor:
        return (f"miners refuse validators below validator_min_stake {floor:,.0f}; this hotkey's stake (the "
                f"metagraph's S) is {own.stake:,.0f}")
    return None


def missing_references(cfg: ValidatorConfig, cat: catalog.Catalog) -> list[str]:
    return [m.model_id for m in cat.models if m.status == "active" and m.model_id not in cfg.references]


async def references_down(references: dict[str, ReferenceEngine]) -> list[str]:
    """The models whose reference engine does not answer HTTP yet (any reply counts: the engine is up)."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        async def down(model: str, reference: ReferenceEngine) -> str | None:
            try:
                await client.get(reference.url.rstrip("/") + "/v1/models")
            except httpx.HTTPError:
                return model
            return None

        return sorted(m for m in await asyncio.gather(*(down(m, r) for m, r in references.items())) if m)


def wait_for_references(references: dict[str, ReferenceEngine], timeout_s: float = REFERENCE_WAIT_S,
                        poll_s: float = 5.0) -> None:
    """Wait until every reference engine is up, at most ``timeout_s``: engines restarted with the validator load
    their models meanwhile, and an epoch started before would compute no canary."""
    deadline, logged = time.monotonic() + timeout_s, None
    while down := asyncio.run(references_down(references)):
        if time.monotonic() >= deadline:
            log.warning("reference engines still down after %.0f s: %s; starting without them", timeout_s,
                        ", ".join(down))
            return
        if down != logged:
            log.info("waiting for the reference engines of %s", ", ".join(down))
            logged = down
        time.sleep(poll_s)


async def run_epoch(cfg: ValidatorConfig, st: bt.Subtensor, wallet: bt.Wallet, secret: bytes, state: ScoreState,
                    epoch: int, blocks_left: int, throughput: Throughput, current: CatalogFollower,
                    board: StatusBoard) -> None:
    cat = current.for_epoch(st, epoch * current.catalog.epoch_blocks)
    sc, policy = cat.scoring, cat.audit_policy()
    neurons, hotkey_of_uid = chain.snapshot(st, cfg.netuid)
    if why := refused_by_miners(neurons, wallet.hotkey.ss58_address, throughput.auditor_hotkey, cat):
        log.critical("epoch %d: %s. Every miner refuses this validator, so it sends no canaries and sets no weights "
                     "this epoch", epoch, why)
        return
    transition = transitioning(state.pins, cat, epoch)
    if transition:
        log.warning("  %s: the pinned weights or engine changed in place; answers are not checked for %d epochs after",
                    ", ".join(sorted(transition)), sc.revision_transition_epochs)
    references = {m: r.engine() for m, r in cfg.references.items()
                  if cat.get(m) is not None and cat.get(m).status != "retired" and m not in transition}
    for m, engine in list(references.items()):
        if why := await wrong_revision(engine, cat.get(m)):
            log.error("  the %s reference engine is not used: %s; its endpoints are not audited", m, why)
            await references.pop(m).close()
    tolerances = {m.model_id: m.tolerance for m in cat.models}
    # No pool limit: a miner's hanging connections must not delay requests to the others.
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=256)
    async with httpx.AsyncClient(limits=limits) as client:
        miners = await discover(neurons, cfg.netuid, cfg.mechid, client, {cat.digest(), *current.accepted()},
                                throughput.manifests)
        throughput.update_slots(miners, neurons, epoch)
        endpoints = [e for m in miners for e in m.endpoints() if e.endpoint.tier == "open"
                     and cat.get(e.endpoint.model_id) is not None and cat.get(e.endpoint.model_id).status != "retired"]
        log.info("epoch %d: %d miners, %d open endpoints", epoch, len(miners), len(endpoints))
        results = await audit_epoch(endpoints, epoch=epoch, secret=secret, keypair=wallet.hotkey,
                                    references=references, tolerances=tolerances, client=client, netuid=cfg.netuid,
                                    mechid=cfg.mechid, pool_size=sc.canary_pool_size,
                                    per_endpoint=sc.canaries_per_endpoint,
                                    budget_s=blocks_left * BLOCK_TIME_S * sc.canary_spread_fraction,
                                    holding=throughput.holding, request_timeout_s=policy.request_timeout_s,
                                    identities=throughput.identities)
        verdicts = None if policy.mode == "observe" else await throughput.verdicts(client, epoch)
        evidence = []
        if verdicts is not None:
            fresh = new_evidence(state, [f"{e.test_id}/{e.hotkey}/{e.endpoint_id}" for e in verdicts.invalid],
                                 epoch, policy.credit_ttl_epochs + 1)
            evidence = await throughput.proven_invalid(
                [e for e in verdicts.invalid if f"{e.test_id}/{e.hotkey}/{e.endpoint_id}" in fresh], references, cat,
                budget_s=EVIDENCE_SHARE * blocks_left * BLOCK_TIME_S)
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
    shares = epoch_shares(verdicts, cat)
    log.info("  emission shares: %s", ", ".join(f"{m} {s:.1%}" for m, s in sorted(shares.items())))
    factors: dict[str, float] = {}
    scores, failed = epoch_scores(results, shares=shares, credited=credited, latency_ms=proven.latency_ms, state=state,
                                  failed_hotkeys=invalid_hotkeys, user_latency_ms=proven.user_latency_ms,
                                  user_availability=proven.user_availability, max_latency_s=policy.max_latency_s,
                                  factors=factors)
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
    # What the board publishes is exactly what is set: light validators set the same.
    uid_weights = weights_to_set(weights(state, hotkey_of_uid), hotkey_of_uid, throughput.auditor_hotkey)
    board.publish(status.build(
        epoch=epoch, epoch_blocks=cat.epoch_blocks, endpoints=endpoints, results=results, scores=scores, failed=failed,
        invalid_hotkeys=invalid_hotkeys, credited=credited,
        scored_models={m.model_id for m in cat.models if m.status == "active"}, state=state, model_shares=shares,
        weights=uid_weights, drifted=proven.drifted, user_availability=proven.user_availability,
        latency_factors=factors), epoch=epoch, epoch_blocks=cat.epoch_blocks, model_shares=shares, weights=uid_weights)
    # The epoch is scored; a failure from here on must not run it again.
    try:
        set_weights(cfg, st, wallet, uid_weights)
    except Exception:
        log.exception("setting weights failed this epoch")


def weights_to_set(w: dict[int, float], hotkey_of_uid: dict[int, str], owner_hotkey: str) -> dict[int, float]:
    """``w``, or, if no miner has a positive score, all weight on the subnet owner's UID (none if it holds no UID)."""
    if w:
        return w
    owner = next((uid for uid, hotkey in hotkey_of_uid.items() if hotkey == owner_hotkey), None)
    if owner is None:
        log.warning("no miner has a positive score and the subnet owner hotkey holds no UID; no weights set")
        return {}
    return {owner: 1.0}


def set_weights(cfg: ValidatorConfig, st: bt.Subtensor, wallet: bt.Wallet, w: dict[int, float]) -> bool:
    """Set ``w`` on the configured mechanism, retried once per block up to ``WEIGHT_ATTEMPTS`` times; in a dry run, or
    before the mechanism exists on chain, only logged. Whether that is done (with nothing to set, it is)."""
    if not w:
        return True
    if cfg.dry_run or not chain.mechanism_ready(st, cfg.netuid, cfg.mechid):
        log.info("dry run (mechanism %d %s): weights %s", cfg.mechid,
                 "not enabled on chain" if not cfg.dry_run else "configured dry", w)
        return True
    for attempt in range(1, WEIGHT_ATTEMPTS + 1):
        if chain.submit_weights(st, wallet, cfg.netuid, cfg.mechid, w):
            return True
        log.warning("setting weights failed (attempt %d of %d); retrying in one block", attempt, WEIGHT_ATTEMPTS)
        time.sleep(BLOCK_TIME_S)
    return False


def main() -> None:
    if sys.argv[1:2] in (["init"], ["uninstall"]):
        from verathos.install import common

        common.main("validator", sys.argv[1], sys.argv[2:])
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(epilog="Set up this machine as a validator: verathos-validator init. Remove it: "
                                            "verathos-validator uninstall.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--once", action="store_true", help="Run one epoch immediately and exit")
    args = parser.parse_args()

    cfg = ValidatorConfig.load(args.config)
    state_dir = cfg.state_dir.expanduser()
    state_dir.mkdir(parents=True, exist_ok=True)
    wallet = bt.Wallet(name=cfg.wallet_name, hotkey=cfg.hotkey_name)
    st = bt.Subtensor(network=cfg.network)
    current = CatalogFollower(st, cfg.network, cfg.netuid, catalog.CatalogStore(cfg.catalog_dir), cfg.catalog_mirrors,
                              state_path=cfg.state_dir.expanduser() / "catalog_followed.json")
    if cfg.mode == "light":  # no references, no miners: the source's verified weights
        current.start()
        light.run(cfg, st, wallet, current, set_weights=lambda w: set_weights(cfg, st, wallet, w), once=args.once)
        st.close()
        return
    secret = load_secret(state_dir)
    state_path = state_dir / "scores.json"
    state = ScoreState.load(state_path)
    if missing := missing_references(cfg, current.catalog):
        raise SystemExit(f"no reference engine configured for the active models {missing}")
    throughput = Throughput.start(network=cfg.network, netuid=cfg.netuid, mechid=cfg.mechid,
                                  current=lambda: current.catalog, st=st, wallet=wallet, references=cfg.references,
                                  screens=cfg.audit_screens, state_dir=state_dir)
    board = StatusBoard(state_dir / "status.json")
    if why := refused_by_miners(chain.snapshot(st, cfg.netuid)[0], wallet.hotkey.ss58_address,
                                throughput.auditor_hotkey, current.catalog):
        log.critical("%s. Every miner refuses this validator's requests, and it sets no weights until that changes",
                     why)
    is_owner = wallet.hotkey.ss58_address == st.get_subnet_owner_hotkey(cfg.netuid)
    if is_owner or throughput.auditor is not None or cfg.public_host:
        if not cfg.public_host:
            raise SystemExit("this validator serves the catalog or the throughput verdicts: set public_host")
        if is_owner:
            current.store.put(current.catalog)
        start_service(st=st, wallet=wallet, netuid=cfg.netuid, public_host=cfg.public_host, port=cfg.port,
                      store=current.store if is_owner else None, auditor=throughput.auditor, board=board,
                      probation=lambda: [key for key, left in list(state.probation.items()) if left > 0],
                      gateways=lambda: current.catalog.gateways)
    pinned = {m.model_id: (m.hf_revision, m.engine.revision) for m in current.catalog.models}

    def adopt(cat: catalog.Catalog) -> None:
        if missing := missing_references(cfg, cat):
            log.critical("catalog %s activates %s, which this validator has no reference engine for; stopping",
                         cat.digest()[:12], missing)
            os._exit(1)
        for m in cat.models:
            if m.model_id in cfg.references and pinned.get(m.model_id, (m.hf_revision, m.engine.revision)) != (
                    m.hf_revision, m.engine.revision):
                log.critical("catalog %s pins %s at weights %s and engine %s: restart its reference engine on them "
                             "within %d epochs", cat.digest()[:12], m.model_id, m.hf_revision, m.engine.revision,
                             cat.scoring.revision_transition_epochs)
            pinned[m.model_id] = (m.hf_revision, m.engine.revision)
        throughput.set_policy(cat.audit_policy())

    current.on_change.append(adopt)
    current.start()
    throughput.start_slot_refresh(cfg.network, current.accepted)
    wait_for_references(cfg.references)

    done = state.last_epoch
    while True:
        try:
            block = st.get_current_block()
            epoch_blocks = current.catalog.epoch_blocks
            epoch = block // epoch_blocks
            next_start = (epoch + 1) * epoch_blocks
            if epoch != done:
                if moved := auditor_moved(st, cfg.netuid, throughput.auditor_hotkey):
                    # the roles are set at the start: the supervisor's restart takes up the new auditor
                    log.critical("the subnet owner hotkey is now %s, the auditor; restarting", moved)
                    os._exit(1)
                blocks_left = max(next_start - block, 1)
                trial = copy.deepcopy(state)  # an epoch that fails part way leaves no trace for its retry
                asyncio.run(run_epoch(cfg, st, wallet, secret, trial, epoch, 1 if args.once else blocks_left,
                                      throughput, current, board))
                state = trial
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
