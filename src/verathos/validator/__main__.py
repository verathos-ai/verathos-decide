"""Run a validator: ``verathos-validator --config validator.json``.

Per epoch (``epoch_blocks`` from the catalog, independent of the subnet tempo): snapshot chain, discover miners, audit
every endpoint with the epoch's canaries spread over the first part of the epoch, score, publish the status with the
weights, and set those weights on the catalog's mechanism, or the configured ``mechid`` (dry run until that mechanism
exists).

Every full validator takes part in the throughput audit (see ``validator.throughput``): the auditor runs the tests,
the others apply its signed verdicts; endpoints score in proportion to the capacity they proved. With the catalog's
usage pool, the hotkeys that served paying users get weight on top, from the period record the verdicts state
(``validator.usage``); with its referral pool, the referrers do (``validator.referral``). A light validator (``mode:
light``, ``validator.light``) does none of this: it sets the weights its source validator publishes.

The subnet owner's validator, the auditor, does not start while another auditor with its hotkey answers at the hotkey's
axon, and stops at an epoch's start once another one moved the axon (``validator.singleton``).
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
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

import bittensor as bt
import httpx

from verathos import __version__, catalog, release
from verathos import update as self_update
from verathos.engine import code_version
from verathos.protocol import tls
from verathos.protocol.callers import ValidatorIdentity
from verathos.protocol.manifest import axon_base
from verathos.throughput.feed import Credit, Verdicts, published_shares
from verathos.throughput.policy import BLOCK_TIME_S
from verathos.throughput.referral import contradiction as referral_contradiction
from verathos.throughput.usage import contradiction
from verathos.validator import chain, light, referral, singleton, status, usage
from verathos.validator import egress as egress_module
from verathos.validator.audit import EndpointResult, audit_epoch
from verathos.validator.config import ReferenceEngine, ValidatorConfig
from verathos.validator.discovery import DiscoveredEndpoint, Neuron, discover
from verathos.validator.fallback import Taken, source_results, take
from verathos.validator.follow import CatalogFollower
from verathos.validator.light import Refused
from verathos.validator.references import served_code, transitioning, wrong_revision
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
from verathos.validator.sequential import Runner, Sequence, coverage_order, overdue, waited_after
from verathos.validator.service import start_service
from verathos.validator.status import StatusBoard
from verathos.validator.throughput import Throughput

log = logging.getLogger("verathos.validator")
WEIGHT_ATTEMPTS = 10
_SAID_UNREVEALED: set[tuple[int, int, int]] = set()  # (netuid, mechanism, epoch counter) said at warning level
REFERENCE_WAIT_S = 900.0  # the most the first epoch waits for reference engines still loading their models
EVIDENCE_SHARE = 0.1  # of the epoch, the most rechecking the auditor's evidence may take
OWNER_RESIDENT = ("this is the subnet owner's hotkey: its validator is the auditor, which checks test answers at any "
                  "time and every model each epoch; its references stay resident: set reference_mode to resident "
                  "(scripts/install.sh validator --full --references resident)")
SOURCE_MARGIN_S = 120.0  # references one at a time: the source's results are awaited until this long before the end
REMOTE_CHECK_S = 10.0  # an engine on another machine answers its check within it
_REMOTE_DOWN: list[str] = []  # the engines on other machines not serving at the last epoch's check


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


def manifest_catalogs(cfg: ValidatorConfig, cat: catalog.Catalog, current: CatalogFollower) -> dict[str, int]:
    """The catalogs a manifest may be built for now (``cat``, and those ``current`` accepts), each with the mechanism a
    manifest built for it is for: the catalog's, or the configured ``mechid``."""
    accepted = {**current.accepted(), cat.digest(): cat.mechanism}
    return {d: m if cfg.mechid is None else cfg.mechid for d, m in accepted.items()}


def missing_references(cfg: ValidatorConfig, cat: catalog.Catalog) -> list[str]:
    return [m.model_id for m in cat.models if m.status == "active" and m.model_id not in cfg.references]


async def references_down(references: dict[str, ReferenceEngine]) -> list[str]:
    """The models whose reference engine does not answer HTTP yet (any reply counts: the engine is up)."""
    async def down(model: str, reference: ReferenceEngine) -> str | None:
        engine = reference.engine(timeout_s=5.0)
        try:
            return None if await engine.reachable() else model
        finally:
            await engine.close()

    return sorted(m for m in await asyncio.gather(*(down(m, r) for m, r in references.items())) if m)


async def remote_engines(engines: Mapping[str, ReferenceEngine]) -> tuple[list[str], list[str]]:
    """The engines on other machines among ``engines`` (name -> engine): those that do not serve now, each with why,
    and those that run other engine code than this validator's release (``verathos.engine.code_version``), each with
    the code it runs."""
    own = code_version()

    async def check(name: str, spec: ReferenceEngine) -> tuple[str | None, str | None]:
        engine = spec.engine(timeout_s=REMOTE_CHECK_S)
        try:
            why = await engine.problem()
            codes = served_code(await engine.info()) if why is None else set()
        finally:
            await engine.close()
        if why is not None:
            return f"{name} at {spec.url}: {why}", None
        return None, (f"{name} at {spec.url} runs {', '.join(sorted(codes))}" if codes and own not in codes else None)

    remote = [(name, spec) for name, spec in sorted(engines.items()) if spec.remote()]
    found = await asyncio.gather(*(check(n, s) for n, s in remote))
    return [down for down, _ in found if down], [other for _, other in found if other]


async def check_remote_engines(cfg: ValidatorConfig, epoch: int, auditor: bool) -> None:
    """One line per epoch while an engine on another machine (a reference, or the auditor's screen) does not serve:
    what it checks stays unchecked, not penalized, as when a local reference is down; and one while such an engine runs
    other engine code than this validator's release, which its machine's installer updates."""
    engines = {f"the {m} reference": r for m, r in cfg.references.items()}
    if auditor:
        engines |= {f"the {m} screen": s for m, s in cfg.audit_screens.items()}
    down, other = await remote_engines(engines)
    if down:
        log.warning("epoch %d: engines on other machines not serving: %s; what they check stays unchecked, not "
                    "penalized, until they serve again (docs/VALIDATING.md, When the GPU machine is gone)", epoch,
                    "; ".join(down))
    elif _REMOTE_DOWN:
        log.info("epoch %d: the engines on other machines serve again", epoch)
    _REMOTE_DOWN[:] = down
    if other:
        log.warning("epoch %d: engines on other machines run other engine code than this validator's release (%s): "
                    "%s; update their machine to this release (install.sh reference --yes there)", epoch,
                    code_version(), "; ".join(other))


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


def score_epoch(state: ScoreState, endpoints: list[DiscoveredEndpoint], results: dict[str, EndpointResult], *,
                taken: Taken | None, shares: dict[str, float], proven: Proven, invalid_hotkeys: set[str],
                sc: catalog.Scoring, max_latency_s: float) -> tuple[dict[str, float], set[str], set[str],
                                                                    dict[str, float]]:
    """Score the epoch and fold it into ``state``: this validator's ``results``, and for the models it took from its
    source the source's (``taken``, ``validator.fallback``), which enter the averages, probation and weights as its
    own do. Endpoints without either keep their averages. -> (scores, failed, held, latency factors)"""
    record_window(state, results, sc.statistical_window_epochs)
    factors: dict[str, float] = {}
    scores, failed = epoch_scores(results, shares=shares, credited=proven.credited, latency_ms=proven.latency_ms,
                                  state=state, failed_hotkeys=invalid_hotkeys, user_latency_ms=proven.user_latency_ms,
                                  user_availability=proven.user_availability, max_latency_s=max_latency_s,
                                  factors=factors)
    taken = taken or Taken()
    scored = {key: taken.uids[key] for key in taken.scores if key not in results}  # its own check comes first
    scores |= {key: taken.scores[key] for key in scored}
    failed |= taken.failed & set(scored)
    held = ({e.key for e in endpoints} | taken.held) - set(results) - set(scored)
    update(state, results, scores, failed, alpha=sc.ema_alpha, probation_epochs=sc.probation_epochs, held=held,
           reset_hotkeys=invalid_hotkeys, taken=scored)
    return scores, failed, held, factors


def coverage_line(epoch: int, served: list[str], sequence: Sequence, why: Mapping[str, str], source: str | None,
                  refused: str | None) -> tuple[int, str]:
    """The epoch's line on references one at a time: the models checked, and those taken from the source (or, without
    its results, kept) and why; a warning when a model was not checked."""
    checked = [m for m in served if m in sequence.checked]
    others = [f"{m} ({sequence.why.get(m) or why.get(m) or 'not drawn'})" for m in served if m not in checked]
    line = f"epoch {epoch}: references one at a time: checked {', '.join(checked) or 'none'}"
    if not others:
        return logging.INFO, line
    if source is not None:
        return logging.WARNING, line + f"; taken from {source}: {', '.join(others)}"
    return logging.WARNING, line + (f"; not checked, their averages kept (no results of the source: {refused}): "
                                    f"{', '.join(others)}")


async def run_epoch(cfg: ValidatorConfig, st: bt.Subtensor, wallet: bt.Wallet, secret: bytes, state: ScoreState,
                    epoch: int, blocks_left: int, throughput: Throughput, current: CatalogFollower,
                    board: StatusBoard, runner: Runner | None = None,
                    egress: egress_module.Egress | None = None) -> None:
    """Score one epoch. With a ``runner``, the references run one at a time and the models not checked take the
    source's results. While this validator's published addresses just changed (``egress``), the endpoints its requests
    did not reach are held, not scored."""
    started, began = time.monotonic(), time.time()
    cat = current.for_epoch(st, epoch * current.catalog.epoch_blocks)
    sc, policy, mechanism = cat.scoring, cat.audit_policy(), catalog.mechanism_of(cat, cfg.mechid)
    neurons, hotkey_of_uid = chain.snapshot(st, cfg.netuid)
    if why := refused_by_miners(neurons, wallet.hotkey.ss58_address, throughput.auditor_hotkey, cat):
        log.critical("epoch %d: %s. Every miner refuses this validator, so it sends no canaries and sets no weights "
                     "this epoch", epoch, why)
        return
    await check_remote_engines(cfg, epoch, throughput.auditor is not None)
    transition = transitioning(state.pins, cat, epoch)
    if transition:
        log.warning("  %s: the pinned weights or engine changed in place; answers are not checked for %d epochs after",
                    ", ".join(sorted(transition)), sc.revision_transition_epochs)
    usable = [m for m in cfg.references
              if cat.get(m) is not None and cat.get(m).status != "retired" and m not in transition]
    if runner is None:
        references: dict[str, Any] = {m: cfg.references[m].engine() for m in usable}
        for m, engine in list(references.items()):
            if why := await wrong_revision(engine, cat.get(m)):
                log.error("  the %s reference engine is not used: %s; its endpoints are not audited", m, why)
                await references.pop(m).close()
    else:  # each started when its model is checked, and checked against the pins then
        references = runner.references(cfg.references, cat, usable)
    tolerances = {m.model_id: m.tolerance for m in cat.models}
    sequence, evidence, verdicts = None, [], None
    try:
        # No pool limit: a miner's hanging connections must not delay requests to the others.
        limits = httpx.Limits(max_connections=None, max_keepalive_connections=256)
        async with tls.client(limits=limits) as client:
            miners = await discover(neurons, cfg.netuid, client, manifest_catalogs(cfg, cat, current),
                                    throughput.manifests)
            throughput.update_slots(miners, neurons, epoch)
            endpoints = [e for m in miners for e in m.endpoints() if e.endpoint.tier == "open"
                         and cat.get(e.endpoint.model_id) is not None
                         and cat.get(e.endpoint.model_id).status != "retired"]
            log.info("epoch %d: %d miners, %d open endpoints", epoch, len(miners), len(endpoints))
            if runner is not None:
                checkable = sorted({e.endpoint.model_id for e in endpoints} & set(references))
                sequence = Sequence(references, coverage_order(checkable, state.waited, secret, epoch),
                                    overdue=overdue(checkable, state.waited), per_endpoint=sc.canaries_per_endpoint,
                                    timings=runner.timings)
            results = await audit_epoch(endpoints, epoch=epoch, secret=secret, keypair=wallet.hotkey,
                                        references=references, tolerances=tolerances, client=client,
                                        netuid=cfg.netuid, pool_size=sc.canary_pool_size,
                                        per_endpoint=sc.canaries_per_endpoint,
                                        budget_s=blocks_left * BLOCK_TIME_S * sc.canary_spread_fraction,
                                        holding=throughput.holding, request_timeout_s=policy.request_timeout_s,
                                        identities=throughput.identities, one_at_a_time=sequence,
                                        images={m.model_id: m.images for m in cat.models if m.reads_images()})
            verdicts = None if policy.mode == "observe" else await throughput.verdicts(client, epoch)
            if verdicts is not None:
                fresh = new_evidence(state, [f"{e.test_id}/{e.hotkey}/{e.endpoint_id}" for e in verdicts.invalid],
                                     epoch, policy.credit_ttl_epochs + 1)
                pending = [e for e in verdicts.invalid if f"{e.test_id}/{e.hotkey}/{e.endpoint_id}" in fresh]
                budget_s = EVIDENCE_SHARE * blocks_left * BLOCK_TIME_S
                if runner is not None:  # a reference started for a recheck must fit in the time
                    pending.sort(key=lambda e: e.model_id)  # one start per model
                    for reference in references.values():
                        reference.deadline = time.monotonic() + budget_s
                evidence = await throughput.proven_invalid(pending, references, cat, budget_s=budget_s)
    finally:
        for e in references.values():
            await e.close()
    if policy.mode != "observe" and verdicts is None:
        log.warning("  no throughput verdicts: no endpoint has proven capacity this epoch")
    for e in evidence:
        where = "gateway traffic" if e.from_gateway else f"throughput test {e.test_id[:12]}"
        log.info("  %s/%s gave a wrong answer in %s: %s", e.hotkey, e.endpoint_id, where, e.reason)
    invalid_hotkeys = {e.hotkey for e in evidence}
    taken = None
    if sequence is not None:
        served = sorted({e.endpoint.model_id for e in endpoints})
        source = refused = None
        if any(m not in sequence.checked for m in served):  # the source's results for the others
            unchecked = {m.model_id for m in cat.models if m.status != "retired"} - set(sequence.checked)
            try:
                source_hotkey, network = await source_results(
                    st, netuid=cfg.netuid, hotkey=wallet.hotkey.ss58_address, source_hotkey=cfg.source_hotkey,
                    epoch=epoch, epoch_blocks=cat.epoch_blocks, cat=cat,
                    deadline=started + blocks_left * BLOCK_TIME_S - SOURCE_MARGIN_S)
            except Refused as exc:
                refused = str(exc)
            else:
                taken = take(network, unchecked)
                source = (f"the validator {source_hotkey}" if cfg.source_hotkey
                          else f"the subnet owner's validator {source_hotkey}")
        reasons = {m: "its pinned weights changed in place" if m in transition else "no reference"
                   for m in served if m not in references}
        log.log(*coverage_line(epoch, served, sequence, reasons, source, refused))
        state.waited = waited_after(state.waited, sequence.order, sequence.tried, {m.model_id for m in cat.models})
    proven = credited_capacity(endpoints, verdicts, cat, policy.mode)
    credited = proven.credited
    shares = epoch_shares(verdicts, cat)
    log.info("  emission shares: %s", ", ".join(f"{m} {s:.1%}" for m, s in sorted(shares.items())))
    miner_share = sc.miner_share_at(epoch)
    log.info("  miner share %.6f: the miners' weights sum to it, the subnet owner's UID gets the rest", miner_share)
    record = usage.take(state, verdicts, cat, epoch, check=lambda r: None if throughput.auditor is not None else
                        contradiction(r, st, cfg.netuid))  # the auditor's own record is not checked again
    usage_by_uid = usage.usage_weights(record, cat, epoch, hotkey_of_uid)
    usage_share = round(sum(usage_by_uid.values()), catalog.SHARE_DIGITS) if sc.usage_pool is not None else None
    if sc.usage_pool is not None and record is None:
        log.info("  usage pool: no record of period %d: nothing paid", sc.usage_pool.period(epoch) - 1)
    elif record is not None:
        log.info("  usage pool: period %d sized %.6f (%s), paid %.6f to %d UIDs", record.period, record.share,
                 record.mode, usage_share, len(usage_by_uid))
    if egress is not None and egress.in_grace(began):
        unreached = egress_module.excused(results)
        for key in unreached:
            del results[key]
        if unreached:
            log.warning("  this validator's published addresses changed within %d minutes: the %d endpoints its "
                        "requests did not reach are held, not scored", egress_module.GRACE_S // 60, len(unreached))
    scores, failed, held, factors = score_epoch(state, endpoints, results, taken=taken, shares=shares, proven=proven,
                                                invalid_hotkeys=invalid_hotkeys, sc=sc,
                                                max_latency_s=policy.max_latency_s)
    if held:
        log.info("  %d endpoints not audited this epoch (no usable reference answers); scores held", len(held))
    for key, r in sorted(results.items()):
        log.info("  %s epoch_verdict=%s answered=%d/%d identity=%s mean_abs=%.4f proven_rps=%.1f users=%.2f "
                 "score=%.3f%s",
                 key, r.tally.verdict().value, r.answered, r.sent, r.identity_ok, r.tally.mean_abs,
                 credited.get(key, 0.0), proven.user_availability.get(key, 1.0), scores.get(key, 0.0),
                 f" errors={sorted(set(r.errors))}" if r.errors else "")
    capacity = weights(state, hotkey_of_uid)
    paid = referral.take(state, verdicts, cat, epoch, check=lambda r, before: None if throughput.auditor is not None
                         else referral_contradiction(r, st, cfg.netuid, before))  # the auditor's is not checked again
    referral_by_uid, referrers = referral.referral_weights(paid, cat, epoch, hotkey_of_uid, capacity)
    referral_share = (round(sum(referral_by_uid.values()), catalog.SHARE_DIGITS) if sc.referral_pool is not None
                      else None)
    if sc.referral_pool is not None and paid is None:
        log.info("  referrals: no record of period %d: nothing paid", sc.referral_pool.period(epoch) - 1)
    elif paid is not None:
        unpaid: dict[str, int] = {}
        for r in referrers.values():
            if r.unpaid:
                unpaid[r.unpaid] = unpaid.get(r.unpaid, 0) + 1
        log.info("  referrals: period %d sized %.6f, %d referrers referred %.2f USD over %d periods; paid %.6f to %d "
                 "UIDs%s", paid.period, paid.share, len(referrers), sum(r.referred_micros for r in referrers.values())
                 / 1e6, paid.window_periods, referral_share, len(referral_by_uid),
                 "; not paid: " + ", ".join(f"{n} {why}" for why, n in sorted(unpaid.items())) if unpaid else "")
    # What the board publishes is exactly what is set: light validators set the same.
    uid_weights, parts = weight_parts(capacity, hotkey_of_uid, throughput.auditor_hotkey, miner_share, usage_by_uid,
                                      referral_by_uid)
    board.publish(status.build(
        epoch=epoch, epoch_blocks=cat.epoch_blocks, endpoints=endpoints, results=results, scores=scores, failed=failed,
        invalid_hotkeys=invalid_hotkeys, credited=credited,
        scored_models={m.model_id for m in cat.models if m.status == "active"}, state=state, model_shares=shares,
        weights=uid_weights, drifted=proven.drifted, user_availability=proven.user_availability,
        latency_factors=factors, taken=taken.status if taken else None, miner_share=miner_share,
        usage_share=usage_share, parts=parts, referral_share=referral_share, referrers=referrers,
        hotkey_of_uid=hotkey_of_uid), epoch=epoch, epoch_blocks=cat.epoch_blocks,
        model_shares=shares, weights=uid_weights, miner_share=miner_share, mechanism=mechanism, usage_share=usage_share,
        referral_share=referral_share, referral=None if paid is None else referral.referral_period(paid, cat, epoch))
    # The epoch is scored; a failure from here on must not run it again.
    try:
        set_weights(cfg, st, wallet, uid_weights, mechanism)
    except Exception:
        log.exception("setting weights failed this epoch")


def weights_to_set(w: dict[int, float], hotkey_of_uid: dict[int, str], owner_hotkey: str,
                   miner_share: float = 1.0, usage_by_uid: Mapping[int, float] | None = None,
                   referral_by_uid: Mapping[int, float] | None = None) -> dict[int, float]:
    """The weights to set (``weight_parts``)."""
    return weight_parts(w, hotkey_of_uid, owner_hotkey, miner_share, usage_by_uid, referral_by_uid)[0]


def weight_parts(w: dict[int, float], hotkey_of_uid: dict[int, str], owner_hotkey: str, miner_share: float = 1.0,
                 usage_by_uid: Mapping[int, float] | None = None, referral_by_uid: Mapping[int, float] | None = None
                 ) -> tuple[dict[int, float], dict[str, dict[int, float]]]:
    """The miners' weights ``w`` (summing to 1) scaled to sum to ``miner_share``, the usage and referral pools'
    weights added (they sum to at most ``1 - miner_share``, ``validator.usage``, ``validator.referral``), and the
    rest on the subnet owner's UID, whose incentive the chain does not pay out; all on the owner's UID if no miner has
    a positive score, usage or referral weight. If the owner hotkey holds no UID there is nothing to put the rest on:
    the miners' weights take what the pools leave, and with none, no weights are set, as the chain would scale the
    pools' up (they pay at most what they size). -> (the weights, and the parts of them per UID: ``serving`` from the
    scores, ``usage`` and ``referral`` from the pools)."""
    pools = {name: {uid: value for uid, value in (by_uid or {}).items() if value > 0}
             for name, by_uid in (("usage", usage_by_uid), ("referral", referral_by_uid))}
    extra: dict[int, float] = {}
    for by_uid in pools.values():
        for uid, value in by_uid.items():
            extra[uid] = extra.get(uid, 0.0) + value
    owner = next((uid for uid, hotkey in hotkey_of_uid.items() if hotkey == owner_hotkey), None)
    if owner is None:
        if not w and not extra:
            log.warning("no miner has a positive score and the subnet owner hotkey holds no UID; no weights set")
        elif miner_share < 1:
            log.warning("the subnet owner hotkey holds no UID: the miners get all of the weights, not miner_share "
                        "%.6f", miner_share)
        if not extra:
            return w, {"serving": dict(w), **pools}
        if not w:
            log.warning("no miner has a positive score and the subnet owner hotkey holds no UID: no weights set, as "
                        "the pools' alone would be scaled up")
            return {}, {"serving": {}, "usage": {}, "referral": {}}
        rest = 1.0 - sum(extra.values())
        serving = {uid: value * rest for uid, value in w.items()}
        out = dict(serving)
        for uid, value in extra.items():
            out[uid] = out.get(uid, 0.0) + value
        return out, {"serving": serving, **pools}
    if not w and not extra:
        return {owner: 1.0}, {"serving": {}, **pools}
    if miner_share >= 1 and not extra:
        return w, {"serving": dict(w), **pools}
    serving = {uid: value * miner_share for uid, value in w.items()}
    out = dict(serving)
    for uid, value in extra.items():
        out[uid] = out.get(uid, 0.0) + value
    rest = 1.0 - (miner_share if w else 0.0) - sum(extra.values())
    if rest > 1e-9:
        out[owner] = out.get(owner, 0.0) + rest
    return out, {"serving": serving, **pools}


def set_weights(cfg: ValidatorConfig, st: bt.Subtensor, wallet: bt.Wallet, w: dict[int, float], mechid: int) -> bool:
    """Set ``w`` on mechanism ``mechid`` once the chain's weights rate limit allows, retried once per block up to
    ``WEIGHT_ATTEMPTS`` times; in a dry run, or before the mechanism exists on chain, only logged. Whether that is done
    (with nothing to set, it is)."""
    if not w:
        return True
    if cfg.dry_run or not chain.mechanism_ready(st, cfg.netuid, mechid):
        log.info("dry run (mechanism %d %s): weights %s", mechid,
                 "not enabled on chain" if not cfg.dry_run else "configured dry", w)
        return True
    uid = st.get_uid_for_hotkey_on_subnet(wallet.hotkey.ss58_address, cfg.netuid)
    if uid is None:
        log.warning("weights not set: this hotkey holds no UID on netuid %d", cfg.netuid)
        return False
    for attempt in range(1, WEIGHT_ATTEMPTS + 1):
        if wait := chain.weights_wait(st, cfg.netuid, mechid, uid):
            log.info("the chain allows this UID's next weights on mechanism %d in %d blocks; setting them then",
                     mechid, wait)
            time.sleep(wait * BLOCK_TIME_S)
        try:
            if chain.submit_weights(st, wallet, cfg.netuid, mechid, w):
                return True
        except chain.ReleaseTooOld as exc:
            log.error("weights not set: %s", exc)
            return False
        except chain.CommitsUnrevealed as exc:  # said once per epoch of the subnet: a light validator asks often
            said = (cfg.netuid, mechid, exc.epoch)
            log.log(logging.DEBUG if said in _SAID_UNREVEALED else logging.WARNING, "weights not set: %s", exc)
            _SAID_UNREVEALED.add(said)
            return False
        log.warning("setting weights failed (attempt %d of %d); retrying in one block", attempt, WEIGHT_ATTEMPTS)
        time.sleep(BLOCK_TIME_S)
    return False


def log_auditor_pin(cat: catalog.Catalog, pin: str) -> None:
    """The auditor's check that the catalog names its certificate: miners, validators and gateways reach it by that
    pin only."""
    named = cat.auditor.tls if cat.auditor is not None else None
    if named == pin:
        log.info("catalog %s names this service's TLS pin %s: miners, validators and gateways reach it",
                 cat.digest()[:12], pin)
        return
    log.critical("catalog %s names %s as the auditor's TLS pin, but this service's certificate has pin %s: no miner, "
                 "validator or gateway reaches the auditor until the subnet owner commits a catalog naming it "
                 "(verathos-subnet commit-catalog on this machine fills it in)", cat.digest()[:12],
                 named or "no pin", pin)


def publish_own_identity(cfg: ValidatorConfig, st: bt.Subtensor, wallet: bt.Wallet, pin: str | None) -> None:
    """A full validator's chain commitment: its service's pin, if it serves, and its egress addresses
    (``protocol.callers``); nothing to publish without either."""
    identity = ValidatorIdentity(pin, tuple(cfg.egress_ips))
    if not cfg.egress_ips:
        log.warning("no egress_ips configured: miners that admit callers by address do not admit this validator's "
                    "requests (docs/VALIDATING.md)")
    if identity.tls is None and not identity.egress:
        return
    try:
        chain.publish_identity(st, wallet, cfg.netuid, identity)
    except Exception as exc:
        log.error("the validator identity was not committed on chain: %s", exc)


def watch_egress(cfg: ValidatorConfig, wallet: bt.Wallet, current: CatalogFollower, pin: str | None,
                 is_owner: bool) -> egress_module.Egress:
    """Follow where this validator's requests leave from (``validator.egress``): a full validator publishes a new
    address itself; the auditor's are the catalog's, which only the subnet owner commits, so it says what to do."""
    from verathos.install.host import detect_public_ip

    if is_owner:
        named = list(current.catalog.auditor.egress) if current.catalog.auditor is not None else []

        def publish(identity: ValidatorIdentity) -> None:
            log.critical("the auditor's requests leave from %s, which catalog %s does not name in auditor.egress: "
                         "miners whose firewall enforces drop its tests and canaries until the subnet owner commits a "
                         "catalog naming it (verathos-subnet commit-catalog on this machine fills it in)",
                         identity.egress[0], current.catalog.digest()[:12])

        watch = egress_module.Egress(named, None, publish, detect_public_ip)
    else:
        def publish(identity: ValidatorIdentity) -> None:
            st = bt.Subtensor(network=cfg.network)  # its own: the epoch loop holds the other
            try:
                chain.publish_identity(st, wallet, cfg.netuid, identity)
            finally:
                st.close()
            log.warning("this validator's requests leave from %s, which it did not publish: it now publishes %s; "
                        "miners admit it from there within their next refresh (egress_ips in validator.json keeps it "
                        "across restarts)", identity.egress[0], ", ".join(identity.egress))

        watch = egress_module.Egress(cfg.egress_ips, pin, publish, detect_public_ip)
    watch.start()
    return watch


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
    log.info("verathos %s, weights version key %d", __version__, release.weights_version_key())
    current = CatalogFollower(st, cfg.network, cfg.netuid, catalog.CatalogStore(cfg.catalog_dir), cfg.catalog_mirrors,
                              state_path=cfg.state_dir.expanduser() / "catalog_followed.json")
    if cfg.mode == "light":  # no references, no miners: the source's verified weights
        current.start()
        updater = None if args.once else self_update.start(cfg.auto_update, "validator")
        light.run(cfg, st, wallet, current, set_weights=lambda w, mechid: set_weights(cfg, st, wallet, w, mechid),
                  once=args.once, between=updater.at_safe_point if updater else None)
        st.close()
        return
    try:
        chain.weights_version_key(st, cfg.netuid)
    except chain.ReleaseTooOld as exc:
        log.critical("%s", exc)
    secret = load_secret(state_dir)
    state_path = state_dir / "scores.json"
    state = ScoreState.load(state_path)
    if missing := missing_references(cfg, current.catalog):
        raise SystemExit(f"no reference engine configured for the active models {missing}")
    if remote := sorted(f"the {m} {kind} at {e.url}" for kind, section in (("reference", cfg.references),
                                                                           ("screen", cfg.audit_screens))
                        for m, e in section.items() if e.remote()):
        log.info("engines on other machines: %s", ", ".join(remote))
    runner = None
    is_owner = wallet.hotkey.ss58_address == st.get_subnet_owner_hotkey(cfg.netuid)
    if cfg.reference_mode == "sequential":
        if is_owner:
            raise SystemExit(OWNER_RESIDENT)
        if cfg.source_hotkey == wallet.hotkey.ss58_address:
            raise SystemExit("source_hotkey is this validator's own hotkey: name the validator whose results the "
                             "models it does not check take")
        runner = Runner.at(state_dir)
        source = f"the validator {cfg.source_hotkey}" if cfg.source_hotkey else "the subnet owner's validator"
        log.info("references one at a time: %s, each started for its model's canaries and stopped after; a model not "
                 "checked in an epoch takes the results of %s", ", ".join(sorted(cfg.references)), source)
    own = axon_base(cfg.public_host, cfg.port) if cfg.public_host else None  # where the auditor points its axon

    def other_auditor() -> str | None:
        """Where another auditor with this hotkey answers (``validator.singleton``); None for other validators."""
        if not is_owner:
            return None
        return singleton.another_auditor(st, cfg.netuid, wallet.hotkey.ss58_address, own, current.catalog)

    if other := other_auditor():  # before its tests, its axon and its weights
        raise SystemExit(singleton.REFUSED.format(url=other))
    throughput = Throughput.start(network=cfg.network, netuid=cfg.netuid, current=lambda: current.catalog, st=st,
                                  wallet=wallet, references=cfg.references, screens=cfg.audit_screens,
                                  state_dir=state_dir)
    board = StatusBoard(state_dir / "status.json")
    if why := refused_by_miners(chain.snapshot(st, cfg.netuid)[0], wallet.hotkey.ss58_address,
                                throughput.auditor_hotkey, current.catalog):
        log.critical("%s. Every miner refuses this validator's requests, and it sets no weights until that changes",
                     why)
    identity = None
    if is_owner or throughput.auditor is not None or cfg.public_host:
        if not cfg.public_host:
            raise SystemExit("this validator serves the catalog or the throughput verdicts: set public_host")
        if is_owner:
            current.store.put(current.catalog)
        identity = tls.identity(state_dir)
        start_service(st=st, wallet=wallet, netuid=cfg.netuid, public_host=cfg.public_host, port=cfg.port,
                      identity=identity, store=current.store if is_owner else None, auditor=throughput.auditor,
                      board=board, probation=lambda: [key for key, left in list(state.probation.items()) if left > 0],
                      gateways=lambda: current.catalog.gateway_hotkeys())
    if is_owner:
        log_auditor_pin(current.catalog, identity.pin)
    else:
        publish_own_identity(cfg, st, wallet, identity.pin if identity is not None else None)
    egress = None if args.once else watch_egress(cfg, wallet, current, identity.pin if identity is not None else None,
                                                 is_owner)
    if is_owner and cfg.auto_update:
        log.info("self-update: the auditor does not update itself (its restart would stop the throughput tests); "
                 "update it by hand")
    updater = None if args.once else self_update.start(cfg.auto_update and not is_owner, "validator")
    pinned = {m.model_id: (m.hf_revision, m.engine.revision) for m in current.catalog.models}
    mechanism = [catalog.mechanism_of(current.catalog, cfg.mechid)]
    log.info("setting weights on mechanism %d (%s)", mechanism[0],
             "the catalog's" if cfg.mechid is None else "mechid in the configuration")

    def adopt(cat: catalog.Catalog) -> None:
        if missing := missing_references(cfg, cat):
            log.critical("catalog %s activates %s, which this validator has no reference engine for; stopping",
                         cat.digest()[:12], missing)
            os._exit(1)
        for m in cat.models:
            if m.model_id in cfg.references and pinned.get(m.model_id, (m.hf_revision, m.engine.revision)) != (
                    m.hf_revision, m.engine.revision):
                log.critical("catalog %s pins %s at weights %s and engine %s: %s within %d epochs",
                             cat.digest()[:12], m.model_id, m.hf_revision, m.engine.revision,
                             "restart its reference engine on them" if runner is None else
                             "update this validator's references to them (scripts/install.sh validator --yes)",
                             cat.scoring.revision_transition_epochs)
            pinned[m.model_id] = (m.hf_revision, m.engine.revision)
        if (now := catalog.mechanism_of(cat, cfg.mechid)) != mechanism[0]:
            log.warning("catalog %s moves the network to mechanism %d: epochs scored under it set weights there, and "
                        "manifests for mechanism %d stay accepted for an epoch", cat.digest()[:12], now, mechanism[0])
            mechanism[0] = now
        throughput.set_policy(cat.audit_policy())
        if is_owner:
            log_auditor_pin(cat, identity.pin)

    current.on_change.append(adopt)
    current.start()
    throughput.start_slot_refresh(cfg.network, lambda: manifest_catalogs(cfg, current.catalog, current))
    wait_for_references({m: r for m, r in cfg.references.items() if runner is None or not r.command})

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
                if other := other_auditor():  # its restart is refused while that one runs
                    log.critical("%s", singleton.STOPPED.format(url=other))
                    os._exit(1)
                blocks_left = max(next_start - block, 1)
                trial = copy.deepcopy(state)  # an epoch that fails part way leaves no trace for its retry
                asyncio.run(run_epoch(cfg, st, wallet, secret, trial, epoch, 1 if args.once else blocks_left,
                                      throughput, current, board, runner, egress))
                state = trial
                done = state.last_epoch = epoch
                state.save(state_path)
                if args.once:
                    break
            if updater is not None:  # between epochs: scored, weights set, state saved
                updater.at_safe_point()
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
