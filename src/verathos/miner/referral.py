"""``verathos-miner referral``: a hotkey's referral codes and its figures as a referrer, run where the hotkey is.

    verathos-miner referral code [NAME]          a code for the hotkey: NAME (4 to 24 characters of a-z, digits and
                                                 single hyphens, starting with a letter), else a random one
    verathos-miner referral move --from HOTKEY   every code of HOTKEY, another hotkey of this coldkey, to this one
    verathos-miner referral status [--days 30]   the codes, their figures per day, and the hotkey's part of the
                                                 referral pool in the auditor's last epoch
    verathos-miner referral dashboard [--hours 24]
                                                 a link to the web app's referral page, valid that long (a day at
                                                 most)

The hotkey is the configuration's (``--config miner.json``), or ``--wallet``, ``--hotkey`` and ``--netuid`` (and
``--network``) for a hotkey that runs no miner. The web app is ``--site``, or ``referral_site`` in the configuration.
The hotkey signs each request (``protocol.referral``); the coldkey is never needed. The web app checks on chain that
the hotkey holds a UID (a move: that both hotkeys have one coldkey) and passes the request on to the gateway, which
keeps the codes.

The web app's API, as this command uses it: ``POST /api/referral/codes`` with a signed registration and ``POST
/api/referral/move`` with a signed move (the gateway's answer, as ``GET /admin/v1/referral-codes`` and ``POST
/admin/v1/referral-codes/move`` give it), ``GET /api/referral/dashboard?days=`` with ``Authorization: Referral
<view token>`` (``GET /admin/v1/referrers/{hotkey}``), and the page ``/referrals#<view token>``.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
from bittensor_wallet import Keypair

from verathos.protocol import tls
from verathos.protocol.referral import (
    VIEW_MAX_MS,
    CodeMove,
    CodeRegistration,
    ReferralView,
    SignedMove,
    SignedRegistration,
    random_code,
    site_origin,
    view_token,
    well_formed,
)
from verathos.protocol.status import HotkeyStatus

CODES_PATH = "/api/referral/codes"
MOVE_PATH = "/api/referral/move"
DASHBOARD_PATH = "/api/referral/dashboard"
PAGE_PATH = "/referrals"
LINK_PATH = "/r/{code}"
STATUS_TOKEN_MS = 60_000  # the view token ``status`` signs for its own request
Post = Callable[[str, bytes], tuple[int, Any]]
Get = Callable[[str, str], tuple[int, Any]]


def post_json(url: str, body: bytes) -> tuple[int, Any]:
    """POST ``body`` (JSON) to the web app. -> (the status, the answer's JSON or its text)."""
    with tls.sync_client(timeout=30.0) as client:
        r = client.post(url, content=body, headers={"Content-Type": "application/json"},
                        extensions=tls.use(tls.CA))
    return r.status_code, _answer(r)


def get_with_token(url: str, token: str) -> tuple[int, Any]:
    """GET ``url`` of the web app with a view token. -> (the status, the answer's JSON or its text)."""
    with tls.sync_client(timeout=30.0) as client:
        r = client.get(url, headers={"Authorization": f"Referral {token}"}, extensions=tls.use(tls.CA))
    return r.status_code, _answer(r)


def _answer(r: httpx.Response) -> Any:
    try:
        return r.json()
    except ValueError:
        return r.text[:300]


def refusal(status: int, answer: Any) -> str:
    """The web app's refusal in a line."""
    error = answer.get("error") if isinstance(answer, dict) else None
    if isinstance(error, dict):
        return f"{status} {error.get('type', '')}: {error.get('message', '')}".strip()
    return f"{status}: {answer}"


def register(keypair: Keypair, netuid: int, site: str, code: str | None = None, *, post: Post = post_json,
             now_ms: int | None = None) -> tuple[int, Any]:
    """Sign a registration of ``code`` (a random one without it) and post it. -> (status, answer)."""
    registration = CodeRegistration(netuid=netuid, hotkey=keypair.ss58_address, code=code or random_code(),
                                    site=site, issued_ms=now_ms if now_ms is not None else int(time.time() * 1000))
    body = SignedRegistration.create(registration, keypair).model_dump_json().encode()
    return post(site + CODES_PATH, body)


def move(keypair: Keypair, netuid: int, site: str, from_hotkey: str, *, post: Post = post_json,
         now_ms: int | None = None) -> tuple[int, Any]:
    """Sign the move of every code of ``from_hotkey`` to this hotkey and post it. -> (status, answer)."""
    signed = SignedMove.create(CodeMove(netuid=netuid, from_hotkey=from_hotkey, to_hotkey=keypair.ss58_address,
                                        site=site, issued_ms=now_ms if now_ms is not None else int(time.time() * 1000)),
                               keypair)
    return post(site + MOVE_PATH, signed.model_dump_json().encode())


def token(keypair: Keypair, netuid: int, site: str, valid_ms: int, now_ms: int | None = None) -> str:
    """A view token of the hotkey's figures, valid ``valid_ms`` (a day at most)."""
    now = now_ms if now_ms is not None else int(time.time() * 1000)
    return view_token(ReferralView(netuid=netuid, hotkey=keypair.ss58_address, site=site,
                                   expires_ms=now + min(valid_ms, VIEW_MAX_MS)), keypair)


def dashboard_link(keypair: Keypair, netuid: int, site: str, hours: float = 24.0, now_ms: int | None = None) -> str:
    """The web app's referral page for the hotkey, with a view token in the fragment (which no server log holds)."""
    return f"{site}{PAGE_PATH}#{token(keypair, netuid, site, int(hours * 3_600_000), now_ms)}"


def usd(micros: int) -> str:
    return f"{micros / 1_000_000:,.2f} USD"


def status_lines(hotkey: str, site: str, figures: dict[str, Any] | None, status: HotkeyStatus | None,
                 why: str | None = None) -> list[str]:
    """What ``status`` prints: the codes and their figures (the web app's, ``figures``), and the hotkey's part of the
    referral pool (the auditor's ``status``; ``why`` it is missing)."""
    lines = []
    if figures is not None:
        codes = figures.get("codes") or []
        lines.append(f"referral codes of {hotkey}:" if codes else f"{hotkey} has no referral code yet "
                     "(verathos-miner referral code)")
        for c in codes:
            revoked = c.get("revoked_at")
            state = f"revoked {str(revoked)[:10]}" if revoked else f"since {str(c['created_at'])[:10]}"
            lines.append(f"  {c['code']:<24} {site}{LINK_PATH.format(code=c['code'])}  {state}, "
                         f"{c.get('referred', 0)} accounts referred")
        t = figures.get("totals") or {}
        if codes:
            lines.append(f"last {figures.get('days')} days: {t.get('referred', 0)} accounts referred, "
                         f"{t.get('paying', 0)} bought credit; top-ups {usd(t.get('topups_micros', 0))}, refunds "
                         f"{usd(t.get('refunds_micros', 0))}; {usd(t.get('counted_micros', 0))} counted for the pool")
    if status is None:
        lines.append(f"referral pool: the auditor's status is not available ({why})" if why else
                     "referral pool: the auditor's status is not available")
        return lines
    r = status.referral
    if r is None:
        if status.epoch is None:
            lines.append("referral pool: the auditor's last epoch does not list this hotkey")
        elif status.referral_share is not None:
            lines.append(f"referral pool in epoch {status.epoch}: no top-up of this hotkey's referred accounts counts "
                         "in the period it pays")
        else:  # a catalog committed since takes effect once the auditor follows it
            lines.append(f"referral pool in epoch {status.epoch}: off in the catalog the auditor scored that epoch "
                         "under")
        return lines
    sums = ", ".join(f"{code} {usd(sum(periods))}" for code, periods in sorted(r.codes.items()))
    lines.append(f"referral pool in epoch {status.epoch}: {usd(r.referred_micros)} referred over the window ({sums}); "
                 f"its share {r.share:.6f} of the miners' emission")
    lines.append(f"  not paid this epoch: {r.unpaid}" if r.unpaid else
                 f"  paid {status.referral_weight:.6f} of the weights this epoch (all referrers: "
                 f"{status.referral_share or 0:.6f})")
    return lines


def auditor_status(network: str, netuid: int, hotkey: str, catalog_dir: Path,
                   mirrors: list[str]) -> tuple[HotkeyStatus | None, str | None]:
    """The hotkey's status at the auditor, signed by the subnet owner hotkey. -> (status, why there is none). Finding
    the auditor reads the chain, which needs the Bittensor SDK (the ``chain`` extra)."""
    try:
        import bittensor as bt
    except ImportError:
        return None, ("reading the chain needs the Bittensor SDK: with uvx add --with bittensor==10.5.0; the "
                      "dashboard link (referral dashboard) shows the pool part without it")

    from verathos.catalog import CatalogStore
    from verathos.protocol.http import get_json
    from verathos.protocol.status import MAX_STATUS_BYTES, STATUS_DOMAIN, STATUS_PATH, unverified
    from verathos.subnet.chain import axon_url, current_catalog

    st = bt.Subtensor(network=network)
    try:
        owner = st.get_subnet_owner_hotkey(netuid)
        url = axon_url(st, netuid, owner)
        if url is None:
            return None, "the subnet owner's validator serves no axon"
        pin = current_catalog(st, netuid, CatalogStore(catalog_dir), mirrors).auditor_pin()
    finally:
        st.close()
    document = get_json(url + STATUS_PATH.format(hotkey=hotkey), limit=MAX_STATUS_BYTES, timeout=15, tls=pin)
    if why := unverified(document, domain=STATUS_DOMAIN, auditor=owner, netuid=netuid,
                         now_ms=int(time.time() * 1000)):
        return None, why
    return HotkeyStatus.model_validate(document), None


def main(argv: list[str]) -> None:
    from verathos.miner.config import MinerConfig

    parser = argparse.ArgumentParser(prog="verathos-miner referral", description="A hotkey's referral codes and its "
                                     "figures as a referrer (docs/MINING.md, Referrals).")
    parser.add_argument("action", choices=["code", "move", "status", "dashboard"])
    parser.add_argument("name", nargs="?", help="code: the code to register; none: a random one")
    parser.add_argument("--config", type=Path, help="the miner's configuration (miner.json)")
    parser.add_argument("--wallet", help="without --config: the wallet holding the hotkey")
    parser.add_argument("--hotkey", help="without --config: the hotkey")
    parser.add_argument("--netuid", type=int, help="without --config: the subnet")
    parser.add_argument("--network", default="finney", help="without --config: the chain")
    parser.add_argument("--site", help="the web app (https://host); default: referral_site in the configuration")
    parser.add_argument("--from", dest="from_hotkey", help="move: the hotkey whose codes move to this one")
    parser.add_argument("--days", type=int, default=30, help="status: the days of figures, 1 to 400")
    parser.add_argument("--hours", type=float, default=24.0, help="dashboard: how long the link is valid, at most 24")
    args = parser.parse_args(argv)
    catalog_dir, mirrors = Path("~/.verathos/catalog"), []
    if args.config is not None:
        config = MinerConfig.load(args.config)
        wallet, hotkey, netuid, network = config.wallet_name, config.hotkey_name, config.netuid, config.network
        site, catalog_dir, mirrors = args.site or config.referral_site, config.catalog_dir, config.catalog_mirrors
    elif None in (args.wallet, args.hotkey, args.netuid):
        parser.error("name the hotkey: --config miner.json, or --wallet, --hotkey and --netuid")
    else:
        wallet, hotkey, netuid, network, site = args.wallet, args.hotkey, args.netuid, args.network, args.site
    if not site:
        parser.error("name the web app: --site https://<host>, or referral_site in the configuration")
    try:
        site = site_origin(site)
    except ValueError as exc:
        parser.error(str(exc))
    if args.action == "code" and args.name is not None and not well_formed(args.name.lower()):
        parser.error(f"{args.name!r} is not a code: 4 to 24 characters of a-z, digits and single hyphens, starting "
                     "with a letter")
    if args.action == "move" and not args.from_hotkey:
        parser.error("move needs --from <the hotkey whose codes move here>")
    if not 0 < args.hours <= 24 or not 1 <= args.days <= 400:
        parser.error("--hours is at most 24, --days 1 to 400")
    from bittensor_wallet import Wallet  # the base package's: no Bittensor SDK needed

    keypair = Wallet(name=wallet, hotkey=hotkey).hotkey
    sys.exit(run(args, keypair, netuid, network, site, catalog_dir, mirrors))


def run(args: argparse.Namespace, keypair: Keypair, netuid: int, network: str, site: str, catalog_dir: Path,
        mirrors: list[str], *, post: Post = post_json, get: Get = get_with_token,
        status: Callable[..., tuple[HotkeyStatus | None, str | None]] = auditor_status) -> int:
    """One action (see the module); -> the exit status."""
    hotkey = keypair.ss58_address
    if args.action == "dashboard":
        print(dashboard_link(keypair, netuid, site, args.hours))
        print(f"valid for {args.hours:g} hours; anyone with the link sees this hotkey's referral figures, read-only")
        return 0
    if args.action in ("code", "move"):
        code, answer = (register(keypair, netuid, site, args.name.lower() if args.name else None, post=post)
                        if args.action == "code" else move(keypair, netuid, site, args.from_hotkey, post=post))
        if code not in (200, 201) or not isinstance(answer, dict):
            print(f"refused: {refusal(code, answer)}", file=sys.stderr)
            return 1
        if args.action == "code":
            print(f"code {answer['code']} {'registered' if code == 201 else 'already registered'} for {hotkey}: "
                  f"{site}{LINK_PATH.format(code=answer['code'])}")
        else:
            print(f"the codes of {answer['from_hotkey']} moved to {hotkey}, which holds now: "
                  f"{', '.join(answer['codes'])}")
        return 0
    code, answer = get(f"{site}{DASHBOARD_PATH}?days={args.days}",
                       token(keypair, netuid, site, STATUS_TOKEN_MS))
    figures = answer if code == 200 and isinstance(answer, dict) else None
    if figures is None:
        print(f"the web app's figures are not available: {refusal(code, answer)}", file=sys.stderr)
    try:
        standing, why = status(network, netuid, hotkey, catalog_dir, mirrors)
    except Exception as exc:  # the chain or the auditor: say so, print the rest
        standing, why = None, f"{type(exc).__name__}: {exc}"
    for line in status_lines(hotkey, site, figures, standing, why):
        print(line)
    return 0 if figures is not None else 1
