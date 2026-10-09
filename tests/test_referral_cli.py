"""``verathos-miner referral``: the hotkey signs codes, moves and view tokens that the web app passes on to the
gateway, and the status shows the web app's figures with the auditor's record."""

from __future__ import annotations

import argparse
import asyncio
import json
import time

import pytest

from tests.test_gateway import ACCOUNTS_TOKEN, admin_call
from tests.test_referral_gateway import SITE, USD, admin
from tests.test_throughput import kp
from verathos.miner import referral as cli
from verathos.protocol.referral import read_view_token
from verathos.protocol.status import HotkeyStatus, ReferrerStatus

pytestmark = pytest.mark.usefixtures("lab_network")


def web_app(s):
    """The web app as the CLI reaches it, passing requests on to the gateway's admin API with the account service's
    token (its checks on chain aside)."""
    routes = {cli.CODES_PATH: "/admin/v1/referral-codes", cli.MOVE_PATH: "/admin/v1/referral-codes/move"}
    calls = []

    def post(url: str, body: bytes):
        path = url.removeprefix(SITE)
        calls.append(path)
        r = asyncio.run(admin_call(s, "POST", routes[path], json_=json.loads(body), token=ACCOUNTS_TOKEN))
        return r.status_code, r.json()

    def get(url: str, token: str):
        view = read_view_token(token, netuid=405, site=SITE, now_ms=int(time.time() * 1000))
        days = url.split("days=")[1]
        r = asyncio.run(admin_call(s, "GET", f"/admin/v1/referrers/{view.hotkey}?days={days}", token=ACCOUNTS_TOKEN))
        return r.status_code, r.json()
    return post, get, calls


def action(what: str, **values) -> argparse.Namespace:
    return argparse.Namespace(**{"action": what, "name": None, "from_hotkey": None, "days": 30, "hours": 24.0,
                                 **values})


def test_codes_signed_by_the_hotkey_are_registered_and_moved_through_the_web_app(tmp_path, capsys) -> None:
    s = admin(tmp_path)
    post, get, calls = web_app(s)
    miner, new = kp(), kp()

    def run(keypair, args) -> int:
        return cli.run(args, keypair, 405, "finney", SITE, tmp_path, [], post=post, get=get,
                       status=lambda *a: (None, "no chain here"))
    assert run(miner, action("code", name="Alice")) == 0
    assert capsys.readouterr().out.strip() == f"code alice registered for {miner.ss58_address}: {SITE}/r/alice"
    assert run(miner, action("code")) == 0  # a random code
    random = capsys.readouterr().out.split()[1]
    assert len(random) == 8 and random != "alice"
    assert run(miner, action("code", name="alice")) == 0 and "already registered" in capsys.readouterr().out
    assert run(new, action("code", name="a1ice")) == 1
    assert "refused: 409 conflict: 'a1ice' reads as the code 'alice', which is taken" in capsys.readouterr().err
    assert run(new, action("move", from_hotkey=miner.ss58_address)) == 0
    assert capsys.readouterr().out.strip() == (f"the codes of {miner.ss58_address} moved to {new.ss58_address}, which "
                                               f"holds now: alice, {random}")  # oldest first
    assert calls == [cli.CODES_PATH] * 4 + [cli.MOVE_PATH]

    assert run(new, action("status", days=7)) == 0
    out = capsys.readouterr().out
    assert f"referral codes of {new.ss58_address}:" in out and f"{SITE}/r/alice" in out
    assert "last 7 days: 0 accounts referred, 0 bought credit" in out
    assert "the auditor's status is not available (no chain here)" in out
    assert run(kp(), action("status")) == 0 and "has no referral code yet" in capsys.readouterr().out


def test_the_dashboard_link_carries_a_view_token_valid_a_day_at_most(tmp_path, capsys) -> None:
    miner, now = kp(), int(time.time() * 1000)
    link = cli.dashboard_link(miner, 405, SITE, hours=6, now_ms=now)
    page, token = link.split("#")
    assert page == f"{SITE}/referrals"
    view = read_view_token(token, netuid=405, site=SITE, now_ms=now)
    assert (view.hotkey, view.expires_ms) == (miner.ss58_address, now + 6 * 3_600_000)
    longest = read_view_token(cli.token(miner, 405, SITE, 10 * 86_400_000, now), netuid=405, site=SITE, now_ms=now)
    assert longest.expires_ms == now + 86_400_000
    assert cli.run(action("dashboard", hours=1.5), miner, 405, "finney", SITE, tmp_path, []) == 0
    assert "valid for 1.5 hours" in capsys.readouterr().out


def test_the_status_shows_the_referrers_part_of_the_pool_from_the_auditors_record() -> None:
    hotkey = kp().ss58_address
    figures = {"days": 30, "codes": [{"code": "alice", "created_at": "2026-10-09T10:00:00", "revoked_at": None,
                                      "referred": 3}],
               "totals": {"referred": 3, "paying": 2, "topups_micros": 120 * USD, "refunds_micros": 0,
                          "counted_micros": 120 * USD}}
    status = HotkeyStatus(hotkey=hotkey, uid=12, epoch=45406, referral_share=0.004, referral_weight=0.0021,
                          referral=ReferrerStatus(codes={"alice": [0, 70 * USD, 50 * USD]}, referred_micros=120 * USD,
                                                  share=0.0021))
    lines = cli.status_lines(hotkey, SITE, figures, status)
    assert lines[1].split() == ["alice", f"{SITE}/r/alice", "since", "2026-10-09,", "3", "accounts", "referred"]
    assert lines[2] == ("last 30 days: 3 accounts referred, 2 bought credit; top-ups 120.00 USD, refunds 0.00 USD; "
                        "120.00 USD counted for the pool")
    assert lines[3] == ("referral pool in epoch 45406: 120.00 USD referred over the window (alice 120.00 USD); its "
                        "share 0.002100 of the miners' emission")
    assert lines[4] == "  paid 0.002100 of the weights this epoch (all referrers: 0.004000)"
    unpaid = status.model_copy(update={"referral": status.referral.model_copy(update={"unpaid": "holds no UID"})})
    assert cli.status_lines(hotkey, SITE, None, unpaid)[-1] == "  not paid this epoch: holds no UID"
    none = HotkeyStatus(hotkey=hotkey, epoch=45406, referral_share=0.004)
    assert "no top-up of this hotkey's referred accounts counts" in cli.status_lines(hotkey, SITE, None, none)[-1]
    assert "off in the committed catalog" in cli.status_lines(hotkey, SITE, None, HotkeyStatus(hotkey=hotkey))[-1]


def test_the_command_names_the_hotkey_and_the_site_and_refuses_what_is_not_a_code(tmp_path, monkeypatch,
                                                                                    capsys) -> None:
    def refused(*argv: str) -> str:
        with pytest.raises(SystemExit):
            cli.main(list(argv))
        return capsys.readouterr().err
    assert "name the hotkey" in refused("code", "alice", "--site", SITE)
    assert "name the web app" in refused("code", "alice", "--wallet", "w", "--hotkey", "h", "--netuid", "405")
    assert "https origin" in refused("code", "alice", "--wallet", "w", "--hotkey", "h", "--netuid", "405", "--site",
                                     "http://referrals.example")
    assert "is not a code" in refused("code", "Al_ice", "--wallet", "w", "--hotkey", "h", "--netuid", "405",
                                      "--site", SITE)
    assert "--from" in refused("move", "--wallet", "w", "--hotkey", "h", "--netuid", "405", "--site", SITE)
    assert "at most 24" in refused("dashboard", "--wallet", "w", "--hotkey", "h", "--netuid", "405", "--site", SITE,
                                   "--hours", "48")
    config = tmp_path / "miner.json"
    config.write_text(json.dumps({"netuid": 405, "wallet_name": "w", "hotkey_name": "h", "public_host": "8.8.8.8",
                                  "port": 8091, "referral_site": SITE + "/",
                                  "endpoints": [{"endpoint_id": "gpu0", "model_id": "kev-4b",
                                                 "engine_url": "http://127.0.0.1:8008"}]}))
    from types import SimpleNamespace

    import bittensor as bt

    miner = kp()
    monkeypatch.setattr(bt, "Wallet", lambda name, hotkey: SimpleNamespace(hotkey=miner))
    with pytest.raises(SystemExit) as done:
        cli.main(["dashboard", "--config", str(config), "--hours", "2"])
    assert done.value.code == 0 and capsys.readouterr().out.startswith(f"{SITE}/referrals#")


def test_a_miner_logs_its_part_as_a_referrer_each_epoch() -> None:
    from verathos.miner.status import StatusWatcher

    watcher = StatusWatcher(hotkey="hk", netuid=405, auditor_url=lambda: None, auditor_pin=lambda: "",
                            auditor_hotkey=lambda: None, listed=lambda: [])
    status = HotkeyStatus(hotkey="hk", epoch=5, referral_weight=0.0021, referral=ReferrerStatus(
        codes={"alice": [0, 5 * USD]}, referred_micros=5 * USD, share=0.0021))
    assert ("  referrals: 5.00 USD referred over the window with alice; this hotkey's part 0.2100% of the miners' "
            "emission") in watcher.lines(status)
    unpaid = status.model_copy(update={"epoch": 6, "referral": status.referral.model_copy(
        update={"unpaid": "excluded"})})
    assert "  referrals: 5.00 USD referred over the window with alice; not paid this epoch: excluded" in (
        watcher.lines(unpaid))
