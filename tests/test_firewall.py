"""The miner's firewall: the allowlist it derives and learns, when it enforces, its kernel rules and the root helper,
its listening-socket filter and middleware, and its setup; with real sockets, and with nft in an unprivileged network
namespace where the machine permits one."""

from __future__ import annotations

import io
import json
import logging
import os
import secrets
import shutil
import socket
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest
from bittensor_wallet import Keypair
from fastapi.testclient import TestClient

from tests.mock_engine import MockEngine
from verathos import catalog
from verathos.install import firewall as setup
from verathos.miner import firewall, nftables, sockfilter
from verathos.miner.app import create_app
from verathos.miner.config import MinerConfig
from verathos.miner.firewall import ENFORCE, OBSERVE, Allowed, Check, Roster
from verathos.miner.nftables import FirewallError
from verathos.protocol.callers import ValidatorIdentity
from verathos.protocol.manifest import commitment_value
from verathos.protocol.requests import sign_request
from verathos.throughput.policy import HDR_REQUEST_ID, Session, session_headers
from verathos.validator.discovery import Neuron

PIN = "0f" * 32
GATEWAY = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
OWN_GATEWAY = "5DAAnrj7VHTznn2AWBemMuyBwZWs6FNFjdyVXUeYum3PTXFy"
OWNER = "5HGjWAeFDfFCWPsjFQdVV2Msvz2XtMktvgocEZcCj68kUMaw"
HOTKEY = "5CiPPseXPECbkjWCa6MnjNokrgYjMqmKndv2rSnekmSK2DjL"
KEEP_S = 7 * 86400.0


def network(firewall_mode: str = "enforce", **scoring) -> catalog.Catalog:
    document = catalog.load().document()
    return catalog.Catalog.parse({
        **document, "auditor": {"tls": PIN, "egress": ["9.9.9.9", "2620:fe::fe"]}, "firewall": firewall_mode,
        "gateways": [{"hotkey": GATEWAY, "egress": ["1.1.1.1"]}],
        "scoring": {**document["scoring"], "validator_min_stake": 1000.0, **scoring}})


def validator(uid: int, egress_ips: tuple[str, ...] | None, *, stake: float = 5000.0, permit: bool = True,
              axon: str = "0.0.0.0") -> Neuron:
    commitment = ValidatorIdentity(None, egress_ips).commitment() if egress_ips is not None else None
    return Neuron(uid, f"hk{uid}", axon, 0, commitment, stake, permit)


def miner_config(tmp_path: Path | None = None, **fields) -> MinerConfig:
    return MinerConfig.model_validate({
        "netuid": 405, "wallet_name": "w", "hotkey_name": "h", "public_host": "203.0.113.5", "port": 8091,
        "state_dir": str(tmp_path or Path("/nonexistent")),
        "endpoints": [{"endpoint_id": "gpu0", "model_id": "kev-4b", "engine_url": "http://127.0.0.1:8008"}],
        **fields})


# hk1, hk2 publish addresses; hk3 is below the stake, hk4 has no permit; hk5 publishes an identity without an address
# (it calls miners, from an address the miner must learn); hk6 is a light validator; then a miner and the auditor
NEURONS = [validator(1, ("8.8.8.8",), stake=6000.0), validator(2, ("8.8.4.4", "2001:4860:4860::8888")),
           validator(3, ("4.4.4.4",), stake=999.0), validator(4, ("5.5.5.5",), permit=False),
           validator(5, (), stake=3000.0), validator(6, None, stake=50_000.0),
           Neuron(7, HOTKEY, "198.51.100.20", 8091, commitment_value("0" * 64), 0.0, False),
           Neuron(0, OWNER, "9.9.9.10", 8091, None, 1e6, True)]


# ---- the allowlist ----

def test_the_allowlist_holds_published_learned_and_own_addresses(lab_network) -> None:
    """Published: the catalog's auditor and gateways, the validators with a permit and the minimum stake that publish
    their addresses, the miner's own gateways. Learned: where permitted callers' verified requests came from. Own:
    this machine's address and, on the primary, its peers; on a worker, the hotkey's axon (its primary). A light
    validator never calls miners, so it does not count towards the coverage; one that publishes an identity without
    an address does until the miner learns where it calls from."""
    roster = Roster()
    cat = network()
    primary = miner_config(peers=["198.51.100.7:20017", "[2001:db8::7]:20017"],
                           gateways=[{"hotkey": OWN_GATEWAY, "egress": ["2.2.2.2"]}])
    seen = firewall.allowed(NEURONS, cat, primary, HOTKEY, roster, OWNER, KEEP_S)
    assert seen.published == ("1.1.1.1", "2.2.2.2", "2001:4860:4860::8888", "2620:fe::fe", "8.8.4.4", "8.8.8.8",
                              "9.9.9.9")
    assert seen.learned == () and seen.machines == ("198.51.100.7", "2001:db8::7", "203.0.113.5")
    assert (seen.validators, seen.stake, seen.covered, seen.uncovered) == (3, 14000.0, 11000.0, ("hk5 (UID 5)",))
    assert seen.auditor and round(seen.coverage, 3) == 0.786
    assert seen.addresses == ("1.1.1.1", "2.2.2.2", "8.8.4.4", "8.8.8.8", "9.9.9.9", "198.51.100.7", "203.0.113.5",
                              "2001:db8::7", "2001:4860:4860::8888", "2620:fe::fe")

    roster.learn("hk5", "3.3.3.3")  # a verified request of hk5's
    roster.learn("hk3", "4.4.4.5")  # below the stake now: not a caller
    roster.learn("hk2", "8.8.4.4")  # published already
    learned = firewall.allowed(NEURONS, cat, primary, HOTKEY, roster, OWNER, KEEP_S)
    assert learned.learned == ("3.3.3.3",) and learned.coverage == 1.0 and learned.uncovered == ()

    worker = miner_config(public_host="198.51.100.7", port=20017, role="worker")
    assert firewall.allowed(NEURONS, cat, worker, HOTKEY, roster, OWNER, KEEP_S).machines == (
        "198.51.100.20", "198.51.100.7")
    bare = catalog.Catalog.parse({**catalog.load().document(),
                                  "gateways": [{"hotkey": GATEWAY, "egress": ["1.1.1.1"]}]})
    assert not firewall.allowed(NEURONS, bare, primary, HOTKEY, Roster(), OWNER, KEEP_S).auditor
    roster.learn(OWNER, "9.9.9.11")  # the auditor, learned
    assert firewall.allowed(NEURONS, bare, primary, HOTKEY, roster, OWNER, KEEP_S).auditor


def test_learned_addresses_are_kept_a_while_per_caller_and_saved(tmp_path) -> None:
    clock = [1000.0]
    roster = Roster(tmp_path / "roster.json", lambda: clock[0])
    for i in range(firewall.MAX_LEARNED + 2):
        roster.learn("hk1", f"8.8.8.{i + 1}")
        clock[0] += 1
    assert len(roster.addresses("hk1", KEEP_S)) == firewall.MAX_LEARNED and "8.8.8.1" not in roster.addresses(
        "hk1", KEEP_S)  # the oldest went first
    roster.learn("hk2", "2001:db8::1")
    roster.save(KEEP_S)
    again = Roster(tmp_path / "roster.json", lambda: clock[0])
    assert again.addresses("hk2", KEEP_S) == ["2001:db8::1"] and len(again.addresses("hk1", KEEP_S)) == 16
    clock[0] += KEEP_S + 1
    assert again.addresses("hk2", KEEP_S) == []
    again.save(KEEP_S)
    assert json.loads((tmp_path / "roster.json").read_text()) == {"callers": {}}


# ---- when it enforces ----

def test_the_firewall_enforces_only_when_the_catalog_says_so_and_its_checks_pass() -> None:
    cat, now = network(), 10_000.0
    full = Allowed(published=("1.1.1.1", "9.9.9.9"), auditor=True, validators=2, stake=100.0, covered=60.0)
    listed = Check(listed=("hk1", "8.8.8.8", 1.0))
    assert firewall.decide(full, cat, listed, now) == (
        ENFORCE, "known addresses cover 60% of the stake of the validators that call miners")
    for allowed, cat_, check, why in (
            (None, cat, listed, "no allowlist read yet"),
            (full, network("observe"), listed, "the catalog's firewall is observe"),
            (Allowed(auditor=True), cat, listed, "no caller's address is known"),
            (Allowed(published=("1.1.1.1",)), cat, listed, "the auditor's address is not known"),
            (full, cat, Check(listed=listed.listed, masked=("hk1", "10.0.2.2", now - 60)), "requests of hk1 arrive "
             "from 10.0.2.2, not a public address: a proxy or port forwarding in front of the miner hides"),
            (full, cat, Check(), "no request authenticated as an allowed caller has come from a published address"),
            (Allowed(published=("9.9.9.9",), auditor=True, validators=2, stake=100.0, covered=40.0), cat, listed,
             "known addresses cover 40% of the stake of the validators that call miners, less than 50%")):
        mode, reason = firewall.decide(allowed, cat_, check, now)
        assert mode == OBSERVE and reason.startswith(why), reason
    old_mask = Check(listed=listed.listed, masked=("hk1", "10.0.2.2", now - firewall.MASKED_S - 1))
    assert firewall.decide(full, cat, old_mask, now)[0] == ENFORCE
    assert firewall.decide(Allowed(published=("9.9.9.9",), auditor=True), cat, listed, now)[0] == ENFORCE  # no
    # validator calls miners yet: the auditor's and the gateways' addresses are all there is


class FakeKernel:
    def __init__(self) -> None:
        self.table: list[str] | None = None
        self.calls: list[str] = []
        self.fail: str | None = None

    def apply(self, port: int, addresses) -> bool:
        self.calls.append("apply")
        if self.fail:
            raise FirewallError(self.fail)
        changed, self.table = self.table != list(addresses), list(addresses)
        return changed

    def status(self) -> dict:
        return {"table": self.table is not None, "dropped_packets": 7}

    def remove(self) -> bool:
        self.calls.append("remove")
        had, self.table = self.table is not None, None
        return had


class Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


def keeper(tmp_path: Path, chain: dict, current: list, kernel=None, clock: Clock | None = None) -> firewall.Keeper:
    def neurons():
        if chain.get("down"):
            raise ConnectionError("chain unreachable")
        return chain["neurons"]

    return firewall.Keeper(miner_config(tmp_path), HOTKEY, neurons, lambda: current[0], lambda: OWNER,
                           kernel=kernel, clock=clock or Clock())


def test_the_miner_observes_learns_then_enforces_on_both_layers_and_holds_dropped_addresses(tmp_path, caplog) -> None:
    caplog.set_level(logging.INFO, logger="verathos.miner")
    kernel, clock = FakeKernel(), Clock()
    chain, current = {"neurons": NEURONS}, [network()]
    k = keeper(tmp_path, chain, current, kernel, clock)
    k.refresh()
    assert k.mode == OBSERVE and "no request authenticated as an allowed caller" in k.why
    assert k.allows("203.0.113.99") and kernel.calls == ["remove"]  # observing: nothing dropped, no table

    k.observe("hk5", "::ffff:3.3.3.3")  # learned (an IPv4-mapped source as IPv4), not published
    k.observe(HOTKEY, "4.4.4.4")  # its own measurement
    k.observe("hk1", "127.0.0.1")
    assert k.check.listed is None and k.roster.addresses("hk5", KEEP_S) == ["3.3.3.3"]
    k.observe("hk1", "8.8.8.8")  # from its published address: the self-check passes
    assert k.check.listed[:2] == ("hk1", "8.8.8.8")
    clock.now += 10
    k.refresh()
    assert k.mode == ENFORCE and "firewall enforcing: known addresses cover 100%" in caplog.text
    assert "3.3.3.3" in kernel.table and "8.8.8.8" in kernel.table and "203.0.113.5" in kernel.table
    assert k.allows("3.3.3.3") and k.allows("::1") and not k.allows("203.0.113.99") and not k.allows("testclient")

    chain["neurons"] = [n for n in NEURONS if n.hotkey != "hk1"]  # hk1 leaves
    for _ in range(firewall.HOLD):
        clock.now += firewall.REFRESH_S
        k.refresh()
        assert k.allows("8.8.8.8")  # held for HOLD more refreshes
    clock.now += firewall.REFRESH_S
    k.refresh()
    assert not k.allows("8.8.8.8") and "8.8.8.8" not in kernel.table

    chain["down"] = True  # the last list read stays
    clock.now += firewall.REFRESH_S
    k.refresh()
    assert k.mode == ENFORCE and k.allows("3.3.3.3")
    assert "could not be read from the chain (chain unreachable); the last list read stays" in caplog.text

    k.observe("hk2", "10.0.2.2")  # a caller behind a proxy or port forwarding: observe again, the table goes
    k.refresh()
    assert k.mode == OBSERVE and kernel.table is None and k.allows("203.0.113.99")
    assert "firewall observing: requests of hk2 arrive from 10.0.2.2" in caplog.text


def test_a_restarted_miner_keeps_enforcing_from_its_saved_state_and_puts_a_flushed_table_back(tmp_path) -> None:
    kernel, clock = FakeKernel(), Clock()
    chain, current = {"neurons": NEURONS}, [network()]
    first = keeper(tmp_path, chain, current, kernel, clock)
    first.refresh()
    first.observe("hk1", "8.8.8.8")
    first.refresh()
    assert first.mode == ENFORCE
    saved = json.loads((tmp_path / firewall.STATE_FILE).read_text())
    assert saved["mode"] == ENFORCE and saved["port"] == 8091 and "8.8.8.8" in saved["enforced"]

    chain["down"] = True  # restarted while the chain is unreadable: the saved list and check stand
    again = keeper(tmp_path, chain, current, kernel, clock)
    again.refresh()
    assert again.mode == ENFORCE and again.allows("8.8.8.8")
    kernel.table = None  # a flushed ruleset (nftables.service restarted)
    chain["down"] = False
    clock.now += firewall.REFRESH_S
    again.round()
    assert kernel.table is not None and kernel.calls.count("apply") == 3

    kernel.fail = "nft refused the rules: busy"  # the kernel layer fails: the process filters
    clock.now += firewall.REFRESH_S
    again.refresh()
    assert again.mode == ENFORCE and not again.allows("203.0.113.99")
    report = again.report()
    assert report.startswith("firewall enforcing on port 8091: ") and "the kernel dropped 7 packets" in report


def test_the_listening_socket_drops_connections_from_other_addresses_before_any_handshake() -> None:
    server = socket.socket(socket.AF_INET6)
    server.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
    server.bind(("::", 0))
    sockfilter.attach(server, ["127.0.0.2", "2001:db8::1"])  # before listen, as the miner does
    server.listen(16)
    port = server.getsockname()[1]
    stop = threading.Event()

    def accept() -> None:
        server.settimeout(0.2)
        while not stop.is_set():
            try:
                server.accept()[0].close()
            except OSError:
                pass

    threading.Thread(target=accept, daemon=True).start()

    def connects(source: str) -> bool:
        family = socket.AF_INET6 if ":" in source else socket.AF_INET
        with socket.socket(family) as s:
            s.settimeout(0.5)
            s.bind((source, 0))
            try:
                s.connect(("::1" if family == socket.AF_INET6 else "127.0.0.1", port))
                return True
            except OSError:
                return False

    try:
        assert connects("127.0.0.2") and connects("127.0.0.1") and connects("::1")
        assert not connects("127.0.0.3")
        sockfilter.attach(server, ["127.0.0.3"])  # replaced
        assert connects("127.0.0.3") and not connects("127.0.0.2")
        sockfilter.detach(server)
        assert connects("127.0.0.2")
    finally:
        stop.set()
        server.close()
    with pytest.raises(ValueError, match="more than 4096"):
        sockfilter.program([f"2001:db8::{i:x}" for i in range(1, 500)])


def test_the_service_serves_tls_on_its_filtered_socket(tmp_path) -> None:
    """As the miner runs it: the socket bound first, filtered by the keeper while it enforces, then served."""
    import httpx
    import uvicorn

    from verathos.protocol import tls

    async def app(scope, receive, send) -> None:
        if scope["type"] == "http":
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning",
                            **tls.server_options(tls.identity(tmp_path)))
    listening = config.bind_socket()
    k = firewall.Keeper(miner_config(tmp_path), HOTKEY, lambda: [], network, lambda: OWNER)
    k.mode, k._enforced = ENFORCE, frozenset({"127.0.0.2"})
    k.attach(listening)
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, kwargs={"sockets": [listening]}, daemon=True).start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    url = f"https://127.0.0.1:{listening.getsockname()[1]}/"

    def get(source: str) -> str:
        transport = httpx.HTTPTransport(local_address=source, verify=False)
        try:
            with httpx.Client(transport=transport, timeout=1.0) as client:
                return client.get(url).text
        except httpx.TransportError:
            return "dropped"

    try:
        assert (get("127.0.0.2"), get("127.0.0.1"), get("127.0.0.3")) == ("ok", "ok", "dropped")
        k.mode = OBSERVE
        k._filter_sockets()
        assert get("127.0.0.3") == "ok"
    finally:
        server.should_exit = True


def test_the_service_learns_from_authenticated_requests_and_refuses_other_addresses_first(tmp_path) -> None:
    miner, validator, auditor = (Keypair.create_from_mnemonic(Keypair.generate_mnemonic()) for _ in range(3))
    seen, allowed_now = [], {"8.8.8.8"}
    config = miner_config(tmp_path)
    app = create_app(config=config, keypair=miner, manifest=lambda: None, engines={"gpu0": MockEngine("kev-4b")},
                     authorized=lambda hk: hk == validator.ss58_address, auditors=lambda: {auditor.ss58_address},
                     serving=lambda: {"gpu0"}, catalog_digest=lambda: "0" * 64, mechid=lambda: 0,
                     on_verified=lambda hotkey, address: seen.append((hotkey, address)),
                     allows=lambda address: address in allowed_now)
    request = {"model": "kev-4b", "state": "Hi", "questions": {"churn": {"type": "noul", "instructions": "Cancel?"}}}
    body = json.dumps(request).encode()
    signed = sign_request(validator, "POST", "/e/gpu0/v1/systemone", body, miner.ss58_address)
    client = TestClient(app, client=("8.8.8.8", 40000))
    assert client.post("/e/gpu0/v1/systemone", content=body, headers=signed).status_code == 200
    session = Session(netuid=405, auditor=auditor.ss58_address, miner_hotkey=miner.ss58_address, endpoint_id="gpu0",
                      test_id="t" * 64, not_after_ms=int(time.time() * 1000) + 30_000)
    test = {**session_headers(session, auditor), HDR_REQUEST_ID: secrets.token_hex(16)}
    assert client.post("/e/gpu0/v1/systemone", json=request, headers=test).status_code == 200
    assert seen == [(validator.ss58_address, "8.8.8.8"), (auditor.ss58_address, "8.8.8.8")]
    stranger = TestClient(app, client=("203.0.113.99", 40000))
    refused = stranger.post("/e/gpu0/v1/systemone", content=body, headers=signed)
    assert (refused.status_code, refused.json()) == (403, {"error": "source address not allowed"})
    assert stranger.get("/health").status_code == 403 and len(seen) == 2


# ---- the kernel layer ----

GOLDEN = textwrap.dedent("""\
    add table inet verathos_miner
    delete table inet verathos_miner
    table inet verathos_miner {
    \tcounter dropped {
    \t}
    \tset ports {
    \t\ttype inet_service
    \t\telements = { 8091 }
    \t}
    \tset allow4 {
    \t\ttype ipv4_addr
    \t\tflags interval
    \t\telements = { 8.8.4.4, 8.8.8.8 }
    \t}
    \tset allow6 {
    \t\ttype ipv6_addr
    \t\tflags interval
    \t\telements = { 2001:4860:4860::8888 }
    \t}
    \tchain prerouting {
    \t\ttype filter hook prerouting priority -300; policy accept;
    \t\tiif "lo" accept
    \t\ttcp dport @ports ip saddr @allow4 accept
    \t\ttcp dport @ports ip6 saddr @allow6 accept
    \t\ttcp dport @ports counter name "dropped" drop
    \t}
    }
    """)


def test_the_rules_replace_the_table_in_one_transaction_and_drop_all_but_the_allowed_on_the_port() -> None:
    assert nftables.ruleset(8091, ["8.8.4.4", "8.8.8.8", "2001:4860:4860::8888"]) == GOLDEN
    only4 = nftables.ruleset(20017, ["8.8.8.8"])
    assert "\tset allow6 {\n\t\ttype ipv6_addr\n\t\tflags interval\n\t}\n" in only4 and "{ 20017 }" in only4
    with pytest.raises(ValueError, match="not a TCP port"):
        nftables.ruleset(0, ["8.8.8.8"])


def test_the_helper_takes_only_unicast_addresses_in_their_normal_form_and_never_none() -> None:
    assert nftables.addresses(["2001:db8::1", "8.8.8.8", "", "1.1.1.1 ", "8.8.8.8", "10.0.0.7"]) == [
        "1.1.1.1", "8.8.8.8", "10.0.0.7", "2001:db8::1"]
    for bad, why in ((["8.8.8.8/24"], "not an IP"), (["1.1.1.1; flush ruleset"], "not an IP"),
                     (["2001:DB8::1"], "write the address as 2001:db8::1"), (["127.0.0.1"], "unicast"),
                     (["::1"], "unicast"), (["224.0.0.1"], "unicast"), (["0.0.0.0"], "unicast"),
                     (["::ffff:8.8.8.8"], "unicast"), ([], "no address"), ([" ", ""], "no address"),
                     ([f"10.{i // 65536}.{i // 256 % 256}.{i % 256}" for i in range(nftables.MAX_ADDRESSES + 1)],
                      "more than")):
        with pytest.raises(ValueError, match=why):
            nftables.addresses(bad)


def done(stdout: str = "", returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


ABSENT = done(returncode=1, stderr="Error: No such file or directory\nlist table inet verathos_miner\n")


def listing(port: int, v4: list[str], v6: list[str], rules: int = 4, prio: int = -300) -> str:
    def named(name: str, elem: list) -> dict:
        return {"set": {"family": "inet", "name": name, "table": "verathos_miner", "elem": elem}}

    return json.dumps({"nftables": [
        {"metainfo": {"version": "1.0.2"}}, {"table": {"family": "inet", "name": "verathos_miner"}},
        {"counter": {"family": "inet", "name": "dropped", "table": "verathos_miner", "packets": 2, "bytes": 120}},
        named("ports", [port]), named("allow4", v4), named("allow6", v6),
        {"chain": {"family": "inet", "table": "verathos_miner", "name": "prerouting", "type": "filter",
                   "hook": "prerouting", "prio": prio, "policy": "accept"}},
        *({"rule": {"chain": "prerouting", "expr": []}} for _ in range(rules))]})


class FakeNft:
    """nft as the helper runs it: records each call, answers in order."""

    def __init__(self, *answers: subprocess.CompletedProcess) -> None:
        self.calls: list[tuple[list[str], str | None]] = []
        self.answers = list(answers)

    def __call__(self, args: list[str], stdin: str | None = None) -> subprocess.CompletedProcess:
        self.calls.append((args, stdin))
        return self.answers.pop(0) if self.answers else done()


def run_helper(argv: list[str], stdin: str, nft: FakeNft) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    saved = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = out, err
    try:
        code = nftables.main(argv, io.StringIO(stdin), nft)
    finally:
        sys.stdout, sys.stderr = saved
    return code, out.getvalue(), err.getvalue()


def test_the_helper_replaces_a_missing_or_different_table_and_refuses_what_it_must_not_apply() -> None:
    stdin = "8.8.8.8\n8.8.4.4\n2001:4860:4860::8888\n"
    nft = FakeNft(ABSENT)
    assert run_helper(["apply", "8091"], stdin, nft) == (0, "port 8091: 3 addresses allowed\n", "")
    assert nft.calls == [(["-j", "list", "table", "inet", "verathos_miner"], None), (["-f", "-"], GOLDEN)]
    same = FakeNft(done(listing(8091, ["8.8.8.8", "8.8.4.4"], ["2001:4860:4860::8888"])))
    assert run_helper(["apply", "8091"], stdin, same) == (0, "port 8091: 3 addresses allowed (unchanged)\n", "")
    assert len(same.calls) == 1
    for differs in (listing(8091, ["8.8.8.8"], ["2001:4860:4860::8888"]),  # another list
                    listing(8092, ["8.8.4.4", "8.8.8.8"], ["2001:4860:4860::8888"]),  # another port
                    listing(8091, ["8.8.4.4", "8.8.8.8"], ["2001:4860:4860::8888"], rules=3),  # edited
                    listing(8091, ["8.8.4.4", "8.8.8.8"], ["2001:4860:4860::8888"], prio=0)):
        nft = FakeNft(done(differs))
        assert run_helper(["apply", "8091"], stdin, nft)[0] == 0 and nft.calls[-1] == (["-f", "-"], GOLDEN)

    for argv, text, why in ((["apply", "8091"], "", "refused: no address"),
                            (["apply", "8091"], "8.8.8.8\n}; flush ruleset\n", "refused: '}; flush ruleset'"),
                            (["apply", "22x"], "8.8.8.8\n", "not a TCP port"), (["apply"], "", "usage"),
                            (["flush"], "", "usage"), (["status", "x"], "", "usage")):
        nft = FakeNft()
        code, _, err = run_helper(argv, text, nft)
        assert code == 2 and nft.calls == [] and why in err  # nothing reached nft

    refused = FakeNft(ABSENT, done(returncode=1, stderr="/dev/stdin:3:1-9: Error: Could not process rule: Operation "
                                                        "not permitted\nadd table\n^^^^\n"))
    assert run_helper(["apply", "8091"], "8.8.8.8\n", refused) == (
        1, "", "nft refused the rules: /dev/stdin:3:1-9: Error: Could not process rule: Operation not permitted\n")
    assert nftables.status(FakeNft(done(listing(8091, ["1.1.1.1"], ["2001:db8::2"])))) == {
        "table": True, "ports": [8091], "allowed": ["1.1.1.1", "2001:db8::2"], "dropped_packets": 2,
        "hook": ["prerouting", -300], "rules": 4}
    assert nftables.status(FakeNft(ABSENT)) == {"table": False}
    assert nftables.remove(FakeNft(ABSENT)) is False and nftables.remove(FakeNft(done())) is True
    with pytest.raises(FirewallError, match="Operation not permitted"):
        nftables.status(FakeNft(done(returncode=1, stderr="Operation not permitted (you must be root)\n")))


def test_the_kernel_layer_is_the_helper_through_sudo_or_nftables_as_root_or_none_with_the_reason(tmp_path) -> None:
    helper = tmp_path / "verathos-firewall"

    def runner(sudo_ok: bool = True, helper_out: subprocess.CompletedProcess | None = None):
        calls = []

        def run(cmd, **kwargs):
            calls.append((cmd, kwargs.get("input")))
            if cmd[:3] == ["sudo", "-n", "-l"]:
                return done(returncode=0 if sudo_ok else 1)
            return helper_out or done(json.dumps({"table": False}))
        return run, calls

    assert firewall.detect(8091, platform="darwin") == (None, "nftables runs on Linux only")
    backend, why = firewall.detect(8091, Path("/home/m/.verathos/miner.json"), euid=lambda: 1000, helper=str(helper))
    assert backend is None and why == ("its helper is not set up on this machine; to set it up: verathos-miner "
                                       "firewall install --config /home/m/.verathos/miner.json (once, as a user with "
                                       "sudo), then restart the miner")
    helper.touch()
    run, calls = runner(sudo_ok=False)
    backend, why = firewall.detect(8091, euid=lambda: 1000, helper=str(helper), run=run)
    assert backend is None and "may not run its helper for port 8091" in why
    assert calls == [(["sudo", "-n", "-l", str(helper), "apply", "8091"], None)]
    run, calls = runner()
    backend, why = firewall.detect(8091, euid=lambda: 1000, helper=str(helper), run=run)
    assert isinstance(backend, firewall.Helper) and why == ""
    assert backend.apply(8091, ["8.8.8.8", "2001:db8::1"])
    assert calls[-1] == (["sudo", "-n", str(helper), "apply", "8091"], "8.8.8.8\n2001:db8::1\n")
    run, _ = runner(helper_out=done(returncode=1, stderr="nft did not list the table: Operation not permitted\n"))
    backend, why = firewall.detect(8091, euid=lambda: 1000, helper=str(helper), run=run)
    assert backend is None and "a container without NET_ADMIN?" in why

    class Root:
        def __init__(self, error: str | None) -> None:
            self.error = error

        def status(self) -> dict:
            if self.error:
                raise FirewallError(self.error)
            return {"table": False}

    assert isinstance(firewall.detect(8091, euid=lambda: 0, in_process=lambda: Root(None))[0], Root)
    assert firewall.detect(8091, euid=lambda: 0, in_process=lambda: Root("nft not found (install nftables)"))[1] == (
        "nft not found (install nftables): apt-get install -y nftables, then restart the miner")
    (tmp_path / "off").write_text("0\n")
    (tmp_path / "on").write_text("1\n")
    assert firewall.syncookies(tmp_path / "off").startswith("tcp_syncookies is 0")
    assert firewall.syncookies(tmp_path / "on") is None and firewall.syncookies(tmp_path / "missing") is None


def test_a_miner_with_the_firewall_off_removes_its_table_and_one_without_root_filters_in_its_process(
        tmp_path, monkeypatch, caplog, lab_network) -> None:
    caplog.set_level(logging.INFO, logger="verathos.miner")
    kernel = FakeKernel()
    kernel.table = ["8.8.8.8"]
    monkeypatch.setattr(firewall, "detect", lambda port, path: (kernel, ""))
    assert firewall.start(miner_config(tmp_path, firewall=False), None, HOTKEY, network, lambda: OWNER, enabled=False,
                          why_off="firewall is false in the configuration") is None
    assert kernel.table is None and "firewall off (firewall is false in the configuration): its kernel table was " \
                                    "removed" in caplog.text
    monkeypatch.setattr(firewall, "detect", lambda port, path: (None, "its helper is not set up on this machine"))
    monkeypatch.setattr(firewall.Keeper, "start", lambda self: None)
    started = firewall.start(miner_config(tmp_path), None, HOTKEY, network, lambda: OWNER, enabled=True,
                             neurons=lambda: NEURONS)
    assert started is not None and started.mode == OBSERVE and started.kernel is None
    assert "the kernel layer is off (its helper is not set up on this machine); connections are filtered in the " \
           "miner process" in caplog.text


# ---- setting it up ----

def test_the_sudo_rule_lets_the_miners_user_run_exactly_the_helpers_three_commands() -> None:
    assert setup.sudoers("miner", 8091) == (
        "# The miner's firewall (verathos-miner): its user applies the allowlist, reads and removes the table.\n"
        "miner ALL=(root) NOPASSWD: /usr/local/sbin/verathos-firewall apply 8091, /usr/local/sbin/verathos-firewall "
        "status, /usr/local/sbin/verathos-firewall remove\n")
    for user in ("ALL", "a b", "x,y", "root ALL=(ALL)", ""):
        with pytest.raises(ValueError):
            setup.sudoers(user, 8091)
    if shutil.which("visudo"):
        rule = subprocess.run(["visudo", "-cqf", "-"], input=setup.sudoers("miner", 8091), capture_output=True,
                              text=True)
        assert rule.returncode == 0, rule.stderr


def test_the_installer_reports_the_firewall_in_one_line(tmp_path) -> None:
    (tmp_path / "miner.json").write_text(json.dumps({"port": 8091, "state_dir": str(tmp_path)}))
    assert setup.miner_state(tmp_path / "miner.json") is None
    assert setup.summary(None, None, 8091) == "firewall: the miner has not run with it yet"
    observing = {"port": 8091, "mode": "observe", "why": "the catalog's firewall is observe", "kernel": "on",
                 "last": {"published": ["1.1.1.1", "9.9.9.9"], "learned": ["3.3.3.3"]}}
    (tmp_path / firewall.STATE_FILE).write_text(json.dumps(observing))
    assert setup.summary(setup.miner_state(tmp_path / "miner.json"), {"table": False}, 8091) == (
        "firewall observing, dropping nothing: the catalog's firewall is observe (2 published and 1 learned caller "
        "addresses so far); kernel layer ready")
    enforcing = {**observing, "mode": "enforce", "enforced": ["1.1.1.1", "3.3.3.3", "9.9.9.9"]}
    assert setup.summary(enforcing, {"table": True, "dropped_packets": 12}, 8091) == (
        "firewall enforcing: port 8091 answers 3 addresses (the network's callers and this hotkey's machines) and "
        "loopback, and drops the rest; kernel table on, 12 packets dropped (sudo nft list table inet verathos_miner)")
    assert setup.summary({**enforcing, "kernel": "off: no root"}, None, 8091).endswith("; kernel layer off: no root")


def test_the_setup_turns_the_firewall_on_by_default_and_removes_it_when_turned_off(monkeypatch, capsys) -> None:
    from verathos.install import miner as miner_setup

    calls = []
    monkeypatch.setattr(setup, "setup", lambda user, port: calls.append(("setup", port)))
    monkeypatch.setattr(setup, "remove", lambda: calls.append(("remove",)) or ["/usr/local/sbin/verathos-firewall"])
    monkeypatch.setattr(setup, "installed", lambda: True)
    assert miner_setup.set_up_firewall(True, 8091, False) is True and calls == [("setup", 8091)]
    assert miner_setup.set_up_firewall(False, 8091, True) is False and calls[-1] == ("remove",)
    assert "firewall off: removed /usr/local/sbin/verathos-firewall" in capsys.readouterr().out

    def unavailable(user, port):
        raise setup.Unavailable("this machine does not permit nftables rules even to root")

    monkeypatch.setattr(setup, "setup", unavailable)
    assert miner_setup.set_up_firewall(True, 8091, False) is False
    assert "the kernel layer is off (this machine does not permit nftables rules even to root); connections are " \
           "filtered in the miner process" in capsys.readouterr().out
    config = miner_setup.miner_config({}, network="finney", netuid=96, wallet_name="w", hotkey_name="h",
                                      public_ip="203.0.113.5", port=8091, endpoints={"gpu0": ("kev-4b", 8008)},
                                      managed_before=set(), home=Path("/h"), firewall=False)
    assert config["firewall"] is False and MinerConfig.model_validate(config).firewall is False
    assert miner_setup.firewall_line(True, True).startswith("firewall on")
    assert miner_setup.firewall_line(True, False).endswith("without root it filters in the miner process")


def test_the_miners_own_gateways_name_their_addresses_and_the_catalog_its_firewall_switch() -> None:
    assert miner_config().firewall is True and miner_config().gateways == []
    for gateways, why in (([{"hotkey": OWN_GATEWAY}], "egress"),
                          ([{"hotkey": OWN_GATEWAY, "egress": ["10.0.0.1"]}], "public"),
                          ([{"hotkey": OWN_GATEWAY, "egress": ["2.2.2.2"]}] * 2, "unique")):
        with pytest.raises(ValueError, match=why):
            miner_config(gateways=gateways)
    assert catalog.load().firewall == OBSERVE and network().firewall == ENFORCE
    with pytest.raises(ValueError):
        network("drop")


# ---- nft itself, in a network namespace of an unprivileged user ----

def namespaces() -> str | None:
    """Why the real nftables tests cannot run here, or None."""
    if not sys.platform.startswith("linux"):
        return "Linux only"
    for tool in ("unshare", "nsenter", "ip"):
        if shutil.which(tool, path=f"{os.environ.get('PATH', '')}:{nftables.SYSTEM_PATH}") is None:
            return f"{tool} not found"
    nft = shutil.which("nft", path=nftables.SYSTEM_PATH)
    if nft is None:
        return "nft not found"
    try:
        probe = subprocess.run(["unshare", "-rn", nft, "list", "tables"], capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        return str(exc)
    return None if probe.returncode == 0 else "no user and network namespaces here"


WHY_NOT = namespaces()
needs_nft = pytest.mark.skipif(WHY_NOT is not None, reason=f"nftables in a namespace: {WHY_NOT}")
HELPER_PYTHON = "/usr/bin/python3" if Path("/usr/bin/python3").exists() else sys.executable  # as installed


def in_namespace(script: str, timeout: float = 60) -> subprocess.CompletedProcess:
    return subprocess.run(["unshare", "-rn", "--fork", "sh", "-ec", script], capture_output=True, text=True,
                          timeout=timeout, env={**os.environ, "PATH": f"{os.environ.get('PATH', '')}:"
                                                f"{nftables.SYSTEM_PATH}"})


@needs_nft
def test_the_helper_applies_keeps_replaces_reads_and_removes_the_table_in_nft() -> None:
    helper = f"{HELPER_PYTHON} -I {nftables.__file__}"
    result = in_namespace(f"""
        printf '8.8.8.8\\n8.8.8.9\\n2001:4860:4860::8888\\n' | {helper} apply 8091
        printf '8.8.8.9\\n2001:4860:4860::8888\\n8.8.8.8\\n' | {helper} apply 8091
        {helper} status
        nft flush ruleset
        printf '8.8.8.8\\n8.8.8.9\\n2001:4860:4860::8888\\n' | {helper} apply 8091
        printf '1.1.1.1\\n' | {helper} apply 8091
        printf '' | {helper} apply 8091 || echo refused $?
        {helper} status
        {helper} remove
        {helper} remove
        {helper} status
        nft list tables
    """)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[:2] == ["port 8091: 3 addresses allowed", "port 8091: 3 addresses allowed (unchanged)"]
    assert json.loads(lines[2]) == {"table": True, "ports": [8091], "allowed": ["8.8.8.8", "8.8.8.9",
                                                                                 "2001:4860:4860::8888"],
                                    "dropped_packets": 0, "hook": ["prerouting", -300], "rules": 4}
    assert lines[3:5] == ["port 8091: 3 addresses allowed", "port 8091: 1 addresses allowed"]  # put back after a flush
    assert lines[5] == "refused 2" and json.loads(lines[6])["allowed"] == ["1.1.1.1"]  # the last one stays
    assert lines[7:] == ["table removed", "no table", '{"table": false}']


@needs_nft
def test_only_allowed_sources_reach_the_port_over_ipv4_and_ipv6(tmp_path) -> None:
    """Two namespaces joined by a veth pair: the miner's (listening on the port and on another) and a caller's.
    Before the table the caller connects; while it is not allowed its connections to the port are dropped and
    counted, other ports still answer and loopback passes; allowed, it connects again."""
    server = tmp_path / "server.py"
    server.write_text(textwrap.dedent("""\
        import socket, threading, time
        def serve(port):
            s = socket.socket(socket.AF_INET6)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
            s.bind(("::", port))
            s.listen(64)
            while True:
                s.accept()[0].close()
        for port in (8091, 9000):
            threading.Thread(target=serve, args=(port,), daemon=True).start()
        time.sleep(50)
        """))
    probe = tmp_path / "probe.py"
    probe.write_text(textwrap.dedent("""\
        import socket, sys
        out = []
        for target in sys.argv[1:]:
            address, port = target.rsplit(":", 1)
            s = socket.socket(socket.AF_INET6 if ":" in address else socket.AF_INET)
            s.settimeout(1.0)
            try:
                s.connect((address, int(port)))
                out.append("ok")
            except OSError:
                out.append("dropped")
            s.close()
        print(" ".join(out))
        """))
    helper = f"{HELPER_PYTHON} -I {nftables.__file__}"
    py = sys.executable
    result = in_namespace(f"""
        unshare -n sleep 55 &
        CALLER=$!
        sleep 0.3
        ip link set lo up
        ip link add miner type veth peer name caller
        ip link set caller netns $CALLER
        ip addr add 8.8.8.1/24 dev miner
        ip -6 addr add 2001:db8::1/64 dev miner nodad
        ip link set miner up
        nsenter -t $CALLER -n sh -c 'ip link set lo up; ip addr add 8.8.8.8/24 dev caller; \
            ip -6 addr add 2001:db8::2/64 dev caller nodad; ip link set caller up'
        {py} {server} &
        SERVER=$!
        sleep 0.5
        probe() {{ nsenter -t $CALLER -n {py} {probe} "$@"; }}
        probe 8.8.8.1:8091 2001:db8::1:8091
        printf '9.9.9.9\\n2620:fe::fe\\n' | {helper} apply 8091 > /dev/null
        probe 8.8.8.1:8091 2001:db8::1:8091 8.8.8.1:9000
        {py} {probe} 127.0.0.1:8091 8.8.8.1:8091
        {helper} status
        printf '8.8.8.8\\n2001:db8::2\\n' | {helper} apply 8091 > /dev/null
        probe 8.8.8.1:8091 2001:db8::1:8091
        kill -9 $CALLER $SERVER  # background jobs here ignore SIGTERM
    """, timeout=90)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[0] == "ok ok"  # no table yet
    assert lines[1] == "dropped dropped ok"  # not allowed: the port drops it, another port answers
    assert lines[2] == "ok ok"  # from the machine itself, over loopback
    assert json.loads(lines[3])["dropped_packets"] >= 2
    assert lines[4] == "ok ok"  # allowed
