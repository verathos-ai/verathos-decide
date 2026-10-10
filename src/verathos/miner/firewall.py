"""The miner's firewall: its service port answers only the network's callers and the hotkey's machines.

The allowlist (``allowed``) holds:
- the published addresses: the auditor's and the gateways' from the catalog, those of each validator with a permit and
  ``validator_min_stake`` from its commitment (``protocol.callers.allowlist``), and those of the miner's own
  ``gateways``;
- the learned addresses (``Roster``): the public source address of every request whose signature (or test session)
  verified for a permitted caller, kept ``KEEP_EPOCHS`` after its last such request, for as long as the caller is
  permitted; they keep a validator whose requests leave from another address than it publishes working;
- this machine's addresses (it measures itself through its service) and the hotkey's other machines: on the primary
  its peers, on a worker the hotkey's axon, its primary.

It is read every ``REFRESH_S`` with the chain's permits, and at once when the catalog changes; an address that drops
out stays allowed for ``HOLD`` more refreshes, and when the chain cannot be read the last list read (saved in
``STATE_FILE``) stays.

Two layers enforce it: the kernel's (``miner.nftables``, applied by the root helper the installer sets up, or by this
process as root), and always this process's own (``miner.sockfilter`` on the listening socket, and the service's first
middleware, ``allows``), for a miner without root or in a container. Both enforce only in ``ENFORCE`` mode
(``decide``): when the committed catalog's ``firewall`` is ``enforce`` (the subnet owner's switch), the list holds the
auditor, a request authenticated as an allowed caller arrived from a published address (no proxy or port forwarding
hides the callers' addresses), and the known addresses cover at least ``COVERAGE`` of the stake of the validators that
call miners. Otherwise the miner observes: it learns and reports, and drops nothing. ``firewall: false`` in the
configuration turns all of it off.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import os
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit

from verathos.catalog import Catalog
from verathos.miner import nftables, sockfilter
from verathos.miner.config import MinerConfig
from verathos.miner.nftables import FirewallError
from verathos.protocol.callers import Caller, allowlist, validator_identity
from verathos.protocol.manifest import is_public
from verathos.subnet.chain import subtensor
from verathos.throughput.policy import BLOCK_TIME_S

log = logging.getLogger("verathos.miner")
OBSERVE, ENFORCE = "observe", "enforce"
REFRESH_S = 600.0  # as the permits
RETRY_S = 60.0  # after a failed read
TICK_S = 30.0
HOLD = 2  # refreshes an address stays allowed after it dropped out of the list
COVERAGE = 0.5  # of the stake of the validators that call miners, at least, before enforcing
KEEP_EPOCHS = 280  # a learned address stays this many epochs after its caller's last request from it (about a week)
MAX_LEARNED = 16  # addresses per caller
MASKED_S = 2 * REFRESH_S  # a request from a non-public address keeps the miner observing this long
STATE_FILE = "firewall.json"
ROSTER_FILE = "firewall-roster.json"


class Neuron(Caller, Protocol):
    uid: int
    axon_ip: str


def normal(address: str | None) -> str | None:
    """``address`` as the lists hold it (an IPv4-mapped IPv6 address as IPv4), or None if it is no IP address."""
    try:
        ip = ipaddress.ip_address((address or "").split("%")[0])
    except ValueError:
        return None
    if ip.version == 6 and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return str(ip)


def unicast(value: str) -> bool:
    """Whether ``value`` is an address the lists can hold (not loopback, which always passes)."""
    try:
        ip = ipaddress.ip_address(value)
    except ValueError:
        return False
    return not (ip.is_loopback or ip.is_unspecified or ip.is_multicast)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")
    tmp.replace(path)


class Roster:
    """Where the permitted callers' verified requests came from: hotkey -> public address -> when last (unix s),
    at most ``MAX_LEARNED`` addresses per caller, the oldest dropped first. Kept in ``path``."""

    def __init__(self, path: Path | None = None, clock: Callable[[], float] = time.time) -> None:
        self.path, self.clock = path, clock
        self._lock = threading.Lock()
        self._seen: dict[str, dict[str, float]] = {}
        try:
            loaded = json.loads(path.read_text()) if path is not None else {}
            self._seen = {str(h): {str(a): float(t) for a, t in seen.items() if normal(a) == a}
                          for h, seen in loaded.get("callers", {}).items()}
        except (OSError, ValueError, AttributeError, TypeError):
            pass

    def learn(self, hotkey: str, address: str) -> None:
        with self._lock:
            seen = self._seen.setdefault(hotkey, {})
            seen[address] = self.clock()
            if len(seen) > MAX_LEARNED:
                del seen[min(seen, key=seen.get)]

    def addresses(self, hotkey: str, keep_s: float) -> list[str]:
        now = self.clock()
        with self._lock:
            return sorted(a for a, t in self._seen.get(hotkey, {}).items() if now - t <= keep_s)

    def save(self, keep_s: float) -> None:
        """Drop what is older than ``keep_s`` and write the roster."""
        now = self.clock()
        with self._lock:
            self._seen = {h: kept for h, seen in self._seen.items()
                          if (kept := {a: t for a, t in seen.items() if now - t <= keep_s})}
            snapshot = {"callers": {h: dict(seen) for h, seen in self._seen.items()}}
        if self.path is not None:
            _write(self.path, snapshot)


@dataclass(frozen=True)
class Allowed:
    """One reading of the allowlist (see the module)."""

    published: tuple[str, ...] = ()
    learned: tuple[str, ...] = ()
    machines: tuple[str, ...] = ()
    auditor: bool = False  # the auditor's address is among them
    validators: int = 0  # validators that call miners (commit a validator identity, or were seen calling)
    stake: float = 0.0  # theirs
    covered: float = 0.0  # of it, that of the validators with a known address
    uncovered: tuple[str, ...] = ()  # those without one

    @property
    def addresses(self) -> tuple[str, ...]:
        everything = {*self.published, *self.learned, *self.machines}
        return tuple(nftables.addresses(everything)) if everything else ()

    @property
    def coverage(self) -> float:
        return self.covered / self.stake if self.stake > 0 else 1.0

    def describe(self) -> str:
        return (f"{len(self.addresses)} addresses ({len(self.published)} published, {len(self.learned)} learned, "
                f"{len(self.machines)} of this hotkey's machines); known addresses cover {self.coverage:.0%} of the "
                f"stake of the {self.validators} validators that call miners")


def answered(n: Caller, cat: Catalog) -> bool:
    """Whether the miner answers ``n`` as a validator: a permit and the catalog's ``validator_min_stake``."""
    return n.permit and n.stake >= cat.scoring.validator_min_stake


def allowed(neurons: Sequence[Neuron], cat: Catalog, config: MinerConfig, hotkey: str, roster: Roster,
            auditor: str | None, keep_s: float) -> Allowed:
    """The allowlist for this machine under ``cat`` (see the module): ``neurons`` is the chain's snapshot
    (``validator.chain.snapshot``), ``hotkey`` the miner's, ``auditor`` the subnet owner's hotkey."""
    own_gateways = [g for g in config.gateways if g.hotkey not in cat.gateway_hotkeys()]
    published = {*allowlist(neurons, cat), *(a for g in own_gateways for a in g.egress)}
    validators = [n for n in neurons if answered(n, cat) and n.hotkey not in (hotkey, auditor)]
    permitted = {*(n.hotkey for n in validators), *cat.gateway_hotkeys(), *(g.hotkey for g in own_gateways),
                 *([auditor] if auditor else [])}
    learned = {a for h in permitted for a in roster.addresses(h, keep_s)}
    machines = {config.public_host, config.listen_host}
    if config.role == "primary":
        machines.update(urlsplit(p).hostname or "" for p in config.peers)
    else:
        machines.update(n.axon_ip for n in neurons if n.hotkey == hotkey and is_public(n.axon_ip))
    calling, stake, covered, uncovered = 0, 0.0, 0.0, []
    for n in validators:
        identity = validator_identity(n.commitment)
        known = bool(identity is not None and identity.egress) or bool(roster.addresses(n.hotkey, keep_s))
        if identity is None and not known:
            continue  # a light validator: it never calls miners
        calling, stake = calling + 1, stake + n.stake
        if known:
            covered += n.stake
        else:
            uncovered.append(f"{n.hotkey} (UID {n.uid})")
    auditor_known = bool(cat.auditor is not None and cat.auditor.egress) or bool(
        auditor and roster.addresses(auditor, keep_s))
    return Allowed(published=tuple(sorted(published)), learned=tuple(sorted(learned - published)),
                   machines=tuple(sorted({m for m in machines if unicast(m)} - published - learned)),
                   auditor=auditor_known, validators=calling, stake=stake, covered=covered,
                   uncovered=tuple(uncovered))


@dataclass
class Check:
    """What the requests authenticated as allowed callers showed: one from a published address (``listed``), and the
    last from an address that is not public (``masked``): each (hotkey, address, unix time)."""

    listed: tuple[str, str, float] | None = None
    masked: tuple[str, str, float] | None = None


def decide(allowed: Allowed | None, cat: Catalog, check: Check, now: float) -> tuple[str, str]:
    """(``ENFORCE`` or ``OBSERVE``, why)."""
    if allowed is None:
        return OBSERVE, "no allowlist read yet"
    if cat.firewall != ENFORCE:
        return OBSERVE, ("the catalog's firewall is observe: the subnet owner switches it to enforce once the "
                         "validators publish their addresses")
    if not allowed.published and not allowed.learned:
        return OBSERVE, "no caller's address is known"
    if not allowed.auditor:
        return OBSERVE, "the auditor's address is not known"
    if check.masked is not None and now - check.masked[2] < MASKED_S:
        hotkey, address, _ = check.masked
        return OBSERVE, (f"requests of {hotkey} arrive from {address}, not a public address: a proxy or port "
                         "forwarding in front of the miner hides the callers' addresses")
    if check.listed is None:
        return OBSERVE, "no request authenticated as an allowed caller has come from a published address yet"
    if allowed.coverage < COVERAGE:
        return OBSERVE, (f"known addresses cover {allowed.coverage:.0%} of the stake of the validators that call "
                         f"miners, less than {COVERAGE:.0%}")
    return ENFORCE, f"known addresses cover {allowed.coverage:.0%} of the stake of the validators that call miners"


# ---- the kernel layer ----

class Backend(Protocol):
    def apply(self, port: int, addresses: Sequence[str]) -> bool: ...

    def status(self) -> dict: ...

    def remove(self) -> bool: ...


class Helper:
    """The root helper, through ``sudo -n``."""

    def __init__(self, *, path: str = nftables.HELPER, run=subprocess.run) -> None:
        self.command = ["sudo", "-n", path]
        self._run = run

    def _call(self, args: list[str], stdin: str | None = None) -> str:
        try:
            result = self._run([*self.command, *args], input=stdin, capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.SubprocessError) as exc:
            raise FirewallError(f"{' '.join(self.command)} {args[0]}: {exc}") from None
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip().splitlines()
            raise FirewallError(detail[-1] if detail else f"{' '.join(self.command)} {args[0]} exited with "
                                f"{result.returncode}")
        return result.stdout

    def apply(self, port: int, addresses: Sequence[str]) -> bool:
        return not self._call(["apply", str(port)], "".join(f"{a}\n" for a in addresses)).strip().endswith(
            "(unchanged)")

    def status(self) -> dict:
        try:
            return json.loads(self._call(["status"]))
        except ValueError as exc:
            raise FirewallError(f"the helper's status is not JSON ({exc})") from None

    def remove(self) -> bool:
        return self._call(["remove"]).strip() == "table removed"


class InProcess:
    """nftables from this process, which runs as root."""

    @staticmethod
    def _do(call: Callable[[], object]):
        try:
            return call()
        except ValueError as exc:
            raise FirewallError(str(exc)) from None
        except (OSError, subprocess.SubprocessError) as exc:
            raise FirewallError(f"nft: {exc}") from None

    def apply(self, port: int, addresses: Sequence[str]) -> bool:
        return self._do(lambda: nftables.apply(port, list(addresses)))

    def status(self) -> dict:
        return self._do(nftables.status)

    def remove(self) -> bool:
        return self._do(nftables.remove)


def install_command(config_path: Path | None) -> str:
    where = f" --config {config_path}" if config_path is not None else ""
    return f"verathos-miner firewall install{where} (once, as a user with sudo), then restart the miner"


def detect(port: int, config_path: Path | None = None, *, euid: Callable[[], int] = os.geteuid,
           platform: str = sys.platform, helper: str = nftables.HELPER, run=subprocess.run,
           in_process: Callable[[], Backend] = InProcess) -> tuple[Backend | None, str]:
    """The kernel layer this process can use, or None and why not."""
    if not platform.startswith("linux"):
        return None, "nftables runs on Linux only"
    if euid() == 0:
        backend = in_process()
    elif not Path(helper).exists():
        return None, f"its helper is not set up on this machine; to set it up: {install_command(config_path)}"
    else:
        try:
            permitted = run(["sudo", "-n", "-l", helper, "apply", str(port)], capture_output=True, text=True,
                            timeout=30).returncode == 0
        except (OSError, subprocess.SubprocessError):
            permitted = False
        if not permitted:
            return None, (f"this user may not run its helper for port {port}; to set it up: "
                          f"{install_command(config_path)}")
        backend = Helper(path=helper, run=run)
    try:
        backend.status()
    except FirewallError as exc:
        text = str(exc)
        if "not found" in text:
            return None, f"{text}: apt-get install -y nftables, then restart the miner"
        if "not permitted" in text.lower() or "permission" in text.lower():
            return None, "this machine does not permit nftables rules here (a container without NET_ADMIN?)"
        return None, text
    return backend, ""


def syncookies(path: Path = Path("/proc/sys/net/ipv4/tcp_syncookies")) -> str | None:
    """A warning if SYN cookies are off (the kernel's default is on), else None."""
    try:
        if path.read_text().strip() == "0":
            return ("tcp_syncookies is 0: a SYN flood fills the service's listen queue; turn it on "
                    "(sysctl -w net.ipv4.tcp_syncookies=1, and in /etc/sysctl.d)")
    except OSError:
        pass
    return None


# ---- keeping it current ----

@dataclass
class Keeper:
    """Keeps the firewall of ``config.port`` current (see the module). ``neurons`` reads the chain's snapshot (raises
    when it cannot), ``catalog`` gives the catalog the miner serves, ``auditor`` the subnet owner's hotkey; ``kernel``
    is the kernel layer, None if there is none (``kernel_off`` says why)."""

    config: MinerConfig
    hotkey: str
    neurons: Callable[[], Sequence[Neuron]]
    catalog: Callable[[], Catalog]
    auditor: Callable[[], str | None]
    kernel: Backend | None = None
    kernel_off: str = ""
    clock: Callable[[], float] = time.time
    mode: str = OBSERVE
    why: str = ""
    last: Allowed | None = None  # the last list read
    check: Check = field(default_factory=Check)
    sockets: list = field(default_factory=list)
    _enforced: frozenset[str] = frozenset()  # what the service accepts while enforcing
    _recent: deque = field(default_factory=lambda: deque(maxlen=HOLD + 1))
    _read_at: float | None = None
    _tried_at: float = float("-inf")
    _reported_at: float = float("-inf")
    _digest: str | None = None
    _kernel_table: bool | None = None  # whether the kernel holds the table; None: not known yet
    _said: dict[str, str] = field(default_factory=dict)
    _seen: dict[str, tuple[str | None, bool, bool]] = field(default_factory=dict)  # address -> (normal, loopback,
    # public), for the service's requests

    def __post_init__(self) -> None:
        state = self.config.state_dir.expanduser()
        self.state_path = state / STATE_FILE
        self.roster = Roster(state / ROSTER_FILE, self.clock)
        try:
            saved = json.loads(self.state_path.read_text())
            if saved.get("port") == self.config.port and saved.get("hotkey") == self.hotkey:
                self.last = Allowed(**{k: tuple(v) if isinstance(v, list) else v for k, v in saved["last"].items()})
                listed = saved.get("check", {}).get("listed")
                self.check.listed = tuple(listed) if listed else None
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            pass

    def _say(self, key: str, level: int, text: str) -> None:
        if self._said.get(key) != text:
            log.log(level, text)
            self._said[key] = text

    def keep_s(self) -> float:
        return KEEP_EPOCHS * self.catalog().epoch_blocks * BLOCK_TIME_S

    # ---- from the service ----

    def _classify(self, address: str | None) -> tuple[str | None, bool, bool]:
        """(``normal(address)``, whether it is loopback, whether it is public), remembered for the next request."""
        known = self._seen.get(address or "")
        if known is None:
            a = normal(address)
            known = (a, a is not None and ipaddress.ip_address(a).is_loopback, a is not None and is_public(a))
            if len(self._seen) >= 4096:
                self._seen.clear()
            self._seen[address or ""] = known
        return known

    def observe(self, hotkey: str, address: str | None) -> None:
        """A request authenticated as ``hotkey`` (a permitted caller) came from ``address``: learn it, and check that
        callers arrive from their published addresses."""
        a, loopback, public = self._classify(address)
        if a is None or loopback or hotkey == self.hotkey or (self.last is not None and a in self.last.machines):
            return
        if not public:
            self.check.masked = (hotkey, a, self.clock())
            return
        self.roster.learn(hotkey, a)
        if self.check.listed is None and self.last is not None and a in self.last.published:
            self.check.listed = (hotkey, a, self.clock())
            log.info("firewall: a request of %s came from %s, a published address", hotkey, a)

    def allows(self, address: str | None) -> bool:
        """Whether the service answers a connection from ``address`` (its first middleware)."""
        if self.mode != ENFORCE:
            return True
        a, loopback, _ = self._classify(address)
        return loopback or a in self._enforced

    def attach(self, sock) -> None:
        """Filter the service's listening socket ``sock`` as the mode says, now and at every change."""
        self.sockets.append(sock)
        self._filter_sockets()

    def _filter_sockets(self) -> None:
        for sock in self.sockets:
            if self.mode != ENFORCE:
                sockfilter.detach(sock)
                continue
            try:
                sockfilter.attach(sock, self._enforced)
                self._said.pop("socket", None)
            except (ValueError, OSError) as exc:
                sockfilter.detach(sock)
                self._say("socket", logging.WARNING, f"firewall: the listening socket is not filtered ({exc}); the "
                          "service refuses other addresses once it accepted them")

    # ---- the refresh ----

    def refresh(self) -> None:
        """Read the allowlist, decide the mode and apply it to both layers."""
        now = self._tried_at = self.clock()
        cat = self.catalog()
        try:
            neurons = self.neurons()
        except Exception as exc:
            text = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
            self._say("read", logging.WARNING, f"firewall: the allowlist could not be read from the chain ({text}); "
                      + ("the last list read stays" if self.last is not None else "the miner observes until it is"))
        else:
            self._said.pop("read", None)
            self.last = allowed(neurons, cat, self.config, self.hotkey, self.roster, self.auditor(), self.keep_s())
            self._read_at, self._digest = now, cat.digest()
        if self.last is not None:
            self._recent.append(self.last.addresses)
        mode, why = decide(self.last, cat, self.check, now)
        if (mode, why) != (self.mode, self.why):
            level = logging.WARNING if self.mode == ENFORCE and mode == OBSERVE else logging.INFO
            log.log(level, "firewall %s: %s%s", "enforcing" if mode == ENFORCE else "observing", why,
                    f"; {self.last.describe()}" if self.last is not None else "")
        self.mode, self.why = mode, why
        self._enforced = frozenset(a for addresses in self._recent for a in addresses) if mode == ENFORCE \
            else frozenset()
        self._filter_sockets()
        self._apply_kernel()
        if mode == ENFORCE and self.last.uncovered:
            self._say("uncovered", logging.WARNING, "firewall: validators that publish no egress address and have not "
                      f"reached this miner, whose requests it drops: {', '.join(self.last.uncovered)}")
        self.roster.save(self.keep_s())
        self._save()

    def _apply_kernel(self) -> None:
        if self.kernel is None:
            return
        try:
            if self.mode == ENFORCE:
                if self.kernel.apply(self.config.port, sorted(self._enforced)):
                    log.info("firewall: the kernel drops connections to port %d from other addresses than the %d "
                             "allowed", self.config.port, len(self._enforced))
                self._kernel_table = True
            elif self._kernel_table is not False:
                if self.kernel.remove():
                    log.info("firewall: the kernel table is removed while the miner observes")
                self._kernel_table = False
            self._said.pop("kernel", None)
        except FirewallError as exc:
            self._say("kernel", logging.WARNING, f"firewall: the kernel layer failed ({exc}); connections are "
                      "filtered in the miner process")

    def _save(self) -> None:
        try:
            _write(self.state_path, {
                "port": self.config.port, "hotkey": self.hotkey, "mode": self.mode, "why": self.why,
                "kernel": "on" if self.kernel is not None else f"off: {self.kernel_off}",
                "last": asdict(self.last) if self.last is not None else None, "enforced": sorted(self._enforced),
                "check": {"listed": self.check.listed, "masked": self.check.masked}, "updated": self.clock()})
        except OSError as exc:
            self._say("save", logging.WARNING, f"firewall: its state could not be saved ({exc})")

    def report(self) -> str:
        """The line the miner logs once an epoch."""
        line = (f"firewall {'enforcing' if self.mode == ENFORCE else 'observing'} on port {self.config.port}"
                + ("" if self.mode == ENFORCE else f" ({self.why})"))
        if self.last is not None:
            line += f": {self.last.describe()}"
        if self.kernel is None:
            line += f"; kernel layer off ({self.kernel_off})"
        elif self.mode == ENFORCE:
            try:
                line += f"; the kernel dropped {self.kernel.status().get('dropped_packets', 0)} packets"
            except FirewallError as exc:
                line += f"; kernel state unknown ({exc})"
        return line

    def round(self) -> None:
        """One round of the background loop."""
        now = self.clock()
        stale = self._read_at is None or now - self._read_at >= REFRESH_S or self.catalog().digest() != self._digest
        if stale and now - self._tried_at >= RETRY_S:
            self.refresh()
        if now - self._reported_at >= self.catalog().epoch_blocks * BLOCK_TIME_S:
            self._reported_at = now
            log.info(self.report())

    def start(self) -> None:
        def loop() -> None:
            while True:
                time.sleep(TICK_S)
                try:
                    self.round()
                except Exception:
                    log.exception("firewall: refresh failed; trying again")

        threading.Thread(target=loop, name="firewall", daemon=True).start()


def start(config: MinerConfig, config_path: Path | None, hotkey: str, catalog: Callable[[], Catalog],
          auditor: Callable[[], str | None], *, enabled: bool, why_off: str = "",
          neurons: Callable[[], Sequence[Neuron]] | None = None) -> Keeper | None:
    """Set up the firewall before the service starts (see the module); None if it is off."""
    kernel, why = detect(config.port, config_path)
    if not enabled:
        removed = False
        if kernel is not None:
            try:
                removed = kernel.remove()
            except FirewallError as exc:
                log.error("firewall: its kernel table could not be removed (%s)", exc)
        log.info("firewall off (%s)%s", why_off, ": its kernel table was removed" if removed else "")
        return None
    if warning := syncookies():
        log.warning(warning)
    if kernel is None:
        log.info("firewall: the kernel layer is off (%s); connections are filtered in the miner process", why)
    keeper = Keeper(config, hotkey, neurons or chain_neurons(config.network, config.netuid), catalog, auditor,
                    kernel=kernel, kernel_off=why)
    keeper.refresh()
    keeper.start()
    return keeper


def chain_neurons(network: str, netuid: int) -> Callable[[], Sequence[Neuron]]:
    """Reads the chain's snapshot over a connection of its own, made again after a failure."""
    import bittensor as bt

    from verathos.validator.chain import snapshot

    connection: list[bt.Subtensor | None] = [None]

    def read() -> Sequence[Neuron]:
        try:
            connection[0] = connection[0] or subtensor(network)
            return snapshot(connection[0], netuid)[0]
        except Exception:
            connection[0] = None
            raise

    return read
