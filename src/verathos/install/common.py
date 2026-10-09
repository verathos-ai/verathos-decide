"""What the miner and validator setups share: questions, the record of what was set up, wallet and chain checks,
waiting for services, and uninstalling."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import httpx
from pydantic import ValidationError

from verathos.install import host, services
from verathos.protocol.tls import chain_network

DEFAULT_HOME = Path("~/.verathos")
ROLES = ("miner", "validator", "reference")
TESTNET_NETUID = 405
MAINNET_NETUID = 96
INSTALLER = Path(__file__).resolve().parents[3] / "scripts" / "install.sh"  # the installer of this checkout
SOURCE_FILE = ".install-source"  # in a checkout the installer made from a tarball: the tarball (scripts/install.sh)
ENGINES_FILE = "reference-engines.json"  # a reference machine's engines, for its callers


# ---- output and questions ----

def step(text: str) -> None:
    print(f"\n==> {text}", flush=True)


def info(text: str) -> None:
    print(f"    {text}", flush=True)


def warn(text: str) -> None:
    print(f"    WARNING: {text}", flush=True)


class Prompter:
    """Asks on a terminal; with ``--yes`` or without a terminal, takes every default."""

    def __init__(self, assume_yes: bool) -> None:
        self.interactive = not assume_yes and sys.stdin.isatty()

    def ask(self, question: str, default: str, check: Callable[[str], str | None] = lambda v: None) -> str:
        """``check`` returns why an answer is not acceptable, or None."""
        while True:
            answer = default
            if self.interactive:
                answer = input(f"    {question} [{default}]: ").strip() or default
            problem = check(answer)
            if problem is None:
                if not self.interactive:
                    info(f"{question}: {answer}")
                return answer
            if not self.interactive:
                raise SystemExit(f"{question}: {problem}")
            print(f"    {problem}")

    def choose(self, question: str, options: list[str], default: str) -> str:
        if len(options) == 1:
            info(f"{question}: {options[0]}")
            return options[0]
        if self.interactive:
            for i, option in enumerate(options, 1):
                print(f"      {i}) {option}")

        def check(answer: str) -> str | None:
            if answer in options or (answer.isdigit() and 1 <= int(answer) <= len(options)):
                return None
            return f"answer 1-{len(options)} or a name"

        answer = self.ask(question, default, check)
        return options[int(answer) - 1] if answer.isdigit() and answer not in options else answer

    def confirm(self, question: str) -> bool:
        if not self.interactive:
            return True
        return input(f"    {question} [Y/n]: ").strip().lower() in ("", "y", "yes")


def common_arguments(parser: argparse.ArgumentParser, *, wallet: bool = True) -> None:
    """The options every setup takes; ``wallet``: also the wallet's and the self-update's (a reference machine holds
    no wallet and is updated with the installer)."""
    parser.add_argument("--yes", "-y", action="store_true", help="Take the default (or the given option) for every "
                        "question")
    if wallet:
        parser.add_argument("--wallet", help="Wallet name (directory under ~/.bittensor/wallets)")
        parser.add_argument("--hotkey", help="Hotkey name in that wallet")
    parser.add_argument("--network", help="finney (default), test, a wss:// subtensor node, or a node on this "
                        "machine (local, ws://127.0.0.1:<port>)")
    parser.add_argument("--netuid", type=int, help=f"Subnet (default {MAINNET_NETUID}; {TESTNET_NETUID} on test)")
    parser.add_argument("--no-service", action="store_true", help="Install no system services and need no root: "
                        "set up, then print the commands to run under a process manager you run (pm2 or another)")
    parser.add_argument("--home", type=Path, default=DEFAULT_HOME, help="Configuration, state, engines and weights "
                        "(default ~/.verathos)")
    if wallet:
        parser.add_argument("--auto-update", action=argparse.BooleanOptionalAction, default=None, help="Update the "
                            "code to each new release of the repository it came from, by itself (default: on, except "
                            "for the subnet owner's validator; a later run keeps the choice)")


def auto_update(args: argparse.Namespace, existing: dict, default: bool = True) -> bool:
    """Whether the process updates itself (``auto_update``): as asked, else as configured before, else ``default``."""
    return args.auto_update if args.auto_update is not None else bool(existing.get("auto_update", default))


def update_line(on: bool) -> str:
    if on:
        return ("self-update on: the code follows the releases (v<version> tags) of the repository it came from, "
                "restarting at a quiet moment (--no-auto-update turns it off)")
    return "self-update off: update the code with the installer (--auto-update turns it on)"


# ---- the record of what was set up ----

@dataclass
class Record:
    """What a setup installed, kept in ``<home>/install/<role>.json``: the defaults of the next run and what
    ``uninstall`` removes."""

    role: str
    manager: str | None = None  # None: the services, if any, run under a process manager the operator runs
    services: list[str] = field(default_factory=list)
    config: str = ""
    engines: dict[str, dict] = field(default_factory=dict)  # per endpoint or model: model, device, port
    paths: list[str] = field(default_factory=list)  # engine checkouts used
    firewall: bool = False  # the miner's firewall helper is set up (``install.firewall``)
    options: dict = field(default_factory=dict)  # choices the next run keeps, which the configuration does not hold

    @staticmethod
    def path(home: Path, role: str) -> Path:
        return home / "install" / f"{role}.json"

    @classmethod
    def load(cls, home: Path, role: str) -> Record:
        try:
            return cls(**json.loads(cls.path(home, role).read_text()))
        except FileNotFoundError:
            return cls(role=role)

    def save(self, home: Path) -> None:
        path = self.path(home, self.role)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}


def write_json(path: Path, value: dict, *, private: bool = False) -> None:
    """Write ``value`` to ``path``, replacing it atomically; ``private``: readable by its owner only from the start."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600 if private else 0o644)
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def invalid(path: Path, exc: ValidationError) -> SystemExit:
    """The stop for a configuration that would not be valid: each problem's place and reason, never the values, which
    may hold keys."""
    problems = "\n".join(f"  {'.'.join(str(p) for p in e['loc']) or 'the configuration'}: {e['msg']}"
                         for e in exc.errors(include_url=False, include_input=False))
    return SystemExit(f"{path} would not be valid; fix or remove it:\n{problems}")


# ---- wallet and chain ----

def network_and_netuid(args: argparse.Namespace, existing: dict) -> tuple[str, int]:
    network = args.network or existing.get("network") or "finney"
    try:
        chain_network(network)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    default_netuid = existing.get("netuid") if network == existing.get("network") else None
    netuid = args.netuid or default_netuid or (TESTNET_NETUID if network == "test" else MAINNET_NETUID)
    return network, netuid


def choose_wallet(ui: Prompter, args: argparse.Namespace, existing: dict) -> tuple[str, str]:
    """The wallet and hotkey: given, configured before, or chosen among the hotkeys on this machine."""
    found = host.wallets()
    previous = (existing.get("wallet_name"), existing.get("hotkey_name"))
    if args.wallet or args.hotkey:
        wallet = args.wallet or previous[0] or "default"
        return wallet, args.hotkey or (previous[1] if wallet == previous[0] else None) or "default"
    if not found and None in previous:
        raise SystemExit(
            "no hotkey on this machine. Copy the hotkey and coldkeypub.txt of a registered wallet here (never the "
            "coldkey):\n  ~/.bittensor/wallets/<wallet>/coldkeypub.txt\n  ~/.bittensor/wallets/<wallet>/hotkeys/"
            "<hotkey>\nthen run this again.")
    options = [f"{w}/{h}" for w, h in found]
    default = "/".join(previous) if None not in previous else options[0]
    if default not in options:
        options.insert(0, default)
    chosen = ui.choose("Wallet/hotkey", options, default)
    wallet, _, hotkey = chosen.partition("/")
    return wallet, hotkey or "default"


def load_hotkey(wallet_name: str, hotkey_name: str):
    """The wallet with its hotkey loaded; checks coldkeypub.txt is there. The coldkey is never opened."""
    import bittensor as bt

    wallet = bt.Wallet(name=wallet_name, hotkey=hotkey_name)
    if not wallet.hotkey_file.exists_on_device():
        raise SystemExit(f"hotkey file {wallet.hotkey_file.path} not found; copy it from the machine holding the "
                         "wallet")
    if not wallet.coldkeypub_file.exists_on_device():
        raise SystemExit(f"{wallet.coldkeypub_file.path} not found; copy it from the machine holding the wallet "
                         "(the public key only, never the coldkey)")
    try:
        hotkey = wallet.hotkey
    except Exception as exc:
        raise SystemExit(f"cannot load hotkey {wallet_name}/{hotkey_name}: {exc}") from None
    info(f"hotkey {wallet_name}/{hotkey_name}: {hotkey.ss58_address}")
    if wallet.coldkey_file.exists_on_device():
        warn("a coldkey is stored on this machine; it is not needed here, keep it offline")
    return wallet


def register_command(network: str, netuid: int, wallet: str, hotkey: str) -> str:
    extra = "" if network == "finney" else f" --network {network}"
    return f"btcli subnet register --netuid {netuid} --wallet.name {wallet} --wallet.hotkey {hotkey}{extra}"


def connect(network: str):
    import bittensor as bt

    try:
        return bt.Subtensor(network=network)
    except Exception as exc:
        raise SystemExit(f"cannot connect to the {network} chain: {exc}") from None


# ---- running and waiting ----

def operator() -> str:
    return getpass.getuser()


def manager_for(args: argparse.Namespace, record: Record) -> services.Manager | None:
    """The service manager to install the services with; None with ``--no-service``, which needs no root. One copy
    of each service runs: system services set up before are refused with ``--no-service``, and replaced by those of
    another manager otherwise."""
    previous = services.by_name(record.manager)
    if args.no_service:
        present = [n for n in record.services if previous is not None and previous.path(n).exists()]
        if present:
            raise SystemExit(f"the {record.role} runs as {previous.name} services set up before ({', '.join(present)}):"
                             f" remove them first ({uninstall_command(record.role)}; the configuration and state "
                             "stay), or run this without --no-service to keep them")
        return None
    manager = services.detect()
    if manager is None:
        raise SystemExit("neither systemd nor supervisord found: install supervisor (apt-get install supervisor) or "
                         "pass --no-service and run the printed commands under your own process manager")
    if previous is not None and previous.name != manager.name:
        previous.remove(record.services)
    elif previous is None and record.services:
        warn(f"set up before to run under a process manager of your own: stop {', '.join(record.services)} there "
             f"(pm2 delete {' '.join(record.services)}), or two copies run")
    return manager


def uninstall_command(role: str) -> str:
    return f"bash {INSTALLER} uninstall" if INSTALLER.exists() else f"verathos-{role} uninstall"


def show_run(what: str, run: list[services.Service], auto_update: bool = False) -> None:
    """How to run ``run`` under a process manager the operator runs, which restarts each when it exits."""
    step("Run (no services installed)")
    info(f"start {what} under your process manager, which restarts each when it exits"
         + (" (a self-update exits to start the new release)" if auto_update else "") + "; with pm2:")
    for s in run:
        print(f"      {services.pm2_command(s)}", flush=True)
    print("      pm2 save", flush=True)
    info("with another, each as:")
    for s in run:
        print(f"      {services.shell_command(s)}", flush=True)
    info(f"after an update by hand or a change of the configuration, restart them (pm2 restart "
         f"{' '.join(s.name for s in run)})")


def script_commands(role: str, home: Path, options: str = "") -> list[str]:
    """How to update and uninstall this setup with the installer of this checkout."""
    script = INSTALLER
    if not script.exists():
        return []
    extra = "" if home == DEFAULT_HOME.expanduser() else f" --home {shlex.quote(str(home))}"
    return [f"update:  bash {script} {role} --yes{options}{extra}", f"uninstall: bash {script} uninstall{extra}"]


def without_credentials(url: str) -> str:
    """An http(s) URL without the user name and password it may carry (``https://<token>@host/...``)."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or "@" not in parts.netloc:
        return url
    return urlunsplit(parts._replace(netloc=parts.netloc.rpartition("@")[2]))


def code_source(checkout: Path = INSTALLER.parents[1]) -> str | None:
    """Where the code of ``checkout`` came from, as the installer records it: the git origin (a URL, without its
    credentials, or a git bundle), else the tarball; None if neither is recorded."""
    if (checkout / ".git").exists():
        try:
            origin = subprocess.run(["git", "-C", str(checkout), "remote", "get-url", "origin"], capture_output=True,
                                    text=True, timeout=30).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
        return without_credentials(origin) or None
    try:
        return (checkout / SOURCE_FILE).read_text().strip() or None
    except OSError:
        return None


def reinstall_command(role: str, home: Path, config: Path, *, no_service: bool = False) -> str:
    """The installer's command that sets ``role`` up again with the configuration (at ``config``) and state that
    ``uninstall`` keeps, from the source the code came from."""
    words = ["install.sh", role, "--yes", *(["--no-service"] if no_service else [])]
    if source := code_source():
        words += ["--source", source]
    if config != home / f"{role}.json":
        words += ["--config", str(config)]
    if home != DEFAULT_HOME.expanduser():
        words += ["--home", str(home)]
    return shlex.join(words)


def engine_state(why: str | None, served: set[str], pinned: str, codes: set[str] = frozenset()) -> str:
    """What an engine serves, for a setup's output: why it does not serve (``why``), else the revisions it names
    (``served``) against the one the catalog pins, and the engine code it names (``codes``, ``verathos-engine``'s)
    against this release's."""
    from verathos.engine import code_version

    if why is not None:
        return why
    if pinned in served:
        line = f"serves the pinned revision {pinned[:12]}"
    elif served:
        line = (f"serves revision {', '.join(r[:12] for r in sorted(served))}, the catalog pins {pinned[:12]}: not "
                "used until it serves the pinned one")
    else:
        line = "names no revision"
    if codes:
        own = code_version()
        line += (f", engine code {own} (this release's)" if own in codes else
                 f", engine code {', '.join(sorted(codes))}, not this release's {own}: update its machine")
    return line


def engine_ready(port: int) -> bool:
    try:
        return httpx.get(f"http://127.0.0.1:{port}/v1/models", timeout=5).status_code == 200
    except httpx.HTTPError:
        return False


def show_tail(manager: services.Manager, name: str, lines: list[str], count: int = 30) -> None:
    print(f"    last lines of {name}:")
    for line in lines[-count:]:
        print(f"      {line}")
    print(f"    full log: {manager.log_command(name)}")


class Health:
    """Whether a service keeps failing: a third process (it restarted twice), or a minute without one."""

    DOWN_S = 60.0

    def __init__(self, manager: services.Manager, name: str) -> None:
        self.manager, self.name = manager, name
        self.pids: set[int] = set()
        self.down_since: float | None = None

    def failing(self) -> bool:
        pid = self.manager.pid(self.name)
        if pid:
            self.pids.add(pid)
            self.down_since = None
        elif self.down_since is None:
            self.down_since = time.monotonic()  # the clock starts
        elif time.monotonic() - self.down_since > self.DOWN_S:
            return True
        return len(self.pids) >= 3


def wait_engines(manager: services.Manager, engines: dict[str, int], timeout_s: float) -> None:
    """Wait until every engine (service name -> port) answers; stops with its log if one keeps failing."""
    logs: dict[str, list[str]] = {name: [] for name in engines}
    health = {name: Health(manager, name) for name in engines}
    pending = dict(engines)
    started = time.monotonic()
    while pending:
        for name, port in list(pending.items()):
            logs[name] += manager.lines(name)
            if engine_ready(port):
                info(f"{name} is ready ({time.monotonic() - started:.0f} s)")
                del pending[name]
            elif health[name].failing():
                show_tail(manager, name, logs[name])
                raise SystemExit(f"{name} keeps stopping")
        if pending and time.monotonic() - started > timeout_s:
            for name in pending:
                show_tail(manager, name, logs[name])
            raise SystemExit(f"not ready after {timeout_s / 60:.0f} minutes: {', '.join(pending)}")
        if pending:
            time.sleep(5)


def follow(manager: services.Manager, name: str, done: Callable[[str], bool], failed: Callable[[str], bool],
           timeout_s: float, required: bool = True, hide: Callable[[str], bool] = lambda line: False) -> bool:
    """Show ``name``'s log (except lines to ``hide``) until a line ``done`` accepts (True); the lines after it
    stay for the next ``follow``. Stops the setup if a line is ``failed`` or the service keeps failing; after
    ``timeout_s`` stops it too, or returns False if not ``required``."""
    deadline, health, seen = time.monotonic() + timeout_s, Health(manager, name), []
    while True:
        lines = manager.lines(name)
        for i, line in enumerate(lines):
            seen.append(line)
            if not hide(line):
                print(f"      {line}", flush=True)
            if done(line):
                manager.unread(name, lines[i + 1:])
                return True
            if failed(line):
                print(f"    full log: {manager.log_command(name)}")
                raise SystemExit(f"{name} failed")
        if health.failing():
            show_tail(manager, name, seen)
            raise SystemExit(f"{name} keeps stopping")
        if time.monotonic() > deadline:
            if not required:
                return False
            raise SystemExit(f"{name} did not get there within {timeout_s / 60:.0f} minutes; it keeps running: "
                             f"{manager.log_command(name)}")
        time.sleep(2)


def main(role: str, command: str, argv: list[str]) -> None:
    """Run ``init`` or ``uninstall`` and exit (the chain SDK leaves threads that would keep the process alive)."""
    if command == "init":
        from verathos.install import miner, reference, validator

        run = {"miner": miner, "validator": validator, "reference": reference}[role].init
    else:
        def run(args: list[str]) -> None:
            uninstall(role, args)
    try:
        run(argv)
        code = 0
    except SystemExit as exc:
        if isinstance(exc.code, str):
            print(f"\nverathos-{role} {command}: {exc.code}", file=sys.stderr)
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    except KeyboardInterrupt:
        print(f"\nverathos-{role} {command}: interrupted; run it again to continue", file=sys.stderr)
        code = 130
    except Exception:
        traceback.print_exc()
        print(f"\nverathos-{role} {command} failed; run it again once the cause is fixed", file=sys.stderr)
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


# ---- uninstall ----

def uninstall(role: str, argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog=f"verathos-{role} uninstall",
                                     description=f"Stop and remove the {role}'s services. Wallets are never touched.")
    parser.add_argument("--purge", action="store_true", help="Also delete the configuration and state (scores, "
                        "measurements, canary secret)")
    parser.add_argument("--home", type=Path, default=DEFAULT_HOME)
    parser.add_argument("--yes", "-y", action="store_true", help=argparse.SUPPRESS)  # nothing is asked
    args = parser.parse_args(argv)
    home = args.home.expanduser().absolute()
    record = Record.load(home, role)
    manager = services.by_name(record.manager)
    if manager is not None and record.services:
        manager.remove(record.services)
        info(f"removed services: {', '.join(record.services)}")
    elif record.services:
        info(f"stop what runs under your process manager (pm2: pm2 delete {' '.join(record.services)})")
    from verathos.install import firewall

    if record.firewall or (role == "miner" and firewall.installed()):
        removed = firewall.remove()
        info(f"removed the firewall: {', '.join(removed)}" if removed else "no firewall to remove")
    still_used = {path for other in ROLES if other != role for path in Record.load(home, other).paths}
    for path in record.paths:
        if path not in still_used:
            shutil.rmtree(path, ignore_errors=True)
            info(f"removed {path}")
    if not still_used:
        for name in ("engines", "hf"):
            shutil.rmtree(home / name, ignore_errors=True)
        info(f"removed engines and model weights in {home}")
    Record.path(home, role).unlink(missing_ok=True)
    config = Path(record.config) if record.config else home / f"{role}.json"
    if args.purge:
        state = Path(read_json(config).get("state_dir", home / role)).expanduser()
        extra = [home / ENGINES_FILE] if role == "reference" else []  # it holds the callers' key
        for path in (config, state, *extra):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
            info(f"removed {path}")
    else:
        info(f"kept the configuration and state in {home} (--purge deletes them); the installer sets the {role} up "
             "again with them:")
        no_service = record.manager is None and bool(record.services)
        print(f"      {reinstall_command(role, home, config, no_service=no_service)}", flush=True)
