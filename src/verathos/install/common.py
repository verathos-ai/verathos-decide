"""What the miner and validator setups share: questions, the record of what was set up, wallet and chain checks,
waiting for services, and uninstalling."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import shutil
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx

from verathos.install import host, services

DEFAULT_HOME = Path("~/.verathos")
TESTNET_NETUID = 405
MAINNET_NETUID = 96


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


def common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--yes", "-y", action="store_true", help="Take the default (or the given option) for every "
                        "question")
    parser.add_argument("--wallet", help="Wallet name (directory under ~/.bittensor/wallets)")
    parser.add_argument("--hotkey", help="Hotkey name in that wallet")
    parser.add_argument("--network", help="finney (default), test, or a subtensor websocket URL")
    parser.add_argument("--netuid", type=int, help=f"Subnet (default {MAINNET_NETUID}; {TESTNET_NETUID} on test)")
    parser.add_argument("--no-service", action="store_true", help="Set up and print the commands to run, without "
                        "installing services")
    parser.add_argument("--home", type=Path, default=DEFAULT_HOME, help="Configuration, state, engines and weights "
                        "(default ~/.verathos)")


# ---- the record of what was set up ----

@dataclass
class Record:
    """What a setup installed, kept in ``<home>/install/<role>.json``: the defaults of the next run and what
    ``uninstall`` removes."""

    role: str
    manager: str | None = None
    services: list[str] = field(default_factory=list)
    config: str = ""
    engines: dict[str, dict] = field(default_factory=dict)  # per endpoint or model: model, device, port
    paths: list[str] = field(default_factory=list)  # engine checkouts used

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


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


# ---- wallet and chain ----

def network_and_netuid(args: argparse.Namespace, existing: dict) -> tuple[str, int]:
    network = args.network or existing.get("network") or "finney"
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
    if args.no_service:
        return None
    manager = services.detect()
    if manager is None:
        raise SystemExit("neither systemd nor supervisord found: install supervisor (apt-get install supervisor) or "
                         "pass --no-service and run the printed commands under your own supervisor")
    if record.manager and record.manager != manager.name:
        previous = services.by_name(record.manager)
        if previous is not None:
            previous.remove(record.services)
    return manager


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


def wait_engines(manager: services.Manager, engines: dict[str, int], timeout_s: float) -> None:
    """Wait until every engine (service name -> port) answers; stops with its log if one stopped twice."""
    logs: dict[str, list[str]] = {name: [] for name in engines}
    pending = dict(engines)
    deadline, stops, was_active = time.monotonic() + timeout_s, dict.fromkeys(engines, 0), dict.fromkeys(engines, True)
    started = time.monotonic()
    while pending:
        for name, port in list(pending.items()):
            logs[name] += manager.new_lines(name)
            if engine_ready(port):
                info(f"{name} is ready ({time.monotonic() - started:.0f} s)")
                del pending[name]
                continue
            active = manager.active(name)
            stops[name] += was_active[name] and not active
            was_active[name] = active
            if stops[name] >= 2:
                show_tail(manager, name, logs[name])
                raise SystemExit(f"{name} keeps stopping")
        if pending and time.monotonic() > deadline:
            for name in pending:
                show_tail(manager, name, logs[name])
            raise SystemExit(f"not ready after {timeout_s / 60:.0f} minutes: {', '.join(pending)}")
        if pending:
            time.sleep(5)


def follow(manager: services.Manager, name: str, done: Callable[[str], bool], failed: Callable[[str], bool],
           timeout_s: float, required: bool = True, hide: Callable[[str], bool] = lambda line: False) -> bool:
    """Show ``name``'s log (except lines to ``hide``) until a line ``done`` accepts (True). Stops the setup if a
    line is ``failed`` or the service stopped twice; after ``timeout_s`` stops it too, or returns False if not
    ``required``."""
    deadline, stops, was_active, seen = time.monotonic() + timeout_s, 0, True, []
    while True:
        for line in manager.new_lines(name):
            seen.append(line)
            if not hide(line):
                print(f"      {line}", flush=True)
            if done(line):
                return True
            if failed(line):
                print(f"    full log: {manager.log_command(name)}")
                raise SystemExit(f"{name} failed")
        active = manager.active(name)
        stops += was_active and not active
        was_active = active
        if stops >= 2:
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
        from verathos.install import miner, validator

        run = (miner if role == "miner" else validator).init
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
    home = args.home.expanduser()
    record = Record.load(home, role)
    manager = services.by_name(record.manager)
    if manager is not None and record.services:
        manager.remove(record.services)
        info(f"removed services: {', '.join(record.services)}")
    other = Record.load(home, "validator" if role == "miner" else "miner")
    still_used = set(other.paths)
    for path in record.paths:
        if path not in still_used:
            shutil.rmtree(path, ignore_errors=True)
            info(f"removed {path}")
    if not other.paths:
        for name in ("engines", "hf"):
            shutil.rmtree(home / name, ignore_errors=True)
        info(f"removed engines and model weights in {home}")
    Record.path(home, role).unlink(missing_ok=True)
    if args.purge:
        config = Path(record.config) if record.config else home / f"{role}.json"
        state = Path(read_json(config).get("state_dir", home / role)).expanduser()
        for path in (config, state):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
            info(f"removed {path}")
    else:
        info(f"kept the configuration and state in {home} (--purge deletes them)")
