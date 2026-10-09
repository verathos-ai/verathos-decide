"""Setting up the kernel layer of the miner's firewall (``miner.firewall``) on a machine: the root helper
(``miner.nftables``, copied to ``HELPER``) and the sudo rule that lets the miner's user run exactly its three commands
without a password; and removing both with the table.

    verathos-miner firewall install [--config miner.json]   needs root or sudo, once
    verathos-miner firewall status
    verathos-miner firewall remove [--config miner.json]
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from verathos.install.services import privileged
from verathos.miner import nftables
from verathos.miner.firewall import STATE_FILE

HELPER = Path(nftables.HELPER)
SUDOERS = Path(nftables.SUDOERS)
PYTHON = Path("/usr/bin/python3")  # the helper's interpreter: the system's, never the miner's environment
SOURCE = Path(nftables.__file__)
_USER = re.compile(r"^[a-z_][a-z0-9_.-]{0,31}$")


class Unavailable(Exception):
    """This machine cannot have the kernel layer; the message says why."""


def sudoers(user: str, port: int) -> str:
    """The sudo rule: ``user`` runs the helper's apply for ``port``, its status and its remove as root, nothing
    else."""
    if not _USER.match(user):
        raise ValueError(f"{user!r} is not a user name sudo rules take")
    commands = [f"{HELPER} apply {nftables.port_number(port)}", f"{HELPER} status", f"{HELPER} remove"]
    return ("# The miner's firewall (verathos-miner): its user applies the allowlist, reads and removes the table.\n"
            f"{user} ALL=(root) NOPASSWD: {', '.join(commands)}\n")


def _nft() -> str | None:
    return shutil.which("nft", path=nftables.SYSTEM_PATH)


def _quiet(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(privileged(cmd), capture_output=True, text=True, **kwargs)


def _apt_install(package: str) -> None:
    if not shutil.which("apt-get"):
        raise Unavailable(f"{package} is not installed and there is no apt-get to install it")
    result = _quiet(["env", "DEBIAN_FRONTEND=noninteractive", "apt-get", "install", "-y", "-qq",
                     "--no-install-recommends", package])
    if result.returncode != 0:
        raise Unavailable(f"{package} could not be installed: {(result.stderr or result.stdout).strip()[-300:]}")


def setup(user: str, port: int) -> None:
    """Install the helper and, for a user other than root, its sudo rule (as root or through sudo). Raises
    Unavailable when this machine cannot filter: not Linux, or nftables not permitted (a container)."""
    if not sys.platform.startswith("linux"):
        raise Unavailable("nftables runs on Linux only")
    if _nft() is None:
        _apt_install("nftables")
    if not PYTHON.exists():
        _apt_install("python3")
    probe = _quiet([_nft() or "nft", "list", "tables"])
    if probe.returncode != 0:
        raise Unavailable("this machine does not permit nftables rules even to root (a container without NET_ADMIN?)")
    subprocess.run(privileged(["install", "-o", "root", "-g", "root", "-m", "755", str(SOURCE), str(HELPER)]),
                   check=True)
    if user == "root":
        return
    text = sudoers(user, port)
    with tempfile.NamedTemporaryFile("w", suffix=".sudoers", delete=False) as tmp:
        tmp.write(text)
    try:
        check = subprocess.run([shutil.which("visudo", path=nftables.SYSTEM_PATH) or "visudo", "-cqf", tmp.name],
                               capture_output=True, text=True)
        if check.returncode != 0:
            raise Unavailable(f"visudo refused the rule: {(check.stderr or check.stdout).strip()}")
        subprocess.run(privileged(["install", "-o", "root", "-g", "root", "-m", "440", tmp.name, str(SUDOERS)]),
                       check=True)
    finally:
        os.unlink(tmp.name)


def installed() -> bool:
    return HELPER.exists()


def kernel_status() -> dict | None:
    """The table as the helper reads it (``nftables.parse_status``), or None if it cannot be read."""
    if not installed():
        return None
    result = _quiet([str(HELPER), "status"])
    try:
        return json.loads(result.stdout) if result.returncode == 0 else None
    except ValueError:
        return None


def miner_state(config: Path) -> dict | None:
    """What the miner last wrote of its firewall (``miner.firewall.STATE_FILE``), or None."""
    try:
        state_dir = Path(json.loads(config.expanduser().read_text()).get("state_dir", "~/.verathos/miner"))
        return json.loads((state_dir.expanduser() / STATE_FILE).read_text())
    except (OSError, ValueError, AttributeError, TypeError):
        return None


def remove() -> list[str]:
    """Delete the table, the helper and its sudo rule (as root or through sudo); -> what was removed."""
    removed = []
    if installed() or _nft() is not None:
        result = _quiet([str(HELPER), "remove"] if installed() else [_nft() or "nft", "delete", "table", "inet",
                                                                     nftables.TABLE])
        if result.returncode == 0 and "no table" not in result.stdout:
            removed.append(f"the table inet {nftables.TABLE}")
    for path in (SUDOERS, HELPER):
        if path.exists() or _quiet(["test", "-e", str(path)]).returncode == 0:
            subprocess.run(privileged(["rm", "-f", str(path)]), check=True)
            removed.append(str(path))
    return removed


def summary(miner: dict | None, kernel: dict | None, port: int) -> str:
    """One line on the firewall of ``port``: the miner's mode (``miner_state``) and the kernel's table
    (``kernel_status``)."""
    if miner is None or miner.get("port") != port:
        return "firewall: the miner has not run with it yet"
    last = miner.get("last") or {}
    if miner.get("mode") == "enforce":
        line = (f"firewall enforcing: port {port} answers {len(miner.get('enforced', []))} addresses (the network's "
                f"callers and this hotkey's machines) and loopback, and drops the rest")
    else:
        line = f"firewall observing, dropping nothing: {miner.get('why')}"
        if last:
            line += (f" ({len(last.get('published', []))} published and {len(last.get('learned', []))} learned "
                     "caller addresses so far)")
    if kernel is None:
        return line + f"; kernel layer {miner.get('kernel', 'off')}"
    if kernel.get("table"):
        return line + (f"; kernel table on, {kernel.get('dropped_packets', 0)} packets dropped (sudo nft list table "
                       f"inet {nftables.TABLE})")
    return line + "; kernel layer ready"


def _port(config: Path) -> int:
    try:
        return int(json.loads(config.expanduser().read_text())["port"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f"cannot read the port from {config}: {exc}") from None


def _set_switch(config: Path, on: bool) -> None:
    path = config.expanduser()
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return
    if value.get("firewall", True) != on:
        value["firewall"] = on
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(value, indent=2) + "\n")
        tmp.replace(path)


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="verathos-miner firewall", description="The miner's firewall: its port "
                                     "answers only the network's callers and the hotkey's machines.")
    parser.add_argument("command", choices=["install", "status", "remove"])
    parser.add_argument("--config", type=Path, default=Path("~/.verathos/miner.json"))
    parser.add_argument("--user", default=getpass.getuser(), help="The user the miner runs as (default: this one)")
    args = parser.parse_args(argv)
    port = _port(args.config)
    if args.command == "install":
        try:
            setup(args.user, port)
        except (Unavailable, ValueError) as exc:
            raise SystemExit(f"firewall not set up: {exc}") from None
        _set_switch(args.config, True)
        print(f"firewall set up for port {port} ({HELPER}, {SUDOERS}): restart the miner to apply it")
    elif args.command == "remove":
        removed = remove()
        _set_switch(args.config, False)
        print("removed " + ", ".join(removed) if removed else "nothing to remove")
        print(f"firewall false in {args.config}: the miner runs without it")
    else:
        print(summary(miner_state(args.config), kernel_status(), port))
