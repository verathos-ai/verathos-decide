#!/usr/bin/python3 -I
"""The kernel layer of the miner's firewall in nftables, and the root helper that applies it.

Table ``inet verathos_miner`` drops TCP to the miner's service port (set ``ports``) from every source but loopback and
the allowed addresses (sets ``allow4``, ``allow6``), counting the drops (counter ``dropped``). Its chain hooks
prerouting at priority -300, before connection tracking and any port forwarding, so a dropped packet costs no state
and a forwarded port is covered as well. Other ports and other traffic pass untouched; other tables still apply to
what this one accepts. An apply replaces the whole table in one transaction, so the port is never open or closed in
between, and only when the table is missing or differs from what it would write.

The installer copies this file to ``HELPER``, owned by root, and lets the miner's user run exactly ``HELPER apply
<port>``, ``HELPER status`` and ``HELPER remove`` through sudo without a password (``SUDOERS``). It runs on the
system's Python and imports only its standard library, never the miner's environment:

    verathos-firewall apply <port>   allowed addresses on stdin, one per line (IPv4 or IPv6, normal form)
    verathos-firewall status         the table as JSON: ports, allowed, dropped_packets
    verathos-firewall remove         delete the table: the port is open again
"""

from __future__ import annotations

import ipaddress
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterable

TABLE = "verathos_miner"
HELPER = "/usr/local/sbin/verathos-firewall"
SUDOERS = "/etc/sudoers.d/verathos-firewall"
SYSTEM_PATH = "/usr/sbin:/usr/bin:/sbin:/bin"
HOOK = ("prerouting", -300)
RULES = 4
MAX_ADDRESSES = 8192  # far above a subnet's callers
MAX_INPUT_BYTES = 1 << 20


class FirewallError(Exception):
    """nft refused, or is missing or not permitted."""


Run = Callable[[list[str], str | None], subprocess.CompletedProcess]


def run_nft(args: list[str], stdin: str | None = None) -> subprocess.CompletedProcess:
    """``nft`` with ``args`` (and ``stdin``), from the system's directories. Raises FirewallError if it is missing."""
    nft = shutil.which("nft", path=SYSTEM_PATH)
    if nft is None:
        raise FirewallError("nft not found (install nftables)")
    return subprocess.run([nft, *args], input=stdin, capture_output=True, text=True, timeout=30)


def addresses(lines: Iterable[str]) -> list[str]:
    """The allowed addresses: each line an IPv4 or IPv6 unicast address in its normal form (loopback always passes);
    IPv4 first, then IPv6, each in order, without repeats. Raises ValueError for anything else, for none and for more
    than ``MAX_ADDRESSES``."""
    out = set()
    for line in lines:
        value = line.strip()
        if not value:
            continue
        try:
            address = ipaddress.ip_address(value)
        except ValueError:
            raise ValueError(f"{value[:64]!r} is not an IP address") from None
        if str(address) != value:
            raise ValueError(f"{value!r}: write the address as {address}")
        if address.is_multicast or address.is_unspecified or address.is_loopback or (
                address.version == 6 and address.ipv4_mapped is not None):
            raise ValueError(f"{value} is not a unicast address")
        out.add(address)
    if not out:
        raise ValueError("no address: an empty allowlist would drop every caller")
    if len(out) > MAX_ADDRESSES:
        raise ValueError(f"more than {MAX_ADDRESSES} addresses")
    return [str(a) for a in sorted(out, key=lambda a: (a.version, a))]


def port_number(value: str | int) -> int:
    text = str(value)
    if not text.isdigit() or not 0 < int(text) < 65536:
        raise ValueError(f"{text[:16]!r} is not a TCP port")
    return int(text)


def ruleset(port: int, allowed: list[str]) -> str:
    """The nft script that replaces the table with one filtering ``port`` to ``allowed`` (``addresses``), in one
    transaction."""
    lines = [f"add table inet {TABLE}", f"delete table inet {TABLE}", f"table inet {TABLE} {{",
             "\tcounter dropped {", "\t}",
             "\tset ports {", "\t\ttype inet_service", f"\t\telements = {{ {port_number(port)} }}", "\t}"]
    for family, kind in (("4", "ipv4_addr"), ("6", "ipv6_addr")):
        values = [a for a in allowed if (":" in a) == (family == "6")]
        lines += [f"\tset allow{family} {{", f"\t\ttype {kind}", "\t\tflags interval",
                  *([f"\t\telements = {{ {', '.join(values)} }}"] if values else []), "\t}"]
    lines += ["\tchain prerouting {", f"\t\ttype filter hook {HOOK[0]} priority {HOOK[1]}; policy accept;",
              '\t\tiif "lo" accept',
              "\t\ttcp dport @ports ip saddr @allow4 accept",
              "\t\ttcp dport @ports ip6 saddr @allow6 accept",
              '\t\ttcp dport @ports counter name "dropped" drop', "\t}", "}", ""]
    return "\n".join(lines)


def _check(result: subprocess.CompletedProcess, what: str) -> None:
    if result.returncode != 0:
        lines = [line.strip() for line in (result.stderr or result.stdout or "").splitlines() if line.strip()]
        errors = [line for line in lines if "Error" in line] or lines
        raise FirewallError(f"{what}: {errors[0] if errors else f'nft exited with {result.returncode}'}")


def _absent(result: subprocess.CompletedProcess) -> bool:
    return result.returncode != 0 and "No such file or directory" in (result.stderr or "")


def _values(elements: list) -> list[str]:
    out = []
    for e in elements:
        value = e.get("elem", {}).get("val") if isinstance(e, dict) else e
        out.append(str(value) if isinstance(value, (str, int)) else json.dumps(e, sort_keys=True))
    return out


def parse_status(document: dict) -> dict:
    """``nft -j list table`` -> {"table": True, "ports", "allowed", "dropped_packets", "hook" ([name, priority] of its
    chain), "rules"}."""
    sets: dict[str, list[str]] = {}
    packets, hook, rules = 0, None, 0
    for item in document.get("nftables", []):
        if "set" in item:
            sets[item["set"].get("name", "")] = _values(item["set"].get("elem", []))
        elif "counter" in item and item["counter"].get("name") == "dropped":
            packets = int(item["counter"].get("packets", 0))
        elif "chain" in item:
            chain = item["chain"]
            hook = [chain.get("hook"), chain.get("prio")] if chain.get("policy") == "accept" else None
        elif "rule" in item:
            rules += 1
    return {"table": True, "ports": sorted(int(p) for p in sets.get("ports", []) if str(p).isdigit()),
            "allowed": sets.get("allow4", []) + sets.get("allow6", []), "dropped_packets": packets, "hook": hook,
            "rules": rules}


def status(run: Run = run_nft) -> dict:
    """The table's state (``parse_status``), or {"table": False}. Raises FirewallError."""
    result = run(["-j", "list", "table", "inet", TABLE], None)
    if _absent(result):
        return {"table": False}
    _check(result, "nft did not list the table")
    try:
        return parse_status(json.loads(result.stdout))
    except (ValueError, TypeError, AttributeError) as exc:
        raise FirewallError(f"nft listed the table in a form this helper does not read ({exc})") from None


def current(state: dict, port: int, allowed: list[str]) -> bool:
    """Whether the table ``state`` (``status``) is the one ``ruleset(port, allowed)`` writes."""
    return (state.get("table") is True and state.get("ports") == [port] and state.get("hook") == list(HOOK)
            and state.get("rules") == RULES and sorted(state.get("allowed", [])) == sorted(allowed))


def apply(port: int, allowed: list[str], run: Run = run_nft) -> bool:
    """Make the table filter ``port`` to ``allowed``, replacing it if it is missing or differs (a flushed ruleset, a
    changed list); -> whether it was replaced. Raises FirewallError; the table is then as before."""
    target = addresses(allowed)
    if current(status(run), port_number(port), target):
        return False
    _check(run(["-f", "-"], ruleset(port, target)), "nft refused the rules")
    return True


def remove(run: Run = run_nft) -> bool:
    """Delete the table; False if there was none. Raises FirewallError."""
    result = run(["delete", "table", "inet", TABLE], None)
    if _absent(result):
        return False
    _check(result, "nft did not delete the table")
    return True


USAGE = "usage: verathos-firewall apply <port> (addresses on stdin) | status | remove"


def main(argv: list[str] | None = None, stdin=None, run: Run = run_nft) -> int:
    """The helper's command line; exit 0, 1 (nft failed), 2 (usage or input refused)."""
    args = sys.argv[1:] if argv is None else argv
    stdin = sys.stdin if stdin is None else stdin
    try:
        if args[:1] == ["apply"] and len(args) == 2:
            port = port_number(args[1])
            text = stdin.read(MAX_INPUT_BYTES + 1)
            if len(text) > MAX_INPUT_BYTES:
                raise ValueError("input too large")
            allowed = addresses(text.splitlines())
            replaced = apply(port, allowed, run)
            print(f"port {port}: {len(allowed)} addresses allowed" + ("" if replaced else " (unchanged)"))
        elif args == ["status"]:
            print(json.dumps(status(run)))
        elif args == ["remove"]:
            print("table removed" if remove(run) else "no table")
        else:
            print(USAGE, file=sys.stderr)
            return 2
    except ValueError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    except (FirewallError, OSError, subprocess.SubprocessError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    os.umask(0o077)
    sys.exit(main())
