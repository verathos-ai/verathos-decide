"""What the machine has: GPUs, memory, its public IP address, free ports and the wallets on it."""

from __future__ import annotations

import ipaddress
import re
import shutil
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path

import httpx

GiB = 1 << 30
MiB = 1 << 20
IP_SERVICES = ("https://api.ipify.org", "https://checkip.amazonaws.com", "https://ifconfig.me/ip")
WALLETS = Path("~/.bittensor/wallets")


@dataclass(frozen=True)
class Gpu:
    index: int
    name: str
    memory_total: int  # bytes
    memory_free: int  # bytes

    def describe(self) -> str:
        return f"GPU {self.index} ({self.name}, {self.memory_total / GiB:.0f} GiB)"


def parse_gpus(csv: str) -> list[Gpu]:
    """``nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits`` output; a
    memory nvidia-smi does not report (``[N/A]``) counts as none."""

    def mib(value: str) -> int:
        try:
            return int(float(value)) * MiB
        except ValueError:
            return 0

    gpus = []
    for line in csv.strip().splitlines():
        index, rest = line.split(",", 1)
        name, total, free = rest.rsplit(",", 2)
        gpus.append(Gpu(int(index), name.strip(), mib(total.strip()), mib(free.strip())))
    return gpus


def gpus() -> list[Gpu]:
    """The NVIDIA GPUs (none without a working driver)."""
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True,
                             timeout=30).stdout
    except (subprocess.SubprocessError, OSError):
        return []
    return parse_gpus(out)


def driver_cuda_version() -> tuple[int, int] | None:
    """The newest CUDA version the installed driver supports (``nvidia-smi``'s header), or None."""
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out)
    return (int(match[1]), int(match[2])) if match else None


def memory_available(meminfo: Path = Path("/proc/meminfo"), cgroups: Path = Path("/sys/fs/cgroup"),
                     own: Path = Path("/proc/self/cgroup")) -> tuple[int, int]:
    """(total, available) RAM in bytes: the machine's, within this process's control group limit where one is set (a
    container's: /proc/meminfo shows the host's memory there)."""
    values = {}
    for line in meminfo.read_text().splitlines():
        key, _, rest = line.partition(":")
        values[key] = int(rest.split()[0]) * 1024
    total, available = values["MemTotal"], values["MemAvailable"]
    limited = memory_limit(cgroups, own)
    if limited is not None and limited[0] < total:
        limit, used = limited
        total, available = limit, max(0, min(available, limit - used))
    return total, available


def memory_limit(cgroups: Path = Path("/sys/fs/cgroup"),
                 own: Path = Path("/proc/self/cgroup")) -> tuple[int, int] | None:
    """(limit, used) of the memory this process's control group may use, the tightest limit along its path (cgroup
    v2), or its memory group's (v1); used leaves out the page cache the kernel can reclaim (inactive files). None
    where no limit is set or none can be read."""
    try:
        lines = own.read_text().splitlines()
    except OSError:
        return None
    found: list[tuple[int, int]] = []
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        _, controllers, path = parts
        if controllers == "":  # v2: memory.max at each level from the group up to the root
            group = cgroups / path.lstrip("/")
            for level in [group, *group.parents]:
                if not level.is_relative_to(cgroups):
                    break
                limit = _read_bytes(level / "memory.max")
                if limit is not None:
                    found.append((limit, _used(level / "memory.current", level / "memory.stat", "inactive_file")))
        elif "memory" in controllers.split(","):  # v1
            group = cgroups / "memory" / path.lstrip("/")
            limit = _read_bytes(group / "memory.limit_in_bytes")
            if limit is not None and limit < 1 << 60:  # v1 writes "no limit" as a huge number
                found.append((limit, _used(group / "memory.usage_in_bytes", group / "memory.stat",
                                           "total_inactive_file")))
    return min(found, default=None, key=lambda limit_used: limit_used[0] - limit_used[1])


def _read_bytes(path: Path) -> int | None:
    try:
        text = path.read_text().strip()
    except OSError:
        return None
    return int(text) if text.isdigit() else None  # "max": no limit


def _used(current: Path, stat: Path, inactive: str) -> int:
    used = _read_bytes(current) or 0
    try:
        for line in stat.read_text().splitlines():
            key, _, value = line.partition(" ")
            if key == inactive and value.strip().isdigit():
                return max(0, used - int(value))
    except OSError:
        pass
    return used


def detect_public_ip(timeout_s: float = 5.0) -> str | None:
    """This machine's public IPv4 address as seen from outside, or None."""
    for url in IP_SERVICES:
        try:
            text = httpx.get(url, timeout=timeout_s).text.strip()
            address = ipaddress.ip_address(text)
        except (httpx.HTTPError, ValueError):
            continue
        if address.version == 4 and address.is_global:
            return str(address)
    return None


def port_free(port: int, host: str = "0.0.0.0") -> bool:
    """Whether a listening socket can bind ``host:port`` now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
        except OSError:
            return False
    return True


def wallets(root: Path = WALLETS) -> list[tuple[str, str]]:
    """(wallet, hotkey) pairs whose hotkey file is on this machine."""
    root = root.expanduser()
    if not root.is_dir():
        return []
    return sorted((w.name, h.name) for w in root.iterdir() if (w / "hotkeys").is_dir()
                  for h in (w / "hotkeys").iterdir() if h.is_file() and not h.name.endswith("pub.txt"))
