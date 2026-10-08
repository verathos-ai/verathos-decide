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
    """``nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits`` output."""
    gpus = []
    for line in csv.strip().splitlines():
        index, name, total, free = (part.strip() for part in line.split(","))
        gpus.append(Gpu(int(index), name, int(float(total)) * MiB, int(float(free)) * MiB))
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


def memory_available(meminfo: Path = Path("/proc/meminfo")) -> tuple[int, int]:
    """(total, available) RAM in bytes."""
    values = {}
    for line in meminfo.read_text().splitlines():
        key, _, rest = line.partition(":")
        values[key] = int(rest.split()[0]) * 1024
    return values["MemTotal"], values["MemAvailable"]


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


def is_public(value: str) -> bool:
    try:
        return ipaddress.ip_address(value).is_global
    except ValueError:
        return False


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
