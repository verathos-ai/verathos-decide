"""The userspace layer of the miner's firewall: a classic BPF filter on the service's listening socket.

The kernel runs it on each packet the socket receives, before TCP answers it, so a connection from an address not on
the list is dropped at its first packet, as the kernel layer drops it: no handshake, no TLS, no Python. Setting it
takes no privilege, so it works for a miner without root and in a container. Connections accepted before a change keep
the filter they were accepted under. Loopback's own addresses (127.0.0.1, ::1) always pass.
"""

from __future__ import annotations

import ctypes
import ipaddress
import socket
import struct
from collections.abc import Iterable

SO_ATTACH_FILTER = getattr(socket, "SO_ATTACH_FILTER", 26)
SO_DETACH_FILTER = getattr(socket, "SO_DETACH_FILTER", 27)
MAX_INSTRUCTIONS = 4096  # the kernel's limit for a classic filter
_NET = -0x100000  # SKF_NET_OFF: offsets from the network header
_LDW, _LDB, _RSH, _JEQ, _JA, _RET, _ST, _LDM = 0x20, 0x30, 0x74, 0x15, 0x05, 0x06, 0x02, 0x60
_ACCEPT, _DROP = 0xFFFFFFFF, 0


def _insn(code: int, k: int, jt: int = 0, jf: int = 0) -> bytes:
    return struct.pack("HBBI", code, jt, jf, k & 0xFFFFFFFF)


def program(allowed: Iterable[str]) -> bytes:
    """The filter: accept a packet whose source is loopback's own address or in ``allowed``, drop any other. Raises
    ValueError if it would exceed ``MAX_INSTRUCTIONS``."""
    addresses = {ipaddress.ip_address(a) for a in allowed} | {ipaddress.ip_address("127.0.0.1"),
                                                              ipaddress.ip_address("::1")}
    v4 = sorted(int(a) for a in addresses if a.version == 4)
    v6 = sorted(struct.unpack("!IIII", a.packed) for a in addresses if a.version == 6)
    body4 = [_insn(_LDW, _NET + 12)]
    for a in v4:
        body4 += [_insn(_JEQ, a, 0, 1), _insn(_RET, _ACCEPT)]
    body4.append(_insn(_RET, _DROP))
    body6 = [b for i in range(4) for b in (_insn(_LDW, _NET + 8 + 4 * i), _insn(_ST, i))]
    for words in v6:
        for i, word in enumerate(words):  # on a mismatch, past this address's remaining checks and its accept
            body6 += [_insn(_LDM, i), _insn(_JEQ, word, 0, 2 * (3 - i) + 1)]
        body6.append(_insn(_RET, _ACCEPT))
    body6.append(_insn(_RET, _DROP))
    # the IP version: 4 goes on to the IPv4 checks, anything else jumps to the IPv6 ones
    head = [_insn(_LDB, _NET), _insn(_RSH, 4), _insn(_JEQ, 4, 1, 0), _insn(_JA, len(body4))]
    instructions = head + body4 + body6
    if len(instructions) > MAX_INSTRUCTIONS:
        raise ValueError(f"{len(addresses)} addresses take {len(instructions)} filter instructions, more than "
                         f"{MAX_INSTRUCTIONS}")
    return b"".join(instructions)


def attach(sock: socket.socket, allowed: Iterable[str]) -> None:
    """Filter ``sock`` (a listening socket) to ``allowed``, replacing any filter it has. Raises ValueError (too many
    addresses) or OSError (the kernel refused)."""
    code = program(allowed)
    buffer = ctypes.create_string_buffer(code)
    sock.setsockopt(socket.SOL_SOCKET, SO_ATTACH_FILTER, struct.pack("HL", len(code) // 8, ctypes.addressof(buffer)))


def detach(sock: socket.socket) -> None:
    """Remove ``sock``'s filter, if it has one."""
    try:
        sock.setsockopt(socket.SOL_SOCKET, SO_DETACH_FILTER, 0)
    except OSError:
        pass  # none attached
