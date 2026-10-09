"""The images of the standard request mix: drawn from a seed and encoded as canonical PNG.

A request of the mix that carries images draws a 32-byte seed from its own randomness; everything about its images
follows from that seed, so every image is new and the load generator (``native/loadgen/src/images.rs``) makes the
same bytes from the same seed. The seed is stretched by SHA-256 in counter mode (``Stream``). The request's images
share one background colour, from ``BACKGROUNDS``; each image has its width and height drawn from the mix's sides and
one to ``MAX_RECTANGLES`` rectangles in other colours of ``COLOURS``, each up to 3/8 of the image's width and height,
so the background stays the larger part.

The PNG is canonical as the gateway makes images (``protocol.media``): 8-bit RGB, not interlaced, its image data
right after its header, its sides as drawn. Its image data is one deflate block with the fixed Huffman codes: a row
that repeats the row above is filtered with Up (all zeros), any other row is unfiltered and coded as runs of one
colour (each run's first pixel as literals, the rest as copies three bytes back).
"""

from __future__ import annotations

import base64
import hashlib
import struct
import zlib
from collections.abc import Sequence
from dataclasses import dataclass

from verathos.protocol.media import PNG

SEED_BYTES = 32
# white, black, blue, yellow, grey and green (a request's background is one of these), red and orange
COLOURS = ((250, 250, 250), (20, 20, 20), (40, 90, 220), (240, 210, 40), (150, 150, 150), (40, 170, 70),
           (220, 40, 40), (245, 140, 30))
BACKGROUNDS = ("white", "black", "blue", "yellow", "grey", "green")  # the names of the first COLOURS
MAX_RECTANGLES = 3
_PNG_START = b"\x89PNG\r\n\x1a\n"
_ZLIB_HEADER = b"\x78\x01"  # deflate with a 32 KiB window
_OVERHEAD = 63  # the signature, the IHDR, IDAT and IEND chunks without the image data, the zlib header and checksum
_LENGTH_BASE = (3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195,
                227, 258)
_LENGTH_EXTRA = (0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0)
_MAX_COPY = 258
_RUN_BITS = 45  # a run's first pixel (three literals of at most 9 bits) and its last, shorter copy (at most 18 bits)
_FULL_COPY_BITS = 13  # a copy of 258 bytes, three bytes or one byte back
_UP_ROW_BITS = 52  # an Up row's two literals and its last two copies, at most


class Stream:
    """Bytes drawn from a seed: SHA-256 of the seed and a 64-bit big-endian block counter, block after block."""

    def __init__(self, seed: bytes) -> None:
        self._seed, self._counter, self._block = seed, 0, b""

    def take(self, n: int) -> bytes:
        while len(self._block) < n:
            self._block += hashlib.sha256(self._seed + self._counter.to_bytes(8, "big")).digest()
            self._counter += 1
        out, self._block = self._block[:n], self._block[n:]
        return out

    def below(self, n: int) -> int:
        """A number in [0, ``n``): the next 8 bytes as a big-endian number, times ``n``, over 2 ** 64."""
        return int.from_bytes(self.take(8), "big") * n >> 64


@dataclass(frozen=True)
class Rectangle:
    x: int
    y: int
    width: int
    height: int
    colour: int  # in COLOURS


def item(png: bytes) -> dict[str, str]:
    """An image as a request carries it."""
    return {"content_type": PNG, "base64": base64.b64encode(png).decode()}


def draw(seed: bytes, count: tuple[int, int], sides: Sequence[int]) -> list[bytes]:
    """The PNGs of a request's images: from ``count[0]`` to ``count[1]`` of them, each side from ``sides``."""
    stream = Stream(seed)
    images = count[0] + stream.below(count[1] - count[0] + 1)
    background = stream.below(len(BACKGROUNDS))
    pngs = []
    for _ in range(images):
        width, height = sides[stream.below(len(sides))], sides[stream.below(len(sides))]
        rectangles = []
        for _ in range(1 + stream.below(MAX_RECTANGLES)):
            w = max(1, width // 8) + stream.below(width // 4 + 1)
            h = max(1, height // 8) + stream.below(height // 4 + 1)
            x, y = stream.below(width - w + 1), stream.below(height - h + 1)
            colour = stream.below(len(COLOURS) - 1)
            rectangles.append(Rectangle(x, y, w, h, colour + (colour >= background)))
        pngs.append(png(width, height, background, rectangles))
    return pngs


def _runs(width: int, row: int, background: int, rectangles: Sequence[Rectangle]) -> list[tuple[int, int]]:
    """A row's pixels as (colour, length) runs, each as long as it goes: the background under the rectangles that
    cross the row, the later ones on top."""
    crossing = [r for r in rectangles if r.y <= row < r.y + r.height]
    edges = sorted({0, width, *(r.x for r in crossing), *(r.x + r.width for r in crossing)})
    out: list[tuple[int, int]] = []
    for start, end in zip(edges, edges[1:]):
        colour = next((r.colour for r in reversed(crossing) if r.x <= start < r.x + r.width), background)
        if out and out[-1][0] == colour:
            out[-1] = (colour, out[-1][1] + end - start)
        else:
            out.append((colour, end - start))
    return out


def _reversed(code: int, bits: int) -> int:
    return int(f"{code:0{bits}b}"[::-1], 2)


def _literal(value: int) -> tuple[int, int]:
    """(bits, as written; their count) of a literal byte in the fixed Huffman code."""
    return (_reversed(0x30 + value, 8), 8) if value < 144 else (_reversed(0x190 + value - 144, 9), 9)


def _copy(length: int, distance: int) -> tuple[int, int]:
    """A copy of ``length`` (3 to 258) bytes from ``distance`` (1 to 4) bytes back in the fixed Huffman code."""
    index = max(i for i, base in enumerate(_LENGTH_BASE) if base <= length)
    symbol, extra = 257 + index, _LENGTH_EXTRA[index]
    code, bits = (_reversed(symbol - 256, 7), 7) if symbol < 280 else (_reversed(0xC0 + symbol - 280, 8), 8)
    code |= (length - _LENGTH_BASE[index]) << bits
    return code | _reversed(distance - 1, 5) << bits + extra, bits + extra + 5


_LITERALS = [_literal(v) for v in range(256)]
_END = (0, 7)  # the end of the block


def _copies(length: int) -> list[int]:
    """The lengths ``length`` (at least 3) bytes are copied in: 258 each, but the last two shortened as far as needed
    to keep each at least 3."""
    out = []
    while length:
        step = min(length, _MAX_COPY)
        if 0 < length - step < 3:
            step = length - 3
        out.append(step)
        length -= step
    return out


class _Bits:
    """Bits written into bytes from the least significant bit up, as deflate writes them."""

    def __init__(self) -> None:
        self.out, self._value, self._count = bytearray(), 0, 0

    def put(self, value: int, count: int) -> None:
        self._value |= value << self._count
        self._count += count
        if self._count >= 4096:
            whole = self._count // 8
            self.out += (self._value & ((1 << 8 * whole) - 1)).to_bytes(whole, "little")
            self._value >>= 8 * whole
            self._count -= 8 * whole

    def done(self) -> bytes:
        self.out += self._value.to_bytes((self._count + 7) // 8, "little")
        return bytes(self.out)


def _chain(codes: Sequence[tuple[int, int]]) -> tuple[int, int]:
    value = count = 0
    for code, bits in codes:
        value |= code << count
        count += bits
    return value, count


def _unfiltered(row: Sequence[tuple[int, int]]) -> tuple[int, int]:
    codes = [_LITERALS[0]]
    for colour, length in row:
        codes += [_LITERALS[v] for v in COLOURS[colour]]
        codes += [_copy(n, 3) for n in _copies(3 * (length - 1))] if length > 1 else []
    return _chain(codes)


def _up(width: int) -> tuple[int, int]:
    """A row that repeats the one above, filtered with Up: the filter type and zeros."""
    zeros = 3 * width - 1
    return _chain([_LITERALS[2], _LITERALS[0],
                   *([_copy(n, 1) for n in _copies(zeros)] if zeros >= 3 else [_LITERALS[0]] * zeros)])


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def png(width: int, height: int, background: int, rectangles: Sequence[Rectangle]) -> bytes:
    """The canonical PNG of a background colour (in ``COLOURS``) with ``rectangles`` on it (see the module)."""
    bits, checksum, above = _Bits(), 1, None
    bits.put(0b011, 3)  # the last block, with the fixed Huffman codes
    up, up_bytes = _up(width), b"\x02" + bytes(3 * width)
    changes = {0, *(r.y for r in rectangles), *(r.y + r.height for r in rectangles)}  # rows that may differ
    for y in range(height):
        row = _runs(width, y, background, rectangles) if y in changes else above
        if row == above:
            bits.put(*up)
            checksum = zlib.adler32(up_bytes, checksum)
            continue
        bits.put(*_unfiltered(row))
        checksum = zlib.adler32(b"\0" + b"".join(bytes(COLOURS[c]) * n for c, n in row), checksum)
        above = row
    bits.put(*_END)
    data = _ZLIB_HEADER + bits.done() + checksum.to_bytes(4, "big")
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return _PNG_START + _chunk(b"IHDR", header) + _chunk(b"IDAT", data) + _chunk(b"IEND", b"")


def png_bound(width: int, height: int) -> int:
    """The most bytes ``png`` writes for an image of this size: at most 2 x ``MAX_RECTANGLES`` + 1 of its rows are
    coded as runs (the first, and where a rectangle starts or ends), each of at most 2 x ``MAX_RECTANGLES`` + 1 runs;
    the others repeat the row above."""
    coded_rows = most_runs = 2 * MAX_RECTANGLES + 1
    coded = 8 + _RUN_BITS * most_runs + _FULL_COPY_BITS * (3 * width // _MAX_COPY)
    up = _UP_ROW_BITS + _FULL_COPY_BITS * ((3 * width - 1) // _MAX_COPY)
    bits = 3 + coded_rows * coded + height * up + _END[1]
    return _OVERHEAD + (bits + 7) // 8
