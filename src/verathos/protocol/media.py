"""Images in SystemOne requests: what users send, the limits, and the canonical form the network passes on.

A request embeds its images (Clef's ``images``): each a base64 data URL (``data:image/png;base64,...``, likewise
``image/jpeg`` and ``image/webp``) or ``{"content_type", "base64"}``, in strict base64 (the standard alphabet with its
padding, no whitespace). Nothing in the network fetches a URL or opens a path: an image travels inside its request.

The gateway makes every image canonical before a miner receives it (``canonical_image``, in its decode workers): the
first frame, turned upright by its EXIF orientation and converted to RGB as the models' processors convert images (an
alpha channel is dropped, not composited), resized with bicubic resampling to the size ``smart_resize`` gives within
the model's pixel budget (each side a multiple of ``token_px``), and encoded without metadata: PNG sources as PNG,
JPEG and WebP sources as JPEG at quality 95 without chroma subsampling. Miners, validators and the auditor take
canonical images only, as ``{"content_type": "image/png" | "image/jpeg", "base64": ...}`` (``canonical_sizes``), and
engines give the model's processor their pixels as they are. An image costs ``image_tokens`` input tokens, and a
request with images one more (the line break after them).

Pillow is imported only where images are decoded or encoded: the gateway's workers and the engines. Reading sizes and
checking the canonical form needs no decoder (``header``).
"""

from __future__ import annotations

import binascii
import io
import math
import re
import struct
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, model_validator

from verathos.protocol.api import MAX_REQUEST_BYTES, ImageObject

if TYPE_CHECKING:
    from PIL.Image import Image

PNG, JPEG, WEBP = "image/png", "image/jpeg", "image/webp"
USER_TYPES = (PNG, JPEG, WEBP)  # what users send
CANONICAL_TYPES = (PNG, JPEG)  # what the gateway forwards
_PIL_FORMATS = {PNG: "PNG", JPEG: "JPEG", WEBP: "WEBP"}
MAX_ASPECT = 200  # the models' processors take no image whose longer side exceeds 200 times its shorter side
JPEG_QUALITY = 95
HEADER_BYTES = 96 << 10  # the start of a user's image read for its size: room for EXIF and ICC segments before it
CANONICAL_HEADER_BYTES = 4 << 10  # the start of a canonical image read for its size: its header takes a few hundred
MAX_IMAGE_REQUEST_BYTES = 15 << 20  # an image model's max_request_bytes at most: a sample of such a request fits a
# post of sampled exchanges to the auditor (protocol.gateway.MAX_SAMPLES_BYTES)
MAX_IMAGE_TOKENS = 1 << 20  # the image tokens of one request an image model's limits may allow, at most
# Engine families whose models read images -> the side in pixels of one image token: their processors' patch size
# times merge size.
ENGINE_TOKEN_PX = {"clef": 32}
_ORIENTATION = 0x0112  # the EXIF orientation tag; 5 to 8 swap width and height
_DATA_URL = re.compile(r"data:(image/[a-z]+);base64,", re.IGNORECASE)
_HEADER_STEPS = 1024  # markers and fill bytes a JPEG header is read through, at most
_NOT_BASE64 = "not base64: the standard alphabet with its padding, without whitespace"


class ImageLimits(BaseModel):
    """What a model that reads images takes per request: the catalog's ``images`` of a model whose
    ``input_modalities`` list ``image``."""

    max_count: int = Field(4, ge=1, description="Images per request")
    max_bytes: int = Field(4 << 20, ge=1, description="Bytes of one image as sent (decoded from base64)")
    max_total_bytes: int = Field(8 << 20, ge=1, description="Bytes of a request's images as sent, together")
    max_input_pixels: int = Field(16_777_216, ge=1, description="Pixels of one image as sent")
    max_pixels: int = Field(1_048_576, ge=1, description="Pixels of one image as the model reads it: larger ones are "
                            "downscaled to fit, keeping their aspect ratio")
    min_pixels: int = Field(65_536, ge=1, description="Pixels of one image as the model reads it, at least: smaller "
                            "ones are upscaled (the model's processor's minimum)")
    token_px: int = Field(32, ge=1, description="The side in pixels of one image token: the model's processor's patch "
                          "size times its merge size")
    max_request_bytes: int = Field(13 << 20, ge=MAX_REQUEST_BYTES, le=MAX_IMAGE_REQUEST_BYTES,
                                   description="A request's body, as the gateway receives it and as miners do")

    @model_validator(mode="after")
    def _consistent(self) -> ImageLimits:
        if self.max_bytes > self.max_total_bytes:
            raise ValueError("images: max_bytes cannot exceed max_total_bytes")
        if not self.token_px ** 2 <= self.min_pixels <= self.max_pixels <= self.max_input_pixels:
            raise ValueError("images: token_px squared <= min_pixels <= max_pixels <= max_input_pixels")
        if self.max_tokens() > MAX_IMAGE_TOKENS:
            raise ValueError(f"images: a request could carry {self.max_tokens()} image tokens, over {MAX_IMAGE_TOKENS}")
        return self

    def max_tokens(self) -> int:
        """The most image tokens a request can carry."""
        return self.max_count * (self.max_pixels // self.token_px ** 2 + 2) + 1

    def document(self) -> dict[str, Any]:
        """The limits as users read them (``GET /v1/models``)."""
        return {"formats": list(USER_TYPES), **self.model_dump(), "max_aspect_ratio": MAX_ASPECT}


class ImageRefused(ValueError):
    """An image the network does not take: ``status`` 400 (not an image it reads) or 413 (beyond a limit)."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


class Incomplete(ValueError):
    """The bytes end before the image's header does."""


# ---- what users send ----

def wire_image(item: Any) -> tuple[str, str]:
    """(media type in lower case, base64) of an image as a request embeds it. Raises ImageRefused."""
    if isinstance(item, str):
        if item[:8].lower().startswith(("http://", "https://")):
            raise ImageRefused(400, "remote URLs are not accepted: embed the image as base64")
        match = _DATA_URL.match(item)
        if match is None:
            raise ImageRefused(400, 'not an image: a data URL (data:image/png;base64,...) or {"content_type", '
                                    '"base64"}')
        return match[1].lower(), item[match.end():]
    image = item if isinstance(item, ImageObject) else ImageObject.model_validate(item)
    return image.content_type.lower(), image.base64


def decoded_size(data: str) -> int:
    """The bytes strict base64 ``data`` decodes to, from its length. Raises ImageRefused if no base64 has it."""
    if len(data) % 4:
        raise ImageRefused(400, _NOT_BASE64)
    return len(data) // 4 * 3 - (2 if data.endswith("==") else 1 if data.endswith("=") else 0)


def b64decode(data: str, limit: int | None = None) -> bytes:
    """Strict base64 ``data`` decoded, or with ``limit`` its start: at least that many bytes, if it holds them. Raises
    ImageRefused."""
    if limit is not None:
        data = data[: -(-limit // 3) * 4]
    try:
        return binascii.a2b_base64(data, strict_mode=True)
    except (binascii.Error, ValueError):
        raise ImageRefused(400, _NOT_BASE64) from None


def sniff(data: bytes) -> str | None:
    """The media type the bytes start with: PNG, JPEG or WebP; None for anything else."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return PNG
    if data.startswith(b"\xff\xd8\xff"):
        return JPEG
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return WEBP
    return None


# ---- headers ----

@dataclass(frozen=True)
class Header:
    content_type: str
    width: int
    height: int
    canonical: bool  # in the canonical form (see ``header``)


def header(data: bytes) -> Header:
    """The media type and size of an image from its first bytes, read without a decoder, and whether it has the form
    canonical images take: a PNG of 8-bit RGB, not interlaced, whose image data follows its header chunk; a baseline
    JPEG of 8 bits and three components without subsampling, whose segments before its scan are a JFIF header without
    thumbnail, its frame header and its tables (no EXIF, ICC profile, XMP or comment). Raises Incomplete if the bytes
    end before the header, ValueError if they are not an image's."""
    kind = sniff(data)
    if kind is None:
        raise ValueError("not a PNG, JPEG or WebP image")
    width, height, canonical = {PNG: _png, JPEG: _jpeg, WEBP: _webp}[kind](data)
    if width < 1 or height < 1:
        raise ValueError("an image without pixels")
    return Header(kind, width, height, canonical)


def _png(data: bytes) -> tuple[int, int, bool]:
    if len(data) < 29:
        raise Incomplete("the PNG ends inside its header")
    if struct.unpack_from(">I4s", data, 8) != (13, b"IHDR"):
        raise ValueError("a PNG starts with its header chunk")
    width, height, *form = struct.unpack_from(">IIBBBBB", data, 16)
    # the next chunk's type follows the header chunk's 13 bytes and checksum, and its own length
    return width, height, form == [8, 2, 0, 0, 0] and data[37:41] == b"IDAT"


_SOF = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}  # frame headers
_STANDALONE = {0x01, *range(0xD0, 0xD8)}  # markers without a length
_TABLES = {0xC4, 0xDB, 0xDD}  # Huffman and quantization tables, restart interval


def _jpeg(data: bytes) -> tuple[int, int, bool]:
    """Read through the segments to the frame header, and while the image may be canonical on to the scan."""
    pos, canonical, size = 2, True, None
    for _ in range(_HEADER_STEPS):
        if pos + 4 > len(data):
            raise Incomplete("the JPEG ends inside its header")
        if data[pos] != 0xFF:
            raise ValueError("a JPEG segment starts with 0xFF")
        marker = data[pos + 1]
        if marker == 0xFF:  # a fill byte
            pos += 1
            continue
        if marker in _STANDALONE:
            pos, canonical = pos + 2, False
            continue
        if marker in (0xD9, 0xDA):
            if size is None:
                raise ValueError("a JPEG without a frame header before its scan")
            return size[0], size[1], canonical and marker == 0xDA
        (length,) = struct.unpack_from(">H", data, pos + 2)
        if length < 2:
            raise ValueError("a JPEG segment shorter than its length field")
        if marker in _SOF:
            if size is not None:
                raise ValueError("a JPEG with two frame headers")
            if pos + 2 + length > len(data):
                raise Incomplete("the JPEG ends inside its frame header")
            if length < 8:
                raise ValueError("a JPEG frame header without its size")
            precision, height, width, components = struct.unpack_from(">BHHB", data, pos + 4)
            size = (width, height)
            canonical = (canonical and marker == 0xC0 and (precision, components, length) == (8, 3, 17)
                         and all(data[pos + 11 + 3 * c] == 0x11 for c in range(3)))
            if not canonical:
                return width, height, False
        elif marker == 0xE0:  # a JFIF header of 16 bytes holds no thumbnail; it comes first
            canonical = canonical and pos == 2 and length == 16 and data[pos + 4:pos + 9] == b"JFIF\0"
        elif marker not in _TABLES:
            canonical = False
        pos += 2 + length
    raise Incomplete("the JPEG's header continues beyond what is read of it")


def _webp(data: bytes) -> tuple[int, int, bool]:
    if len(data) < 30:
        raise Incomplete("the WebP ends inside its header")
    kind = data[12:16]
    if kind == b"VP8 ":
        if data[23:26] != b"\x9d\x01\x2a":
            raise ValueError("a lossy WebP without its frame's start code")
        width, height = struct.unpack_from("<HH", data, 26)
        return width & 0x3FFF, height & 0x3FFF, False
    if kind == b"VP8L":
        if data[20] != 0x2F:
            raise ValueError("a lossless WebP without its signature")
        bits = int.from_bytes(data[21:25], "little")
        return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1, False
    if kind == b"VP8X":
        return int.from_bytes(data[24:27], "little") + 1, int.from_bytes(data[27:30], "little") + 1, False
    raise ValueError("a WebP whose first chunk is not VP8, VP8L or VP8X")


# ---- sizes and tokens ----

def check_size(width: int, height: int, limits: ImageLimits) -> None:
    """Raises ImageRefused for an image of more pixels than ``max_input_pixels``, or of an aspect ratio beyond
    ``MAX_ASPECT``."""
    if width * height > limits.max_input_pixels:
        raise ImageRefused(413, f"{width} x {height} pixels, more than the model takes ({limits.max_input_pixels})")
    if max(width, height) > MAX_ASPECT * min(width, height):
        raise ImageRefused(400, f"{width} x {height} pixels: an aspect ratio beyond {MAX_ASPECT}:1")


def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """(height, width) the models' processors resize an image to (transformers' Qwen2-VL ``smart_resize``): each side
    a multiple of ``factor``, the pixels within [``min_pixels``, ``max_pixels``] as far as those multiples allow, and
    the aspect ratio kept as closely as they allow."""
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def image_tokens(width: int, height: int, token_px: int) -> int:
    """The input tokens of a canonical image: one per ``token_px`` square, and the marks of its start and end."""
    return (width // token_px) * (height // token_px) + 2


def request_image_tokens(sizes: list[tuple[int, int]], token_px: int) -> int:
    """The input tokens of a request's canonical images (``(width, height)`` each): theirs, and the line break after
    them; 0 without images."""
    return sum(image_tokens(w, h, token_px) for w, h in sizes) + 1 if sizes else 0


# ---- the canonical form ----

@dataclass(frozen=True)
class Canonical:
    content_type: str
    width: int
    height: int
    data: bytes


def canonical_image(data: bytes, content_type: str, limits: ImageLimits) -> Canonical:
    """The canonical form of an image a user sent with ``content_type`` (see the module), decoded by Pillow as that
    format only, its size checked from its header before a pixel is decoded and Pillow's own pixel limit set to the
    model's (a decompression bomb is refused, not decoded). Raises ImageRefused."""
    from PIL import Image, ImageOps

    kind = sniff(data)
    if kind is None:
        raise ImageRefused(400, "not a PNG, JPEG or WebP image")
    if kind != content_type:
        raise ImageRefused(400, f"its bytes are {kind}, not {content_type}")
    Image.MAX_IMAGE_PIXELS = limits.max_input_pixels
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data), formats=[_PIL_FORMATS[kind]]) as image:
                check_size(image.width, image.height, limits)
                height, width = smart_resize(image.height, image.width, limits.token_px, limits.min_pixels,
                                             limits.max_pixels)
                if not limits.min_pixels <= width * height <= limits.max_pixels:
                    raise ImageRefused(413, f"{image.width} x {image.height} pixels do not fit the model's pixel "
                                            "budget")
                if kind == JPEG:
                    image.draft("RGB", (width, height))  # decoded at 1/2 to 1/8 scale while that covers the target
                if image.getexif().get(_ORIENTATION, 1) in (5, 6, 7, 8):
                    width, height = height, width
                upright = ImageOps.exif_transpose(image).convert("RGB")
    except ImageRefused:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise ImageRefused(413, f"more pixels than the model takes ({limits.max_input_pixels})") from None
    except MemoryError:
        raise ImageRefused(413, "it takes more memory to decode than the gateway gives one image") from None
    except Exception:
        raise ImageRefused(400, "it could not be decoded") from None
    if upright.size != (width, height):
        upright = upright.resize((width, height), Image.Resampling.BICUBIC)
    upright.info.clear()  # nothing of the source's metadata is written
    out = io.BytesIO()
    if kind == PNG:
        upright.save(out, "PNG")
    else:
        upright.save(out, "JPEG", quality=JPEG_QUALITY, subsampling=0)
    return Canonical(PNG if kind == PNG else JPEG, width, height, out.getvalue())


def canonical_item(item: Any) -> tuple[str, str]:
    """(media type, base64) of an image wrapped as canonical images are: ``{"content_type": "image/png" |
    "image/jpeg", "base64": ...}``. Raises ValueError."""
    if isinstance(item, ImageObject):
        item = item.model_dump()
    if not (isinstance(item, dict) and item.get("content_type") in CANONICAL_TYPES
            and isinstance(item.get("base64"), str)):
        raise ValueError('not {"content_type": "image/png" | "image/jpeg", "base64": ...}')
    return item["content_type"], item["base64"]


def canonical_header(data: bytes, content_type: str, *, token_px: int, min_pixels: int,
                     max_pixels: int) -> tuple[int, int]:
    """(width, height) of a canonical image from its first bytes: of ``content_type``, in the canonical form
    (``header``), each side a multiple of ``token_px`` and its pixels within [``min_pixels``, ``max_pixels``]. Raises
    ValueError naming what it is not."""
    try:
        found = header(data)
    except Incomplete:
        raise ValueError("its header is incomplete") from None
    if found.content_type != content_type:
        raise ValueError(f"its bytes are {found.content_type}, not {content_type}")
    if not found.canonical:
        raise ValueError("not in the canonical form (8-bit RGB without metadata)")
    if found.width % token_px or found.height % token_px:
        raise ValueError(f"{found.width} x {found.height} pixels: sides not multiples of {token_px}")
    if not min_pixels <= found.width * found.height <= max_pixels:
        raise ValueError(f"{found.width} x {found.height} pixels, outside [{min_pixels}, {max_pixels}]")
    return found.width, found.height


def canonical_sizes(images: list[Any], *, token_px: int, min_pixels: int, max_pixels: int) -> list[tuple[int, int]]:
    """(width, height) of each of a request's images if all are canonical (``canonical_item``, ``canonical_header``),
    read from the start of their base64. Raises ValueError naming the first that is not."""
    sizes = []
    for index, item in enumerate(images):
        try:
            content_type, data = canonical_item(item)
            sizes.append(canonical_header(b64decode(data, CANONICAL_HEADER_BYTES), content_type, token_px=token_px,
                                          min_pixels=min_pixels, max_pixels=max_pixels))
        except ValueError as exc:
            raise ValueError(f"images.{index}: {exc}") from None
    return sizes


def canonical_pixels(images: list[Any], *, token_px: int, min_pixels: int, max_pixels: int) -> list[Image]:
    """A request's canonical images (see ``canonical_sizes``) decoded to RGB by Pillow, as PNG or JPEG only. Raises
    ValueError naming the first that is not canonical or cannot be decoded."""
    from PIL import Image

    pixels = []
    for index, item in enumerate(images):
        try:
            content_type, payload = canonical_item(item)
            data = b64decode(payload)
            size = canonical_header(data, content_type, token_px=token_px, min_pixels=min_pixels,
                                    max_pixels=max_pixels)
            try:
                with Image.open(io.BytesIO(data), formats=[_PIL_FORMATS[content_type]]) as image:
                    image.load()
                    rgb = image.convert("RGB")
            except Exception:
                raise ValueError("it could not be decoded") from None
            if rgb.size != size:
                raise ValueError("its pixels are not the size its header gives")
        except ValueError as exc:
            raise ValueError(f"images.{index}: {exc}") from None
        pixels.append(rgb)
    return pixels
