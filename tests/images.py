"""Images for the tests, drawn with Pillow: a coloured field with a marked top-left quarter, so where the corner ends up
shows how an image was turned."""

from __future__ import annotations

import base64
import io

from verathos.protocol.media import ImageLimits, canonical_image

FIELD, CORNER = (40, 90, 200), (20, 220, 40)


def drawn(width: int, height: int, mode: str = "RGB"):
    from PIL import Image

    image = Image.new("RGB", (width, height), FIELD)
    image.paste(CORNER, (0, 0, max(1, width // 4), max(1, height // 4)))
    return image.convert(mode)


def encoded(width: int, height: int, fmt: str = "PNG", mode: str = "RGB", **save) -> bytes:
    out = io.BytesIO()
    drawn(width, height, mode).save(out, fmt, **save)
    return out.getvalue()


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def data_url(data: bytes, content_type: str) -> str:
    return f"data:{content_type};base64,{b64(data)}"


def item(data: bytes, content_type: str) -> dict:
    return {"content_type": content_type, "base64": b64(data)}


def canonical(width: int, height: int, fmt: str = "PNG") -> dict:
    """A canonical image as the gateway forwards it, of a size that is canonical as it is (sides multiples of 32
    within the default pixel budget)."""
    content_type = {"PNG": "image/png", "JPEG": "image/jpeg"}[fmt]
    made = canonical_image(encoded(width, height, fmt), content_type, ImageLimits())
    assert (made.width, made.height) == (width, height)
    return item(made.data, made.content_type)
