"""Images in requests: the wire items, strict base64, formats read from the bytes, headers read without a decoder, the
canonical form the gateway makes, its sizes and tokens, and the check miners and engines make of it."""

from __future__ import annotations

import io
import random
import struct
import zlib

import pytest
from pydantic import ValidationError

from tests.images import CORNER, FIELD, b64, canonical, data_url, drawn, encoded, item
from verathos.protocol.api import SystemOneRequest
from verathos.protocol.media import (
    MAX_ASPECT,
    Header,
    ImageLimits,
    ImageRefused,
    Incomplete,
    b64decode,
    canonical_image,
    canonical_pixels,
    canonical_sizes,
    decoded_size,
    header,
    image_tokens,
    request_image_tokens,
    smart_resize,
    sniff,
    wire_image,
)

LIMITS = ImageLimits()
BUDGET = {"token_px": 32, "min_pixels": LIMITS.min_pixels, "max_pixels": LIMITS.max_pixels}
QUESTIONS = {"q": {"type": "noul", "instructions": "Is there a green corner?"}}


def make(data: bytes, content_type: str, limits: ImageLimits = LIMITS):
    from PIL import Image

    made = canonical_image(data, content_type, limits)
    return made, Image.open(io.BytesIO(made.data))


def chunk(kind: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))


def png_claiming(width: int, height: int, interlace: int = 0) -> bytes:
    """A PNG whose header claims the size (and interlacing, which Pillow does not write) and whose image data is a few
    bytes."""
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, interlace))
            + chunk(b"IDAT", zlib.compress(b"\0" * 64)) + chunk(b"IEND", b""))


def test_an_image_is_a_data_url_or_an_object_and_never_a_url_or_a_path() -> None:
    png = encoded(64, 48)
    assert wire_image(data_url(png, "image/png")) == ("image/png", b64(png))
    assert wire_image("DATA:IMAGE/JPEG;BASE64,QUJD") == ("image/jpeg", "QUJD")  # the scheme and type in any case
    assert wire_image({"content_type": "Image/WebP", "base64": "QUJD"}) == ("image/webp", "QUJD")
    for url in ("http://169.254.169.254/latest/meta-data/", "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                "HTTPS://example.com/a.png"):
        with pytest.raises(ImageRefused, match="remote URLs are not accepted") as refused:
            wire_image(url)
        assert refused.value.status == 400
    for other in ("/etc/passwd", "file:///etc/passwd", b64(png), "data:image/png,rawbytes", "data:text/plain;base64,"
                  "QUJD"):
        with pytest.raises(ImageRefused, match="not an image") as refused:
            wire_image(other)
        assert refused.value.status == 400


def test_base64_is_strict_and_sizes_are_known_before_decoding() -> None:
    assert b64decode("QUJDRA==") == b"ABCD" and decoded_size("QUJDRA==") == 4 and decoded_size("QUJD") == 3
    assert b64decode(b64(bytes(range(256)) * 400), limit=1000)[:1000] == (bytes(range(256)) * 4)[:1000]
    for bad in ("QUJDRA", "QUJD RA==", "QUJDRA==\n", "QUJD-_==", "=QUJ", "QUJDRA===", "QUJDé==="):
        with pytest.raises(ImageRefused, match="not base64"):
            b64decode(bad)
    with pytest.raises(ImageRefused, match="not base64"):
        decoded_size("QUJDR")


def test_the_bytes_decide_the_format() -> None:
    assert sniff(encoded(8, 8, "PNG")) == "image/png"
    assert sniff(encoded(8, 8, "JPEG")) == "image/jpeg"
    assert sniff(encoded(8, 8, "WEBP")) == "image/webp"
    for fmt in ("GIF", "BMP", "TIFF"):
        assert sniff(encoded(8, 8, fmt)) is None
    assert sniff(b'<svg xmlns="http://www.w3.org/2000/svg"/>') is None
    assert sniff(b"\0\0\0\x18ftypheic\0\0\0\0") is None and sniff(b"\0\0\0\x18ftypavif\0\0\0\0") is None


def test_headers_give_the_size_without_a_decoder() -> None:
    from PIL import Image

    exif = Image.Exif()
    exif[0x0112], exif[0x010F] = 6, "x" * 30_000  # a large EXIF segment before the frame header
    cases = {
        "png": encoded(300, 200, "PNG"), "png interlaced": png_claiming(300, 200, interlace=1),
        "jpeg": encoded(300, 200, "JPEG"), "jpeg with exif": encoded(300, 200, "JPEG", exif=exif.tobytes()),
        "jpeg progressive": encoded(300, 200, "JPEG", progressive=True),
        "webp lossy": encoded(300, 200, "WEBP"), "webp lossless": encoded(300, 200, "WEBP", lossless=True),
        "webp extended": encoded(300, 200, "WEBP", exif=exif.tobytes()),
    }
    for name, data in cases.items():
        found = header(data)
        assert (found.width, found.height) == (300, 200), name
        assert found.content_type == sniff(data)
    with pytest.raises(Incomplete):
        header(cases["jpeg with exif"][:20_000])  # the frame header comes after the EXIF segment
    for data in cases.values():
        with pytest.raises(Incomplete):
            header(data[:24])
    assert header(png_claiming(20_000, 20_000)) == Header("image/png", 20_000, 20_000, True)
    with pytest.raises(ValueError, match="not a PNG, JPEG or WebP"):
        header(encoded(8, 8, "GIF"))


def test_canonical_images_keep_png_and_turn_jpeg_and_webp_into_jpeg_without_metadata() -> None:
    from PIL import Image, PngImagePlugin

    exif = Image.Exif()
    exif[0x010F], exif[0x8825] = "ACME phone", {2: (47.0, 22.0, 0.0)}  # a maker and a GPS position
    text = PngImagePlugin.PngInfo()
    text.add_text("Author", "someone")
    sources = {
        ("image/png", "image/png"): encoded(640, 480, "PNG", pnginfo=text, exif=exif.tobytes(), icc_profile=b"\0" * 64),
        ("image/jpeg", "image/jpeg"): encoded(640, 480, "JPEG", exif=exif.tobytes(), icc_profile=b"\0" * 64,
                                              comment=b"a comment"),
        ("image/webp", "image/jpeg"): encoded(640, 480, "WEBP", exif=exif.tobytes(), icc_profile=b"\0" * 64),
    }
    for (sent, made_as), data in sources.items():
        made, image = make(data, sent)
        assert made.content_type == made_as and image.size == (made.width, made.height) == (640, 480)
        assert header(made.data).canonical and image.mode == "RGB"
        assert not ({"exif", "icc_profile", "comment", "Author", "xmp"} & set(image.info))
        assert b"ACME" not in made.data and b"someone" not in made.data
        if made_as == "image/jpeg":
            assert image.layers == 3 and image.layer[0][1:3] == (1, 1) and image.layer[1][1:3] == (1, 1)  # 4:4:4
        assert image.getpixel((5, 5)) == pytest.approx(CORNER, abs=12)
        assert image.getpixel((600, 400)) == pytest.approx(FIELD, abs=12)


def test_the_exif_orientation_turns_the_image_upright() -> None:
    from PIL import Image

    exif = Image.Exif()
    exif[0x0112] = 6  # turn 90 degrees clockwise to view
    made, image = make(encoded(640, 320, "JPEG", exif=exif.tobytes()), "image/jpeg")
    assert image.size == (320, 640)  # turned: its height is the stored width
    # stored top-left corner, turned 90 degrees clockwise, is at the top right
    assert image.getpixel((300, 5)) == pytest.approx(CORNER, abs=16)
    assert image.getpixel((5, 5)) == pytest.approx(FIELD, abs=16)


def test_transparency_is_dropped_as_the_processor_drops_it() -> None:
    from PIL import Image

    image = Image.new("RGBA", (256, 256), (255, 0, 0, 0))  # transparent pixels that store red
    out = io.BytesIO()
    image.save(out, "PNG")
    _, made = make(out.getvalue(), "image/png")
    assert made.mode == "RGB" and made.getpixel((10, 10)) == (255, 0, 0)  # the colour kept, not composited on white
    palette = drawn(256, 256).convert("P")
    out = io.BytesIO()
    palette.save(out, "PNG", transparency=0)
    assert make(out.getvalue(), "image/png")[1].mode == "RGB"


def test_an_animation_counts_as_its_first_frame() -> None:
    from PIL import Image

    frames = [Image.new("RGB", (256, 256), c) for c in ((250, 0, 0), (0, 0, 250))]
    for fmt, content_type in (("WEBP", "image/webp"), ("PNG", "image/png")):
        out = io.BytesIO()
        frames[0].save(out, fmt, save_all=True, append_images=frames[1:], duration=100, loop=0)
        assert make(out.getvalue(), content_type)[1].getpixel((128, 128)) == pytest.approx((250, 0, 0), abs=8)


def test_sizes_follow_the_pixel_budget_at_multiples_of_32() -> None:
    # (sent, as the model reads it, its tokens); a request adds one token for the line break after its images
    examples = [((1024, 1024), (1024, 1024), 1026), ((1920, 1080), (1344, 768), 1010),
                ((4032, 3024), (1152, 864), 974), ((800, 600), (800, 608), 477), ((512, 512), (512, 512), 258),
                ((100, 100), (256, 256), 66)]
    for (width, height), size, tokens in examples:
        h, w = smart_resize(height, width, 32, LIMITS.min_pixels, LIMITS.max_pixels)
        assert (w, h) == size and image_tokens(w, h, 32) == tokens
    assert request_image_tokens([], 32) == 0
    assert request_image_tokens([(1024, 1024), (512, 512)], 32) == 1026 + 258 + 1
    made, image = make(encoded(4032, 3024, "JPEG"), "image/jpeg")  # decoded at reduced scale, then resized
    assert image.size == (1152, 864) and image.getpixel((10, 10)) == pytest.approx(CORNER, abs=16)
    rng = random.Random(7)
    for _ in range(20_000):
        height = rng.randint(1, 8192)
        width = rng.randint(max(1, -(-height // MAX_ASPECT)), min(8192, height * MAX_ASPECT))
        h, w = smart_resize(height, width, 32, LIMITS.min_pixels, LIMITS.max_pixels)
        assert h % 32 == 0 and w % 32 == 0 and LIMITS.min_pixels <= h * w <= LIMITS.max_pixels


def test_smart_resize_is_the_processors() -> None:
    qwen2_vl = pytest.importorskip("transformers.models.qwen2_vl.image_processing_qwen2_vl")
    rng = random.Random(3)
    for limits in ((32, 65_536, 1_048_576), (32, 65_536, 16_777_216), (28, 3136, 1_003_520)):
        for _ in range(5000):
            height = rng.randint(1, 6000)
            width = rng.randint(max(1, -(-height // MAX_ASPECT)), min(6000, height * MAX_ASPECT))
            assert smart_resize(height, width, *limits) == qwen2_vl.smart_resize(height, width, *limits)


def test_images_beyond_the_limits_are_refused_before_decoding() -> None:
    for data, status, message in ((png_claiming(20_000, 20_000), 413, "more pixels"),
                                  (encoded(4100, 20, "PNG"), 400, "aspect ratio"),
                                  (encoded(64, 64, "GIF"), 400, "not a PNG, JPEG or WebP"),
                                  (encoded(64, 64, "PNG")[:60], 400, "could not be decoded")):
        with pytest.raises(ImageRefused, match=message) as refused:
            canonical_image(data, "image/png" if sniff(data) is None else sniff(data), LIMITS)
        assert refused.value.status == status
    with pytest.raises(ImageRefused, match="its bytes are image/jpeg, not image/png") as refused:
        canonical_image(encoded(64, 64, "JPEG"), "image/png", LIMITS)
    assert refused.value.status == 400
    small = ImageLimits(max_input_pixels=1_048_576, max_pixels=262_144)
    with pytest.raises(ImageRefused, match="more pixels") as refused:
        canonical_image(encoded(1100, 1000, "JPEG"), "image/jpeg", small)
    assert refused.value.status == 413
    made, image = make(encoded(1000, 1000, "JPEG"), "image/jpeg", small)
    assert image.size == (512, 512)


def test_miners_take_canonical_images_only() -> None:
    from PIL import Image

    good = [canonical(256, 320), canonical(1024, 1024, "JPEG")]
    assert canonical_sizes(good, **BUDGET) == [(256, 320), (1024, 1024)]
    pixels = canonical_pixels(good, **BUDGET)
    assert [p.size for p in pixels] == [(256, 320), (1024, 1024)] and {p.mode for p in pixels} == {"RGB"}
    exif = Image.Exif()
    exif[0x0112] = 6
    not_canonical = {
        "a data URL": data_url(encoded(256, 256), "image/png"),
        "webp": item(encoded(256, 256, "WEBP"), "image/webp"),
        "a type that is not the bytes'": item(encoded(256, 256, "JPEG"), "image/png"),
        "sides not multiples of 32": item(encoded(250, 300), "image/png"),
        "beyond the budget": item(encoded(1056, 1024), "image/png"),
        "below the budget": item(encoded(128, 128), "image/png"),
        "alpha": item(encoded(256, 256, mode="RGBA"), "image/png"),
        "a palette": item(encoded(256, 256, mode="P"), "image/png"),
        "interlaced": item(png_claiming(256, 256, interlace=1), "image/png"),
        "a text chunk": item(encoded(256, 256, dpi=(72, 72)), "image/png"),
        "subsampled": item(encoded(256, 256, "JPEG", subsampling=2), "image/jpeg"),
        "progressive": item(encoded(256, 256, "JPEG", subsampling=0, progressive=True), "image/jpeg"),
        "an orientation": item(encoded(256, 256, "JPEG", subsampling=0, exif=exif.tobytes()), "image/jpeg"),
        "a comment": item(encoded(256, 256, "JPEG", subsampling=0, comment=b"hi"), "image/jpeg"),
        "grey": item(encoded(256, 256, "JPEG", mode="L"), "image/jpeg"),
        "not base64": {"content_type": "image/png", "base64": "iVBORw0KGgo AAAA"},
        "upper-case type": {**canonical(256, 256), "content_type": "image/PNG"},
    }
    for image in not_canonical.values():
        with pytest.raises(ValueError, match=r"images\.1: "):
            canonical_sizes([good[0], image], **BUDGET)
        with pytest.raises(ValueError, match=r"images\.1: "):
            canonical_pixels([good[0], image], **BUDGET)
    truncated = canonical(256, 256)
    truncated["base64"] = b64(b64decode(truncated["base64"])[:400])
    assert canonical_sizes([truncated], **BUDGET) == [(256, 256)]  # its header is whole
    with pytest.raises(ValueError, match="could not be decoded"):
        canonical_pixels([truncated], **BUDGET)


def test_the_limits_hold_together() -> None:
    assert LIMITS.max_tokens() == 4 * 1026 + 1
    assert LIMITS.document()["formats"] == ["image/png", "image/jpeg", "image/webp"]
    for bad in ({"max_pixels": 2 << 20, "max_input_pixels": 1 << 20}, {"min_pixels": 512}, {"max_bytes": 9 << 20},
                {"max_request_bytes": 16 << 20}, {"max_request_bytes": 512 << 10},
                {"max_count": 64, "max_pixels": 16_777_216}):
        with pytest.raises(ValidationError):
            ImageLimits(**bad)


def test_requests_carry_images_but_never_videos_or_processor_arguments() -> None:
    request = {"model": "clef-flash", "state": "s", "questions": QUESTIONS}
    images = [data_url(b"x", "image/png"), item(b"x", "image/png")]
    parsed = SystemOneRequest.model_validate({**request, "images": images})
    assert len(parsed.images) == 2 and SystemOneRequest.model_validate(request).images is None
    for refused, value in (("videos", []), ("videos", None), ("media_kwargs", {"max_pixels": 1 << 30})):
        with pytest.raises(ValidationError, match="videos are not supported|media_kwargs is not accepted"):
            SystemOneRequest.model_validate({**request, refused: value})
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate({**request, "images": [{"url": "https://example.com/a.png"}]})
