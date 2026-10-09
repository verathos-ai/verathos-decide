"""The standard mix's images: drawn from a seed as the load generator draws them, canonical as the gateway makes
images, within the bound the mix's size limit counts; the mix's requests with images, and a text mix's requests
unchanged."""

from __future__ import annotations

import collections
import hashlib
import io
import json
import random
import struct
import zlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from verathos import catalog
from verathos.protocol.api import SystemOneRequest
from verathos.protocol.canonical import canonical_json, digest
from verathos.protocol.media import PNG, ImageLimits, b64decode, canonical_image, canonical_sizes, header
from verathos.throughput import images
from verathos.throughput.images import BACKGROUNDS, COLOURS, Rectangle, draw, png, png_bound
from verathos.throughput.loadgen import mix_plan
from verathos.throughput.measure import fingerprint
from verathos.throughput.workload import (
    IMAGE_QUESTIONS,
    WORDS,
    Workload,
    WorkloadImages,
    in_mix_form,
    make_request,
)

VECTORS = Path(__file__).resolve().parents[1] / "native" / "loadgen" / "tests" / "fixtures" / "images.json"
CLEF = catalog.load().get("clef-flash")
KEV = catalog.load().get("kev-4b")
SIDES = list(range(256, 513, 32))


def pixels(data: bytes):
    from PIL import Image

    with Image.open(io.BytesIO(data), formats=["PNG"]) as image:
        image.load()
        return image.convert("RGB")


def seed(index: int) -> bytes:
    return hashlib.sha256(f"seed {index}".encode()).digest()


def test_the_load_generator_draws_the_same_bytes_from_the_same_seed() -> None:
    """The vectors its own tests check (``native/loadgen/src/images.rs``): edge cases among them, sides of 1 to 7
    pixels, one (87) whose zero rows end in a shortened copy, and 2048 pixels."""
    vectors = json.loads(VECTORS.read_text())["vectors"]
    assert len(vectors) >= 10
    for vector in vectors:
        drawn = draw(bytes.fromhex(vector["seed"]), tuple(vector["count"]), vector["sides_px"])
        assert [{"width": struct.unpack_from(">I", p, 16)[0], "height": struct.unpack_from(">I", p, 20)[0],
                 "bytes": len(p), "sha256": hashlib.sha256(p).hexdigest()} for p in drawn] == vector["images"]


def test_a_mix_image_is_canonical_and_the_gateway_would_forward_its_pixels_as_they_are() -> None:
    limits = ImageLimits()
    for index in range(40):
        for data in draw(seed(index), (1, 2), SIDES):
            found = header(data)
            assert found.canonical and found.content_type == PNG
            assert found.width in SIDES and found.height in SIDES
            image = pixels(data)
            assert image.size == (found.width, found.height)
            made = canonical_image(data, PNG, limits)
            assert (made.content_type, made.width, made.height) == (PNG, found.width, found.height)
            assert pixels(made.data).tobytes() == image.tobytes()


def test_a_png_holds_the_pixels_it_was_drawn_with() -> None:
    from PIL import Image

    rectangles = [Rectangle(3, 2, 40, 30, 6), Rectangle(20, 10, 50, 5, 7), Rectangle(0, 25, 96, 3, 1)]
    for width, height in ((96, 40), (87, 33), (1, 1), (2, 7)):
        shown = [r for r in rectangles if r.x + r.width <= width and r.y + r.height <= height]
        expected = Image.new("RGB", (width, height), COLOURS[2])
        for r in shown:
            expected.paste(COLOURS[r.colour], (r.x, r.y, r.x + r.width, r.y + r.height))
        data = png(width, height, 2, shown)
        assert pixels(data).tobytes() == expected.tobytes()
        # the image data inflates as zlib checks it, and only Up and unfiltered rows occur
        idat = data[41:41 + struct.unpack_from(">I", data, 33)[0]]
        rows = zlib.decompress(idat)
        assert len(rows) == height * (3 * width + 1) and {rows[y * (3 * width + 1)] for y in range(height)} <= {0, 2}


def test_a_requests_images_share_a_background_that_covers_most_of_each() -> None:
    backgrounds = collections.Counter()
    for index in range(60):
        drawn = [pixels(data) for data in draw(seed(index), (1, 4), [256, 320])]
        dominant = {max(image.getcolors(1 << 16))[1] for image in drawn}
        assert len(dominant) == 1  # one background for the request's images
        (colour,) = dominant
        assert colour in COLOURS[:len(BACKGROUNDS)]
        assert all(max(image.getcolors(1 << 16))[0] > 0.55 * image.width * image.height for image in drawn)
        backgrounds[colour] += 1
    assert len(backgrounds) == len(BACKGROUNDS)


def test_every_request_draws_new_images() -> None:
    drawn = [data for index in range(400) for data in draw(seed(index), (1, 1), SIDES)]
    assert len(set(drawn)) == len(drawn)
    assert draw(seed(7), (1, 3), SIDES) == draw(seed(7), (1, 3), SIDES)


def test_a_png_stays_within_its_bound() -> None:
    for index in range(300):
        for data in draw(seed(index), (1, 1), [512]):
            assert len(data) <= png_bound(512, 512)
    # the most runs: three rectangles crossing every row, each of a different colour, edges apart
    widest = [Rectangle(10 + 40 * i, 0, 30, 512, 6 + i % 2 + (i == 2)) for i in range(images.MAX_RECTANGLES)]
    staggered = [Rectangle(5 * i, 50 * i, 300, 100 + 50 * i, i + 1) for i in range(images.MAX_RECTANGLES)]
    for rectangles in (widest, staggered):
        for side in (1, 2, 87, 256, 512, 1024):
            shown = [r for r in rectangles if r.x + r.width <= side and r.y + r.height <= side]
            assert len(png(side, side, 0, shown)) <= png_bound(side, side)
    assert png_bound(512, 512) < 9000  # tight enough to bound a request's size usefully


def test_a_share_of_the_mix_carries_images_and_asks_what_colour_their_background_is() -> None:
    rng, workload = random.Random(11), CLEF.workload
    limits = CLEF.images
    requests = [make_request("clef-flash", workload, rng) for _ in range(800)]
    with_images = [r for r in requests if "images" in r]
    assert 0.2 < len(with_images) / len(requests) < 0.3  # the share, 0.25
    assert len({r["state"] for r in requests}) == len(requests)
    for request in requests:
        assert SystemOneRequest.model_validate(request) and in_mix_form(request["state"])
        assert len(canonical_json(request)) <= workload.max_request_bytes("clef-flash")
        if "images" not in request:
            assert request["questions"] == workload.questions
            continue
        assert request["questions"] == {**workload.questions, **IMAGE_QUESTIONS}
        assert len(request["images"]) == 1  # the shipped mix's count
        sizes = canonical_sizes(request["images"], token_px=limits.token_px, min_pixels=limits.min_pixels,
                                max_pixels=limits.max_pixels)
        assert all(w in SIDES and h in SIDES for w, h in sizes)
    pngs = [b64decode(r["images"][0]["base64"]) for r in with_images]
    assert len(set(pngs)) == len(pngs)


def test_images_change_nothing_for_a_mix_without_them() -> None:
    """The same draws, the same largest request, the same plan for the load generator and the same fingerprint of a
    miner's measurement, so no text model is measured again."""
    rng, again = random.Random(3), random.Random(3)
    for _ in range(50):
        request = make_request("kev-4b", KEV.workload, rng)
        words = " ".join(again.choice(WORDS) for _ in range(again.randint(*KEV.workload.state_words)))
        assert request["state"].endswith(f". {words}.") and set(request) == {"model", "state", "questions"}
    assert KEV.workload.document() == {"state_words": [150, 1200], "questions": KEV.workload.questions}
    longest = " ".join([max(WORDS, key=len)] * 1200)
    assert KEV.workload.max_request_bytes("kev-4b") == len(canonical_json(
        {"model": "kev-4b", "state": f"Ticket {'0' * 16}. {longest}.", "questions": KEV.workload.questions}))
    config = {"model_id": "kev-4b", "engine_url": "http://127.0.0.1:8008"}
    assert fingerprint(config, None, KEV.workload) == digest(
        {"endpoint": config, "engine": None, "workload": {"state_words": [150, 1200],
                                                          "questions": KEV.workload.questions}})
    assert fingerprint(config, None, KEV.workload) != fingerprint(config, None, CLEF.workload)
    assert mix_plan(KEV.workload) == {"state_words": [150, 1200], "questions": KEV.workload.questions}


def test_the_load_generator_gets_the_mix_and_the_questions_of_a_request_with_images() -> None:
    plan = mix_plan(CLEF.workload)
    assert plan["images"] == {"share": 0.25, "count": [1, 1], "sides_px": SIDES,
                              "questions": {**CLEF.workload.questions, **IMAGE_QUESTIONS}}
    assert set(IMAGE_QUESTIONS["background"]["criteria"]) == set(BACKGROUNDS)
    json.dumps(plan)  # as the plan goes to the load generator


def test_a_mix_with_images_is_checked() -> None:
    questions = KEV.workload.questions
    for part, why in (({"share": 0, "count": [1, 1], "sides_px": [256]}, "share"),
                      ({"share": 1.5, "count": [1, 1], "sides_px": [256]}, "share"),
                      ({"share": 0.5, "count": [0, 1], "sides_px": [256]}, "images.count"),
                      ({"share": 0.5, "count": [2, 1], "sides_px": [256]}, "images.count"),
                      ({"share": 0.5, "count": [1, 17], "sides_px": [256]}, "images.count"),
                      ({"share": 0.5, "count": [1, 1], "sides_px": []}, "sides_px"),
                      ({"share": 0.5, "count": [1, 1], "sides_px": [256, 0]}, "images.sides_px"),
                      ({"share": 0.5, "count": [1, 1], "sides_px": [8192]}, "images.sides_px")):
        with pytest.raises(ValidationError, match=why):
            Workload(state_words=(1, 2), questions=questions, images=part)
    with pytest.raises(ValidationError, match="background"):
        Workload(state_words=(1, 2), questions={**questions, **IMAGE_QUESTIONS},
                 images=WorkloadImages(share=0.5, count=(1, 1), sides_px=[256]))
    assert Workload(state_words=(1, 2), questions={**questions, **IMAGE_QUESTIONS}).images is None
