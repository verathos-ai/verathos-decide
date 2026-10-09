"""Images at the gateway: per-model modality, the limits, canonical images in decode workers, what is forwarded and
what its receipt binds, the billing bounds, the image limits users read, the sampler's reservoirs and the size of
a sample post."""

from __future__ import annotations

import asyncio
import json
import os
import random
import sys

import httpx
import pytest

from tests.images import b64, data_url, encoded, item
from tests.mock_engine import MockEngine
from tests.test_gateway import Setup
from tests.test_throughput import kp
from verathos import catalog
from verathos.gateway.app import Rejected, check_request, with_images
from verathos.gateway.media import CanonicalImage, Decoders, DecodersBusy, _Worker
from verathos.gateway.sampler import BATCH_BYTES, Sampler, batches
from verathos.gateway.upstream import Exchange
from verathos.protocol.api import MAX_RESPONSE_BYTES
from verathos.protocol.canonical import canonical_json, digest
from verathos.protocol.gateway import GatewaySample, Route
from verathos.protocol.media import (
    MAX_IMAGE_REQUEST_BYTES,
    ImageLimits,
    ImageRefused,
    b64decode,
    header,
    image_tokens,
)
from verathos.protocol.receipts import SignedReceipt

pytestmark = pytest.mark.usefixtures("lab_network")

QUESTIONS = {"corner": {"type": "noul", "instructions": "Is there a green corner?"},
             "kind": {"type": "choice", "criteria": {"photo": "A photo", "drawing": "A drawing"}}}
REQUEST = {"model": "clef-flash", "state": "Look at the image.", "questions": QUESTIONS}


def reading_images(**limits) -> catalog.Catalog:
    """The shipped catalog with clef-flash reading images."""
    document = catalog.load().document()
    entry = {"input_modalities": ["text", "image"], **({"images": limits} if limits else {})}
    return catalog.Catalog.parse({**document, "models": [{**m, **entry} if m["model_id"] == "clef-flash" else m
                                                          for m in document["models"]]})


class Recording(MockEngine):
    """Answers as the mock engine, keeps the requests it receives, and signs the input tokens it is given."""

    def __init__(self, model: str, tokens: int | None = None) -> None:
        super().__init__(model)
        self.received: list[dict] = []
        self.tokens = tokens

    async def answer(self, request):
        self.received.append(request)
        response = await super().answer(request)
        if self.tokens is not None:
            response["usage"] = {"input_tokens": self.tokens, "output_tokens": 0}
        return response


@pytest.fixture()
def setups():
    made: list[Setup] = []

    def make(tmp_path, engine=None, *, model: str = "clef-flash", cat=None, **kwargs) -> Setup:
        s = Setup(tmp_path, {"10.0.0.1": engine or Recording(model)}, model=model, cat=cat or reading_images(),
                  **kwargs)
        made.append(s)
        return s

    yield make
    for s in made:
        s.gateway.decoders.close()


def error(response: httpx.Response) -> tuple[int, str, str]:
    body = response.json()["error"]
    return response.status_code, body["type"], body["message"]


def test_a_model_that_reads_text_only_refuses_images_and_no_request_has_videos_or_processor_arguments(
        tmp_path, setups) -> None:
    engine = Recording("kev-4b")
    s = setups(tmp_path, engine, model="kev-4b")
    from tests.test_miner_app import REQUEST as TEXT

    image = data_url(encoded(256, 256), "image/png")
    assert error(asyncio.run(s.post({**TEXT, "images": [image]}))) == (400, "invalid_request",
                                                                        "kev-4b reads text only")
    assert error(asyncio.run(s.post({**TEXT, "videos": [image]}))) == (400, "invalid_request",
                                                                        "videos are not supported")
    status, kind, message = error(asyncio.run(s.post({**TEXT, "media_kwargs": {"do_resize": False}})))
    assert (status, kind) == (400, "invalid_request") and "media_kwargs is not accepted" in message
    assert engine.received == []  # none reached a miner
    answered = asyncio.run(s.post({**TEXT, "images": [], "user": "someone"}))
    assert answered.status_code == 200 and "image_tokens" not in answered.json()["usage"]
    assert engine.received == [TEXT]  # forwarded as its model, state and questions


def test_images_are_forwarded_canonical_and_billed_their_exact_tokens(tmp_path, setups) -> None:
    engine = Recording("clef-flash")
    s = setups(tmp_path, engine)
    photo, screenshot = encoded(4032, 3024, "JPEG"), encoded(800, 600, "PNG")
    request = {**REQUEST, "images": [data_url(photo, "image/jpeg"), item(screenshot, "image/png")], "extra": 1}
    r = asyncio.run(s.post(request))
    assert r.status_code == 200, r.text
    (forwarded,) = engine.received
    assert set(forwarded) == {"model", "state", "questions", "images"}  # other fields are dropped
    sizes = []
    for sent, expected in zip(forwarded["images"], (("image/jpeg", 1152, 864), ("image/png", 800, 608))):
        found = header(b64decode(sent["base64"]))
        assert (sent["content_type"], found.width, found.height) == expected and found.canonical
        sizes.append((found.width, found.height))
    tokens = sum(image_tokens(w, h, 32) for w, h in sizes) + 1
    assert tokens == 974 + 477 + 1
    body = r.json()
    receipt = SignedReceipt.model_validate(body["receipt"])
    assert receipt.binds(forwarded, body["answers"]) and not receipt.binds(request, body["answers"])
    usage = body["usage"]
    assert usage["image_tokens"] == tokens and usage["input_tokens"] >= tokens  # what the miner signed, within bounds


def test_the_signed_input_tokens_are_bounded_by_the_exact_image_tokens(tmp_path, setups) -> None:
    images = [data_url(encoded(1024, 1024, "PNG"), "image/png")]
    request = {**REQUEST, "images": images}
    text_bytes = len(canonical_json(REQUEST))
    for signed, billed in ((0, len(REQUEST["state"]) // 8 + 1027), (10 ** 9, text_bytes + 1027), (1200, 1200)):
        s = setups(tmp_path / str(signed), Recording("clef-flash", tokens=signed))
        assert asyncio.run(s.post(request)).json()["usage"] == {"input_tokens": billed, "image_tokens": 1027}


@pytest.mark.parametrize(("images", "status", "message"), [
    (lambda: ["http://169.254.169.254/latest/meta-data/"], 400, "images.0: remote URLs are not accepted"),
    (lambda: ["https://www.youtube.com/watch?v=dQw4w9WgXcQ"], 400, "images.0: remote URLs are not accepted"),
    (lambda: ["/etc/passwd"], 400, "images.0: not an image"),
    (lambda: [b64(encoded(64, 64))], 400, "images.0: not an image"),
    (lambda: [{"content_type": "image/gif", "base64": b64(encoded(64, 64, "GIF"))}], 400, "content_type must be"),
    (lambda: [item(encoded(64, 64, "GIF"), "image/png")], 400, "images.0: not a PNG, JPEG or WebP image"),
    (lambda: [item(encoded(64, 64, "JPEG"), "image/png")], 400, "images.0: its bytes are image/jpeg"),
    (lambda: [{"content_type": "image/png", "base64": "iVBO Rw0K"}], 400, "images.0: not base64"),
    (lambda: [item(encoded(64, 64), "image/png"), item(encoded(4100, 20), "image/png")], 400,
     "images.1: 4100 x 20 pixels: an aspect ratio beyond 200:1"),
    (lambda: [item(encoded(64, 64), "image/png")] * 5, 413, "clef-flash takes at most 4 images per request"),
    (lambda: [item(encoded(4097, 4096), "image/png")], 413, "images.0: 4097 x 4096 pixels, more than"),
    (lambda: [item(os.urandom(5 << 20), "image/png")], 413, "images.0: 5242880 bytes, more than"),
    (lambda: [item(b"\x89PNG\r\n\x1a\n" + b"\0" * 64, "image/png")], 400, "images.0: it could not be decoded"),
])
def test_images_beyond_the_limits_are_refused(tmp_path, setups, images, status, message) -> None:
    engine = Recording("clef-flash")
    s = setups(tmp_path, engine)
    got = error(asyncio.run(s.post({**REQUEST, "images": images()})))
    assert got[0] == status and got[1] == ("too_large" if status == 413 else "invalid_request")
    assert message in got[2]
    assert engine.received == []


def test_the_body_limit_is_the_models(tmp_path, setups) -> None:
    from tests.test_miner_app import REQUEST as TEXT

    s = setups(tmp_path, Recording("clef-flash"), cat=reading_images(max_total_bytes=12 << 20, max_bytes=4 << 20))
    big = [item(os.urandom(3 << 20), "image/png")] * 3  # undecodable, but the body is read in full for clef-flash
    assert s.gateway.body_limit() == 13 << 20
    assert "images.0" in error(asyncio.run(s.post({**REQUEST, "images": big})))[2]
    padded = json.dumps({**TEXT, "state": "x" * 1000}).encode() + b" " * (2 << 20)
    assert error(asyncio.run(s.post(content_override=padded))) == (
        413, "too_large", f"the body exceeds {1 << 20} bytes, kev-4b's limit")
    over = json.dumps({**REQUEST, "images": [item(b"x" * (10 << 20), "image/png")]}).encode()
    assert error(asyncio.run(s.post(content_override=over)))[:2] == (413, "too_large")


def test_busy_decoders_and_ones_that_cannot_start_answer_503(tmp_path, setups) -> None:
    s = setups(tmp_path, Recording("clef-flash"))

    class Unavailable:
        def __init__(self, error: Exception) -> None:
            self.error = error

        async def canonical(self, images, limits):
            raise self.error

        def close(self) -> None:
            pass

    from verathos.gateway.media import DecodingUnavailable

    request = {**REQUEST, "images": [data_url(encoded(256, 256), "image/png")]}
    s.gateway.decoders.close()
    for raised, expected in ((DecodersBusy(), (503, "busy", "the gateway's image decoders are all busy")),
                             (DecodingUnavailable(), (503, "unavailable", "the gateway cannot decode images "
                                                                          "right now"))):
        s.gateway.decoders = Unavailable(raised)
        r = asyncio.run(s.post(request))
        assert error(r) == expected and int(r.headers["Retry-After"]) >= 1


def test_the_models_list_says_what_each_model_reads(tmp_path, setups) -> None:
    s = setups(tmp_path, Recording("clef-flash"), cat=reading_images(max_count=2))

    async def models():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=s.app), base_url="http://gw") as c:
            return {m["id"]: m for m in (await c.get("/v1/models")).json()["data"]}

    listed = asyncio.run(models())
    assert listed["kev-4b"]["input_modalities"] == ["text"] and listed["kev-4b"]["images"] is None
    clef = listed["clef-flash"]
    assert clef["input_modalities"] == ["text", "image"]
    assert clef["images"] == {**ImageLimits(max_count=2).model_dump(), "max_aspect_ratio": 200,
                              "formats": ["image/png", "image/jpeg", "image/webp"]}


def test_the_forwarded_body_is_the_canonical_json_of_the_forwarded_request() -> None:
    images = [CanonicalImage("image/png", 256, 256, b64(encoded(256, 256))),
              CanonicalImage("image/jpeg", 512, 256, b64(encoded(512, 256, "JPEG")))]
    forwarded = {"model": "clef-flash", "state": {"note": "é   \"quoted\""}, "questions": QUESTIONS}
    body = with_images(canonical_json(forwarded), images)
    as_sent = {**forwarded, "images": [{"content_type": i.content_type, "base64": i.base64} for i in images]}
    assert body == canonical_json(as_sent) and json.loads(body) == as_sent


def test_check_request_gives_the_bounds_of_what_is_billed(tmp_path, setups) -> None:
    s = setups(tmp_path, Recording("clef-flash"))
    checked = asyncio.run(check_request(s.gateway, json.dumps(
        {**REQUEST, "images": [item(encoded(512, 512), "image/png")]}).encode()))
    assert checked.image_tokens == 258 + 1
    assert checked.max_tokens == len(canonical_json(REQUEST)) + 259
    assert checked.min_tokens == len(REQUEST["state"]) // 8 + 259
    assert checked.request_sha256 == digest(checked.raw) and checked.body == canonical_json(checked.raw)
    text = asyncio.run(check_request(s.gateway, json.dumps({**REQUEST, "images": []}).encode()))
    assert (text.image_tokens, text.max_tokens, text.raw) == (0, len(canonical_json(REQUEST)), REQUEST)
    with pytest.raises(Rejected, match="reads text only"):
        asyncio.run(check_request(s.gateway, json.dumps({**REQUEST, "model": "kev-4b", "images": [
            item(encoded(512, 512), "image/png")]}).encode()))


def exchange(request: dict, hotkey: str = "5F" + "x" * 46) -> Exchange:
    route = Route(hotkey=hotkey, endpoint_id="gpu0", model_id="clef-flash", url="https://203.0.113.5:8091/e/gpu0",
                  tls="ab" * 32, credited_rps=10.0, latency_ms=100.0)
    return Exchange(route, request, '{"answers": {}}', {}, {}, 10)


def test_the_sampler_keeps_answers_with_images_apart_from_text_answers() -> None:
    sampler = Sampler(kp(), httpx.AsyncClient(), auditor=None, auditor_hotkey="a", window_s=600,
                      rng=random.Random(5))
    with_image = {**REQUEST, "images": [{"content_type": "image/png", "base64": "QUJD"}]}
    for _ in range(50):
        sampler.offer(exchange(REQUEST))
    sampler.offer(exchange(with_image))  # one answer with an image among many: it is picked all the same
    picks = list(sampler._picked.values())
    assert len(picks) == 2 and sorted(bool(e.request.get("images")) for e in picks) == [False, True]


def test_the_largest_request_an_image_model_takes_fits_a_sample_post() -> None:
    request = {**REQUEST, "images": [{"content_type": "image/png", "base64": ""}]}
    request["images"][0]["base64"] = "A" * (MAX_IMAGE_REQUEST_BYTES - len(canonical_json(request)))
    assert len(canonical_json(request)) == MAX_IMAGE_REQUEST_BYTES
    response = json.dumps({"answers": {"q": {"type": "noul", "noul": 0.5}}, "pad": '"' * (MAX_RESPONSE_BYTES - 80)})
    sample = GatewaySample(hotkey="5F" + "x" * 46, endpoint_id="gpu0", model_id="clef-flash", request=request,
                           response=response[:MAX_RESPONSE_BYTES], test_id="ab" * 32).model_dump_json()
    assert len(sample) <= BATCH_BYTES and batches([sample]) == [[sample]]
    with pytest.raises(ValueError, match="max_request_bytes"):
        ImageLimits(max_request_bytes=MAX_IMAGE_REQUEST_BYTES + 1)


@pytest.mark.skipif(sys.platform != "linux", reason="reads the worker's state from /proc")
def test_a_decoder_runs_apart_with_limits_and_none_of_the_gateways_secrets(monkeypatch) -> None:
    monkeypatch.setenv("GATEWAY_SECRET", "hunter2")
    worker = _Worker(1 << 30)
    try:
        worker.ready(30)
        proc = f"/proc/{worker.process.pid}"
        assert sorted(os.listdir(f"{proc}/fd")) == ["0", "1", "2"]
        with open(f"{proc}/environ", "rb") as f:
            assert b"GATEWAY_SECRET" not in f.read()
        with open(f"{proc}/limits") as f:
            limits = {line[:26].strip(): line[26:].split() for line in f.read().splitlines()[1:]}
        assert limits["Max core file size"][:2] == ["0", "0"]
        assert limits["Max address space"][:2] == [str(1 << 30)] * 2
        assert os.readlink(f"{proc}/cwd") == "/"
    finally:
        worker.close()


def test_a_decoder_is_replaced_when_an_image_takes_too_long_or_too_much_memory(caplog, monkeypatch) -> None:
    from verathos.gateway import media
    from verathos.gateway.logs import Throttle

    monkeypatch.setattr(media, "_throttled", Throttle())
    limits, big = ImageLimits(), b64(encoded(4096, 4096, mode="RGBA"))

    async def canonical(decoders: Decoders, data: str):
        return await decoders.canonical([("image/png", data)], limits)

    for decoders, why in ((Decoders(1, 2, 0.05), "takes too long"), (Decoders(1, 2, 5.0, memory_bytes=200 << 20),
                                                                       "more memory")):
        try:
            with pytest.raises(ImageRefused, match=why) as refused:
                asyncio.run(canonical(decoders, big))
            assert refused.value.status == 413
            (made,) = asyncio.run(canonical(decoders, b64(encoded(256, 256))))  # by a new worker
            assert (made.width, made.height) == (256, 256)
        finally:
            decoders.close()
    replaced = [r.getMessage() for r in caplog.records if "decoder is replaced" in r.getMessage()]
    assert len(replaced) == 2 and all("images.0 of a request" in m for m in replaced)
    assert not any(big[1000:1100] in r.getMessage() for r in caplog.records)  # no image data in a log line
    decoders = Decoders(1, 2, 10.0)  # the defaults decode the largest image they take
    try:
        (made,) = asyncio.run(canonical(decoders, big))
        assert (made.width, made.height) == (1024, 1024)
    finally:
        decoders.close()


def test_requests_beyond_the_decoders_in_flight_limit_are_told_they_are_busy() -> None:
    decoders = Decoders(1, 1, 10.0)
    slow = b64(encoded(4096, 4096, mode="RGBA"))

    async def two():
        first = asyncio.create_task(decoders.canonical([("image/png", slow)], ImageLimits()))
        await asyncio.sleep(0)
        with pytest.raises(DecodersBusy):
            await decoders.canonical([("image/png", b64(encoded(64, 64)))], ImageLimits())
        return await first

    try:
        assert asyncio.run(two())[0].width == 1024
    finally:
        decoders.close()


def test_a_decoders_reply_is_checked_before_it_is_trusted() -> None:
    decoders = Decoders(1, 2, 5.0)
    worker = decoders._take()
    decoders._release(worker)
    real = worker.exchange

    def lying(header, payload, timeout_s):  # a reply whose image is not the size it claims
        head, data = real(header, payload, timeout_s)
        return {**head, "width": head["width"] + 32}, data

    worker.exchange = lying
    try:
        with pytest.raises(ImageRefused, match="could not be decoded"):
            asyncio.run(decoders.canonical([("image/png", b64(encoded(256, 256)))], ImageLimits()))
        assert not worker.alive()  # replaced
    finally:
        decoders.close()
