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
from verathos.gateway.media import CanonicalImage, Decoders, DecodersBusy, _Worker, landlock_abi
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

pytestmark = [pytest.mark.usefixtures("lab_network"),
              pytest.mark.skipif(landlock_abi() < 1, reason="the gateway decodes images only where the kernel offers "
                                                            "Landlock (Linux 5.13 or later)")]

QUESTIONS = {"corner": {"type": "noul", "instructions": "Is there a green corner?"},
             "kind": {"type": "choice", "criteria": {"photo": "A photo", "drawing": "A drawing"}}}
REQUEST = {"model": "clef-flash", "state": "Look at the image.", "questions": QUESTIONS}


def reading_images(max_state_tokens: int | None = None, **limits) -> catalog.Catalog:
    """The shipped catalog with clef-flash reading images."""
    document = catalog.load().document()
    entry = {"input_modalities": ["text", "image"], **({"images": limits} if limits else {}),
             **({"max_state_tokens": max_state_tokens} if max_state_tokens else {})}
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


def test_the_body_limit_is_the_models_and_beyond_the_text_limit_a_body_holds_base64(tmp_path, setups) -> None:
    from tests.test_miner_app import REQUEST as TEXT

    engine = Recording("clef-flash")
    s = setups(tmp_path, engine, cat=reading_images(max_total_bytes=12 << 20, max_bytes=4 << 20))
    big = [item(os.urandom(3 << 20), "image/png")] * 3  # undecodable, but the body is read in full for clef-flash
    assert s.gateway.body_limit() == 13 << 20
    assert "images.0" in error(asyncio.run(s.post({**REQUEST, "images": big})))[2]
    # a text model's limit holds first, as it did when bodies were read up to it: before kev-4b's 503 here
    long = json.dumps({**TEXT, "state": "x" * (2 << 20)}).encode()
    assert error(asyncio.run(s.post(content_override=long))) == (413, "too_large", f"the body exceeds {1 << 20} bytes")
    # beyond the text limit, a body is base64 but for what a text body may hold: no costly structure is parsed
    for junk in (b"[" + b"{}," * (1 << 20) + b"{}]", json.dumps({**REQUEST, "state": " " * (2 << 20)}).encode()):
        assert error(asyncio.run(s.post(content_override=junk))) == (
            413, "too_large", f"the body exceeds {1 << 20} bytes besides its images' base64")
    over = json.dumps({**REQUEST, "images": [item(b"x" * (10 << 20), "image/png")]}).encode()
    assert error(asyncio.run(s.post(content_override=over)))[:2] == (413, "too_large")
    assert engine.received == []


def test_the_request_without_its_images_is_held_to_the_text_limit_before_any_image_is_decoded(tmp_path,
                                                                                               setups) -> None:
    class Untouched:
        async def canonical(self, images, limits):
            raise AssertionError("decoded")

        def close(self) -> None:
            pass

    s = setups(tmp_path, Recording("clef-flash"), cat=reading_images(max_state_tokens=1 << 20))
    s.gateway.decoders.close()
    s.gateway.decoders = Untouched()
    state = "\u0007" * ((1 << 20) // 6 + 100)  # each character canonical JSON as six bytes
    r = asyncio.run(s.post({**REQUEST, "state": state, "images": [item(encoded(256, 256), "image/png")]}))
    assert error(r) == (413, "too_large", f"the request without its images exceeds {1 << 20} bytes")


def test_credit_is_reserved_before_images_are_decoded(tmp_path, setups) -> None:
    """A request reserves the most its images may cost before they are decoded, so a prepaid account whose credit the
    requests in flight hold is refused before its next request's images are."""

    class Holding:
        calls = 0
        go = asyncio.Event()

        async def canonical(self, images, limits):
            Holding.calls += 1
            await Holding.go.wait()
            raise ImageRefused(400, "images.0: refused here")

        def close(self) -> None:
            pass

    s = setups(tmp_path, Recording("clef-flash"))
    s.gateway.decoders.close()
    s.gateway.decoders = Holding()
    asyncio.run(s.accounts.set_account("acme", prepaid=True))
    asyncio.run(s.accounts.add_credit("acme", 100, kind="grant", reference="100 micro-USD"))
    request = {**REQUEST, "images": [item(encoded(256, 256), "image/png")]}

    async def two():
        first = asyncio.create_task(s.post(request))
        while not Holding.calls:
            await asyncio.sleep(0.01)
        second = await s.post(request)  # the first holds more than the account's 100 micro-USD
        Holding.go.set()
        return await first, second

    first, second = asyncio.run(two())
    assert (first.status_code, second.status_code, Holding.calls) == (400, 402, 1)


def test_a_body_that_arrives_too_slowly_gets_408(tmp_path, setups) -> None:
    s = setups(tmp_path, Recording("clef-flash"), body_timeout_s=0.2)

    async def slow():
        yield json.dumps(REQUEST).encode()[:10]
        await asyncio.sleep(1.0)
        yield json.dumps(REQUEST).encode()[10:]

    async def post():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=s.app), base_url="http://gw") as c:
            return await c.post("/v1/systemone", content=slow(), headers={"Authorization": f"Bearer {s.key}"})

    assert error(asyncio.run(post())) == (408, "timeout", "the body took longer than 0.2 s")


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


def test_the_images_held_for_the_auditor_are_bounded_with_one_proof_with_images_per_endpoint(monkeypatch) -> None:
    from verathos.gateway import sampler as sampler_module

    monkeypatch.setattr(sampler_module, "IMAGES_HELD_BYTES", 2500)
    sampler = Sampler(kp(), httpx.AsyncClient(), auditor=None, auditor_hotkey="a", window_s=600,
                      rng=random.Random(5))
    with_image = {**REQUEST, "images": [{"content_type": "image/png", "base64": "A" * 1000}]}
    a, b, c, d = ("5F" + x * 46 for x in "abcd")
    sampler.proof(exchange(with_image, a))
    sampler.proof(exchange(with_image, a))  # the same endpoint's: one proves it
    sampler.proof(exchange(REQUEST, a))
    sampler.offer(exchange(with_image, b))  # 2,000 bytes of images held
    sampler.proof(exchange(with_image, c))  # 3,000 would be beyond
    sampler.offer(exchange(with_image, d))
    sampler.offer(exchange(REQUEST, d))  # text is not bounded
    assert [bool(e.request.get("images")) for e in sampler._proofs] == [True, False]
    assert sorted(sampler._picked) == [(f"{b}/gpu0", True), (f"{d}/gpu0", False)]


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
    import subprocess

    spawned: dict = {}
    popen = subprocess.Popen

    def spawn(*args, **kwargs):
        spawned.update(kwargs)
        return popen(*args, **kwargs)

    monkeypatch.setenv("GATEWAY_SECRET", "hunter2")
    monkeypatch.setattr(subprocess, "Popen", spawn)
    worker = _Worker(1 << 30)
    try:
        worker.ready(30)
        assert "GATEWAY_SECRET" not in spawned["env"] and spawned["cwd"] == "/" and spawned["close_fds"]
        proc = f"/proc/{worker.process.pid}"
        with open(f"{proc}/limits") as f:
            limits = {line[:26].strip(): line[26:].split() for line in f.read().splitlines()[1:]}
        assert limits["Max core file size"][:2] == ["0", "0"]
        assert limits["Max address space"][:2] == [str(1 << 30)] * 2
        with pytest.raises(PermissionError):  # not dumpable: its memory is closed to other processes
            open(f"{proc}/environ", "rb").read()
        with open(f"{proc}/status") as f:
            status = dict(line.split(":", 1) for line in f.read().splitlines())
        assert status["NoNewPrivs"].strip() == "1" and status["Seccomp"].strip() == "2"  # confined before it is ready
    finally:
        worker.close()


def test_a_confined_worker_reads_no_file_makes_no_socket_signals_no_process_and_runs_nothing(tmp_path) -> None:
    import socket
    import subprocess

    config = tmp_path / "gateway.json"
    config.write_text('{"database_url": "postgresql+asyncpg://gateway:hunter2@db/gateway"}')
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    code = f"""
import os, socket, sys
from verathos.gateway.media import confine
tcp = socket.socket()  # made before: where Landlock handles TCP, it still connects nowhere
confine()
for name, attempt in (("read", lambda: open({str(config)!r}).read()),
                      ("socket", lambda: socket.socket(socket.AF_INET, socket.SOCK_DGRAM)),
                      ("signal", lambda: os.kill(os.getppid(), 0)),
                      ("run", lambda: os.execv(sys.executable, [sys.executable, "-c", "print('ran')"])),
                      ("connect", lambda: tcp.connect(("127.0.0.1", {listener.getsockname()[1]})))):
    try:
        attempt()
        print(name, "done")
    except PermissionError:
        print(name, "refused")
"""
    try:
        run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    finally:
        listener.close()
    assert run.returncode == 0, run.stderr
    assert dict(line.split() for line in run.stdout.splitlines()) == {
        "read": "refused", "socket": "refused", "signal": "refused", "run": "refused",
        "connect": "refused" if landlock_abi() >= 4 else "done"}


@pytest.mark.parametrize(("unconfinable", "why"), [
    ("media.landlock_abi = lambda: 0", "the kernel offers no Landlock"),
    ("media.platform.machine = lambda: 'riscv64'", "no system call filter for riscv64"),
])
def test_a_worker_that_cannot_be_confined_reads_no_image(unconfinable, why) -> None:
    import subprocess

    from verathos.gateway.media import _frame

    job = _frame({"content_type": "image/png", "limits": ImageLimits().model_dump()}, b64(encoded(64, 64)).encode())
    run = subprocess.run([sys.executable, "-c", f"import verathos.gateway.media as media\n{unconfinable}\n"
                                                "media.serve(1 << 30)\n"], input=job, capture_output=True, timeout=60)
    assert run.stdout == b""  # neither its ready mark nor a reply
    assert f"an image decoder cannot be confined: {why}".encode() in run.stderr


def test_confined_workers_make_every_kind_of_image_canonical_as_the_gateway_itself_would() -> None:
    """All a worker decodes with is loaded before it is confined."""
    import binascii
    import io

    from PIL import Image

    from tests.images import drawn
    from verathos.protocol.media import JPEG, PNG, WEBP, canonical_image

    def saved(image, fmt: str, **params) -> bytes:
        out = io.BytesIO()
        image.save(out, fmt, **params)
        return out.getvalue()

    field, frames = drawn(320, 200), [drawn(96, 64), drawn(96, 64).transpose(Image.Transpose.ROTATE_180)]
    turned = Image.Exif()
    turned[0x0112] = 6
    images = {
        "png rgba": (PNG, saved(field.convert("RGBA"), "PNG")),
        "png palette, transparent": (PNG, saved(field.convert("P"), "PNG", transparency=0)),
        "png 16-bit grey": (PNG, saved(field.convert("I;16"), "PNG")),
        "png grey and alpha": (PNG, saved(field.convert("LA"), "PNG")),
        "png animated": (PNG, saved(frames[0], "PNG", save_all=True, append_images=frames[1:])),
        "jpeg turned by exif": (JPEG, saved(field, "JPEG", exif=turned)),
        "jpeg cmyk": (JPEG, saved(field.convert("CMYK"), "JPEG")),
        "jpeg grey, progressive": (JPEG, saved(field.convert("L"), "JPEG", progressive=True)),
        "jpeg decoded at a smaller scale": (JPEG, saved(drawn(3000, 2000), "JPEG")),
        "webp lossless rgba": (WEBP, saved(field.convert("RGBA"), "WEBP", lossless=True)),
        "webp animated": (WEBP, saved(frames[0], "WEBP", save_all=True, append_images=frames[1:])),
    }
    limits, decoders = ImageLimits(), Decoders(2, 4, 10.0)
    try:
        for name, (content_type, data) in images.items():
            (made,) = asyncio.run(decoders.canonical([(content_type, b64(data))], limits))
            here = canonical_image(data, content_type, limits)
            assert (made.content_type, made.width, made.height) == (here.content_type, here.width, here.height), name
            assert binascii.a2b_base64(made.base64) == here.data, name
    finally:
        decoders.close()


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


@pytest.mark.parametrize("lie", [
    lambda head, data: ({**head, "width": head["width"] + 32}, data),  # not the size its image has
    lambda head, data: ({**head, "width": float(head["width"])}, data),
    lambda head, data: ({**head, "content_type": "image/jpeg"}, data),
    # valid base64 first, then what would break out of the forwarded JSON
    lambda head, data: (head, data + b'"},{"base64":"QUJD","content_type":"image/png'),
    lambda head, data: (head, data[:-4]),
])
def test_a_decoders_reply_is_checked_before_it_is_trusted(lie) -> None:
    decoders = Decoders(1, 2, 5.0)
    worker = decoders._take()
    decoders._release(worker)
    real = worker.exchange
    worker.exchange = lambda header, payload, timeout_s: lie(*real(header, payload, timeout_s))
    try:
        with pytest.raises(ImageRefused, match="could not be decoded"):
            asyncio.run(decoders.canonical([("image/png", b64(encoded(256, 256)))], ImageLimits()))
        assert not worker.alive()  # replaced
    finally:
        decoders.close()


def test_a_workers_reply_is_one_frame_of_strict_json() -> None:
    import threading

    from verathos.gateway.media import _FRAME

    def answered(reply: bytes):
        job_r, job_w = os.pipe()
        reply_r, reply_w = os.pipe()
        worker = object.__new__(_Worker)
        worker._stdin, worker._stdout = job_w, reply_r
        os.set_blocking(job_w, False)
        os.set_blocking(reply_r, False)

        def play():
            os.read(job_r, 1 << 16)
            os.write(reply_w, reply)

        threading.Thread(target=play).start()
        try:
            return worker.exchange({"content_type": "image/png"}, b"QUJD", 5.0)
        finally:
            for fd in (job_r, job_w, reply_r, reply_w):
                os.close(fd)

    head = b'{"content_type": "image/png", "width": 256, "height": 256}'
    assert answered(_FRAME.pack(len(head), 4) + head + b"QUJD") == (json.loads(head), b"QUJD")
    for reply in (_FRAME.pack(55, 0) + b'{"content_type": "image/png", "width": Infinity, "x": 1}',
                  _FRAME.pack(len(head), 0) + head + b"more",
                  _FRAME.pack(1 << 30, 0)):
        with pytest.raises(ValueError):
            answered(reply)
