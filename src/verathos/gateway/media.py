"""The gateway's image decoders: worker processes that make users' images canonical (``protocol.media``).

Users' image files are decoded only here, each by a worker process: a fresh interpreter (no copy of the gateway's
memory, its hotkey included, and none of its descriptors) whose address space is ``WORKER_MEMORY_BYTES``, that writes no
core dump, is killed and replaced when an image takes it longer than ``timeout_s``, and is replaced after
``WORKER_IMAGES`` images. Workers start with the first image. A request's images are decoded at once, each by a
worker of its own as one is free; at most ``max_in_flight`` requests are decoded or wait for a worker at a time, and
more are told the decoders are busy.

A worker's reply is data, never code: a length-prefixed JSON header and the canonical image's base64, whose form and
size the gateway checks again (``protocol.media.canonical_header``) before it forwards the image or counts its tokens.

    python -m verathos.gateway.media <memory bytes>   a worker: jobs on stdin, replies on stdout
"""

from __future__ import annotations

import asyncio
import binascii
import json
import logging
import os
import selectors
import struct
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from verathos.gateway import metrics
from verathos.gateway.logs import Throttle
from verathos.protocol.media import (
    CANONICAL_HEADER_BYTES,
    JPEG,
    MAX_IMAGE_REQUEST_BYTES,
    PNG,
    ImageLimits,
    ImageRefused,
    b64decode,
    canonical_header,
    canonical_image,
)

log = logging.getLogger(__name__)
_throttled = Throttle()
WORKER_MEMORY_BYTES = 1 << 30
WORKER_IMAGES = 256
START_TIMEOUT_S = 30.0  # for a worker to import its decoders
READY = b"R"
_FRAME = struct.Struct(">II")  # the lengths of a frame's JSON header and of its payload
MAX_HEADER_BYTES = 64 << 10
_ENVIRONMENT = ("PATH", "PYTHONPATH", "LANG", "LC_ALL")  # what a worker keeps of the gateway's environment


class DecodersBusy(Exception):
    """As many requests as the decoders take are being decoded or wait for a worker."""


class DecodingUnavailable(Exception):
    """No worker could be started."""


@dataclass(frozen=True)
class CanonicalImage:
    content_type: str
    width: int
    height: int
    base64: str


def _frame(header: dict[str, Any], payload: bytes) -> bytes:
    head = json.dumps(header).encode()
    return _FRAME.pack(len(head), len(payload)) + head + payload


class _Worker:
    """One worker process and its pipes, which the gateway side uses without blocking (``exchange``)."""

    def __init__(self, memory_bytes: int) -> None:
        self.images = 0
        self.process = subprocess.Popen(
            [sys.executable, "-m", "verathos.gateway.media", str(memory_bytes)], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, cwd="/", close_fds=True,
            env={key: os.environ[key] for key in _ENVIRONMENT if key in os.environ})
        self._stdin, self._stdout = self.process.stdin.fileno(), self.process.stdout.fileno()
        os.set_blocking(self._stdin, False)
        os.set_blocking(self._stdout, False)

    def alive(self) -> bool:
        return self.process.poll() is None

    def ready(self, timeout_s: float) -> None:
        """Wait for the worker's ready mark. Raises TimeoutError, EOFError or ValueError."""
        if self._read(timeout_s, lambda got: len(got) >= len(READY)) != READY:
            raise ValueError("the worker did not start as a decoder")

    def exchange(self, header: dict[str, Any], payload: bytes, timeout_s: float) -> tuple[dict[str, Any], bytes]:
        """Send one job and read its reply within ``timeout_s``. Raises TimeoutError, EOFError (the worker is gone) or
        ValueError (a reply that is not one frame within the bounds)."""
        deadline = time.monotonic() + timeout_s
        out = memoryview(_frame(header, payload))
        with selectors.DefaultSelector() as selector:
            selector.register(self._stdin, selectors.EVENT_WRITE)
            while out:
                if not selector.select(max(0.0, deadline - time.monotonic())):
                    raise TimeoutError
                try:
                    out = out[os.write(self._stdin, out[: 1 << 16]):]
                except BlockingIOError:
                    continue
                except BrokenPipeError:
                    raise EOFError from None
        reply = self._read(deadline - time.monotonic(), _whole_frame)
        head_size, size = _FRAME.unpack_from(reply)
        if len(reply) != _FRAME.size + head_size + size:
            raise ValueError("more than one frame")
        head = json.loads(reply[_FRAME.size:_FRAME.size + head_size])
        if not isinstance(head, dict):
            raise ValueError("a reply header that is not an object")
        return head, bytes(reply[_FRAME.size + head_size:])

    def _read(self, timeout_s: float, complete) -> bytearray:
        deadline = time.monotonic() + timeout_s
        got = bytearray()
        with selectors.DefaultSelector() as selector:
            selector.register(self._stdout, selectors.EVENT_READ)
            while not complete(got):
                if not selector.select(max(0.0, deadline - time.monotonic())):
                    raise TimeoutError
                try:
                    chunk = os.read(self._stdout, 1 << 20)
                except BlockingIOError:
                    continue
                if not chunk:
                    raise EOFError
                got += chunk
        return got

    def close(self) -> None:
        if self.alive():
            self.process.kill()
        self.process.wait()
        self.process.stdin.close()
        self.process.stdout.close()


def _whole_frame(got: bytearray) -> bool:
    """Whether ``got`` holds a whole frame; raises ValueError for one beyond the bounds."""
    if len(got) < _FRAME.size:
        return False
    head_size, size = _FRAME.unpack_from(got)
    if head_size > MAX_HEADER_BYTES or size > MAX_IMAGE_REQUEST_BYTES:
        raise ValueError("a frame beyond the bounds")
    return len(got) >= _FRAME.size + head_size + size


class Decoders:
    """The worker pool (see the module). Its jobs run in threads of its own, one per worker, so they wait for workers
    off the event loop."""

    def __init__(self, workers: int, max_in_flight: int, timeout_s: float, *,
                 memory_bytes: int = WORKER_MEMORY_BYTES) -> None:
        self.workers, self.max_in_flight, self.timeout_s, self.memory_bytes = (workers, max_in_flight, timeout_s,
                                                                               memory_bytes)
        self._threads = ThreadPoolExecutor(workers, thread_name_prefix="image-decoder")
        self._idle: list[_Worker] = []
        self._lock = threading.Lock()
        self._in_flight = 0
        self._closed = False

    async def canonical(self, images: list[tuple[str, str]], limits: ImageLimits) -> list[CanonicalImage]:
        """The canonical form of a request's images ((media type, base64) as sent), in their order. Raises
        DecodersBusy, DecodingUnavailable, or ImageRefused for the first image refused, its message naming the image
        (``images.<index>``)."""
        if self._in_flight >= self.max_in_flight:
            raise DecodersBusy
        self._in_flight += 1
        try:
            loop = asyncio.get_running_loop()
            results = await asyncio.gather(*(loop.run_in_executor(self._threads, self._decode, index, content_type,
                                                                  data, limits)
                                             for index, (content_type, data) in enumerate(images)),
                                           return_exceptions=True)
        finally:
            self._in_flight -= 1
        for result in results:
            if isinstance(result, DecodingUnavailable):
                raise result
        for index, result in enumerate(results):
            if isinstance(result, ImageRefused):
                raise ImageRefused(result.status, f"images.{index}: {result}")
            if isinstance(result, BaseException):
                raise result
        return results

    def _decode(self, index: int, content_type: str, data: str, limits: ImageLimits) -> CanonicalImage:
        worker = self._take()
        started = time.monotonic()
        try:
            head, payload = worker.exchange({"content_type": content_type, "limits": limits.model_dump()},
                                            data.encode("ascii"), self.timeout_s)
        except TimeoutError:
            self._retire(worker, f"an image took it longer than {self.timeout_s:g} s", index)
            raise ImageRefused(413, "it takes too long to decode") from None
        except Exception as exc:  # the worker is gone, or its reply is not one frame
            self._retire(worker, f"it failed on an image ({type(exc).__name__})", index)
            raise ImageRefused(400, "it could not be decoded") from None
        if "status" in head:
            if head.get("retire"):
                self._retire(worker, "it ran out of memory on an image", index)
            else:
                self._release(worker)
            metrics.IMAGES.labels("refused").inc()
            raise ImageRefused(413 if head["status"] == 413 else 400, str(head.get("message", ""))[:300])
        try:
            made = CanonicalImage(str(head["content_type"]), int(head["width"]), int(head["height"]),
                                  payload.decode("ascii"))
            expected = PNG if content_type == PNG else JPEG
            size = canonical_header(b64decode(made.base64, CANONICAL_HEADER_BYTES), expected,
                                    token_px=limits.token_px, min_pixels=limits.min_pixels,
                                    max_pixels=limits.max_pixels)
            if made.content_type != expected or size != (made.width, made.height):
                raise ValueError("a reply that does not describe its image")
        except (KeyError, TypeError, ValueError) as exc:
            self._retire(worker, f"its reply is not a canonical image: {exc}", index)
            raise ImageRefused(400, "it could not be decoded") from None
        self._release(worker)
        metrics.IMAGES.labels("canonical").inc()
        metrics.IMAGE_SECONDS.observe(time.monotonic() - started)
        metrics.IMAGE_BYTES.observe(len(payload) // 4 * 3)
        return made

    def _take(self) -> _Worker:
        with self._lock:
            while self._idle:
                worker = self._idle.pop()
                if worker.alive():
                    return worker
                worker.close()
        worker = None
        try:
            worker = _Worker(self.memory_bytes)
            worker.ready(START_TIMEOUT_S)
        except (OSError, TimeoutError, EOFError, ValueError) as exc:
            if worker is not None:
                worker.close()
            log.error("an image decoder could not start (%s: %s): is Pillow installed (the gateway extra)?",
                      type(exc).__name__, exc)
            raise DecodingUnavailable from None
        return worker

    def _release(self, worker: _Worker) -> None:
        worker.images += 1
        with self._lock:
            if worker.images < WORKER_IMAGES and not self._closed:
                self._idle.append(worker)
                return
        worker.close()

    def _retire(self, worker: _Worker, why: str, index: int) -> None:
        """Replace a worker that failed on a request's image ``index``."""
        metrics.IMAGES.labels("failed").inc()
        if times := _throttled(why):
            log.warning("an image decoder is replaced: %s (images.%d of a request; %d times since last logged)", why,
                        index, times)
        worker.close()

    def close(self) -> None:
        """Stop the idle workers; busy ones stop when their image is done."""
        with self._lock:
            self._closed, idle, self._idle = True, self._idle, []
        for worker in idle:
            worker.close()
        self._threads.shutdown(wait=False, cancel_futures=True)


# ---- the worker ----

def _job(header: dict[str, Any], payload: bytes) -> tuple[dict[str, Any], bytes]:
    """One image: (reply header, the canonical image's base64). ``retire`` in the header: the worker exits after."""
    limits = ImageLimits.model_validate(header["limits"])
    try:
        data = b64decode(payload)
        if len(data) > limits.max_bytes:
            raise ImageRefused(413, f"{len(data)} bytes, more than the model takes ({limits.max_bytes})")
        made = canonical_image(data, header["content_type"], limits)
        encoded = binascii.b2a_base64(made.data, newline=False)
        if len(encoded) > limits.max_request_bytes:
            raise ImageRefused(413, "made canonical, it exceeds the model's request size (max_request_bytes)")
    except ImageRefused as refused:
        return {"status": refused.status, "message": str(refused)}, b""
    except MemoryError:
        return {"status": 413, "message": "it takes more memory to decode than the gateway gives one image",
                "retire": True}, b""
    return {"content_type": made.content_type, "width": made.width, "height": made.height}, encoded


def serve(memory_bytes: int) -> None:
    """Take jobs from stdin until it closes, one at a time, under the resource limits (see the module)."""
    import resource

    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    from PIL import Image

    Image.init()  # the decoders are loaded before the address space is limited
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    jobs, replies = sys.stdin.buffer, sys.stdout.buffer
    replies.write(READY)
    replies.flush()
    while len(lengths := jobs.read(_FRAME.size)) == _FRAME.size:
        head_size, size = _FRAME.unpack(lengths)
        if head_size > MAX_HEADER_BYTES or size > MAX_IMAGE_REQUEST_BYTES:
            return
        head, payload = jobs.read(head_size), jobs.read(size)
        if len(payload) != size:
            return
        try:
            reply, data = _job(json.loads(head), payload)
        except Exception:
            reply, data = {"status": 400, "message": "it could not be decoded"}, b""
        replies.write(_frame(reply, data))
        replies.flush()
        if reply.get("retire"):
            return


if __name__ == "__main__":
    serve(int(sys.argv[1]))
