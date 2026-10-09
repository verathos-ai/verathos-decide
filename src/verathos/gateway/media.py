"""The gateway's image decoders: worker processes that make users' images canonical (``protocol.media``).

Users' image files are decoded only here, each by a worker process: a fresh interpreter (no copy of the gateway's
memory, its hotkey included, and none of its descriptors) whose address space is ``WORKER_MEMORY_BYTES``, that writes no
core dump, is killed and replaced when an image takes it longer than ``timeout_s``, and is replaced after
``WORKER_IMAGES`` images. Once its decoders are loaded, and before it reads an image, a worker is confined for good
(``confine``): it opens and executes no file, so it reads none of the gateway's keys or configuration, creates no
socket and signals no process. Where the kernel offers no Landlock, no worker starts and no image is decoded. Workers
start with the first image. A request's images are decoded at once, each by a worker of its own as one is free; at
most ``max_in_flight`` requests are decoded or wait for a worker at a time, and more are told the decoders are busy.

A worker's reply is data, never code: a length-prefixed JSON header and the canonical image's base64, which the
gateway decodes strictly and whose form and size it checks again (``protocol.media.canonical_header``) before it
forwards the image or counts its tokens: a worker that a decoder's flaw took over can at worst send a wrong image.

    python -m verathos.gateway.media <memory bytes>   a worker: jobs on stdin, replies on stdout
"""

from __future__ import annotations

import asyncio
import binascii
import ctypes
import errno
import importlib
import json
import logging
import os
import platform
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
    JPEG,
    MAX_IMAGE_REQUEST_BYTES,
    PNG,
    ImageLimits,
    ImageRefused,
    b64decode,
    canonical_header,
    canonical_image,
    keep_out_of_dumps,
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


def _no_constants(name: str) -> Any:
    raise ValueError(f"{name} is not JSON")


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
        head = json.loads(reply[_FRAME.size:_FRAME.size + head_size], parse_constant=_no_constants)
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
            made = _checked(head, payload, PNG if content_type == PNG else JPEG, limits)
        except Exception as exc:  # whatever the reply holds, only a canonical image is forwarded
            self._retire(worker, f"its reply is not a canonical image ({type(exc).__name__})", index)
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
            if times := _throttled("start"):
                log.error("an image decoder could not start (%s: %s; %d times since last logged): it needs Pillow "
                          "(the gateway extra) and a kernel with Landlock", type(exc).__name__, exc, times)
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


_ENDS = {PNG: b"\0\0\0\0IEND\xaeB`\x82", JPEG: b"\xff\xd9"}  # a whole PNG ends with its end chunk, a JPEG with EOI


def _checked(head: dict[str, Any], payload: bytes, content_type: str, limits: ImageLimits) -> CanonicalImage:
    """The canonical image a worker's reply describes, checked: its media type the one expected, its base64 strict
    (so the forwarded JSON needs no escaping), its form and size those its header gives, and whole. Raises
    ValueError."""
    width, height = head["width"], head["height"]
    if head["content_type"] != content_type or type(width) is not int or type(height) is not int:
        raise ValueError("a reply that does not describe an image")
    data = b64decode(payload)
    size = canonical_header(data, content_type, token_px=limits.token_px, min_pixels=limits.min_pixels,
                            max_pixels=limits.max_pixels)
    if size != (width, height) or not data.endswith(_ENDS[content_type]):
        raise ValueError("a reply that does not describe its image")
    return CanonicalImage(content_type, width, height, payload.decode("ascii"))


# ---- the worker ----

_SYS_LANDLOCK_CREATE_RULESET, _SYS_LANDLOCK_RESTRICT_SELF = 444, 446  # the same on every architecture
_LANDLOCK_CREATE_RULESET_VERSION = 1
# the rights a Landlock ruleset handles, by the ABI version that added them: every access to files, and TCP's bind and
# connect (4); a ruleset without rules refuses all of them
_LANDLOCK_FS = {1: (1 << 13) - 1, 2: 1 << 13, 3: 1 << 14, 5: 1 << 15}
_LANDLOCK_NET = {4: 0b11}
_PR_SET_NO_NEW_PRIVS, _PR_SET_SECCOMP, _SECCOMP_MODE_FILTER = 38, 22, 2
_SECCOMP_RET_KILL_PROCESS, _SECCOMP_RET_ERRNO, _SECCOMP_RET_ALLOW = 0x80000000, 0x00050000, 0x7FFF0000
# per architecture: its audit identifier and the system calls a worker is refused (socket, socketpair, io_uring_setup,
# and those that signal a process: kill, tkill, tgkill, rt_sigqueueinfo, rt_tgsigqueueinfo, pidfd_send_signal)
_REFUSED_CALLS = {"x86_64": (0xC000003E, (41, 53, 425, 62, 200, 234, 129, 297, 424)),
                  "aarch64": (0xC00000B7, (198, 199, 425, 129, 130, 131, 138, 240, 424))}
_X32_CALLS = 0x40000000  # x86_64's x32 numbering, refused whole


class _RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64), ("handled_access_net", ctypes.c_uint64)]


class _SockFilter(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint16), ("jt", ctypes.c_uint8), ("jf", ctypes.c_uint8), ("k", ctypes.c_uint32)]


class _SockFprog(ctypes.Structure):
    _fields_ = [("len", ctypes.c_uint16), ("filter", ctypes.POINTER(_SockFilter))]


def _libc() -> ctypes.CDLL:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    return libc


def landlock_abi() -> int:
    """The version of Landlock the kernel offers; 0 for none."""
    if sys.platform != "linux":
        return 0
    return max(0, _libc().syscall(ctypes.c_long(_SYS_LANDLOCK_CREATE_RULESET), ctypes.c_void_p(None),
                                  ctypes.c_size_t(0), ctypes.c_uint32(_LANDLOCK_CREATE_RULESET_VERSION)))


def confine() -> None:
    """Confine this process for good: it opens and executes no file (Landlock), nor binds or connects a TCP port where
    Landlock handles them (Linux 6.7 and later), and it creates no socket and signals no process (seccomp; a system
    call of another architecture ends it). Raises OSError where the kernel offers no Landlock or the architecture is
    not one of ``_REFUSED_CALLS``."""
    abi, machine = landlock_abi(), platform.machine()
    if abi < 1:
        raise OSError("the kernel offers no Landlock (Linux 5.13 or later, with landlock among its security modules)")
    if machine not in _REFUSED_CALLS:
        raise OSError(f"no system call filter for {machine}")
    libc = _libc()
    attr = _RulesetAttr(sum(bits for version, bits in _LANDLOCK_FS.items() if version <= abi),
                        sum(bits for version, bits in _LANDLOCK_NET.items() if version <= abi))
    ruleset = libc.syscall(ctypes.c_long(_SYS_LANDLOCK_CREATE_RULESET), ctypes.byref(attr),
                           ctypes.c_size_t(ctypes.sizeof(attr)), ctypes.c_uint32(0))
    if ruleset < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset failed")
    try:
        if libc.prctl(_PR_SET_NO_NEW_PRIVS, ctypes.c_ulong(1), ctypes.c_ulong(0), ctypes.c_ulong(0),
                      ctypes.c_ulong(0)) != 0:
            raise OSError(ctypes.get_errno(), "prctl(PR_SET_NO_NEW_PRIVS) failed")
        filters = _filters(*_REFUSED_CALLS[machine])
        program = _SockFprog(len(filters), ctypes.cast(filters, ctypes.POINTER(_SockFilter)))
        if libc.prctl(_PR_SET_SECCOMP, ctypes.c_ulong(_SECCOMP_MODE_FILTER), ctypes.byref(program), ctypes.c_ulong(0),
                      ctypes.c_ulong(0)) != 0:
            raise OSError(ctypes.get_errno(), "prctl(PR_SET_SECCOMP) failed")
        if libc.syscall(ctypes.c_long(_SYS_LANDLOCK_RESTRICT_SELF), ctypes.c_long(ruleset), ctypes.c_uint32(0)) != 0:
            raise OSError(ctypes.get_errno(), "landlock_restrict_self failed")
    finally:
        os.close(ruleset)


def _filters(arch: int, refused: tuple[int, ...]) -> ctypes.Array:
    """The seccomp program: a system call of another architecture ends the process, the refused ones and x32's fail
    with EPERM, any other is allowed."""
    # BPF_LD|BPF_W|BPF_ABS, BPF_JMP|BPF_JEQ|BPF_K, BPF_JMP|BPF_JGE|BPF_K, BPF_RET|BPF_K
    load, equal, at_least, ret = 0x20, 0x15, 0x35, 0x06
    checks =[(at_least, _X32_CALLS), *((equal, number) for number in refused)]
    program = [(load, 0, 0, 4), (equal, 1, 0, arch), (ret, 0, 0, _SECCOMP_RET_KILL_PROCESS), (load, 0, 0, 0)]
    program += [(code, len(checks) - i, 0, k) for i, (code, k) in enumerate(checks)]  # a match: to the refusal
    program += [(ret, 0, 0, _SECCOMP_RET_ALLOW), (ret, 0, 0, _SECCOMP_RET_ERRNO | errno.EPERM)]
    return (_SockFilter * len(program))(*(_SockFilter(*instruction) for instruction in program))


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
    """Take jobs from stdin until it closes, one at a time, confined and under the resource limits (see the module).
    Where it cannot be confined, it returns without its ready mark."""
    import resource

    keep_out_of_dumps()
    from PIL import Image

    importlib.import_module("PIL.ImageOps")  # what canonical_image imports, loaded while modules may still be opened
    Image.init()  # every decoder, loaded before the address space is limited and the worker confined
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    try:
        confine()
    except OSError as exc:
        sys.stderr.write(f"an image decoder cannot be confined: {exc}\n")
        return
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
