"""HTTP server around a decision engine: ``POST /v1/systemone`` and ``GET /v1/models``.

A request's body is at most ``MAX_BODY_BYTES`` (413 beyond): the largest request a catalog model may take, with room
for the miner's encoding of it, which may escape characters its canonical JSON does not.
Requests are prepared (validated and tokenized) in worker threads as they arrive, then queued for one device thread,
which takes everything waiting, up to ``max_batch`` requests, lets the engine split it into passes and runs them one
after the other, answering each pass's requests when it ends. A request that arrives while passes run joins the next
batch, so batches grow with load and a lone request waits for nobody. A failed pass of several requests runs them
again one at a time, so only a request that fails on its own (one too long for the memory, say) gets an error.
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
from concurrent.futures import Future
from typing import Any, Protocol

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from verathos.protocol.api import SystemOneRequest
from verathos.protocol.media import MAX_IMAGE_REQUEST_BYTES

log = logging.getLogger(__name__)
MAX_BODY_BYTES = 2 * MAX_IMAGE_REQUEST_BYTES


class Engine(Protocol):
    def prepare(self, body: dict[str, Any]) -> Any: ...

    def passes(self, items: list[Any]) -> list[list[int]]:
        """The items' indexes grouped into the passes that run them."""
        ...

    def run(self, items: list[Any]) -> list[dict[str, Any]]: ...


class Batcher:
    """One device thread running queued prepared requests in passes."""

    def __init__(self, engine: Engine, max_batch: int = 64) -> None:
        self.engine = engine
        self.max_batch = max_batch
        self.queue: queue.Queue[tuple[Any, Future]] = queue.Queue()
        self.thread = threading.Thread(target=self._work, name="engine-device", daemon=True)
        self.thread.start()

    def submit(self, item: Any) -> Future:
        future: Future = Future()
        self.queue.put((item, future))
        return future

    def _work(self) -> None:
        while True:
            pending = [self.queue.get()]
            while len(pending) < self.max_batch:
                try:
                    pending.append(self.queue.get_nowait())
                except queue.Empty:
                    break
            pending = [(item, future) for item, future in pending if future.set_running_or_notify_cancel()]
            if not pending:
                continue
            try:
                passes = self.engine.passes([item for item, _ in pending])
            except Exception:
                log.exception("grouping %d requests into passes failed; running them one at a time", len(pending))
                passes = [[i] for i in range(len(pending))]
            for indexes in passes:
                self._run([pending[i] for i in indexes])

    def _run(self, pending: list[tuple[Any, Future]]) -> None:
        try:
            results = self.engine.run([item for item, _ in pending])
        except Exception as exc:
            if len(pending) > 1:
                log.warning("a pass of %d requests failed (%s: %s); running them one at a time", len(pending),
                            type(exc).__name__, exc)
                for one in pending:
                    self._run([one])
                return
            if isinstance(exc, ValueError):
                log.info("request refused: %s", exc)
            else:
                log.exception("request failed")
            pending[0][1].set_exception(exc)
            return
        for (_, future), result in zip(pending, results):
            future.set_result(result)


def create_app(engine: Engine, served_models: list[dict[str, Any]] | None = None, max_batch: int = 64) -> FastAPI:
    """``served_models`` describe what the engine serves (``GET /v1/models``): id, revision, device and dtype, so a
    miner can tell when its engine changed."""
    app = FastAPI(title="verathos engine", docs_url=None, redoc_url=None, openapi_url=None)
    batcher = Batcher(engine, max_batch=max_batch)

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {"data": served_models or []}

    @app.post("/v1/systemone")
    async def systemone(request: Request) -> JSONResponse:
        declared, raw = request.headers.get("content-length", ""), bytearray()
        too_large = JSONResponse({"error": f"request larger than {MAX_BODY_BYTES} bytes"}, status_code=413)
        if declared.isdigit() and int(declared) > MAX_BODY_BYTES:
            return too_large
        async for chunk in request.stream():
            raw += chunk
            if len(raw) > MAX_BODY_BYTES:
                return too_large
        try:
            body = json.loads(raw)
            SystemOneRequest.model_validate(body)
            item = await asyncio.to_thread(engine.prepare, body)
        except (ValueError, ValidationError, RecursionError) as exc:
            return JSONResponse({"error": f"invalid request: {exc}"}, status_code=422)
        try:
            return JSONResponse(await asyncio.wrap_future(batcher.submit(item)))
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=422)

    return app
