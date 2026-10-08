"""HTTP server around a decision engine: ``POST /v1/systemone`` and ``GET /v1/models``.

Requests are prepared (validated and tokenized) in worker threads as they arrive, then queued for one device thread,
which takes everything waiting, up to ``max_batch`` requests, and runs it as one ``engine.run`` call. A request that
arrives while a batch runs joins the next one, so batches grow with load and a lone request waits for nobody.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from concurrent.futures import Future
from typing import Any, Protocol

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from verathos.protocol.api import SystemOneRequest

log = logging.getLogger(__name__)


class Engine(Protocol):
    def prepare(self, body: dict[str, Any]) -> Any: ...

    def run(self, items: list[Any]) -> list[dict[str, Any]]: ...


class Batcher:
    """One device thread running queued prepared requests in batches."""

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
            try:
                results = self.engine.run([item for item, _ in pending])
            except Exception as exc:  # a failed batch fails each of its requests, the thread keeps serving
                log.exception("batch of %d failed", len(pending))
                for _, future in pending:
                    future.set_exception(exc)
                continue
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
        try:
            body = await request.json()
            SystemOneRequest.model_validate(body)
            item = await asyncio.to_thread(engine.prepare, body)
        except (ValueError, ValidationError) as exc:
            return JSONResponse({"error": f"invalid request: {exc}"}, status_code=422)
        try:
            return JSONResponse(await asyncio.wrap_future(batcher.submit(item)))
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=422)

    return app
