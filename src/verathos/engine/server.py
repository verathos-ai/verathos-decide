"""HTTP server around a model runner: ``POST /v1/systemone`` and ``GET /v1/models``."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from verathos.protocol.api import SystemOneRequest

Runner = Callable[[dict[str, Any]], dict[str, Any]]


def create_app(runner: Runner, served_models: list[str] | None = None) -> FastAPI:
    app = FastAPI(title="verathos engine", docs_url=None, redoc_url=None, openapi_url=None)
    lock = threading.Lock()  # one forward pass at a time on the device

    def run(body: dict[str, Any]) -> dict[str, Any]:
        with lock:
            return runner(body)

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {"data": [{"id": m} for m in served_models or []]}

    @app.post("/v1/systemone")
    async def systemone(request: Request) -> JSONResponse:
        try:
            body = await request.json()
            SystemOneRequest.model_validate(body)
        except (ValueError, ValidationError) as exc:
            return JSONResponse({"error": f"invalid request: {exc}"}, status_code=422)
        try:
            return JSONResponse(await asyncio.to_thread(run, body))
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=422)

    return app
