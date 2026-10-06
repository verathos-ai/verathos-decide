"""The auditor's load generator: ``verathos-loadgen`` (``native/loadgen``), built in place by
``verathos-build-loadgen``.

It reads one plan as JSON on stdin (targets with their URL, open-loop rate, headers and request mix), sends every
target its traffic at once, and writes per-target counts, latencies and a random sample of completed requests as JSON
on stdout. A request counts as delivered when its response (HTTP 200, a JSON object of at most 1 MiB) completes
inside the measured window; ``sent`` and ``late_sends`` count the sends scheduled inside it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

BINARY_PATH = Path(__file__).resolve().parent / "verathos-loadgen"
CRATE = Path(__file__).resolve().parents[3] / "native" / "loadgen"


class LoadgenError(RuntimeError):
    pass


class Latency(BaseModel):
    """Milliseconds over the requests completed in the window; None when none completed."""

    p50: float | None = None
    p95: float | None = None
    max: float | None = None


class Sample(BaseModel):
    request: dict[str, Any]
    request_id: str
    status: int
    response: Any
    sent_at_ms: int
    completed_at_ms: int


class TargetResult(BaseModel):
    id: str
    sent: int = Field(ge=0)
    skipped: int = Field(ge=0)
    completed: int = Field(ge=0)
    delivered_rps: float = Field(ge=0)
    latency_ms: Latency = Latency()
    errors: dict[str, int] = {}
    max_send_lag_ms: float = 0.0
    late_sends: int = Field(0, ge=0)
    samples: list[Sample] = []


class LoadResult(BaseModel):
    started_at_ms: int
    late_start_ms: int = 0
    targets: list[TargetResult]


async def run(plan: dict[str, Any], *, binary: Path = BINARY_PATH, timeout_s: float) -> LoadResult:
    """Run one plan. Raises LoadgenError if the binary is missing, fails, or returns something unexpected."""
    if not binary.exists():
        raise LoadgenError(f"{binary} not found; build it with verathos-build-loadgen")
    proc = await asyncio.create_subprocess_exec(str(binary), stdin=asyncio.subprocess.PIPE,
                                                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(proc.communicate(json.dumps(plan).encode()), timeout_s)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        raise LoadgenError(f"no result within {timeout_s:.0f} s") from None
    if proc.returncode != 0:
        raise LoadgenError(f"exit code {proc.returncode}: {err.decode(errors='replace')[-500:]}")
    try:
        result = LoadResult.model_validate_json(out)
    except ValueError as exc:
        raise LoadgenError(f"unreadable result: {exc}") from None
    if [t.id for t in result.targets] != [t["id"] for t in plan["targets"]]:
        raise LoadgenError("result targets do not match the plan")
    return result


def build() -> None:
    """``verathos-build-loadgen``: build the load generator from a source checkout and install it in place."""
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    if not (CRATE / "Cargo.toml").exists():
        raise SystemExit(f"{CRATE} not found; the load generator builds from a source checkout")
    subprocess.run(["cargo", "build", "--release", "--locked"], cwd=CRATE, check=True)
    shutil.copy2(CRATE / "target" / "release" / BINARY_PATH.name, BINARY_PATH)
    print(f"installed {BINARY_PATH}")
