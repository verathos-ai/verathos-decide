"""SystemOne engines for model families that ship no server of their own: ``verathos-engine``.

A miner's endpoint and a validator's reference both reach their engine over ``POST /v1/systemone``. Families with
an upstream server (Kev: ``kev.serve``) use it; the others are served here.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def code_version() -> str:
    """A digest of this package's source. ``GET /v1/models`` reports it and the installer writes it into the engine's
    service, so new engine code restarts the engine and the miner measures it again."""
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()[:12]
