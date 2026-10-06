"""Clef models (Cloudflare): the model repository ships its own inference code (``joint_schema_model.py``) with a
SystemOne entry point; it is loaded from the pinned revision."""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def load(repo: str, revision: str, *, device: str, dtype: str):
    import torch
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(repo, revision=revision))
    spec = importlib.util.spec_from_file_location("joint_schema_model", path / "joint_schema_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model, processor = module.load_release_model(str(path), device=device, dtype=getattr(torch, dtype))
    model.eval()
    log.info("loaded %s@%s on %s (%s)", repo, revision[:12], device, dtype)

    def run(request: dict[str, Any]) -> dict[str, Any]:
        return module.systemone(model, processor, request)

    return run
