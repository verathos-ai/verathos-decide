"""Whether a reference engine may define a model's right answers now.

A reference must serve what the catalog pins. An engine that names the revision of the weights it serves in its
``GET /v1/models`` document (``verathos-engine`` does) is not used while it names another one; one that names none
is taken at its operator's word. When the catalog pins other weights or another engine for a model in place, miners
and references update at their own pace: for ``scoring.revision_transition_epochs`` epochs the model's answers are not
checked, so neither side is failed for the other's lag. A new revision is best published as a new model.
"""

from __future__ import annotations

from typing import Any

from verathos.catalog import Catalog, CatalogModel
from verathos.miner.engine import Engine


def served_revisions(info: Any) -> set[str]:
    """The revisions an engine's ``GET /v1/models`` document names for the models it serves."""
    models = (info.get("data") or info.get("models")) if isinstance(info, dict) else None
    if not isinstance(models, list):
        return set()
    return {m["revision"] for m in models if isinstance(m, dict) and isinstance(m.get("revision"), str)}


async def wrong_revision(engine: Engine, model: CatalogModel) -> str | None:
    """Why ``engine`` must not answer for ``model`` now: it names the weights it serves, and they are not the pinned
    ones. None if it may."""
    served = served_revisions(await engine.info())
    if served and model.hf_revision not in served:
        names = ", ".join(sorted(r[:12] for r in served))
        return f"it serves revision {names}, the catalog pins {model.hf_revision[:12]}"
    return None


def transitioning(pins: dict[str, list], cat: Catalog, epoch: int) -> set[str]:
    """The models whose pinned weights or engine changed less than ``scoring.revision_transition_epochs`` before
    ``epoch``. Keeps ``pins`` (model -> [what it pins, the epoch that changed it]) up to date; a model first seen is
    not in transition."""
    window, out = cat.scoring.revision_transition_epochs, set()
    for m in cat.models:
        pinned = f"{m.hf_revision}/{m.engine.revision}"
        if m.model_id not in pins:
            pins[m.model_id] = [pinned, epoch - window]
        elif pins[m.model_id][0] != pinned:
            pins[m.model_id] = [pinned, epoch]
        if epoch < pins[m.model_id][1] + window:
            out.add(m.model_id)
    return out
