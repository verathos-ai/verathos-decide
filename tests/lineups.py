"""Catalogs with a pinned line-up of models, for tests of how a line-up is treated (emission shares, candidates,
references per model) that must not change with the line-up the shipped catalog has."""

from __future__ import annotations

from verathos import catalog


def two_active_and_a_candidate() -> catalog.Catalog:
    """The shipped catalog with exactly kev-4b and kev-9b active, anchored at half the emission each, and clef-flash a
    candidate, each model otherwise as the shipped catalog defines it."""
    document = catalog.load().document()
    shipped = {m["model_id"]: m for m in document["models"]}
    lineup = {"kev-4b": ("active", 0.5), "kev-9b": ("active", 0.5), "clef-flash": ("candidate", 0.0)}
    return catalog.Catalog.parse({**document, "models": [{**shipped[m], "status": status, "anchor_share": anchor}
                                                         for m, (status, anchor) in lineup.items()]})
