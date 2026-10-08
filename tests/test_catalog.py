from __future__ import annotations

import json

from verathos import catalog
from verathos.protocol.canonical import digest


def test_shipped_catalog_loads_and_digest_is_stable() -> None:
    c = catalog.load()
    assert c.get("kev-4b") is not None
    assert c.digest() == catalog.load().digest()
    assert catalog.committed_digest(catalog.commitment_value(c.digest())) == c.digest()


def variant(**changes) -> catalog.Catalog:
    return catalog.Catalog.parse({**catalog.load().document(), **changes})


def test_digest_changes_with_content() -> None:
    assert variant(revision="next").digest() != catalog.load().digest()


def test_digest_covers_fields_this_code_does_not_know() -> None:
    newer = variant(future_setting={"x": 1})
    assert newer.get("kev-4b") is not None
    assert newer.digest() != catalog.load().digest()


def test_rejects_non_catalog_commitment() -> None:
    assert catalog.committed_digest("verathos/manifest/v1 sha256:" + "a" * 64) is None
    assert catalog.committed_digest(None) is None


def test_resolve_prefers_shipped_then_store_then_verified_download(tmp_path, monkeypatch) -> None:
    import pytest

    shipped = catalog.load()
    store = catalog.CatalogStore(tmp_path)
    assert catalog.resolve(shipped.digest(), store=store, sources=[]) is not None

    newer = variant(revision="next", future_setting=True)
    served = {"good": newer.document(), "forged": variant(revision="x").document()}
    monkeypatch.setattr(catalog, "get_json", lambda url, **_: served[url.split("/")[2]])
    with pytest.raises(catalog.CatalogUnavailable):
        catalog.resolve(newer.digest(), store=store, sources=["http://forged/{digest}"])
    got = catalog.resolve(newer.digest(), store=store, sources=["http://forged/{digest}", "http://good/{digest}"])
    assert got.digest() == newer.digest() and store.get(newer.digest()).digest() == newer.digest()


def test_the_committed_shipped_catalog_outlives_its_release(tmp_path, monkeypatch) -> None:
    """A release shipping another catalog still finds the one committed, with every source down."""
    shipped = catalog.load()
    store = catalog.CatalogStore(tmp_path)
    catalog.resolve(shipped.digest(), store=store, sources=[])
    next_release, load = variant(revision="next-release"), catalog.load
    monkeypatch.setattr(catalog, "load", lambda path=None: load(path) if path else next_release)
    assert catalog.resolve(shipped.digest(), store=store, sources=[]).digest() == shipped.digest()


def test_changed_copies_are_digested_from_their_fields() -> None:
    c = catalog.load()
    assert c.model_copy(update={"revision": "other"}).digest() != c.digest()


def test_a_fetched_catalog_is_checked_against_the_commitment_before_parsing(tmp_path, monkeypatch) -> None:
    """A source serving anything but the committed document is skipped, whatever its content would do to the
    parser."""
    import verathos.catalog as catalog_module

    real = catalog_module.load()
    doc = json.loads(json.dumps(real.document()))
    doc["revision"] = "next"
    wanted = catalog_module.Catalog.parse(doc).digest()
    hostile = {**doc, "throughput_audit": {**doc["throughput_audit"], "lead_blocks": "5"}}
    served = {"https://a/": hostile, "https://b/": doc}
    monkeypatch.setattr(catalog_module, "get_json", lambda url, **kw: served[url.split(wanted)[0]])
    got = catalog_module.resolve(wanted, store=catalog_module.CatalogStore(tmp_path),
                                 sources=["https://a/{digest}", "https://b/{digest}"])
    assert got.digest() == wanted


def test_a_stored_catalog_of_an_older_format_does_not_stop_a_node(tmp_path) -> None:
    """A node upgraded before the owner commits the catalog for its release finds the old one unreadable: it is
    not available here, and the node keeps looking (and waiting) instead of failing."""
    import pytest

    old = {**catalog.load().document()}
    old["models"] = [{k: v for k, v in m.items() if k != "price"} for m in old["models"]]
    old_digest = digest(old)
    store = catalog.CatalogStore(tmp_path)
    store.path(old_digest).parent.mkdir(parents=True, exist_ok=True)
    store.path(old_digest).write_text(json.dumps(old))
    assert store.get(old_digest) is None
    with pytest.raises(catalog.CatalogUnavailable):
        catalog.resolve(old_digest, store=store, sources=[])


def test_manifests_of_the_previous_catalog_stay_accepted_across_a_validator_restart(tmp_path, monkeypatch) -> None:
    from bittensor_wallet import Keypair

    from verathos.validator import follow

    first = catalog.load()
    second = catalog.Catalog.parse({**first.document(), "gateways": [Keypair.create_from_uri("//gw").ss58_address]})
    committed = {"catalog": first}
    monkeypatch.setattr(follow, "current_catalog", lambda *args, **kwargs: committed["catalog"])
    store = catalog.CatalogStore(tmp_path)

    def start() -> follow.CatalogFollower:
        return follow.CatalogFollower(None, "test", 405, store, [], state_path=tmp_path / "followed.json")

    running = start()
    assert running.accepted() == {first.digest()}
    committed["catalog"] = second
    assert running.refresh(None) and running.accepted() == {first.digest(), second.digest()}
    assert start().accepted() == {first.digest(), second.digest()}  # restarted inside the grace
    committed["catalog"] = first  # changed back while the validator was down: miners still announce the second
    assert start().accepted() == {first.digest(), second.digest()}
    state = json.loads((tmp_path / "followed.json").read_text())
    (tmp_path / "followed.json").write_text(json.dumps({**state, "until": 0}))
    assert start().accepted() == {first.digest()}  # the grace is over


def test_settings_this_code_does_not_know_are_found_before_publishing() -> None:
    document = catalog.load().document()
    mistyped = {**document, "scoring": {**document.get("scoring", {}), "min_model_shares": 0.0},
                "models": [{**document["models"][0], "prise": 1.0}] + document["models"][1:]}
    assert catalog.load().unknown_settings() == []
    unknown = catalog.Catalog.parse(mistyped).unknown_settings()
    assert unknown == ["scoring.min_model_shares", f"models.{document['models'][0]['model_id']}.prise"]


def test_a_models_relative_speed_is_positive_and_one_when_absent() -> None:
    import pytest
    from pydantic import ValidationError

    document = catalog.load().document()
    assert {m.model_id: m.relative_speed for m in catalog.load().models} == {"kev-4b": 1.0, "kev-9b": 0.85,
                                                                            "clef-flash": 1.3}
    without = [{k: v for k, v in m.items() if k != "relative_speed"} for m in document["models"]]
    assert {m.relative_speed for m in catalog.Catalog.parse({**document, "models": without}).models} == {1.0}
    for speed in (0, -1.0):
        with pytest.raises(ValidationError, match="relative_speed"):
            catalog.Catalog.parse({**document, "models": [{**document["models"][0], "relative_speed": speed}]})
