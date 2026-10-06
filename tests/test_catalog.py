from __future__ import annotations

from verathos import catalog


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


def test_changed_copies_are_digested_from_their_fields() -> None:
    c = catalog.load()
    assert c.model_copy(update={"revision": "other"}).digest() != c.digest()
