from __future__ import annotations

from verathos import catalog


def test_shipped_catalog_loads_and_digest_is_stable() -> None:
    c = catalog.load()
    assert c.get("kev-4b") is not None
    assert c.digest() == catalog.load().digest()
    assert catalog.committed_digest(catalog.commitment_value(c.digest())) == c.digest()


def test_digest_changes_with_content() -> None:
    c = catalog.load()
    changed = c.model_copy(update={"revision": c.revision + "x"})
    assert changed.digest() != c.digest()


def test_rejects_non_catalog_commitment() -> None:
    assert catalog.committed_digest("verathos/manifest/v1 sha256:" + "a" * 64) is None
    assert catalog.committed_digest(None) is None


def test_resolve_prefers_shipped_then_store_then_verified_download(tmp_path, monkeypatch) -> None:
    import httpx
    import pytest

    shipped = catalog.load()
    store = catalog.CatalogStore(tmp_path)
    assert catalog.resolve(shipped.digest(), store=store, sources=[]) is not None

    newer = shipped.model_copy(update={"revision": "next"})
    served = {"good": newer.model_dump_json(), "forged": shipped.model_copy(update={"revision": "x"}).model_dump_json()}

    def fake_get(url, **_):
        return httpx.Response(200, text=served[url.split("/")[2]], request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    with pytest.raises(catalog.CatalogUnavailable):
        catalog.resolve(newer.digest(), store=store, sources=["http://forged/{digest}"])
    got = catalog.resolve(newer.digest(), store=store, sources=["http://forged/{digest}", "http://good/{digest}"])
    assert got.digest() == newer.digest() and store.get(newer.digest()) is not None


def test_only_active_models_are_scored() -> None:
    c = catalog.load()
    candidate = c.model_copy(update={"models": [c.models[0].model_copy(update={"status": "candidate"})]})
    assert c.scored("kev-4b") and not candidate.scored("kev-4b") and not c.scored("unknown")
