from __future__ import annotations

from decide import catalog


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
    assert catalog.committed_digest("decide/v1 manifest sha256:" + "a" * 64) is None
    assert catalog.committed_digest(None) is None
