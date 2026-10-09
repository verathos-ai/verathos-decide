from __future__ import annotations

import json

from verathos import catalog
from verathos.protocol.canonical import digest


def test_shipped_catalog_loads_and_digest_is_stable() -> None:
    c = catalog.load()
    assert c.get("kev-4b") is not None
    # all three active, sharing the emission equally without demand; publishable: the anchors sum to 1
    assert c.anchor_shares(strict=True) == {"kev-4b": 0.333334, "kev-9b": 0.333333, "clef-flash": 0.333333}
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

    first = catalog.load()  # mechanism 0
    second = catalog.Catalog.parse({**first.document(), "mechanism": 1,
                                    "gateways": [{"hotkey": Keypair.create_from_uri("//gw").ss58_address}]})
    committed = {"catalog": first}
    monkeypatch.setattr(follow, "current_catalog", lambda *args, **kwargs: committed["catalog"])
    store = catalog.CatalogStore(tmp_path)

    def start() -> follow.CatalogFollower:
        return follow.CatalogFollower(None, "test", 405, store, [], state_path=tmp_path / "followed.json")

    running = start()
    assert running.accepted() == {first.digest(): 0}
    committed["catalog"] = second
    # each catalog's manifests are for its own mechanism
    assert running.refresh(None) and running.accepted() == {first.digest(): 0, second.digest(): 1}
    assert start().accepted() == {first.digest(): 0, second.digest(): 1}  # restarted inside the grace
    committed["catalog"] = first  # changed back while the validator was down: miners still announce the second
    assert start().accepted() == {first.digest(): 0, second.digest(): 1}
    state = json.loads((tmp_path / "followed.json").read_text())
    (tmp_path / "followed.json").write_text(json.dumps({**state, "until": 0}))
    assert start().accepted() == {first.digest(): 0}  # the grace is over
    # a state kept by an earlier release names no mechanisms: the stored catalog's
    (tmp_path / "followed.json").write_text(json.dumps({"catalog": second.digest(), "previous": None, "until": 0}))
    store.put(second)
    assert start().accepted() == {first.digest(): 0, second.digest(): 1}


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
    assert {m.model_id: m.relative_speed for m in catalog.load().models} == {"kev-4b": 1.0, "kev-9b": 0.92,
                                                                            "clef-flash": 0.85}
    without = [{k: v for k, v in m.items() if k != "relative_speed"} for m in document["models"]]
    assert {m.relative_speed for m in catalog.Catalog.parse({**document, "models": without}).models} == {1.0}
    for speed in (0, -1.0):
        with pytest.raises(ValidationError, match="relative_speed"):
            catalog.Catalog.parse({**document, "models": [{**document["models"][0], "relative_speed": speed}]})


def test_the_miner_share_is_a_number_or_a_linear_change_and_the_same_for_every_validator() -> None:
    import pytest
    from pydantic import ValidationError

    document = catalog.load().document()

    def with_share(share) -> catalog.Catalog:
        return catalog.Catalog.parse({**document, "scoring": {**document["scoring"], "miner_share": share}})

    shipped = catalog.load().scoring
    assert shipped.miner_share_at(0) == shipped.miner_share_at(10**7) == 0.59  # not started: the owner starts it
    without = {**document, "scoring": {k: v for k, v in document["scoring"].items() if k != "miner_share"}}
    assert catalog.Catalog.parse(without).scoring.miner_share_at(5) == 1.0  # older catalogs: everything to miners
    assert with_share(0.7).scoring.miner_share_at(5) == 0.7
    change = with_share({"from": 0.59, "to": 0.15, "start_epoch": 1000, "epochs": 2400}).scoring
    assert [change.miner_share_at(e) for e in (0, 1000, 1001, 2200, 3399, 3400, 10**7)] == [
        0.59, 0.59, 0.589817, 0.37, 0.150183, 0.15, 0.15]
    rising = with_share({"from": 0.2, "to": 0.8, "start_epoch": 10, "epochs": 3}).scoring
    assert [rising.miner_share_at(e) for e in (10, 11, 12, 13)] == [0.2, 0.4, 0.6, 0.8]
    assert with_share({"from": 0.59, "to": 0.15, "start_epoch": 7, "epochs": 2}).unknown_settings() == []
    assert with_share({"from": 0.59, "to": 0.15, "epochs": 2, "start": 7}).unknown_settings() == [
        "scoring.miner_share.start"]
    for invalid in (0, 1.01, -0.5, {"from": 0.59, "to": 0, "epochs": 1}, {"from": 1.2, "to": 0.5, "epochs": 1},
                    {"from": 0.59, "to": 0.15, "epochs": 0}, {"from": 0.59, "to": 0.15},
                    {"from": 0.59, "to": 0.15, "epochs": 5, "start_epoch": -1}, [0.59, 0.15]):
        with pytest.raises(ValidationError, match="miner_share"):
            with_share(invalid)
    built = catalog.load().model_copy(update={"revision": "built"})  # a catalog built in code keeps the JSON names
    assert built.document()["scoring"]["miner_share"] == document["scoring"]["miner_share"]
    assert catalog.Catalog.parse(built.document()).digest() == built.digest()


def with_models(**changes: dict) -> dict:
    """The shipped catalog's document with fields of some models changed (model id -> fields)."""
    document = catalog.load().document()
    return {**document, "models": [{**m, **changes.get(m["model_id"], {})} for m in document["models"]]}


def test_models_read_text_until_their_entry_lists_image() -> None:
    from verathos.protocol.media import ImageLimits

    shipped = catalog.load()
    assert {m.model_id: m.reads_images() for m in shipped.models} == {"kev-4b": False, "kev-9b": False,
                                                                       "clef-flash": True}
    assert shipped.get("clef-flash").images == ImageLimits()  # the defaults, written out
    assert all("input_modalities" not in m and "images" not in m for m in shipped.document()["models"]
               if m["model_id"] != "clef-flash")
    seeing = catalog.Catalog.parse(with_models(**{"clef-flash": {"input_modalities": ["text", "image"],
                                                                  "images": {"max_count": 2}}}))
    model = seeing.get("clef-flash")
    assert model.reads_images() and model.modalities() == ["text", "image"]
    assert model.images == ImageLimits(max_count=2) and seeing.digest() != shipped.digest()
    assert seeing.input_problems() == [] and seeing.unknown_settings() == []
    newer = catalog.Catalog.parse(with_models(**{"clef-flash": {"input_modalities": ["text", "image", "audio"]}}))
    assert newer.get("clef-flash").modalities() == ["text", "image"]  # what this code serves of what it lists


def test_input_this_code_does_not_serve_is_found_before_publishing() -> None:
    import pytest
    from pydantic import ValidationError

    problems = catalog.Catalog.parse(with_models(
        **{"kev-4b": {"input_modalities": ["text", "image"]},
           "kev-9b": {"input_modalities": ["text", "video"]},
           "clef-flash": {"input_modalities": ["text", "image"], "images": {"token_px": 16}}})).input_problems()
    assert problems == ["models.kev-4b: its kev engine reads no images",
                        "models.kev-9b.input_modalities: video is not served",
                        "models.clef-flash.images.token_px: the clef engine's image tokens are 32 pixels square, not "
                        "16"]
    below = catalog.Catalog.parse(with_models(**{"clef-flash": {"input_modalities": ["text", "image"],
                                                                 "images": {"min_pixels": 16_384}}}))
    assert below.input_problems() == ["models.clef-flash.images: the clef engine takes images of 65536 to 16777216 "
                                      "pixels, not 16384 to 1048576"]
    mistyped = catalog.Catalog.parse(with_models(**{"clef-flash": {"images": {"max_images": 2}}}))
    assert mistyped.unknown_settings() == ["models.clef-flash.images.max_images"]
    for modalities in (["image"], ["text", "text"], []):
        with pytest.raises(ValidationError, match="input_modalities"):
            catalog.Catalog.parse(with_models(**{"clef-flash": {"input_modalities": modalities}}))
    with pytest.raises(ValidationError, match="max_pixels"):
        catalog.Catalog.parse(with_models(**{"clef-flash": {"images": {"max_pixels": 32 << 20}}}))


def test_a_request_mix_carries_images_only_for_a_model_that_reads_them_and_within_its_limits() -> None:
    import pytest
    from pydantic import ValidationError

    shipped = catalog.load()
    mix = shipped.get("clef-flash").workload.document()
    assert mix["images"] == {"share": 0.25, "count": [1, 1], "sides_px": list(range(256, 513, 32))}
    assert all("images" not in m["workload"] for m in shipped.document()["models"] if m["model_id"] != "clef-flash")

    def with_mix(mix_images: dict, model_id: str = "clef-flash", **fields) -> catalog.Catalog:
        workload = {**catalog.load().get(model_id).workload.document(), "images": mix_images}
        return catalog.Catalog.parse(with_models(**{model_id: {"workload": workload, **fields}}))

    assert with_mix({**mix["images"], "count": [1, 4]}).get("clef-flash").workload.images.count == (1, 4)
    for images, fields, why in (
            (mix["images"], {"input_modalities": ["text"]}, "clef-flash reads no images"),
            ({**mix["images"], "count": [1, 3]}, {"images": {"max_count": 2}}, "takes at most 2 images"),
            ({**mix["images"], "sides_px": [256, 300]}, {}, "multiple of images.token_px"),
            ({**mix["images"], "sides_px": [224, 256]}, {}, "outside images.min_pixels"),
            ({**mix["images"], "sides_px": [2048]}, {}, "outside images.min_pixels"),
            ({**mix["images"], "sides_px": [512]}, {"images": {"max_bytes": 4096, "max_total_bytes": 4096}},
             "images.max_bytes"),
            ({**mix["images"], "sides_px": [16, 201 * 16]},
             {"images": {"token_px": 16, "min_pixels": 256, "max_pixels": (201 * 16) ** 2}}, "aspect ratio")):
        with pytest.raises(ValidationError, match=why):
            with_mix(images, **fields)
    with pytest.raises(ValidationError, match="kev-4b reads no images"):
        with_mix(mix["images"], "kev-4b")
    mistyped = with_mix({**mix["images"], "side_px": [256]})
    assert mistyped.unknown_settings() == ["models.clef-flash.workload.images.side_px"]
    assert catalog.Catalog.parse(with_models(**{"kev-4b": {"workload": {
        **catalog.load().get("kev-4b").workload.document(), "image": {}}}})).unknown_settings() == [
        "models.kev-4b.workload.image"]
