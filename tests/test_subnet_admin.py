"""verathos-subnet: what the owner sees before and after committing a catalog."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from verathos import catalog
from verathos.subnet import admin

OWNER = "5Owner"
OWNER_COLDKEY = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"


class Chain:
    def __init__(self, block: int = 1000 * 180, mechanisms: int = 1) -> None:
        self.block, self.mechanisms, self.commitments = block, mechanisms, []
        self.substrate = SimpleNamespace(query=lambda module, name, params: SimpleNamespace(
            value=OWNER_COLDKEY if name == "SubnetOwner" else None))

    def get_subnet_owner_hotkey(self, netuid: int, block: int | None = None) -> str:
        return OWNER

    def get_uid_for_hotkey_on_subnet(self, hotkey: str, netuid: int, block: int | None = None) -> int:
        return 0

    def get_commitment(self, netuid: int, uid: int, block: int | None = None) -> str | None:
        return self.commitments[-1] if self.commitments else None

    def get_current_block(self) -> int:
        return self.block

    def get_mechanism_count(self, netuid: int) -> int:
        return self.mechanisms

    def set_commitment(self, wallet, netuid: int, value: str):
        self.commitments.append(value)
        return SimpleNamespace(success=True)


AUDITOR = {"tls": "ab" * 32, "egress": ["9.9.9.9"]}


def catalog_file(tmp_path, mechanism: int = 0, auditor: dict | None = AUDITOR, **scoring):
    document = catalog.load().document()
    path = tmp_path / "catalog.json"
    document = {**document, "mechanism": mechanism, "scoring": {**document["scoring"], **scoring}}
    if auditor is not None:
        document["auditor"] = auditor
    path.write_text(json.dumps(document))
    return path


def commit(chain: Chain, path, tmp_path, validator_config=None) -> None:
    admin.commit_catalog(chain, SimpleNamespace(hotkey=SimpleNamespace(ss58_address=OWNER)), 405, path,
                         tmp_path / "store", validator_config)


def test_commit_catalog_needs_the_auditors_pin_and_egress_and_the_gateways_egress(tmp_path) -> None:
    chain = Chain()
    for auditor, why in ((None, "the auditor's TLS pin"), ({"tls": "ab" * 32}, "auditor.egress")):
        with pytest.raises(SystemExit, match=why):
            commit(chain, catalog_file(tmp_path, auditor=auditor), tmp_path)
    with pytest.raises(ValueError, match="pattern"):
        commit(chain, catalog_file(tmp_path, auditor={"tls": "AB" * 32, "egress": ["9.9.9.9"]}), tmp_path)
    for egress in (["10.0.0.1"], ["9.9.9.9", "9.9.9.9"], ["::ffff:9.9.9.9"], ["2606:4700:0::1111"]):
        with pytest.raises(ValueError, match="egress|public|unique|write the address"):
            commit(chain, catalog_file(tmp_path, auditor={"tls": "ab" * 32, "egress": egress}), tmp_path)
    path = catalog_file(tmp_path)
    gateway = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
    path.write_text(json.dumps({**json.loads(path.read_text()), "gateways": [{"hotkey": gateway}]}))
    with pytest.raises(SystemExit, match=f"the egress addresses of {gateway}"):
        commit(chain, path, tmp_path)
    assert chain.commitments == []
    path.write_text(json.dumps({**json.loads(path.read_text()), "gateways": [{"hotkey": gateway,
                                                                              "egress": ["2606:4700::1111"]}]}))
    commit(chain, path, tmp_path)
    cat = catalog.load(path)
    assert chain.commitments == [catalog.commitment_value(cat.digest())]
    assert cat.gateway_hotkeys() == [gateway] and cat.auditor_pin() == "ab" * 32


def test_commit_catalog_on_the_auditors_machine_fills_in_its_pin_and_egress(tmp_path, capsys) -> None:
    from verathos.protocol import tls

    chain = Chain()
    state = tmp_path / "validator"
    identity = tls.identity(state)
    config = tmp_path / "validator.json"
    config.write_text(json.dumps({"state_dir": str(state), "egress_ips": ["9.9.9.9", "2606:4700::1111"]}))
    commit(chain, catalog_file(tmp_path, auditor=None), tmp_path, config)
    stored = catalog.CatalogStore(tmp_path / "store").get(chain.commitments[-1].rsplit(":", 1)[1])
    assert stored.auditor.tls == identity.pin and stored.auditor.egress == ["9.9.9.9", "2606:4700::1111"]
    assert f"filled in auditor.tls {identity.pin}" in capsys.readouterr().out
    with pytest.raises(SystemExit, match=f"the auditor's certificate in {state} has pin {identity.pin}"):
        commit(chain, catalog_file(tmp_path), tmp_path, config)  # a pin the auditor does not serve
    commit(chain, catalog_file(tmp_path, auditor={"tls": identity.pin, "egress": ["9.9.9.9"]}), tmp_path, config)
    assert catalog.CatalogStore(tmp_path / "store").get(
        chain.commitments[-1].rsplit(":", 1)[1]).auditor.egress == ["9.9.9.9"]  # the file's own egress stays


def test_commit_catalog_creates_the_auditors_certificate_when_the_auditor_has_not_yet(tmp_path, capsys) -> None:
    """A catalog change the running auditor cannot read stops it before its service creates a certificate: run on the
    auditor's machine, commit-catalog creates it where the service then finds it."""
    from verathos.protocol import tls

    chain = Chain()
    state = tmp_path / "validator"
    config = tmp_path / "validator.json"
    config.write_text(json.dumps({"state_dir": str(state), "egress_ips": ["9.9.9.9"]}))
    assert tls.stored_pin(state) is None
    commit(chain, catalog_file(tmp_path, auditor=None), tmp_path, config)
    pin = tls.stored_pin(state)
    assert pin is not None and tls.identity(state).pin == pin  # the service reuses it
    stored = catalog.CatalogStore(tmp_path / "store").get(chain.commitments[-1].rsplit(":", 1)[1])
    assert stored.auditor.tls == pin
    assert f"created the auditor's TLS certificate in {state}" in capsys.readouterr().out


def test_commit_catalog_states_the_miner_share_now_and_how_it_changes(tmp_path, capsys) -> None:
    chain = Chain()
    commit(chain, catalog_file(tmp_path), tmp_path)
    assert ("miner share 0.590000 in epoch 1000; the change to 0.15 over 2400 epochs has no start_epoch yet"
            in capsys.readouterr().out)
    path = catalog_file(tmp_path, miner_share={"from": 0.59, "to": 0.15, "start_epoch": 400, "epochs": 2400})
    commit(chain, path, tmp_path)
    assert ("miner share 0.480000 in epoch 1000; 0.59 until epoch 400, then linearly to 0.15 at epoch 2800"
            in capsys.readouterr().out)
    assert chain.commitments[-1] == catalog.commitment_value(catalog.load(path).digest())
    commit(chain, catalog_file(tmp_path, miner_share=0.8), tmp_path)
    assert "miner share 0.800000 in epoch 1000\n" in capsys.readouterr().out


def test_commit_catalog_states_the_usage_pool_and_refuses_the_owner_coldkey_as_its_buyback(tmp_path, capsys) -> None:
    chain = Chain()
    commit(chain, catalog_file(tmp_path), tmp_path)
    assert "usage pool: off" in capsys.readouterr().out
    buyback = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
    pool = {"mode": "burned", "fraction": 1.0, "period_epochs": 40, "buyback_coldkey": buyback}
    commit(chain, catalog_file(tmp_path, usage_pool=pool), tmp_path)
    assert (f"usage pool (burned): 1 of the alpha {buyback} burns on the subnet in each period of 40 epochs, paid "
            "over the next period to the hotkeys that served paying users, 0 hotkeys or coldkeys excluded; epoch "
            "1000 is in period 25, which pays period 24's") in capsys.readouterr().out
    with pytest.raises(SystemExit, match="buyback_coldkey is the subnet owner's coldkey"):
        commit(chain, catalog_file(tmp_path, usage_pool={**pool, "buyback_coldkey": OWNER_COLDKEY}), tmp_path)
    with pytest.raises(SystemExit, match="scoring.usage_pool.coldkey"):
        commit(chain, catalog_file(tmp_path, usage_pool={**pool, "coldkey": buyback}), tmp_path)
    with pytest.raises(ValueError, match="buyback_coldkey"):
        commit(chain, catalog_file(tmp_path, usage_pool={"mode": "burned", "fraction": 1.0}), tmp_path)
    assert len(chain.commitments) == 2
    commit(chain, catalog_file(tmp_path, usage_pool={"mode": "revenue", "fraction": 0.9}), tmp_path)
    assert "usage pool (revenue): 0.9 of the paid usage's revenue in alpha" in capsys.readouterr().out


def test_commit_catalog_refuses_an_invalid_or_mistyped_miner_share(tmp_path) -> None:
    chain = Chain()
    path = catalog_file(tmp_path, miner_share={"from": 0.59, "to": 0.15, "start": 400, "epochs": 2400})
    with pytest.raises(SystemExit, match="scoring.miner_share.start"):
        commit(chain, path, tmp_path)
    path = catalog_file(tmp_path, miner_share={"from": 0.59, "to": 0.15, "start_epoch": 400, "epochs": 0})
    with pytest.raises(ValueError, match="miner_share"):
        commit(chain, path, tmp_path)
    assert chain.commitments == []


def test_commit_catalog_needs_the_catalogs_mechanism_on_chain_and_says_when_the_network_moves(tmp_path,
                                                                                            capsys) -> None:
    chain = Chain(mechanisms=1)  # a subnet with mechanism 0 only, as SN96 at 256 UIDs
    with pytest.raises(SystemExit, match="the catalog names mechanism 1, but netuid 405 has 1 mechanism"):
        commit(chain, catalog_file(tmp_path, mechanism=1), tmp_path)
    assert chain.commitments == []
    commit(chain, catalog_file(tmp_path, mechanism=0), tmp_path)
    assert "mechanism 0; active models" in capsys.readouterr().out
    chain.mechanisms = 2  # a second mechanism set up
    commit(chain, catalog_file(tmp_path, mechanism=1), tmp_path)
    out = capsys.readouterr().out
    assert "the network moves from mechanism 0 to 1" in out and len(chain.commitments) == 2
    commit(chain, catalog_file(tmp_path, mechanism=1, miner_share=0.8), tmp_path)
    assert "moves" not in capsys.readouterr().out


def test_commit_catalog_publishes_images_only_where_the_model_and_its_engine_read_them(tmp_path) -> None:
    from pydantic import ValidationError

    chain = Chain()

    def with_input(model_id: str, **fields) -> object:
        path = catalog_file(tmp_path)
        document = json.loads(path.read_text())
        document["models"] = [{**m, **fields} if m["model_id"] == model_id else m for m in document["models"]]
        path.write_text(json.dumps(document))
        return path

    mix = catalog.load().get("clef-flash").workload.document()
    text_mix = {k: v for k, v in mix.items() if k != "images"}
    for model_id, fields, why in (
            ("kev-4b", {"input_modalities": ["text", "image"]}, "kev-4b: its kev engine reads no images"),
            ("kev-9b", {"input_modalities": ["text", "video"]}, "video is not served"),
            ("clef-flash", {"input_modalities": ["text", "image"], "images": {"token_px": 64, "min_pixels": 65_536},
                            "workload": text_mix}, "images.token_px"),
            ("clef-flash", {"input_modalities": ["text", "image"], "images": {"max_count": 4, "canary": 1}},
             "models.clef-flash.images.canary"),
            ("clef-flash", {"workload": {**mix, "images": {**mix["images"], "pixels": [256, 512]}}},
             "models.clef-flash.workload.images.pixels")):
        with pytest.raises(SystemExit, match=why):
            commit(chain, with_input(model_id, **fields), tmp_path)
    with pytest.raises(ValidationError, match="kev-4b reads no images"):
        commit(chain, with_input("kev-4b", workload={**text_mix, "images": mix["images"]}), tmp_path)
    assert chain.commitments == []
    path = with_input("clef-flash", input_modalities=["text", "image"])
    commit(chain, path, tmp_path)
    assert chain.commitments == [catalog.commitment_value(catalog.load(path).digest())]
    assert catalog.load(path).get("clef-flash").reads_images()
