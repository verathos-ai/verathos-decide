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


def catalog_file(tmp_path, mechanism: int = 0, **scoring):
    document = catalog.load().document()
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps({**document, "mechanism": mechanism, "scoring": {**document["scoring"], **scoring}}))
    return path


def commit(chain: Chain, path, tmp_path) -> None:
    admin.commit_catalog(chain, SimpleNamespace(hotkey=SimpleNamespace(ss58_address=OWNER)), 405, path,
                         tmp_path / "store")


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
