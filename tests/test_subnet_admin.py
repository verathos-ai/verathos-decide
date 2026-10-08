"""verathos-subnet: what the owner sees before and after committing a catalog."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from verathos import catalog
from verathos.subnet import admin

OWNER = "5Owner"


class Chain:
    def __init__(self, block: int = 1000 * 180) -> None:
        self.block, self.commitments = block, []

    def get_subnet_owner_hotkey(self, netuid: int) -> str:
        return OWNER

    def get_current_block(self) -> int:
        return self.block

    def set_commitment(self, wallet, netuid: int, value: str):
        self.commitments.append(value)
        return SimpleNamespace(success=True)


def catalog_file(tmp_path, **scoring):
    document = catalog.load().document()
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps({**document, "scoring": {**document["scoring"], **scoring}}))
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


def test_commit_catalog_refuses_an_invalid_or_mistyped_miner_share(tmp_path) -> None:
    chain = Chain()
    path = catalog_file(tmp_path, miner_share={"from": 0.59, "to": 0.15, "start": 400, "epochs": 2400})
    with pytest.raises(SystemExit, match="scoring.miner_share.start"):
        commit(chain, path, tmp_path)
    path = catalog_file(tmp_path, miner_share={"from": 0.59, "to": 0.15, "start_epoch": 400, "epochs": 0})
    with pytest.raises(ValueError, match="miner_share"):
        commit(chain, path, tmp_path)
    assert chain.commitments == []
