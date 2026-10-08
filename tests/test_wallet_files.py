from __future__ import annotations

import json

import bittensor_wallet as bw


def test_hotkey_files_with_and_without_crypto_type_load(tmp_path) -> None:
    """Hotkeys created by a current btcli carry ``cryptoType``; older ones do not. Both must load, or an operator
    with a new hotkey could not run a miner or validator."""
    path = str(tmp_path / "wallets")
    created = bw.Wallet(name="w", hotkey="current", path=path)
    created.create_new_hotkey(n_words=12, use_password=False, overwrite=True, suppress=True)
    current = tmp_path / "wallets" / "w" / "hotkeys" / "current"
    document = json.loads(current.read_text())
    old = {k: v for k, v in document.items() if k != "cryptoType"}
    (current.parent / "old").write_text(json.dumps(old))
    document["cryptoType"] = 1
    current.write_text(json.dumps(document))
    for name in ("current", "old"):
        assert bw.Wallet(name="w", hotkey=name, path=path).hotkey.ss58_address == document["ss58Address"]
