"""``verathos-miner``: the referral commands run with the package's base dependencies alone (no Bittensor SDK), so a
hotkey that serves nothing can register and check its codes with ``uvx``; every other command is the miner's."""

from __future__ import annotations

import sys


def main() -> None:
    if sys.argv[1:2] == ["referral"]:
        from verathos.miner import referral

        referral.main(sys.argv[2:])
        return
    from verathos.miner.__main__ import main as miner_main

    miner_main()
