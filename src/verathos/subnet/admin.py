"""Subnet administration: ``verathos-subnet``. The changing commands are signed with the subnet owner's keys.

    status            mechanisms, emission split, UID cap, the committed catalog and the shipped one
    setup-mechanisms  mechanism count 2 and the emission split (owner coldkey)
    commit-catalog    publish a catalog file: store it where the owner's validator serves it, commit its digest

The owner coldkey comes from a bittensor wallet (``--wallet``, password prompt) or, for unattended test networks,
from a mnemonic in the OS keyring (``--keyring SERVICE:USER``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import bittensor as bt
from bittensor_wallet import Keypair

from verathos import catalog
from verathos.subnet.chain import committed_catalog

MAX_UIDS_TWO_MECHANISMS = 128  # chain rule: max UIDs x mechanism count <= 256
SPLIT_SCALE = 65535


def owner_coldkey(args: argparse.Namespace, st: bt.Subtensor) -> Keypair:
    if args.keyring:
        import keyring

        service, _, user = args.keyring.partition(":")
        mnemonic = keyring.get_password(service, user)
        if not mnemonic:
            raise SystemExit(f"no mnemonic in the keyring for {args.keyring}")
        kp = Keypair.create_from_mnemonic(mnemonic)
    elif args.wallet:
        kp = bt.Wallet(name=args.wallet).coldkey
    else:
        raise SystemExit("pass --wallet or --keyring for the owner coldkey")
    owner = st.substrate.query("SubtensorModule", "SubnetOwner", [args.netuid]).value
    if kp.ss58_address != owner:
        raise SystemExit(f"{kp.ss58_address} is not the owner of netuid {args.netuid} ({owner})")
    return kp


def admin_call(st: bt.Subtensor, kp: Keypair, netuid: int, function: str, params: dict) -> None:
    """Submit an AdminUtils call outside the admin freeze window at the end of each tempo."""
    freeze = int(st.substrate.query("SubtensorModule", "AdminFreezeWindow").value)
    while (left := st.get_next_epoch_start_block(netuid) - st.get_current_block()) <= freeze + 2:
        print(f"inside the admin freeze window ({left} blocks to tempo end); waiting", flush=True)
        time.sleep(12 * (left + 2))
    call = st.substrate.compose_call(call_module="AdminUtils", call_function=function, call_params=params)
    receipt = st.substrate.submit_extrinsic(st.substrate.create_signed_extrinsic(call=call, keypair=kp),
                                            wait_for_inclusion=True, wait_for_finalization=True)
    if not receipt.is_success:
        raise SystemExit(f"{function} failed: {receipt.error_message}")
    print(f"{function} {params}: ok", flush=True)


def status(st: bt.Subtensor, netuid: int) -> None:
    q = st.substrate.query
    committed = committed_catalog(st, netuid)
    local = catalog.load().digest()
    print(json.dumps({
        "max_allowed_uids": q("SubtensorModule", "MaxAllowedUids", [netuid]).value,
        "registered_uids": q("SubtensorModule", "SubnetworkN", [netuid]).value,
        "mechanism_count": q("SubtensorModule", "MechanismCountCurrent", [netuid]).value,
        "mechanism_emission_split": q("SubtensorModule", "MechanismEmissionSplit", [netuid]).value,
        "owner_hotkey": st.get_subnet_owner_hotkey(netuid),
        "committed_catalog": committed,
        "shipped_catalog": local,
        "shipped_catalog_is_committed": committed == local,
    }, indent=2), flush=True)


def setup_mechanisms(st: bt.Subtensor, kp: Keypair, netuid: int, split: str) -> None:
    q = st.substrate.query
    weights = [float(x) for x in split.split(",")]
    if len(weights) != 2 or min(weights) < 0 or sum(weights) <= 0:
        raise SystemExit("--split takes two non-negative weights, e.g. 90,10")
    registered = q("SubtensorModule", "SubnetworkN", [netuid]).value
    if registered > MAX_UIDS_TWO_MECHANISMS:
        raise SystemExit(f"{registered} UIDs registered; two mechanisms need at most {MAX_UIDS_TWO_MECHANISMS}")
    if q("SubtensorModule", "MaxAllowedUids", [netuid]).value > MAX_UIDS_TWO_MECHANISMS:
        admin_call(st, kp, netuid, "sudo_set_max_allowed_uids",
                   {"netuid": netuid, "max_allowed_uids": MAX_UIDS_TWO_MECHANISMS})
    if q("SubtensorModule", "MechanismCountCurrent", [netuid]).value < 2:
        admin_call(st, kp, netuid, "sudo_set_mechanism_count", {"netuid": netuid, "mechanism_count": 2})
    first = round(SPLIT_SCALE * weights[0] / sum(weights))
    admin_call(st, kp, netuid, "sudo_set_mechanism_emission_split",
               {"netuid": netuid, "maybe_split": [first, SPLIT_SCALE - first]})


def commit_catalog(st: bt.Subtensor, wallet: bt.Wallet, netuid: int, path: Path | None, store_dir: Path) -> None:
    """Validate the catalog, put it in the owner validator's catalog store, then commit its digest."""
    if wallet.hotkey.ss58_address != st.get_subnet_owner_hotkey(netuid):
        raise SystemExit(f"{wallet.hotkey.ss58_address} is not the subnet owner hotkey")
    cat = catalog.load(path)
    cat.audit_policy(strict=True)  # nodes ignore settings their code does not know; a publisher must not mistype one
    cat.anchor_shares(strict=True)
    stored = catalog.CatalogStore(store_dir).put(cat)
    print(f"catalog {cat.digest()} stored at {stored}; active models: "
          f"{[m.model_id for m in cat.models if m.status == 'active']}", flush=True)
    value = catalog.commitment_value(cat.digest())
    result = st.set_commitment(wallet, netuid, value)
    if not getattr(result, "success", False):
        raise SystemExit(f"set_commitment failed: {getattr(result, 'message', result)}")
    print(f"committed {value}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--network", required=True)
    parser.add_argument("--netuid", type=int, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    mech = sub.add_parser("setup-mechanisms")
    mech.add_argument("--split", default="90,10", help="relative emission of mechanisms 0 and 1")
    mech.add_argument("--wallet", help="wallet holding the owner coldkey")
    mech.add_argument("--keyring", help="SERVICE:USER of an owner coldkey mnemonic in the OS keyring")
    commit = sub.add_parser("commit-catalog")
    commit.add_argument("--wallet", required=True, help="wallet name of the owner hotkey")
    commit.add_argument("--hotkey", required=True)
    commit.add_argument("--file", type=Path, help="catalog JSON to publish (default: the shipped catalog)")
    commit.add_argument("--catalog-dir", type=Path, default=catalog.DEFAULT_CACHE_DIR,
                        help="the owner validator's catalog_dir, where it serves catalogs from")
    args = parser.parse_args()

    st = bt.Subtensor(network=args.network)
    if args.command == "status":
        status(st, args.netuid)
    elif args.command == "setup-mechanisms":
        setup_mechanisms(st, owner_coldkey(args, st), args.netuid, args.split)
        status(st, args.netuid)
    else:
        commit_catalog(st, bt.Wallet(name=args.wallet, hotkey=args.hotkey), args.netuid, args.file, args.catalog_dir)
    st.close()


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)  # the SDK leaves non-daemon threads behind
