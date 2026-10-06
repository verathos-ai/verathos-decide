"""Subnet owner operations on a TESTNET subnet, signed with the owner coldkey held in the OS keyring.

The mnemonic is read from the Secret Service keyring (service ``verathos-testnet``,
user ``test_sn96-coldkey-mnemonic``) and only lives in memory. Mainnet endpoints are refused.

    status            mechanism count, split, max UIDs
    enable-mechanism  max_allowed_uids -> 128, mechanism count -> 2, emission split
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import bittensor as bt
import keyring
from bittensor_wallet import Keypair

SERVICE, USER = "verathos-testnet", "test_sn96-coldkey-mnemonic"
MAINNET = {"finney", "wss://entrypoint-finney.opentensor.ai:443", "archive"}


def owner_keypair(st: bt.Subtensor, netuid: int) -> Keypair:
    mnemonic = keyring.get_password(SERVICE, USER)
    if not mnemonic:
        raise SystemExit("owner mnemonic not found in the keyring")
    kp = Keypair.create_from_mnemonic(mnemonic)
    owner = st.substrate.query("SubtensorModule", "SubnetOwner", [netuid]).value
    if kp.ss58_address != owner:
        raise SystemExit(f"keyring key {kp.ss58_address} is not the owner of netuid {netuid} ({owner})")
    return kp


def wait_admin_window(st: bt.Subtensor, netuid: int) -> None:
    freeze = int(st.substrate.query("SubtensorModule", "AdminFreezeWindow").value)
    while True:
        left = st.get_next_epoch_start_block(netuid) - st.get_current_block()
        if left > freeze + 2:
            return
        print(f"inside admin freeze window ({left} blocks to epoch end), waiting", flush=True)
        time.sleep(12 * (left + 2))


def call(st: bt.Subtensor, kp: Keypair, netuid: int, function: str, params: dict) -> None:
    wait_admin_window(st, netuid)
    c = st.substrate.compose_call(call_module="AdminUtils", call_function=function, call_params=params)
    ext = st.substrate.create_signed_extrinsic(call=c, keypair=kp)
    r = st.substrate.submit_extrinsic(ext, wait_for_inclusion=True, wait_for_finalization=True)
    ok = r.is_success
    print(f"{function}({params}) -> {'ok' if ok else 'FAILED: ' + str(r.error_message)}", flush=True)
    if not ok:
        raise SystemExit(1)


def status(st: bt.Subtensor, netuid: int) -> None:
    q = st.substrate.query
    print({"MaxAllowedUids": q("SubtensorModule", "MaxAllowedUids", [netuid]).value,
           "SubnetworkN": q("SubtensorModule", "SubnetworkN", [netuid]).value,
           "MechanismCountCurrent": q("SubtensorModule", "MechanismCountCurrent", [netuid]).value,
           "MechanismEmissionSplit": q("SubtensorModule", "MechanismEmissionSplit", [netuid]).value}, flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--network", required=True)
    p.add_argument("--netuid", type=int, required=True)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    e = sub.add_parser("enable-mechanism")
    e.add_argument("--split", default="90,10", help="relative emission weights for mechanisms 0 and 1")
    args = p.parse_args()
    if args.network in MAINNET or "finney" in args.network:
        raise SystemExit("refusing to run against mainnet")

    st = bt.Subtensor(network=args.network)
    if args.cmd == "status":
        status(st, args.netuid)
    else:
        kp = owner_keypair(st, args.netuid)
        n = st.substrate.query("SubtensorModule", "SubnetworkN", [args.netuid]).value
        if n > 128:
            raise SystemExit(f"{n} UIDs registered; trimming is out of scope for this script")
        if st.substrate.query("SubtensorModule", "MaxAllowedUids", [args.netuid]).value > 128:
            call(st, kp, args.netuid, "sudo_set_max_allowed_uids", {"netuid": args.netuid, "max_allowed_uids": 128})
        if st.substrate.query("SubtensorModule", "MechanismCountCurrent", [args.netuid]).value < 2:
            call(st, kp, args.netuid, "sudo_set_mechanism_count", {"netuid": args.netuid, "mechanism_count": 2})
        parts = [float(x) for x in args.split.split(",")]
        if len(parts) != 2 or min(parts) < 0 or sum(parts) <= 0:
            raise SystemExit("split must be two non-negative numbers")
        first = round(65535 * parts[0] / sum(parts))
        call(st, kp, args.netuid, "sudo_set_mechanism_emission_split",
             {"netuid": args.netuid, "maybe_split": [first, 65535 - first]})
        status(st, args.netuid)
    st.close()


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
