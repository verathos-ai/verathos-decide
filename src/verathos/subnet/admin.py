"""Subnet administration: ``verathos-subnet``. The changing commands are signed with the subnet owner's keys.

    status            mechanisms, emission split, UID cap, the committed catalog (its mechanism) and the shipped one,
                      the epoch and the miner share in it, the usage pool
    setup-mechanisms  mechanism count 2 and the emission split (owner coldkey)
    commit-catalog    publish a catalog file: store it where the owner's validator serves it, commit its digest; the
                      mechanism it names must exist on chain, a usage pool's buyback coldkey must not be the subnet
                      owner's coldkey, and it must name the auditor's TLS pin and egress addresses and each gateway's
                      egress addresses; run on the auditor's machine, it fills in the auditor's pin and egress
                      addresses from the validator's configuration and certificate where the file names none

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

from verathos import __version__, catalog, release
from verathos.protocol import tls
from verathos.subnet.chain import committed_catalog, current_catalog

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


def miner_share_line(cat: catalog.Catalog, epoch: int) -> str:
    """The catalog's miner share in ``epoch`` and how it changes."""
    change = cat.scoring.miner_share
    line = f"miner share {cat.scoring.miner_share_at(epoch):.6f} in epoch {epoch}"
    if not isinstance(change, catalog.MinerShareChange):
        return line
    if change.start_epoch is None:
        return line + f"; the change to {change.to:g} over {change.epochs} epochs has no start_epoch yet"
    return line + (f"; {change.from_:g} until epoch {change.start_epoch}, then linearly to {change.to:g} at epoch "
                   f"{change.start_epoch + change.epochs}")


def usage_pool_line(cat: catalog.Catalog, epoch: int) -> str:
    """The catalog's usage pool, and the period ``epoch`` is in."""
    pool = cat.scoring.usage_pool
    if pool is None:
        return "usage pool: off"
    basis = (f"{pool.fraction:g} of the alpha {pool.buyback_coldkey} burns on the subnet" if pool.mode == "burned"
             else f"{pool.fraction:g} of the paid usage's revenue in alpha")
    period = pool.period(epoch)
    return (f"usage pool ({pool.mode}): {basis} in each period of {pool.period_epochs} epochs, paid over the next "
            f"period to the hotkeys that served paying users, {len(pool.excluded)} hotkeys or coldkeys excluded; "
            f"epoch {epoch} is in period {period}, which pays period {period - 1}'s")


def status(st: bt.Subtensor, netuid: int) -> None:
    q = st.substrate.query
    committed = committed_catalog(st, netuid)
    shipped = catalog.load()
    try:
        cat = current_catalog(st, netuid, catalog.CatalogStore(), [])
    except catalog.CatalogUnavailable as exc:
        print(f"the committed catalog is not available: {exc}", file=sys.stderr, flush=True)
        cat = None
    epoch = st.get_current_block() // (cat or shipped).epoch_blocks
    print(json.dumps({
        "max_allowed_uids": q("SubtensorModule", "MaxAllowedUids", [netuid]).value,
        "registered_uids": q("SubtensorModule", "SubnetworkN", [netuid]).value,
        "mechanism_count": q("SubtensorModule", "MechanismCountCurrent", [netuid]).value,
        "mechanism_emission_split": q("SubtensorModule", "MechanismEmissionSplit", [netuid]).value,
        "owner_hotkey": st.get_subnet_owner_hotkey(netuid),
        "committed_catalog": committed,
        "shipped_catalog": shipped.digest(),
        "shipped_catalog_is_committed": committed == shipped.digest(),
        "catalog_mechanism": cat.mechanism if cat else None,
        "auditor": cat.auditor.model_dump() if cat and cat.auditor else None,
        "gateways": [g.model_dump() for g in cat.gateways] if cat else None,
        "epoch": epoch,
        "miner_share": cat.scoring.miner_share_at(epoch) if cat else None,
        "weights_version_key": q("SubtensorModule", "WeightsVersionKey", [netuid]).value,
        "release": __version__,
        "release_weights_version_key": release.weights_version_key(),
        "usage_pool": cat.scoring.usage_pool.model_dump() if cat and cat.scoring.usage_pool else None,
        "usage_period": cat.scoring.usage_pool.period(epoch) if cat and cat.scoring.usage_pool else None,
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


def with_auditor(document: dict, validator_config: Path | None) -> dict:
    """``document`` with the auditor's pin and egress addresses filled in where it names none, from the auditor's
    configuration on this machine (``validator_config``: its ``egress_ips`` and the certificate in its
    ``state_dir``, created there if the auditor has not yet, so the service then serves it). Refuses a pin that differs
    from the certificate's, and a catalog that ends up without the auditor's pin."""
    config: dict = {}
    if validator_config is not None and validator_config.expanduser().exists():
        config = json.loads(validator_config.expanduser().read_text())
    state_dir = Path(config.get("state_dir", "~/.verathos/validator"))
    served = tls.stored_pin(state_dir) if config else None
    auditor = dict(document.get("auditor") or {})
    if config and served is None and "tls" not in auditor:
        served = tls.identity(state_dir.expanduser()).pin
        print(f"created the auditor's TLS certificate in {state_dir} (the auditor serves it from its next start)",
              flush=True)
    if served is not None and auditor.get("tls") not in (None, served):
        raise SystemExit(f"the catalog names {auditor['tls']} as the auditor's TLS pin, but the auditor's certificate "
                         f"in {state_dir} has pin {served}: name the pin the auditor serves (or leave auditor.tls out "
                         "to fill it in)")
    filled = []
    if "tls" not in auditor and served is not None:
        auditor["tls"] = served
        filled.append(f"auditor.tls {served} (from {state_dir})")
    if not auditor.get("egress") and config.get("egress_ips"):
        auditor["egress"] = list(config["egress_ips"])
        filled.append(f"auditor.egress {', '.join(auditor['egress'])} (from {validator_config})")
    for line in filled:
        print(f"filled in {line}", flush=True)
    if not auditor.get("tls"):
        raise SystemExit("the catalog names no auditor.tls, the auditor's TLS pin, and no auditor configuration was "
                         "found to fill it in: run commit-catalog on the auditor's machine (--validator-config) or "
                         "name the pin")
    return {**document, "auditor": auditor}


def commit_catalog(st: bt.Subtensor, wallet: bt.Wallet, netuid: int, path: Path | None, store_dir: Path,
                   validator_config: Path | None = None) -> None:
    """Validate the catalog (with the auditor's pin and egress addresses filled in from ``validator_config`` where it
    names none), put it in the owner validator's catalog store, then commit its digest."""
    if wallet.hotkey.ss58_address != st.get_subnet_owner_hotkey(netuid):
        raise SystemExit(f"{wallet.hotkey.ss58_address} is not the subnet owner hotkey")
    document = json.loads(path.read_text()) if path else catalog.load().document()
    cat = catalog.Catalog.parse(with_auditor(document, validator_config))
    if problems := cat.identity_problems():
        raise SystemExit("the catalog must name: " + "; ".join(problems) + " (docs/SUBNET_ADMIN.md)")
    cat.audit_policy(strict=True)  # nodes ignore settings their code does not know; a publisher must not mistype one
    if unknown := cat.unknown_settings():
        raise SystemExit(f"settings this code does not know: {', '.join(unknown)}")
    cat.anchor_shares(strict=True)
    pool = cat.scoring.usage_pool
    if pool is not None and pool.buyback_coldkey is not None and pool.buyback_coldkey == st.substrate.query(
            "SubtensorModule", "SubnetOwner", [netuid]).value:
        raise SystemExit("the usage pool's buyback_coldkey is the subnet owner's coldkey: name the buyback's own "
                         "coldkey, which signs its purchases and burns, funded with revenue only")
    if (count := st.get_mechanism_count(netuid)) <= cat.mechanism:
        raise SystemExit(f"the catalog names mechanism {cat.mechanism}, but netuid {netuid} has {count} mechanism(s) "
                         "(0 to count - 1): name an existing one, or add it first (setup-mechanisms)")
    store = catalog.CatalogStore(store_dir)
    previous = store.get(committed_catalog(st, netuid) or "")
    if previous is not None and previous.mechanism != cat.mechanism:
        print(f"the network moves from mechanism {previous.mechanism} to {cat.mechanism}: miners publish their "
              "manifests for it within minutes, and validators set weights there from the first epoch scored under "
              "this catalog", flush=True)
    stored = store.put(cat)
    print(f"catalog {cat.digest()} stored at {stored}; mechanism {cat.mechanism}; active models: "
          f"{[m.model_id for m in cat.models if m.status == 'active']}", flush=True)
    print(miner_share_line(cat, st.get_current_block() // cat.epoch_blocks), flush=True)
    print(usage_pool_line(cat, st.get_current_block() // cat.epoch_blocks), flush=True)
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
    commit.add_argument("--validator-config", type=Path, default=Path("~/.verathos/validator.json"),
                        help="the auditor's configuration on this machine, whose certificate (in its state_dir) and "
                        "egress_ips fill in the catalog's auditor section where it names none")
    args = parser.parse_args()

    st = bt.Subtensor(network=args.network)
    if args.command == "status":
        status(st, args.netuid)
    elif args.command == "setup-mechanisms":
        setup_mechanisms(st, owner_coldkey(args, st), args.netuid, args.split)
        status(st, args.netuid)
    else:
        commit_catalog(st, bt.Wallet(name=args.wallet, hotkey=args.hotkey), args.netuid, args.file, args.catalog_dir,
                       args.validator_config)
    st.close()


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)  # the SDK leaves non-daemon threads behind
