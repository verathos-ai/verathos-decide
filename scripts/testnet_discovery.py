"""Exercise chain-native discovery on a live network (bittensor SDK 10.x).

Miner side:
  manifest-build   sign a manifest with the miner hotkey and write it to a file
  axon             serve_axon: point the hotkey's axon at the manifest server
  manifest-commit  set_commitment: commit the manifest digest
Owner side:
  catalog-commit   set_commitment from the subnet owner hotkey: commit the catalog digest
Validator side:
  discover         read axon and commitments from chain, fetch the manifest, accept or reject it,
                   and check the catalog against the owner's commitment
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.request
from pathlib import Path

import bittensor as bt

from verathos import catalog as catalog_mod
from verathos.protocol.manifest import (
    Endpoint,
    Hardware,
    Manifest,
    ManifestRejected,
    SignedManifest,
    accept,
    commitment_value,
    committed_digest,
)

log = logging.getLogger("testnet_discovery")


def subtensor(args: argparse.Namespace) -> bt.Subtensor:
    return bt.Subtensor(network=args.network)


def cmd_manifest_build(args: argparse.Namespace) -> None:
    wallet = bt.Wallet(name=args.wallet, hotkey=args.hotkey)
    st = subtensor(args)
    endpoints = [
        Endpoint(endpoint_id=f"gpu{i}", url=url, model_id=model,
                 hardware=Hardware(kind="cuda", device_id=f"GPU-test-{i}"), max_concurrency=64)
        for i, (url, model) in enumerate(zip(args.endpoint_url, args.model_id))
    ]
    manifest = Manifest(netuid=args.netuid, mechid=args.mechid, hotkey=wallet.hotkey.ss58_address, seq=args.seq,
                        issued_block=st.get_current_block(), catalog_digest=catalog_mod.load().digest(),
                        endpoints=endpoints)
    signed = SignedManifest.create(manifest, wallet.hotkey)
    Path(args.out).write_text(signed.model_dump_json(indent=2))
    log.info("manifest seq=%d digest=%s written to %s", manifest.seq, manifest.digest(), args.out)


def cmd_axon(args: argparse.Namespace) -> None:
    wallet = bt.Wallet(name=args.wallet, hotkey=args.hotkey)
    axon = bt.Axon(wallet=wallet, port=args.port, external_ip=args.ip, external_port=args.port)
    response = subtensor(args).serve_axon(netuid=args.netuid, axon=axon)
    log.info("serve_axon %s:%d -> %s", args.ip, args.port, response)


def cmd_manifest_commit(args: argparse.Namespace) -> None:
    wallet = bt.Wallet(name=args.wallet, hotkey=args.hotkey)
    signed = SignedManifest.model_validate_json(Path(args.file).read_text())
    value = commitment_value(signed.manifest.digest())
    response = subtensor(args).set_commitment(wallet, args.netuid, value)
    log.info("set_commitment %r -> %s", value, response)


def cmd_catalog_commit(args: argparse.Namespace) -> None:
    wallet = bt.Wallet(name=args.wallet, hotkey=args.hotkey)
    value = catalog_mod.commitment_value(catalog_mod.load().digest())
    response = subtensor(args).set_commitment(wallet, args.netuid, value)
    log.info("set_commitment %r -> %s", value, response)


def cmd_discover(args: argparse.Namespace) -> None:
    st = subtensor(args)
    mg = st.metagraph(args.netuid)
    owner = st.get_subnet_owner_hotkey(args.netuid)
    owner_commit = st.get_commitment(args.netuid, mg.hotkeys.index(owner)) if owner in mg.hotkeys else None
    local = catalog_mod.load()
    catalog_ok = catalog_mod.committed_digest(owner_commit) == local.digest()
    print(json.dumps({"block": st.get_current_block(), "owner": owner, "owner_commitment": owner_commit,
                      "local_catalog_digest": local.digest(), "catalog_accepted": catalog_ok}, indent=2))
    commitments = st.get_all_commitments(args.netuid)
    uids = [args.uid] if args.uid is not None else range(int(mg.n))
    for uid in uids:
        axon, hotkey = mg.axons[uid], mg.hotkeys[uid]
        commitment = commitments.get(hotkey)
        if axon.ip in ("0.0.0.0", "") or not axon.port or committed_digest(commitment) is None:
            continue
        url = f"http://{axon.ip}:{axon.port}/v1/manifest"
        result: dict = {"uid": uid, "hotkey": hotkey, "axon": f"{axon.ip}:{axon.port}", "commitment": commitment}
        try:
            body = urllib.request.urlopen(url, timeout=10).read()
            manifest = accept(SignedManifest.model_validate_json(body), hotkey=hotkey, netuid=args.netuid,
                              commitment=commitment)
            result.update(accepted=True, seq=manifest.seq, endpoints=[e.model_dump() for e in manifest.endpoints],
                          catalog_match=manifest.catalog_digest == local.digest())
        except ManifestRejected as e:
            result.update(accepted=False, reason=str(e))
        except Exception as e:  # unreachable axon or malformed body
            result.update(accepted=False, reason=f"{type(e).__name__}: {e}")
        print(json.dumps(result, indent=2))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--network", default="test")
    p.add_argument("--netuid", type=int, default=405)
    sub = p.add_subparsers(dest="cmd", required=True)

    def keyed(name: str) -> argparse.ArgumentParser:
        s = sub.add_parser(name)
        s.add_argument("--wallet", required=True)
        s.add_argument("--hotkey", required=True)
        return s

    s = keyed("manifest-build")
    s.add_argument("--mechid", type=int, default=1)
    s.add_argument("--seq", type=int, required=True)
    s.add_argument("--endpoint-url", action="append", required=True)
    s.add_argument("--model-id", action="append", required=True)
    s.add_argument("--out", required=True)
    s = keyed("axon")
    s.add_argument("--ip", required=True)
    s.add_argument("--port", type=int, required=True)
    s = keyed("manifest-commit")
    s.add_argument("--file", required=True)
    keyed("catalog-commit")
    s = sub.add_parser("discover")
    s.add_argument("--uid", type=int)

    args = p.parse_args()
    {"manifest-build": cmd_manifest_build, "axon": cmd_axon, "manifest-commit": cmd_manifest_commit,
     "catalog-commit": cmd_catalog_commit, "discover": cmd_discover}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
