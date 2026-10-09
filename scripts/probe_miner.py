"""Validator-style probe of one miner on a live network (bittensor SDK 10.x).

Discovers the miner from chain (axon + manifest commitment, over TLS checked against the pin the manifest names),
checks endpoint control with the identity challenge, sends signed SystemOne requests over TLS with the endpoint's pin,
verifies every receipt, and compares the answers with stored reference answers for the same requests
(``tests/fixtures/drift``: 30 generated tickets, 6 questions each).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import secrets
import statistics
import sys
import time
from pathlib import Path

import bittensor as bt
import httpx

from verathos import catalog
from verathos.miner.app import IDENTITY_DOMAIN, identity_payload
from verathos.protocol import signing, tls
from verathos.protocol.api import distributions
from verathos.protocol.manifest import accept, axon_base, endpoint_host
from verathos.protocol.receipts import SignedReceipt
from verathos.protocol.requests import HDR_NONCE, sign_request
from verathos.subnet.chain import current_catalog
from verathos.validator.compare import EndpointTally, Tolerance, compare_request

TOPICS = ["billed twice for March", "the export button returns a 500 error", "I want to upgrade to the enterprise plan",
          "my package arrived broken", "please delete my account and all data", "the app crashes when I open settings",
          "I was promised a discount that never applied", "our API key stopped working this morning",
          "can you send the invoice in German", "the delivery is three weeks late"]
TONES = ["", "This is unacceptable. ", "Thanks a lot for your help! ",
         "If this is not fixed by Friday we will cancel. ", "No rush. "]
QUESTIONS = {
    "department": {"type": "choice", "instructions": "Which team should handle this?",
                   "criteria": {"billing": "Charges and refunds", "technical": "Bugs or outages",
                                "sales": "Pricing and plans", "logistics": "Shipping and delivery",
                                "privacy": "Data and account deletion", "other": "Everything else"}},
    "urgency": {"type": "score", "instructions": "How urgent is this?",
                "criteria": ["Can wait", "This week", "Today", "Right now"]},
    "churn": {"type": "noul", "instructions": "Does the customer threaten to cancel?"},
    "bug": {"type": "noul", "instructions": "Is a product bug reported?"},
    "refund": {"type": "noul", "instructions": "Is a refund requested?"},
    "angry": {"type": "noul", "instructions": "Is the customer angry?"},
}


def states(n: int = 30) -> list[str]:
    """The same tickets the reference fixtures were computed on."""
    rng = random.Random(1234)
    out = []
    for i in range(n):
        t = rng.sample(TOPICS, 2)
        out.append(
            f"Ticket {1000 + i}. {rng.choice(TONES)}Hi team, {t[0]}. Also, {t[1]}. "
            f"{rng.choice(TONES)}Regards, customer {i}."
        )
    return out


def post(client: httpx.Client, url: str, pin: str, path: str, payload: dict, caller, receiver: str) -> dict:
    body = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json", **sign_request(caller, "POST", path, body, receiver)}
    r = client.post(url + path, content=body, headers=headers, extensions=tls.use(pin))
    r.raise_for_status()
    return r.json()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--network", default="test")
    p.add_argument("--netuid", type=int, default=405)
    p.add_argument("--mechid", type=int, default=None, help="the mechanism to expect (default: the catalog's)")
    p.add_argument("--uid", type=int, required=True)
    p.add_argument("--caller-wallet", required=True)
    p.add_argument("--caller-hotkey", required=True)
    p.add_argument("--reference", type=Path, required=True,
                   help="fixture JSON with reference answers, named drift-<model>-<hardware>.json")
    p.add_argument("--endpoint", default=None)
    args = p.parse_args()
    try:
        tls.chain_network(args.network)
    except ValueError as exc:
        p.error(str(exc))

    st = bt.Subtensor(network=args.network)
    mg = st.metagraph(args.netuid, lite=True)
    hotkey, axon = mg.hotkeys[args.uid], mg.axons[args.uid]
    base = axon_base(axon.ip, axon.port)
    cat = current_catalog(st, args.netuid, catalog.CatalogStore(), [])
    mechid = catalog.mechanism_of(cat, args.mechid)
    client = tls.sync_client(timeout=120)
    with client.stream("GET", base + "/v1/manifest", extensions=tls.use(tls.ANY)) as r:
        r.raise_for_status()
        seen = tls.peer_pin(r)
        document = json.loads(r.read())
    manifest = accept(document, hotkey=hotkey, netuid=args.netuid, commitment=st.get_commitment(args.netuid, args.uid),
                      catalogs={cat.digest(): mechid}, axon=base, seen=seen)
    print(f"manifest accepted: catalog {manifest.catalog_digest[:12]}, "
          f"endpoints {[e.endpoint_id for e in manifest.endpoints]}")

    caller = bt.Wallet(name=args.caller_wallet, hotkey=args.caller_hotkey).hotkey
    endpoint = next(e for e in manifest.endpoints if args.endpoint in (None, e.endpoint_id))
    host, prefix = endpoint_host(endpoint.url, endpoint.endpoint_id), f"/e/{endpoint.endpoint_id}"
    pin = manifest.pin(host)
    print(f"endpoint {endpoint.endpoint_id} at {host}, TLS pin {pin}" + (" (the axon)" if host == base else ""))
    challenge = secrets.token_hex(16)
    ident = post(client, host, pin, prefix + "/v1/identity", {"challenge": challenge}, caller, hotkey)
    ok = signing.verify(
        hotkey, IDENTITY_DOMAIN,
        identity_payload(endpoint.endpoint_id, challenge, caller.ss58_address),
        ident["signature"],
    )
    print(f"identity challenge for {endpoint.endpoint_id}: {'verified' if ok else 'FAILED'}")

    if f"-{endpoint.model_id}-" not in args.reference.name:
        raise SystemExit(f"{args.reference.name} holds reference answers for another model than {endpoint.model_id}")
    fixture = json.loads(args.reference.read_text())
    if any("images" in r or r.get("state") != state for r, state in zip(fixture, states(len(fixture)))):
        raise SystemExit(f"{args.reference.name} holds reference answers for other requests than the probe's tickets")
    reference = [distributions(r["answers"]) for r in fixture]
    tolerance = cat.get(endpoint.model_id).tolerance if cat.get(endpoint.model_id) else Tolerance()
    tally, latencies, receipts_ok = EndpointTally(tolerance), [], 0
    for state, ref in zip(states(len(reference)), reference):
        request = {"model": endpoint.model_id, "state": state, "questions": QUESTIONS}
        t0 = time.time()
        body = json.dumps(request).encode()
        path = prefix + "/v1/systemone"
        headers = {"Content-Type": "application/json", **sign_request(caller, "POST", path, body, hotkey)}
        reply = client.post(host + path, content=body, headers=headers, extensions=tls.use(pin))
        reply.raise_for_status()
        response = reply.json()
        latencies.append((time.time() - t0) * 1000)
        receipt = SignedReceipt.model_validate(response["receipt"])
        r = receipt.receipt
        receipts_ok += (  # the checks validators make
            receipt.signature_valid() and receipt.binds(request, response["answers"]) and r.miner_hotkey == hotkey
            and r.endpoint_id == endpoint.endpoint_id and r.model_id == endpoint.model_id
            and r.caller_hotkey == caller.ss58_address and r.request_nonce == headers[HDR_NONCE]
            and r.netuid == args.netuid
        )
        tally.add(compare_request(ref, distributions(response["answers"]), tally.tolerance))
    print(f"requests {tally.requests}, receipts valid {receipts_ok}/{tally.requests}, "
          f"round trip median {statistics.median(latencies):.0f} ms")
    print(f"mean abs diff vs reference {tally.mean_abs:.4f}, hard failures {len(tally.hard_failures)} "
          f"-> verdict {tally.verdict().value}")
    st.close()


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)  # the SDK leaves non-daemon threads behind that would keep the process alive
