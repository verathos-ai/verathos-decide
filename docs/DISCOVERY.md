# Discovery and catalog

Miners and the model catalog are discovered from native chain state only. No EVM, no gas, no central database.

```
                         SUBSTRATE (SN96)
  ┌──────────────────────────────────────────────────────────────────┐
  │  subnet owner hotkey                                             │
  │    commitment: "verathos/catalog/v1 sha256:<catalog digest>"     │
  │                                                                  │
  │  miner hotkey (any UID)                                          │
  │    axon:       ip:port of the miner's primary machine            │
  │    commitment: "verathos/manifest/v1 sha256:<manifest digest>"   │
  │                                                                  │
  │  validator hotkey (full validators)                              │
  │    commitment: "verathos/validator/v1 <pin|-> <egress ips|->"    │
  └──────────────────────────────────────────────────────────────────┘
          ▲ serve_axon / set_commitment              │ read axons + commitments
          │ (free, hotkey-signed, primary only)      ▼
  ┌──────────────────────────┐ GET /v1/manifest TLS ┌───────────────────────────┐
  │ MINER primary (axon)     │ ◄────────────────────│ VALIDATOR / AUDITOR       │
  │  signed manifest:        │                      │  1. hotkeys with manifest │
  │   gpu0 kev-4b, here      │                      │     commitment            │
  │   w1-gpu0 kev-9b, worker │                      │  2. fetch manifest        │
  │   each machine's pin     │  identity, canaries, │  3. signature == hotkey   │
  │  decision engines        │ ◄────────────────────│  4. digest == commitment  │
  └────────────┬─────────────┘  tests,              │  5. catalog digest ==     │
               │                pinned TLS          │     owner commitment      │
               │ GET /v1/endpoints, every minute    │  6. axon's certificate == │
               ▼                                    │     the pin it names      │
  ┌──────────────────────────┐  identity, canaries, │  7. each endpoint signs a │
  │ MINER worker, same hotkey│ ◄────────────────────│     challenge at its URL  │
  │  no chain writes         │  tests,              │     before any traffic    │
  │  decision engines        │  pinned TLS          └───────────────────────────┘
  └──────────────────────────┘
```

Rules:
- Only hotkeys with a manifest commitment and an axon on a public address are contacted.
- Gateways read no manifest: they send users' requests to the endpoints of the auditor's signed routes, over TLS
  with the pins the routes name (`GATEWAY.md`).
- A manifest is accepted only if it is signed by the hotkey that owns the axon, is for this netuid, is built for the
  committed catalog (or, for one epoch after a change, the previous one) and for that catalog's mechanism, its digest
  equals that hotkey's current commitment, every endpoint URL is exactly
  `https://<ip>:<port>/e/<endpoint_id>`: the IP a public unicast address written as the chain writes axons (not
  private, shared, loopback, link-local, multicast, reserved, unspecified or documentation, and not an IPv6 form of
  an IPv4 address), the port 1-65535, nothing else, it names the pin of every endpoint's machine and of the axon
  (`tls`, base URL `https://<ip>:<port>` -> pin), and the axon presented the certificate with that pin when it served
  the manifest. The host is the axon or any other machine of the miner. A manifest names at most 64 machines and
  lists at most 64 endpoints. A miner therefore cannot show different endpoint lists to different validators. The manifest depends only on the miner's configuration, its machines' TLS certificates, its
  capacity measurements and the catalog, so a restart does not change the commitment.
- An endpoint gets traffic only once it proved control at its URL, over TLS with its machine's pin: it signs a fresh
  challenge with the listing hotkey (`POST /e/<endpoint_id>/v1/identity`); a new pin is a new place to prove control
  at. Validators challenge every endpoint each epoch before its first
  canary; the auditor tests an endpoint only if it passed at that URL in the current or the previous epoch, and
  gateways route only to endpoints the auditor credits, at public addresses. At the hotkey's own hosts (the axon,
  while it serves the manifest, and hosts where one of its endpoints passed in this epoch or the last) endpoints are
  challenged together, and a failed challenge is tried again before each canary. Any other host is challenged one
  endpoint at a time; after a failure it gets no further challenge for that hotkey in the epoch. A manifest that
  names someone else's server therefore costs that server at most one challenge per validator and epoch, and no
  other traffic.
- While a hotkey's axon serves no manifest that is accepted (it is down, or restarting), validators and the auditor
  keep using the one it last served, for up to 3 epochs and only while the commitment still names its digest: the
  endpoints on the miner's other machines keep their checks and tests while the machine serving the manifest is out.
- A manifest names the catalog it was built for. Validators accept the current catalog and, for one epoch after a
  change, the previous one, while miners re-publish (they check every minute).
- A catalog is accepted only if its digest equals the subnet owner hotkey's current commitment. A copy ships with
  the code; newer ones come from the owner's axon (`GET /v1/catalog/{digest}`) or any https mirror, since the digest
  proves them. A new commitment takes effect without a code release.
- Signatures and digests cover the JSON documents exactly as published (canonical encoding of the parsed JSON), so
  nodes whose code knows fewer fields still agree on them. Responses from other neurons are size- and time-bounded.
- The throughput audit (declared capacity, see `THROUGHPUT_AUDIT.md`) is a separate check applied after discovery.

## TLS

Every connection between machines is TLS; plain HTTP stays on one machine's loopback (engines, the gateway behind its
reverse proxy). The chain is reached the same way: over wss (`finney`, `test` or a `wss://` node), plain `ws://` only
to a node on the same machine (`local`); every configuration and command refuses anything else. Miners and validators are reached by IP address, where no public CA applies: each serving process
(a miner's service on each machine, a validator's service) creates a TLS key and a self-signed certificate (ECDSA
P-256, no expiry) once in its `state_dir` (`tls.key`, `tls.crt`, mode 600) and reuses them; deleting both rotates
them. A certificate is accepted by its pin, the SHA-256 of its SubjectPublicKeyInfo (DER) as 64 lowercase hex digits,
and nothing else: no CA, name or date check, and a client sends no byte of a request before the handshake shows the
pinned key. `curl --pinnedpubkey` takes the same pin in base64 (`sha256//...`), which the miner installer prints.

Pins come from documents the chain authenticates:
- a miner machine's from its hotkey's manifest (`tls`: each machine's base URL -> its pin);
- the auditor's from the committed catalog (`auditor.tls`, `SUBNET_ADMIN.md`);
- another validator's from its commitment, `verathos/validator/v1 <pin> <egress>`: the pin as unpadded base64url of
  its 32 bytes (`-` for a validator that serves no one) and the addresses its requests to miners come from,
  comma-separated (`-` for none), in one raw field of at most 128 bytes. A full validator commits it at start when it
  changed; the subnet owner's hotkey commits the catalog instead, and light validators commit nothing.

A validator's engines on a reference machine (a GPU machine serving them over TLS) are accepted by the pin in the
file that machine's setup writes, which the operator copies to the validator, and need its key (`VALIDATING.md`).

Documents that are authenticated otherwise are fetched accepting any certificate: a manifest (signed by its hotkey,
its digest committed), whose pin for the axon must then be the certificate the axon presented; a worker's endpoints
document (signed by the hotkey), likewise for the worker; a catalog by its digest. Catalog mirrors with a hostname
need a public CA's certificate.

The auditor's and the gateways' egress addresses in the catalog and the validators' in their commitments give the
addresses miners are called from (`protocol.callers.allowlist`): the auditor, the catalog's gateways and every
validator holding a permit and `validator_min_stake` that publishes its addresses. Miners' firewalls let them through
and drop other connections once the catalog's `firewall` is `enforce` (`MINING.md`).

## Chain limits

Read from subtensor main and testnet 405 on 2026-10-06:
- `serve_axon`: no fee (`Pays::No`), at most once per `ServingRateLimit` = 50 blocks (about 10 minutes). Needed only
  when the manifest server's ip or port changes.
- `set_commitment`: no fee and no deposit; only hotkeys registered on the netuid; up to 3 fields per commitment;
  3,100 bytes per hotkey per epoch, each commit counting at least 100 bytes (about 31 commits per 360-block
  epoch). One current commitment per (netuid, hotkey).
