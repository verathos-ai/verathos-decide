# Discovery and catalog

Miners and the model catalog are discovered from native chain state only. No EVM, no gas, no central database.

```
                         SUBSTRATE (SN96)
  ┌──────────────────────────────────────────────────────────────────┐
  │  owner hotkey (UID 0)                                            │
  │    commitment: "verathos/catalog/v1 sha256:<catalog digest>"     │
  │                                                                  │
  │  miner hotkey (any UID)                                          │
  │    axon:       ip:port of the miner's primary machine            │
  │    commitment: "verathos/manifest/v1 sha256:<manifest digest>"   │
  └──────────────────────────────────────────────────────────────────┘
          ▲ serve_axon / set_commitment              │ read axons + commitments
          │ (free, hotkey-signed, primary only)      ▼
  ┌──────────────────────────┐   GET /v1/manifest   ┌───────────────────────────┐
  │ MINER primary (axon)     │ ◄────────────────────│ VALIDATOR / GATEWAY       │
  │  signed manifest:        │                      │  1. hotkeys with manifest │
  │   gpu0 kev-4b, here      │                      │     commitment            │
  │   w1-gpu0 kev-9b, worker │                      │  2. fetch manifest        │
  │  decision engines        │  identity, canaries, │  3. signature == hotkey   │
  │                          │ ◄────────────────────│  4. digest == commitment  │
  └────────────┬─────────────┘  tests, requests     │  5. catalog digest ==     │
               │ GET /v1/endpoints, every minute    │     owner commitment      │
               ▼                                    │  6. each endpoint signs a │
  ┌──────────────────────────┐  identity, canaries, │     challenge at its URL  │
  │ MINER worker, same hotkey│ ◄────────────────────│     before any traffic    │
  │  no chain writes         │  tests, requests     └───────────────────────────┘
  │  decision engines        │
  └──────────────────────────┘
```

Rules:
- Only hotkeys with a manifest commitment and an axon on a public address are contacted.
- A manifest is accepted only if it is signed by the hotkey that owns the axon, is for this netuid, is built for the
  committed catalog (or, for one epoch after a change, the previous one) and for that catalog's mechanism, its digest
  equals that hotkey's current commitment, and every endpoint URL is exactly
  `http://<ip>:<port>/e/<endpoint_id>`: the IP a public unicast address written as the chain writes axons (not
  private, shared, loopback, link-local, multicast, reserved, unspecified or documentation, and not an IPv6 form of
  an IPv4 address), the port 1-65535, nothing else. The host is the axon or any other machine of the miner. A
  manifest lists at most 64 endpoints on at most 64 hosts. A miner therefore cannot show different endpoint lists to
  different validators. The manifest depends only on the miner's configuration, its capacity measurements and the
  catalog, so a restart does not change the commitment.
- An endpoint gets traffic only once it proved control at its URL: it signs a fresh challenge with the listing
  hotkey (`POST /e/<endpoint_id>/v1/identity`). Validators challenge every endpoint each epoch before its first
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
  the code; newer ones come from the owner's axon (`GET /v1/catalog/{digest}`) or any mirror, since the digest
  proves them. A new commitment takes effect without a code release.
- Signatures and digests cover the JSON documents exactly as published (canonical encoding of the parsed JSON), so
  nodes whose code knows fewer fields still agree on them. Responses from other neurons are size- and time-bounded.
- The throughput audit (declared capacity, see `THROUGHPUT_AUDIT.md`) is a separate check applied after discovery.

Chain limits (read from subtensor main and testnet 405 on 2026-10-06):
- `serve_axon`: no fee (`Pays::No`), at most once per `ServingRateLimit` = 50 blocks (about 10 minutes). Needed only
  when the manifest server's ip or port changes.
- `set_commitment`: no fee and no deposit; only hotkeys registered on the netuid; up to 3 fields per commitment;
  3,100 bytes per hotkey per epoch, each commit counting at least 100 bytes (about 31 commits per 360-block
  epoch). One current commitment per (netuid, hotkey).
