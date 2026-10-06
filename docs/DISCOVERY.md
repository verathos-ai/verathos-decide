# Discovery and catalog

Miners and the model catalog are discovered from native chain state only. No EVM, no gas, no central database.

```
                         SUBSTRATE (SN96)
  ┌──────────────────────────────────────────────────────────────────┐
  │  owner hotkey (UID 0)                                            │
  │    commitment: "verathos/catalog/v1 sha256:<catalog digest>"       │
  │                                                                  │
  │  miner hotkey (any UID)                                          │
  │    axon:       ip:port of the miner's manifest server            │
  │    commitment: "verathos/manifest/v1 sha256:<manifest digest>"     │
  └──────────────────────────────────────────────────────────────────┘
          ▲ serve_axon / set_commitment              │ read axons + commitments
          │ (free, hotkey-signed)                    ▼
  ┌──────────────────────────┐   GET /v1/manifest   ┌───────────────────────────┐
  │ MINER                    │ ◄────────────────────│ VALIDATOR / GATEWAY       │
  │  manifest server (axon)  │                      │  1. hotkeys with manifest │
  │  signed manifest:        │                      │     commitment            │
  │   endpoint gpu0 kev-4b   │                      │  2. fetch manifest        │
  │   endpoint gpu1 kev-9b   │                      │  3. signature == hotkey   │
  │   ...                    │   requests, audits   │  4. digest == commitment  │
  │  decision engines        │ ◄────────────────────│  5. catalog digest ==     │
  └──────────────────────────┘                      │     owner commitment      │
                                                    └───────────────────────────┘
```

Rules:
- A manifest is accepted only if it is signed by the hotkey that owns the axon, is for this netuid and mechanism,
  its digest equals that hotkey's current commitment, and every endpoint URL is `http://<axon ip>:<axon port>/e/<id>`.
  A miner therefore cannot show different endpoint lists to different validators, nor claim hosts other than its
  own axon. The manifest depends only on the miner's configuration and the catalog, so a restart does not change
  the commitment.
- A manifest names the catalog it was built for. Validators accept the current catalog and, for one tempo after a
  change, the previous one, while miners re-publish (they check every minute).
- A catalog is accepted only if its digest equals the subnet owner hotkey's current commitment. A copy ships with
  the code; newer ones come from the owner's axon (`GET /v1/catalog/{digest}`) or any mirror, since the digest
  proves them. A new commitment takes effect without a code release.
- Signatures and digests cover the JSON documents exactly as published (canonical encoding of the parsed JSON), so
  nodes whose code knows fewer fields still agree on them. Responses from other neurons are size- and time-bounded.
- Endpoint control (each endpoint signs a validator nonce with the listing hotkey) and the capacity audit
  (shared hardware) are separate checks applied after discovery.

Chain limits (read from subtensor main and testnet 405 on 2026-10-06):
- `serve_axon`: no fee (`Pays::No`), at most once per `ServingRateLimit` = 50 blocks (about 10 minutes). Needed only
  when the manifest server's ip or port changes.
- `set_commitment`: no fee and no deposit; only hotkeys registered on the netuid; up to 3 fields per commitment;
  3,100 bytes per hotkey per epoch, each commit counting at least 100 bytes (about 31 commits per 360-block
  epoch). One current commitment per (netuid, hotkey).
