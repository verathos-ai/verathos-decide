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
- A manifest is accepted only if it is signed by the hotkey that owns the axon, is for this netuid, and its digest
  equals that hotkey's current commitment. A miner therefore cannot show different endpoint lists to different
  validators, and the commitment block records when each version was published.
- A catalog is accepted only if its digest equals the subnet owner hotkey's current commitment. The file ships
  with the code; anyone can mirror it because the digest proves it.
- Endpoint control (each endpoint signs a validator nonce with the listing hotkey) and the capacity audit
  (shared hardware) are separate checks applied after discovery.

Chain limits (read from subtensor main and testnet 405 on 2026-10-06):
- `serve_axon`: no fee (`Pays::No`), at most once per `ServingRateLimit` = 50 blocks (about 10 minutes). Needed only
  when the manifest server's ip or port changes.
- `set_commitment`: no fee and no deposit; only hotkeys registered on the netuid; up to 3 fields per commitment;
  3,100 bytes per hotkey per epoch, each commit counting at least 100 bytes (about 31 commits per 360-block
  epoch). One current commitment per (netuid, hotkey).

Comparable designs: Targon (SN4) also points the axon at a miner service listing nodes, without a committed
digest; Templar (SN3) and Chutes (SN64) use commitments for pointers and hashes; Chutes and Engy (SN53) keep the
node registry in an owner-run API instead.

## Testnet 405 run, 2026-10-06

Miner hotkey `test_miner96/rehearsal-aug18` (UID 12), owner hotkey `test_sn96` (UID 0), manifest served from a
temporary host, bittensor SDK 10.3.2, `scripts/testnet_discovery.py`.

| Case | Result |
|---|---|
| Signed manifest (seq 1, 2 endpoints), axon set, digest committed | accepted, both endpoints read, catalog matched |
| Catalog digest committed by the owner hotkey | accepted |
| New validly signed manifest (seq 2) served without committing | rejected: digest differs from commitment |
| Manifest content edited after signing | rejected: signature does not verify |
| Manifest signed by another hotkey served at the axon | rejected: hotkey mismatch |
| Seq 2 committed and served | accepted, visible 3 blocks after submission |
| Full discovery pass over all 14 UIDs (commitment pre-filter) | 3 s, one manifest fetched |

Commitment space used: 200 of 3,100 bytes (miner, two commits), 100 (owner, one commit).
