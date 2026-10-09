"""Light validators: they set exactly the weights the source validator published, only from a document that
verifies, and never contact a miner or start a reference engine; the auditor publishes exactly what it sets."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from types import SimpleNamespace

import httpx
import pytest
from bittensor_wallet import Keypair
from fastapi import FastAPI
from fastapi.testclient import TestClient

from verathos import catalog, release
from verathos.protocol.callers import ValidatorIdentity
from verathos.protocol.status import NETWORK_DOMAIN, NETWORK_PATH, STATUS_MAX_AGE_S, NetworkStatus, sign_document
from verathos.validator import chain as validator_chain
from verathos.validator import light
from verathos.validator.config import ValidatorConfig
from verathos.validator.scoring import ScoreState
from verathos.validator.service import add_network_route
from verathos.validator.status import StatusBoard

AUDITOR_PIN = "ab" * 32  # the auditor's certificate, as the catalog names it
CAT = catalog.Catalog.parse({**catalog.load().document(), "auditor": {"tls": AUDITOR_PIN}})
BLOCKS = CAT.epoch_blocks
OFFSET = light.LightValidator.offset(CAT)
SOURCE_AXON = ("198.51.100.20", 8091)
WEIGHTS = {3: 0.5123456789012345, 7: 0.3, 12: 0.18765432109876551}


def kp() -> Keypair:
    return Keypair.create_from_mnemonic(Keypair.generate_mnemonic())


def neuron(uid: int, ip: str = "0.0.0.0", port: int = 0, permit: bool = True) -> SimpleNamespace:
    return SimpleNamespace(uid=uid, is_null=False, validator_permit=permit, axon_info=SimpleNamespace(ip=ip, port=port))


class Chain:
    """The chain as a light validator and the weight-setting code see it. A light validator reads no metagraph and
    no miner's commitment: those are how validators find miners; it reads the commitment of a source validator other
    than the auditor, which names its TLS pin."""

    def __init__(self, owner: str, own: str, block: int) -> None:
        self.owner, self.block = owner, block
        self.neurons = {owner: neuron(0, *SOURCE_AXON), own: neuron(5)}
        self.commitments: dict[int, str] = {}
        self.set: list[dict] = []
        self.last_update: dict[int, list[int]] = {}  # per mechanism, per UID
        self.version_floor = 7  # the subnet's WeightsVersionKey
        self.substrate = SimpleNamespace(query=self.query)

    def query(self, module: str, item: str, params: list):
        if item == "LastUpdate":
            mechanism, netuid = divmod(params[0], 4096)  # the chain keeps each mechanism's apart
            assert netuid == 405
            return SimpleNamespace(value=self.last_update.get(mechanism, []))
        assert item == "WeightsVersionKey"
        return SimpleNamespace(value=self.version_floor)

    def get_current_block(self) -> int:
        return self.block

    def get_subnet_owner_hotkey(self, netuid: int) -> str:
        return self.owner

    def get_neuron_for_pubkey_and_subnet(self, hotkey: str, netuid: int):
        return self.neurons.get(hotkey)

    def get_mechanism_count(self, netuid: int) -> int:
        return 2

    def get_uid_for_hotkey_on_subnet(self, hotkey: str, netuid: int) -> int | None:
        return self.neurons[hotkey].uid if hotkey in self.neurons else None

    def commit_reveal_enabled(self, netuid: int) -> bool:
        return True

    def weights_rate_limit(self, netuid: int) -> int:
        return 100

    def set_weights(self, *args, **kwargs):
        raise AssertionError("the SDK's set_weights checks the rate limit of mechanism 0 only")

    def close(self) -> None:
        pass

    def metagraph(self, *args, **kwargs):
        raise AssertionError("a light validator reads no metagraph")

    def get_all_commitments(self, *args, **kwargs):
        raise AssertionError("a light validator reads no miner's commitments")

    def get_commitment(self, netuid: int, uid: int) -> str:
        return self.commitments.get(uid, "")


@pytest.fixture(autouse=True)
def commits(monkeypatch) -> None:
    """Weights are committed (the subnet has commit-reveal on) into the fake chain's ``set``."""
    def commit(subtensor, wallet, netuid, mechid, uids, weights, **kwargs):
        subtensor.set.append({"netuid": netuid, "uids": uids, "weights": weights, "mechid": mechid,
                              "version_key": kwargs["version_key"]})
        return SimpleNamespace(success=True, message="")

    monkeypatch.setattr(validator_chain, "commit_timelocked_weights_extrinsic", commit)
    monkeypatch.setattr(validator_chain, "set_weights_extrinsic", lambda *a, **k: pytest.fail("commit-reveal is on"))


def config(tmp_path, **changes) -> ValidatorConfig:
    return ValidatorConfig(**{"netuid": 405, "wallet_name": "v", "hotkey_name": "default", "mode": "light",
                              "state_dir": tmp_path, **changes})


def document(signer: Keypair, epoch: int, *, weights: dict[int, float] | None = WEIGHTS, netuid: int = 405,
             at: float | None = None, epoch_blocks: int = BLOCKS) -> dict:
    """A network document as a validator of this release serves it."""
    body = NetworkStatus(epoch=epoch, epoch_blocks=epoch_blocks, weights=weights).model_dump(mode="json")
    if weights is None:
        del body["weights"]  # a validator of an older release
    return sign_document(body, signer, netuid=netuid, domain=NETWORK_DOMAIN,
                         now_ms=int((time.time() if at is None else at) * 1000))


def validator(tmp_path, chain: Chain, own: Keypair, served: list, *, fetched: list | None = None,
              pins: list | None = None, **cfg) -> light.LightValidator:
    """A light validator whose source's axon serves ``served[-1]`` (an exception is raised); ``fetched`` gets the
    URLs it asks, ``pins`` the certificate pin it accepts for each."""
    from verathos.validator import __main__ as validator_main

    settings = config(tmp_path, **cfg)

    def fetch(url: str, **kwargs):
        (fetched if fetched is not None else []).append(url)
        (pins if pins is not None else []).append(kwargs["tls"])
        if isinstance(served[-1], Exception):
            raise served[-1]
        return json.loads(json.dumps(served[-1]))

    wallet = SimpleNamespace(hotkey=own)
    return light.LightValidator(settings, chain, own.ss58_address, SimpleNamespace(catalog=CAT),
                                set_weights=lambda w, mechid: validator_main.set_weights(settings, chain, wallet, w,
                                                                                         mechid),
                                state_path=tmp_path / "light.json", fetch=fetch)


def test_a_light_validator_sets_exactly_the_weights_its_source_published(tmp_path, caplog) -> None:
    caplog.set_level(logging.INFO, logger="verathos.validator")
    owner, own = kp(), kp()
    chain = Chain(owner.ss58_address, own.ss58_address, 100 * BLOCKS + OFFSET + 5)
    fetched: list[str] = []
    pins: list[str] = []
    served = [document(owner, 99)]  # the source has not scored epoch 100 yet
    started = validator(tmp_path, chain, own, served, fetched=fetched, pins=pins)
    assert started.step() == light.POLL_S and chain.set == []  # once epoch 100's are due, 99's are not set
    assert started.why == "the source has not published the results of epoch 100 yet (its latest are for epoch 99)"
    (tmp_path / "light.json").write_text(json.dumps({"epoch": 99, "epoch_blocks": BLOCKS}))
    fetched.clear()
    v = validator(tmp_path, chain, own, served, fetched=fetched)
    assert v.step() == light.POLL_S and chain.set == [] and "set already" in v.why
    served.append(document(owner, 100))
    chain.block += 3
    v.step()
    assert chain.set == [{"netuid": 405, "uids": [3, 7, 12], "weights": [WEIGHTS[3], WEIGHTS[7], WEIGHTS[12]],
                          "mechid": CAT.mechanism, "version_key": release.weights_version_key()}]  # exactly, with
    # this release's version key and on the catalog's mechanism, as full ones
    assert fetched == [f"https://{SOURCE_AXON[0]}:{SOURCE_AXON[1]}{NETWORK_PATH}"] * 2
    assert set(pins) == {AUDITOR_PIN}  # over TLS, accepting the certificate the catalog names for the auditor
    assert (f"epoch 100: set the weights of the subnet owner's validator {owner.ss58_address}: 3 UIDs, top 3: 0.512, "
            "7: 0.300, 12: 0.188") in caplog.messages
    # Done for the epoch: it waits for the next epoch's results, also after a restart.
    wait = v.step()
    assert wait == (101 * BLOCKS + OFFSET - chain.block) * light.BLOCK_TIME_S and len(fetched) == 2
    restarted = validator(tmp_path, chain, own, served, fetched=fetched)
    restarted.step()
    assert len(chain.set) == 1 and len(fetched) == 2

    # The previous epoch's results still come in time (the source finished late); never older ones.
    served.append(document(owner, 101))
    chain.block = 102 * BLOCKS + 2
    restarted.step()
    assert len(chain.set) == 2


def test_a_light_validator_refuses_documents_that_do_not_verify() -> None:
    source, stranger, now = kp(), kp(), time.time()

    def refused(doc: dict, done: int | None = 99 * BLOCKS) -> str:
        with pytest.raises(light.Refused) as exc:
            light.check(doc, source=source.ss58_address, netuid=405, epoch=100, epoch_blocks=BLOCKS, done_block=done,
                        now_ms=int(now * 1000))
        return str(exc.value)

    tampered = document(source, 100)
    tampered["weights"]["7"] = 0.9
    unsigned = NetworkStatus(epoch=100, epoch_blocks=BLOCKS, weights=WEIGHTS).model_dump(mode="json")
    assert refused(tampered) == "its signature does not verify against the source hotkey"
    assert refused(document(stranger, 100)).startswith("it is not signed by the source hotkey")
    assert refused(document(source, 100, netuid=96)) == (f"it is not signed by the source hotkey "
                                                         f"{source.ss58_address} for netuid 405")
    assert "within 300 s" in refused(document(source, 100, at=now - STATUS_MAX_AGE_S - 60))  # stale
    assert "within 300 s" in refused(document(source, 100, at=now + STATUS_MAX_AGE_S + 60))
    assert refused(document(source, 98), done=None) == ("it is for epoch 98, neither the current epoch 100 nor the "
                                                        "previous one")  # an old document, replayed
    assert "epoch 101" in refused(document(source, 101))
    assert "set already" in refused(document(source, 100), done=100 * BLOCKS)  # never set twice
    assert "not signed (the auditor runs an older release)" in refused(unsigned)
    assert refused(document(source, 100, weights=None)) == "it lists no weights (the source runs an older release)"
    assert "180 blocks" in refused(document(source, 50, epoch_blocks=360))
    assert "non-negative" in refused(document(source, 100, weights={3: -1.0}))
    got = light.check(document(source, 99), source=source.ss58_address, netuid=405, epoch=100, epoch_blocks=BLOCKS,
                      done_block=98 * BLOCKS, now_ms=int(now * 1000))
    assert got.weights == WEIGHTS and got.epoch == 99


def test_without_a_fresh_verified_document_a_light_validator_sets_nothing_and_says_so_once(tmp_path, caplog) -> None:
    owner, stranger, own = kp(), kp(), kp()
    chain = Chain(owner.ss58_address, own.ss58_address, 100 * BLOCKS + OFFSET)
    served: list = [httpx.ConnectError("connection refused")]
    v = validator(tmp_path, chain, own, served)
    for block in range(100 * BLOCKS + OFFSET, 101 * BLOCKS + OFFSET, 3):
        chain.block = block
        assert v.step() == light.POLL_S
        served.append(document(stranger, 100) if block > 100 * BLOCKS + 150 else served[-1])
    stale = document(owner, 100, at=time.time() - 3600)
    served.append(stale)
    chain.block = 101 * BLOCKS + OFFSET - 1
    v.step()
    assert chain.set == [] and not caplog.records
    chain.block += 1  # the deadline: the point from which the next epoch's results can come
    v.step()
    chain.block += 3
    v.step()
    assert chain.set == []
    assert [r.getMessage() for r in caplog.records] == [
        f"epoch 100: no weights set: no results of the source verified by block {101 * BLOCKS + OFFSET} (it was not "
        f"signed within {STATUS_MAX_AGE_S:.0f} s of this machine's clock)"]

    served.append(httpx.ConnectError("connection refused"))
    chain.block = 102 * BLOCKS + OFFSET
    v.step()
    assert len(caplog.records) == 2 and caplog.records[-1].getMessage().startswith("epoch 101: no weights set")
    assert v.why == (f"the source's axon https://{SOURCE_AXON[0]}:{SOURCE_AXON[1]} did not serve it: ConnectError: "
                     "connection refused")


def test_a_source_other_than_the_owner_needs_a_permit_an_axon_and_a_committed_pin(tmp_path) -> None:
    owner, other, own = kp(), kp(), kp()
    chain = Chain(owner.ss58_address, own.ss58_address, 100 * BLOCKS + OFFSET)
    chain.neurons[other.ss58_address] = neuron(9, "198.51.100.30", 8091, permit=False)
    served, pins = [document(other, 100)], []
    v = validator(tmp_path, chain, own, served, source_hotkey=other.ss58_address, pins=pins)
    v.step()
    assert chain.set == [] and v.why == f"the source hotkey {other.ss58_address} holds no validator permit"
    chain.neurons[other.ss58_address].validator_permit = True
    v.step()
    assert chain.set == [] and v.why.startswith(f"the source hotkey {other.ss58_address} commits no TLS pin")
    chain.commitments[9] = ValidatorIdentity("cd" * 32, ("8.8.8.8",)).commitment()
    v.step()
    assert len(chain.set) == 1 and pins == ["cd" * 32]  # its service, accepted by the pin it committed
    chain.neurons[own.ss58_address].validator_permit = False
    assert v.permit() == "this hotkey holds no validator permit, which setting weights takes"
    with pytest.raises(ValueError, match="not a hotkey"):
        config(tmp_path, source_hotkey="5Fnot-a-hotkey")


def test_the_auditor_refuses_light_mode(tmp_path) -> None:
    owner, own = kp(), kp()
    chain = Chain(owner.ss58_address, owner.ss58_address, 100 * BLOCKS)
    with pytest.raises(SystemExit, match="this is the subnet owner's hotkey"):
        light.run(config(tmp_path), chain, SimpleNamespace(hotkey=owner), SimpleNamespace(catalog=CAT),
                  set_weights=lambda w: True)
    with pytest.raises(SystemExit, match="own hotkey"):
        light.run(config(tmp_path, source_hotkey=own.ss58_address), Chain(owner.ss58_address, own.ss58_address, 0),
                  SimpleNamespace(hotkey=own), SimpleNamespace(catalog=CAT), set_weights=lambda w: True)
    # The owner hotkey moves to this validator's: it stops, and its restart refuses light mode.
    chain = Chain(owner.ss58_address, own.ss58_address, 100 * BLOCKS + OFFSET)
    v = validator(tmp_path, chain, own, [document(owner, 100)])
    chain.owner = own.ss58_address
    with pytest.raises(SystemExit, match="this is the subnet owner's hotkey"):
        v.step()
    assert chain.set == []


class Recorder:
    """Every HTTP request the process sends; only the source's axon answers."""

    def __init__(self, body: bytes) -> None:
        self.body, self.urls = body, []

    def sync(self, request: httpx.Request) -> httpx.Response:
        self.urls.append(str(request.url))
        if (request.url.host, request.url.port) == SOURCE_AXON:
            return httpx.Response(200, content=self.body, headers={"content-type": "application/json"})
        raise httpx.ConnectError("not the source")

    async def asynchronous(self, request: httpx.Request) -> httpx.Response:
        self.urls.append(str(request.url))
        raise httpx.ConnectError("a light validator sends nothing asynchronously")


def test_a_light_validator_never_contacts_miners_or_loads_references(tmp_path, monkeypatch) -> None:
    """Started as a service starts it, with references still in its configuration: it fetches the source's
    document and sets its weights, and nothing else of a full validator runs."""
    from verathos.validator import __main__ as validator_main

    owner, own = kp(), kp()
    chain = Chain(owner.ss58_address, own.ss58_address, 100 * BLOCKS + OFFSET + 5)
    recorder = Recorder(json.dumps(document(owner, 100)).encode())
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", recorder.sync)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", recorder.asynchronous)
    monkeypatch.setattr(validator_main.bt, "Subtensor", lambda network: chain)
    monkeypatch.setattr(validator_main.bt, "Wallet", lambda name, hotkey: SimpleNamespace(hotkey=own))
    monkeypatch.setattr(validator_main, "CatalogFollower", lambda *args, **kwargs: SimpleNamespace(
        catalog=CAT, start=lambda: None))

    def forbidden(*args, **kwargs):
        raise AssertionError("a light validator runs none of a full validator's checks")

    for name in ("load_secret", "missing_references", "wait_for_references", "references_down", "run_epoch",
                 "audit_epoch", "discover", "start_service"):
        monkeypatch.setattr(validator_main, name, forbidden)
    monkeypatch.setattr(validator_main.Throughput, "start", forbidden)
    path = tmp_path / "validator.json"
    path.write_text(json.dumps({"netuid": 405, "wallet_name": "v", "hotkey_name": "default", "mode": "light",
                                "state_dir": str(tmp_path), "catalog_dir": str(tmp_path / "catalog"),
                                "references": {"kev-4b": "http://127.0.0.1:8101"}}))
    monkeypatch.setattr(sys, "argv", ["verathos-validator", "--config", str(path), "--once"])
    validator_main.main()
    assert recorder.urls == [f"https://{SOURCE_AXON[0]}:{SOURCE_AXON[1]}{NETWORK_PATH}"]
    assert [s["uids"] for s in chain.set] == [[3, 7, 12]]


def auditor_and_light(tmp_path, cat: catalog.Catalog, verdicts=None):
    """The auditor's epochs (run_epoch with no endpoints and ``verdicts``, publishing on its board), and a light
    validator following it: -> (chain, light validator, a function running the auditor's epoch and returning the
    weights it set)."""
    from verathos.validator import __main__ as validator_main
    from verathos.validator.discovery import Identities, ManifestCache

    owner, own = kp(), kp()
    chain = Chain(owner.ss58_address, own.ss58_address, 100 * BLOCKS + OFFSET + 5)
    chain.metagraph = lambda netuid, lite=True: SimpleNamespace(
        hotkeys=[owner.ss58_address, "hkA", "hkB"], axons=[SimpleNamespace(ip="0.0.0.0", port=0)] * 3,
        S=[0.0, 0.0, 0.0], validator_permit=[True, False, False])
    chain.get_all_commitments = lambda netuid: {}

    async def no_verdicts(client, epoch):
        return None if verdicts is None else verdicts(owner.ss58_address, epoch)

    async def no_evidence(evidence, references, cat, budget_s=None):
        return []

    throughput = SimpleNamespace(auditor_hotkey=owner.ss58_address, manifests=ManifestCache(60.0),
                                 identities=Identities(), holding=lambda: False, verdicts=no_verdicts,
                                 update_slots=lambda miners, neurons, epoch: None, auditor=SimpleNamespace(),
                                 proven_invalid=no_evidence)
    current = SimpleNamespace(catalog=cat, for_epoch=lambda st, block: cat,
                              accepted=lambda: {cat.digest(): cat.mechanism})
    auditor_cfg = SimpleNamespace(netuid=405, mechid=None, references={}, dry_run=False)
    board = StatusBoard()
    app = FastAPI()
    add_network_route(app, board, owner, 405)
    client = TestClient(app)
    follower = validator(tmp_path, chain, own, [None])
    follower.current = SimpleNamespace(catalog=cat)
    follower.fetch = lambda url, **kwargs: client.get(NETWORK_PATH).json()

    def auditor_epoch(state: ScoreState, epoch: int) -> dict:
        asyncio.run(validator_main.run_epoch(auditor_cfg, chain, SimpleNamespace(hotkey=owner), b"k" * 32, state,
                                             epoch, 1, throughput, current, board))
        return chain.set[-1]

    return chain, follower, board, auditor_epoch


def test_the_auditor_publishes_exactly_the_weights_it_sets_and_light_validators_set_the_same(tmp_path) -> None:
    """Weights of UIDs whose endpoints are gone this epoch (their averages decay), the owner's UID's part beyond the
    miner share and the fallback to the owner's UID are part of what is set, so they are part of what is
    published."""
    chain, follower, board, auditor_epoch = auditor_and_light(tmp_path, CAT)
    state = ScoreState(ema={"hkA/gpu0": 0.6, "hkB/gpu0": 0.2}, uid_of={"hkA/gpu0": 1, "hkB/gpu0": 2})
    published = auditor_epoch(state, 100)
    network = NetworkStatus.model_validate_json(board.network())
    assert network.hotkeys == [] and network.epoch == 100  # no endpoint this epoch, yet weights are set
    assert network.weights == dict(zip(published["uids"], published["weights"]))
    assert network.miner_share == CAT.scoring.miner_share_at(100) == 0.59  # the shipped share, not stepping down yet
    assert network.weights == pytest.approx({0: 0.41, 1: 0.75 * 0.59, 2: 0.25 * 0.59})
    follower.step()
    assert chain.set[-1] == published

    gone = ScoreState()
    published = auditor_epoch(gone, 101)
    assert published["uids"] == [0] and NetworkStatus.model_validate_json(board.network()).weights == {0: 1.0}
    chain.block = 101 * BLOCKS + OFFSET + 5
    follower.step()
    assert chain.set[-1] == published and len(chain.set) == 4


def test_the_miner_share_steps_down_by_epoch_and_light_validators_set_the_auditors_weights(tmp_path) -> None:
    """Once the owner sets start_epoch, each epoch's share follows from the epoch alone: the auditor sets the miners'
    weights summing to it and the rest on the owner's UID, publishes the share, and a light validator sets exactly
    those weights."""
    document = CAT.document()
    stepping = catalog.Catalog.parse({**document, "scoring": {**document["scoring"], "miner_share": {
        "from": 0.59, "to": 0.15, "start_epoch": 100, "epochs": 4}}})
    chain, follower, board, auditor_epoch = auditor_and_light(tmp_path, stepping)
    state = ScoreState(ema={"hkA/gpu0": 0.6, "hkB/gpu0": 0.2}, uid_of={"hkA/gpu0": 1, "hkB/gpu0": 2})
    for epoch, share in ((100, 0.59), (101, 0.48), (102, 0.37)):
        chain.block = epoch * BLOCKS + OFFSET + 5
        published = auditor_epoch(state, epoch)
        network = NetworkStatus.model_validate_json(board.network())
        assert network.miner_share == share and network.weights == dict(zip(published["uids"], published["weights"]))
        assert network.weights[0] == pytest.approx(1 - share) and sum(network.weights.values()) == pytest.approx(1)
        follower.step()
        assert chain.set[-1] == published
    assert [stepping.scoring.miner_share_at(e) for e in (103, 104, 10**6)] == [0.26, 0.15, 0.15]


def test_the_usage_pool_is_part_of_what_the_auditor_sets_and_light_validators_set_the_same(tmp_path) -> None:
    """The hotkeys that served paying users get the period's usage share on top of their scores; the auditor
    publishes the weights with it and the share, and a light validator sets exactly those."""
    from verathos.throughput.feed import Verdicts
    from verathos.throughput.usage import Burn, Emission, close

    document = CAT.document()
    buyback = kp().ss58_address
    pooled = catalog.Catalog.parse({**document, "scoring": {**document["scoring"], "usage_pool": {
        "mode": "burned", "fraction": 1.0, "period_epochs": 2, "buyback_coldkey": buyback}}})
    emission = Emission(alpha_out=10**9, owner_cut=11796, mechanism=0, mechanisms=1)
    record = close(49, pool=pooled.scoring.usage_pool, cat=pooled, epoch_blocks=BLOCKS,  # epoch 100 pays period 49
                   paid={(hotkey, "gpu0", "kev-4b"): n for hotkey, n in (("hkA", 300), ("hkB", 100), ("hkZ", 100))},
                   coldkeys={}, emission=emission,
                   burns=[Burn(block=50 * 2 * BLOCKS - 9, event_index=0, coldkey=buyback, hotkey="hkX",
                               amount=40 * 10**9)])

    def verdicts(auditor: str, epoch: int) -> Verdicts:
        return Verdicts(netuid=405, auditor=auditor, epoch=epoch, credits=[], invalid=[],
                        usage=record.model_dump(mode="json"))
    chain, follower, board, auditor_epoch = auditor_and_light(tmp_path, pooled, verdicts)
    state = ScoreState(ema={"hkA/gpu0": 0.6, "hkB/gpu0": 0.2}, uid_of={"hkA/gpu0": 1, "hkB/gpu0": 2})
    published = auditor_epoch(state, 100)
    network = NetworkStatus.model_validate_json(board.network())
    share = record.share
    assert 0 < share < 1 - 0.59 and network.usage_share == round(share * 4 / 5, 6)  # hkZ holds no UID
    assert network.weights == dict(zip(published["uids"], published["weights"]))
    assert network.weights == pytest.approx({0: 0.41 - share * 4 / 5, 1: 0.75 * 0.59 + share * 3 / 5,
                                             2: 0.25 * 0.59 + share * 1 / 5})
    follower.step()
    assert chain.set[-1] == published


def test_the_auditor_and_light_validators_set_weights_on_the_catalogs_mechanism(tmp_path) -> None:
    """The mechanism comes from the committed catalog: the auditor sets the epoch's weights on it and publishes it,
    and a light validator sets them on the mechanism of the catalog it follows, or on the one its mechid names."""
    moved = catalog.Catalog.parse({**CAT.document(), "mechanism": 1})
    chain, follower, board, auditor_epoch = auditor_and_light(tmp_path, moved)
    state = ScoreState(ema={"hkA/gpu0": 0.6, "hkB/gpu0": 0.2}, uid_of={"hkA/gpu0": 1, "hkB/gpu0": 2})
    published = auditor_epoch(state, 100)
    assert published["mechid"] == 1 and NetworkStatus.model_validate_json(board.network()).mechanism == 1
    follower.step()
    assert chain.set[-1] == published
    follower.cfg = follower.cfg.model_copy(update={"mechid": 0})
    published = auditor_epoch(state, 101)
    chain.block = 101 * BLOCKS + OFFSET + 5
    follower.step()
    assert chain.set[-1] == {**published, "mechid": 0}


def test_a_light_configuration_needs_no_references(tmp_path) -> None:
    light_config = ValidatorConfig.model_validate({"netuid": 96, "wallet_name": "v", "hotkey_name": "h",
                                                   "mode": "light"})
    assert light_config.references == {} and light_config.source_hotkey is None
    assert ValidatorConfig.model_validate({"netuid": 96, "wallet_name": "v", "hotkey_name": "h"}).mode == "full"
    with pytest.raises(ValueError):
        ValidatorConfig.model_validate({"netuid": 96, "wallet_name": "v", "hotkey_name": "h", "mode": "half"})


def test_weights_wait_out_the_rate_limit_of_their_own_mechanism(tmp_path, monkeypatch, caplog) -> None:
    """The chain limits how often a UID sets weights, per mechanism: a set it would refuse waits the blocks left of
    this mechanism's limit (a set on mechanism 0 just before does not hold back one on mechanism 1), one it allows
    goes at once."""
    from verathos.validator import __main__ as validator_main

    caplog.set_level(logging.INFO, logger="verathos.validator")
    owner, own = kp(), kp()
    chain = Chain(owner.ss58_address, own.ss58_address, 100 * BLOCKS + OFFSET)
    slept: list[float] = []

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        chain.block += round(seconds / light.BLOCK_TIME_S)

    monkeypatch.setattr(validator_main.time, "sleep", sleep)
    chain.last_update = {0: [0] * 6, 1: [0] * 6}
    chain.last_update[0][5] = chain.block - 1
    chain.last_update[1][5] = chain.block - 70  # 30 of the 100 blocks left
    settings, wallet = config(tmp_path), SimpleNamespace(hotkey=own)
    assert validator_main.set_weights(settings, chain, wallet, WEIGHTS, 1)
    assert slept == [30 * light.BLOCK_TIME_S] and len(chain.set) == 1
    assert "the chain allows this UID's next weights on mechanism 1 in 30 blocks; setting them then" in caplog.messages
    chain.last_update[1][5] = chain.block - 100
    slept.clear()
    assert validator_main.set_weights(settings, chain, wallet, WEIGHTS, 1) and slept == [] and len(chain.set) == 2


def test_a_validator_whose_release_the_subnet_retired_sets_no_weights_and_says_why(tmp_path, monkeypatch,
                                                                                    caplog) -> None:
    """The chain refuses weights below the subnet's WeightsVersionKey: a validator of an older release does not try,
    and logs that it needs an update; one at or above the key sets them."""
    from verathos.validator import __main__ as validator_main

    caplog.set_level(logging.INFO, logger="verathos.validator")
    owner, own = kp(), kp()
    chain = Chain(owner.ss58_address, own.ss58_address, 100 * BLOCKS + OFFSET + 5)
    chain.version_floor = release.weights_version_key() + 1  # the owner retired this release
    monkeypatch.setattr(validator_main.time, "sleep", lambda s: pytest.fail("refused at once, not retried"))
    assert not validator_main.set_weights(config(tmp_path), chain, SimpleNamespace(hotkey=own), WEIGHTS, 1)
    too_old = (f"this validator's release is too old to set weights: the subnet's weights version key is "
               f"{chain.version_floor}, above its own {release.weights_version_key()} (verathos ")
    assert chain.set == [] and any(m.startswith("weights not set: " + too_old) for m in caplog.messages)
    v = validator(tmp_path, chain, own, [document(owner, 100)])
    assert v.step() == light.POLL_S and chain.set == [] and v.why.startswith(too_old)
    chain.version_floor = release.weights_version_key()
    v.step()
    assert [s["version_key"] for s in chain.set] == [release.weights_version_key()]
