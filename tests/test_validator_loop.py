"""Full validator loop against in-process miners: discovery, identity, canaries, receipts, comparison, scoring.

Miners are real miner apps served over an in-memory transport; engines are deterministic mocks. The honest miner
uses the reference engine's model; the cheater uses a different one behind the same model label.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
from bittensor_wallet import Keypair

from tests.mock_engine import MockEngine
from verathos import catalog
from verathos.miner.app import create_app
from verathos.miner.config import MinerConfig
from verathos.protocol.manifest import Endpoint, Manifest, SignedManifest, commitment_value
from verathos.validator.audit import audit_epoch
from verathos.validator.canaries import assignment, canary_pool
from verathos.validator.compare import Tolerance, Verdict
from verathos.validator.discovery import Neuron, discover
from verathos.validator.scoring import ScoreState, epoch_scores, record_window, update, weights


class HostRouter(httpx.AsyncBaseTransport):
    """Routes requests to in-process ASGI apps by host name."""

    def __init__(self, apps: dict[str, object]) -> None:
        self._t = {host: httpx.ASGITransport(app=app) for host, app in apps.items()}

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return await self._t[request.url.host].handle_async_request(request)


def kp() -> Keypair:
    return Keypair.create_from_mnemonic(Keypair.generate_mnemonic())


def miner(host: str, keypair: Keypair, validator: Keypair, engine: MockEngine):
    config = MinerConfig.model_validate({
        "netuid": 405, "mechid": 1, "wallet_name": "w", "hotkey_name": "h", "public_host": host, "port": 80,
        "endpoints": [{"endpoint_id": "gpu0", "model_id": "kev-4b", "engine_url": "http://unused",
                       "hardware": {"kind": "cuda", "device_id": f"GPU-{host}"}, "max_concurrency": 8}]})
    digest = catalog.load().digest()
    signed = SignedManifest.create(Manifest(
        netuid=405, mechid=1, hotkey=keypair.ss58_address, catalog_digest=digest,
        endpoints=[Endpoint(endpoint_id="gpu0", url=config.endpoint_url("gpu0"), model_id="kev-4b",
                            hardware=config.endpoints[0].hardware, max_concurrency=8)]), keypair)
    app = create_app(config=config, keypair=keypair, manifest=lambda: signed, engines={"gpu0": engine},
                     authorized=lambda hk: hk == validator.ss58_address)
    return app, commitment_value(signed.manifest.digest())


def test_canary_pool_is_deterministic_and_assignment_rotates() -> None:
    secret = b"s" * 32
    assert canary_pool(secret, 7, "kev-4b", 5) == canary_pool(secret, 7, "kev-4b", 5)
    assert canary_pool(secret, 7, "kev-4b", 5) != canary_pool(secret, 8, "kev-4b", 5)
    assert canary_pool(secret, 7, "kev-4b", 5) != canary_pool(b"t" * 32, 7, "kev-4b", 5)
    a = assignment(secret, 7, "hk/gpu0", 20, 6)
    assert len(a) == 6 and a != assignment(secret, 8, "hk/gpu0", 20, 6)


def test_epoch_scores_honest_and_zeroes_cheater() -> None:
    validator, honest_kp, cheat_kp, ghost_kp = kp(), kp(), kp(), kp()
    honest, cheat, ghost = "10.0.0.1", "10.0.0.2", "10.0.0.3"
    honest_app, honest_commit = miner(honest, honest_kp, validator, MockEngine("kev-4b"))
    cheat_app, cheat_commit = miner(cheat, cheat_kp, validator, MockEngine("kev-4b", salt="other-model"))
    ghost_app, _ = miner(ghost, ghost_kp, validator, MockEngine("kev-4b"))
    neurons = [
        Neuron(1, honest_kp.ss58_address, honest, 80, honest_commit),
        Neuron(2, cheat_kp.ss58_address, cheat, 80, cheat_commit),
        Neuron(3, ghost_kp.ss58_address, ghost, 80, None),  # serves a manifest but never committed it
    ]

    async def run():
        transport = HostRouter({honest: honest_app, cheat: cheat_app, ghost: ghost_app})
        async with httpx.AsyncClient(transport=transport) as client:
            miners = await discover(neurons, 405, 1, client, {catalog.load().digest()})
            endpoints = [e for m in miners for e in m.endpoints()]
            results = await audit_epoch(endpoints, epoch=1, secret=b"k" * 32, keypair=validator,
                                        references={"kev-4b": MockEngine("kev-4b")},
                                        tolerances={"kev-4b": Tolerance(min_probabilities=20)}, client=client,
                                        netuid=405, mechid=1, pool_size=10, per_endpoint=6)
            return miners, results

    miners, results = asyncio.run(run())
    assert {m.uid for m in miners} == {1, 2}
    honest = results[f"{honest_kp.ss58_address}/gpu0"]
    cheat = results[f"{cheat_kp.ss58_address}/gpu0"]
    assert honest.identity_ok and cheat.identity_ok
    assert honest.answered == honest.sent == 6 and not honest.errors
    assert honest.tally.verdict() is Verdict.PASS
    assert cheat.tally.verdict() is Verdict.FAIL_HARD

    state = ScoreState()
    record_window(state, results, 3)
    scores, failed = epoch_scores(results, {"kev-4b": 1.0}, state)
    update(state, results, scores, failed)
    w = weights(state, {1: honest_kp.ss58_address, 2: cheat_kp.ss58_address})
    assert w == {1: pytest.approx(1.0)}
    assert state.probation == {f"{cheat_kp.ss58_address}/gpu0": 5}  # the failing epoch is the first of six

    # probation holds even if the cheater behaves next epoch
    update(state, results, {k: 1.0 for k in scores}, set())
    assert state.ema[f"{cheat_kp.ss58_address}/gpu0"] == 0.0
    assert state.probation[f"{cheat_kp.ss58_address}/gpu0"] == 4


def test_weights_ignore_reassigned_uids() -> None:
    state = ScoreState(ema={"hkA/gpu0": 0.5, "hkB/gpu0": 0.5}, uid_of={"hkA/gpu0": 1, "hkB/gpu0": 2})
    assert weights(state, {1: "hkA", 2: "someone-else"}) == {1: pytest.approx(1.0)}


def test_window_catches_subtle_drift_no_single_epoch_shows() -> None:
    """Each epoch alone has too few probabilities for the statistical bound; three epochs together fail it."""
    from verathos.validator.audit import EndpointResult
    from verathos.validator.compare import RequestComparison
    from verathos.validator.discovery import DiscoveredEndpoint

    tol = Tolerance(min_probabilities=100, mean_abs=0.005)
    ep = DiscoveredEndpoint(1, "hkA", "http://x", Endpoint(endpoint_id="gpu0", url="http://x/e/gpu0", model_id="kev-4b",
                            hardware={"kind": "cuda", "device_id": "GPU-a"}, max_concurrency=1))
    state = ScoreState()
    verdicts = []
    for _ in range(3):
        r = EndpointResult(ep, tol, identity_ok=True, sent=10, answered=10, latencies_ms=[40.0] * 10)
        # mean 0.01, no hard fail
        r.tally.add(RequestComparison(max_abs=0.03, sum_abs=0.6, count=60, confident_flips=0))
        record_window(state, {"hkA/gpu0": r}, 3)
        scores, failed = epoch_scores({"hkA/gpu0": r}, {"kev-4b": 1.0}, state)
        verdicts.append(bool(failed))
    assert verdicts == [False, True, True]  # 60 probabilities: insufficient; 120 and 180: mean 0.01 > 0.005


def test_capacity_gate_and_unauditable_endpoints() -> None:
    """Gated endpoints fail verification (probation); open endpoints without a calibrated GPU class score zero."""
    from verathos.validator.audit import EndpointResult
    from verathos.validator.capacity import capacity_slots, unauditable_endpoints
    from verathos.validator.discovery import DiscoveredEndpoint, DiscoveredMiner

    audited, undeclared = kp().ss58_address, kp().ss58_address
    hardware = {audited: {"kind": "cuda", "device_id": "GPU-a", "gpu_name": "NVIDIA GeForce RTX 5090", "vram_gb": 32},
                undeclared: {"kind": "cuda", "device_id": "GPU-b"}}
    miners = []
    for uid, hotkey in enumerate((audited, undeclared), start=1):
        endpoint = Endpoint(endpoint_id="gpu0", url=f"http://h{uid}/e/gpu0", model_id="kev-4b",
                            hardware=hardware[hotkey], max_concurrency=8)
        miners.append(DiscoveredMiner(uid, hotkey, f"http://h{uid}", Manifest(
            netuid=405, mechid=1, hotkey=hotkey, catalog_digest=catalog.load().digest(),
            endpoints=[endpoint])))
    assert [s.hotkey for s in capacity_slots(miners, 405)] == [audited]
    assert unauditable_endpoints(miners, catalog.load().audit_policy()) == {f"{undeclared}/gpu0"}

    tol = Tolerance(min_probabilities=1)
    results = {}
    for m in miners:
        ep = DiscoveredEndpoint(m.uid, m.hotkey, m.base_url, m.manifest.endpoints[0])
        results[ep.key] = EndpointResult(ep, tol, identity_ok=True, sent=4, answered=4, latencies_ms=[40.0] * 4)
    state = ScoreState()
    record_window(state, results, 3)
    scores, failed = epoch_scores(results, {"kev-4b": 1.0}, state, capacity_gated={f"{audited}/gpu0"},
                                  unauditable={f"{undeclared}/gpu0"})
    assert scores == {f"{audited}/gpu0": 0.0, f"{undeclared}/gpu0": 0.0}
    assert failed == {f"{audited}/gpu0"}


def test_hostile_replies_fail_the_endpoint_without_aborting_the_epoch() -> None:
    """Malformed identity replies, NaN probabilities and junk receipts are recorded against the endpoint."""
    from fastapi import FastAPI
    from fastapi.responses import PlainTextResponse

    from verathos.validator.discovery import DiscoveredEndpoint

    validator, hostile_kp = kp(), kp()
    app = FastAPI()

    @app.post("/e/gpu0/v1/identity")
    async def identity():
        return PlainTextResponse('{"signature": 5}', media_type="application/json")

    @app.post("/e/gpu0/v1/systemone")
    async def systemone():
        return PlainTextResponse('{"answers": {"q": {"noul": NaN}}, "receipt": []}', media_type="application/json")

    ep = DiscoveredEndpoint(1, hostile_kp.ss58_address, "http://10.0.0.9:80", Endpoint(
        endpoint_id="gpu0", url="http://10.0.0.9:80/e/gpu0", model_id="kev-4b",
        hardware={"kind": "cuda", "device_id": "GPU-x"}, max_concurrency=1))

    async def run():
        async with httpx.AsyncClient(transport=HostRouter({"10.0.0.9": app})) as client:
            return await audit_epoch([ep], epoch=1, secret=b"k" * 32, keypair=validator,
                                     references={"kev-4b": MockEngine("kev-4b")},
                                     tolerances={"kev-4b": Tolerance()}, client=client, netuid=405, mechid=1,
                                     pool_size=4, per_endpoint=3)

    result = asyncio.run(run())[ep.key]
    assert not result.identity_ok
    assert result.sent == 3 and len(result.tally.hard_failures) == 3
    assert result.tally.verdict() is Verdict.FAIL_HARD


def test_non_finite_or_out_of_range_probabilities_are_malformed() -> None:
    from verathos.protocol.api import distributions

    for bad in ({"q": {"noul": float("nan")}}, {"q": {"probabilities": {"a": "0.5"}}}, {"q": {"noul": 1.5}},
                {"q": {"noul": True}}, {"q": []}, []):
        with pytest.raises(ValueError):
            distributions(bad)
    assert distributions({"q": {"noul": 0.25}}) == {"q": {"false": 0.75, "true": 0.25}}


def test_renamed_endpoint_inherits_probation() -> None:
    from verathos.validator.audit import EndpointResult
    from verathos.validator.discovery import DiscoveredEndpoint

    def result(endpoint_id: str) -> EndpointResult:
        ep = DiscoveredEndpoint(1, "hkA", "http://x", Endpoint(
            endpoint_id=endpoint_id, url=f"http://x/e/{endpoint_id}", model_id="kev-4b",
            hardware={"kind": "cuda", "device_id": "GPU-a"}, max_concurrency=1))
        return EndpointResult(ep, Tolerance(), identity_ok=True, sent=4, answered=4, latencies_ms=[40.0] * 4)

    state = ScoreState()
    update(state, {"hkA/gpu0": result("gpu0")}, {"hkA/gpu0": 0.0}, {"hkA/gpu0"}, probation_epochs=6)
    update(state, {"hkA/renamed": result("renamed")}, {"hkA/renamed": 1.0}, set(), probation_epochs=6)
    assert state.ema["hkA/renamed"] == 0.0 and state.probation["hkA/renamed"] == 4


def test_uid_value_is_its_best_endpoint_until_capacity_gates() -> None:
    state = ScoreState(ema={"hkA/gpu0": 0.5, "hkA/gpu1": 0.5, "hkB/gpu0": 0.5},
                       uid_of={"hkA/gpu0": 1, "hkA/gpu1": 1, "hkB/gpu0": 2})
    hotkeys = {1: "hkA", 2: "hkB"}
    assert weights(state, hotkeys, per_uid="max") == {1: pytest.approx(0.5), 2: pytest.approx(0.5)}
    assert weights(state, hotkeys, per_uid="sum") == {1: pytest.approx(2 / 3), 2: pytest.approx(1 / 3)}


def test_gone_endpoints_are_forgotten() -> None:
    state = ScoreState(ema={"hkA/gpu0": 1e-5}, uid_of={"hkA/gpu0": 1}, window={"hkA/gpu0": [[0.0, 1.0]]})
    update(state, {}, {}, set())
    assert state.ema == {} and state.uid_of == {} and state.window == {}


def test_endpoints_the_validator_could_not_audit_keep_their_score() -> None:
    state = ScoreState(ema={"hkA/gpu0": 0.5}, uid_of={"hkA/gpu0": 1})
    update(state, {}, {}, set(), held={"hkA/gpu0"})
    assert state.ema == {"hkA/gpu0": 0.5}
    update(state, {}, {}, set())
    assert state.ema["hkA/gpu0"] == pytest.approx(0.4)
