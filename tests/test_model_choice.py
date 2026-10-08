"""The miner setup's model choice: what each active model would pay on each GPU by the auditor's signed network
status, the hysteresis on updates, choices by hand, and the fallback without the network's numbers."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from bittensor_wallet import Keypair

from verathos import catalog
from verathos.install import miner as setup
from verathos.install import plan
from verathos.install.host import GiB, Gpu
from verathos.protocol.status import (
    NETWORK_DOMAIN,
    EndpointStatus,
    HotkeyStatus,
    NetworkStatus,
    sign_document,
)
from verathos.throughput.measure import STORE_FILE, Measurement, MeasurementStore

NEED = {"kev-4b": plan.miner_memory(4_659_865_088), "kev-9b": plan.miner_memory(9_653_104_368),
        "clef-flash": plan.miner_memory(9_409_813_744, "clef")}
KEV = ["kev-4b", "kev-9b"]
SPEED = {"kev-4b": 1.0, "kev-9b": 0.85, "clef-flash": 1.3}


def a100(index: int = 0) -> Gpu:
    return Gpu(index, "NVIDIA A100 80GB PCIe", 80 * GiB, 80 * GiB)


def market(shares: dict[str, float], capacity: dict[str, float], held: dict[str, float] | None = None,
           median_speed: float = 20.0) -> plan.Market:
    return plan.Market(shares, capacity, held or {}, median_speed)


def choose(gpus: list[Gpu], m: plan.Market | None, *, offered: list[str] = KEV, scored=None, requested=(),
           previous=None, **rule) -> dict[int, plan.Choice]:
    return plan.choose_models(gpus, offered, NEED, list(requested), previous or {}, offered if scored is None else
                              scored, market=m, relative=SPEED, **rule)


def models(choices: dict[int, plan.Choice]) -> dict[int, str]:
    return {i: c.model for i, c in choices.items()}


def test_a_gpu_takes_the_model_that_pays_most_and_a_crowded_model_pays_less() -> None:
    """Expected share = model share x rate / (proven capacity + rate), the rate being the GPU's speed times the
    model's relative speed: a larger share loses to a smaller one that fewer serve."""
    chosen = choose([a100()], market({"kev-4b": 0.6, "kev-9b": 0.4}, {"kev-4b": 600.0, "kev-9b": 100.0}))[0]
    assert chosen.model == "kev-9b" and chosen.reason == plan.PAYS
    assert chosen.rates == pytest.approx({"kev-4b": 20.0, "kev-9b": 17.0})
    assert chosen.pays == pytest.approx({"kev-4b": 0.6 * 20 / 620, "kev-9b": 0.4 * 17 / 117})
    assert chosen.source == "as the network's median endpoint"
    crowded = choose([a100()], market({"kev-4b": 0.6, "kev-9b": 0.4}, {"kev-4b": 600.0, "kev-9b": 1000.0}))
    assert models(crowded) == {0: "kev-4b"}
    # a model nobody serves yet pays its whole share to the first endpoint
    empty = choose([a100()], market({"kev-4b": 0.5, "kev-9b": 0.3, "clef-flash": 0.2},
                                    {"kev-4b": 100.0, "kev-9b": 50.0, "clef-flash": 0.0}),
                   offered=["kev-4b", "kev-9b", "clef-flash"])[0]
    assert empty.model == "clef-flash" and empty.pays["clef-flash"] == pytest.approx(0.2)


def test_a_machine_spreads_its_gpus_where_that_pays_counting_its_own_earlier_picks_and_other_machines() -> None:
    shares, capacity = {"kev-4b": 0.5, "kev-9b": 0.5}, {"kev-4b": 100.0, "kev-9b": 80.0}
    two = choose([a100(0), a100(1)], market(shares, capacity))
    assert models(two) == {0: "kev-9b", 1: "kev-4b"}
    # the second GPU on kev-9b would mostly take from the first: the hotkey's part with it, less without
    assert two[1].pays["kev-9b"] == pytest.approx(0.5 * (34 / 114 - 17 / 97))
    assert two[1].network == pytest.approx({"kev-4b": 100.0, "kev-9b": 97.0})
    # the hotkey already serves most of kev-9b on another machine
    assert models(choose([a100()], market(shares, capacity, {"kev-9b": 50.0}))) == {0: "kev-4b"}


def test_an_update_keeps_a_gpus_model_unless_another_pays_a_fifth_more() -> None:
    measured = {0: ("kev-4b", 20.0)}
    slightly = market({"kev-4b": 0.5, "kev-9b": 0.5}, {"kev-4b": 180.0, "kev-9b": 137.0})
    kept = choose([a100()], slightly, previous={0: "kev-4b"}, measured=measured)[0]
    assert kept.model == "kev-4b" and kept.reason == plan.KEPT
    assert 1.1 < kept.pays["kev-9b"] / kept.pays["kev-4b"] < plan.SWITCH_RATIO
    assert kept.source == "by its kev-4b measurement"
    assert models(choose([a100()], slightly, measured=measured)) == {0: "kev-9b"}  # a new GPU takes the best
    clearly = market({"kev-4b": 0.5, "kev-9b": 0.5}, {"kev-4b": 180.0, "kev-9b": 110.0})
    switched = choose([a100()], clearly, previous={0: "kev-4b"}, measured=measured)[0]
    assert switched.model == "kev-9b" and switched.reason == plan.PAYS


def test_a_model_named_by_hand_wins_and_stays_until_auto() -> None:
    m = market({"kev-4b": 0.5, "kev-9b": 0.5}, {"kev-4b": 180.0, "kev-9b": 110.0})
    assert models(choose([a100()], m, requested=["kev-4b"])) == {0: "kev-4b"}
    named = choose([a100()], m, previous={0: "kev-4b"}, manual={0})[0]
    assert named.model == "kev-4b" and named.reason == plan.MANUAL
    assert models(choose([a100()], m, previous={0: "kev-4b"}, manual={0}, requested=[plan.AUTO])) == {0: "kev-9b"}
    # GPUs named by hand count first: the others choose around them
    around = choose([a100(0), a100(1)], market({"kev-4b": 0.5, "kev-9b": 0.5}, {"kev-4b": 100.0, "kev-9b": 80.0}),
                    requested=[plan.AUTO, "kev-9b"])
    assert models(around) == {0: "kev-4b", 1: "kev-9b"}
    with pytest.raises(plan.PlanError, match="or auto"):
        choose([a100()], m, requested=["kev-27b"])


def test_candidates_are_never_chosen_by_the_setup() -> None:
    offered = ["kev-4b", "kev-9b", "clef-flash"]
    m = market({"kev-4b": 0.5, "kev-9b": 0.5}, {"kev-4b": 1000.0, "kev-9b": 1000.0})
    assert models(choose([a100()], m, offered=offered, scored=set(KEV), previous={0: "clef-flash"})) == {
        0: "kev-4b"}
    named = choose([a100()], m, offered=offered, scored=set(KEV), requested=["clef-flash"])[0]
    assert named.model == "clef-flash" and "clef-flash" not in named.pays
    line, _ = plan.choice_lines([a100()], {0: named}, offered, NEED, set(KEV), m.shares)
    assert line.startswith("GPU 0: clef-flash (candidate), as --model asks. Also fit: kev-4b 0.98%, kev-9b 0.84%")
    assert line.endswith("kev-4b would pay about 0.98% on GPU 0: --model kev-4b to switch")


def test_a_gpus_speed_comes_from_its_measurement_its_kind_or_the_network() -> None:
    rtx_4090 = Gpu(2, "NVIDIA GeForce RTX 4090", 24 * GiB, 24 * GiB)
    speeds = plan.gpu_speeds([a100(0), a100(1), rtx_4090], {0: ("kev-9b", 17.0), 5: ("kev-27b", 9.0)}, SPEED, 14.0)
    assert speeds == {0: (pytest.approx(20.0), "by its kev-9b measurement"),
                      1: (pytest.approx(20.0), "as measured on its kind of GPU"),
                      2: (14.0, "as the network's median endpoint")}


def test_the_market_counts_the_scored_capacity_others_serve() -> None:
    owner, mine = "5Other", "5Mine"

    def endpoint(eid: str, model: str, state: str, credited: float) -> EndpointStatus:
        return EndpointStatus(endpoint_id=eid, model_id=model, state=state, credited_rps=credited)

    status = NetworkStatus(epoch=7, model_shares={"kev-4b": 0.5, "kev-9b": 0.5}, hotkeys=[
        HotkeyStatus(hotkey=owner, endpoints=[endpoint("gpu0", "kev-4b", "scored", 30.0),
                                              endpoint("gpu1", "kev-9b", "scored", 17.0),
                                              endpoint("gpu2", "kev-4b", "qualifying", 0.0),
                                              endpoint("gpu3", "kev-4b", "failed", 25.0),
                                              endpoint("gpu4", "clef-flash", "candidate", 9.0)]),
        HotkeyStatus(hotkey=mine, endpoints=[endpoint("gpu0", "kev-4b", "scored", 20.0),
                                             endpoint("w-gpu0", "kev-9b", "scored", 8.5)])])
    m = plan.network_market(status, hotkey=mine, own={"gpu0"}, active=KEV, relative=SPEED)
    assert m.capacity == pytest.approx({"kev-4b": 30.0, "kev-9b": 25.5})  # this machine's gpu0 left out
    assert m.held == pytest.approx({"kev-4b": 0.0, "kev-9b": 8.5})  # the hotkey's worker
    assert m.median_speed == pytest.approx(20.0) and m.epoch == 7  # of 30, 20, 20 and 10 at relative speed 1
    with pytest.raises(ValueError, match="not for the catalog's active models"):
        plan.network_market(status, hotkey=mine, own=(), active=["kev-4b", "kev-9b", "clef-flash"], relative=SPEED)
    with pytest.raises(ValueError, match="no endpoint is scored yet"):
        plan.network_market(NetworkStatus(model_shares={"kev-4b": 0.5, "kev-9b": 0.5}), hotkey=mine, own=(),
                            active=KEV, relative=SPEED)


def test_the_plan_gives_the_numbers_and_the_advice_to_switch() -> None:
    m = market({"kev-4b": 0.5, "kev-9b": 0.5}, {"kev-4b": 180.0, "kev-9b": 137.0})
    chosen = choose([a100()], m, previous={0: "kev-4b"}, measured={0: ("kev-4b", 20.0)})
    line, how = plan.choice_lines([a100()], chosen, KEV, NEED, KEV, m.shares)
    assert line == ("GPU 0: kev-4b, as set up before: about 5.00% of the emission (its share 50.0%, 180.0 requests/s "
                    "proven by other endpoints, about 20.0 on this GPU by its kev-4b measurement). Also fit: kev-9b "
                    "5.52%. kev-9b would pay about 1.10x as much on GPU 0: --model kev-9b to switch")
    assert how == "another choice: --model <id> for every GPU, or one per GPU in order, as in --model kev-9b"
    two = choose([a100(0), a100(1)], m, previous={0: "kev-4b", 1: "kev-4b"}, measured={0: ("kev-4b", 20.0)},
                 requested=["kev-4b", plan.AUTO])
    first, second, _ = plan.choice_lines([a100(0), a100(1)], two, KEV, NEED, KEV, m.shares)
    assert first.endswith("--model kev-9b,auto to switch")
    assert second.startswith("GPU 1: kev-9b, pays best: ")
    assert plan.market_line(plan.Market(m.shares, m.capacity, {}, 14.6, 9)) == (
        "the network in the auditor's epoch 9: kev-4b 50.0% of the emission, 180.0 requests/s proven; kev-9b 50.0% of "
        "the emission, 137.0 requests/s proven; the median endpoint 14.6 requests/s at relative speed 1")


class Chain:
    """The subnet owner's hotkey and its axon, as the chain says."""

    def __init__(self, owner: str, served: bool = True) -> None:
        self.owner, self.served = owner, served

    def get_subnet_owner_hotkey(self, netuid: int) -> str:
        return self.owner

    def get_neuron_for_pubkey_and_subnet(self, hotkey: str, netuid: int):
        axon = SimpleNamespace(ip="203.0.113.10", port=8091) if self.served else SimpleNamespace(ip="0.0.0.0", port=0)
        return SimpleNamespace(is_null=False, uid=0, axon_info=axon)


def test_the_setup_uses_the_network_status_only_as_the_subnet_owner_signed_it_recently(monkeypatch) -> None:
    owner, other = Keypair.create_from_uri("//owner"), Keypair.create_from_uri("//other")
    cat, now = catalog.load(), 1_800_000_000.0
    shares = {m.model_id: 0.5 for m in cat.models if m.status == "active"}
    status = NetworkStatus(epoch=7, model_shares=shares, hotkeys=[HotkeyStatus(hotkey="5Miner", endpoints=[
        EndpointStatus(endpoint_id="gpu0", model_id="kev-4b", state="scored", credited_rps=14.0)])])

    def signed(keypair: Keypair = owner, *, at: float = now, netuid: int = 96, body: NetworkStatus = status) -> dict:
        return sign_document(body.model_dump(mode="json"), keypair, netuid=netuid, domain=NETWORK_DOMAIN,
                             now_ms=int(at * 1000))

    served: dict = {}

    def get_json(url: str, **kwargs):
        assert url == "http://203.0.113.10:8091/v1/network"
        if isinstance(served.get("doc"), Exception):
            raise served["doc"]
        return json.loads(json.dumps(served["doc"]))

    monkeypatch.setattr(setup, "get_json", get_json)

    def read(chain: Chain | None = None):
        return setup.read_market(chain or Chain(owner.ss58_address), 96, "5Mine", {"gpu0"}, cat, clock=lambda: now)

    served["doc"] = signed()
    m, why = read()
    assert why == "" and m.capacity["kev-4b"] == 14.0 and m.median_speed == 14.0
    served["doc"] = signed(at=now - 3600)  # signed once per epoch: an hour old is recent enough
    assert read()[0] is not None
    for doc, reason in [(signed(at=now - 3 * 3600), "not signed within 4320 s"),
                        (signed(other), "not signed by the subnet owner hotkey"),
                        (signed(netuid=405), "not signed by the subnet owner hotkey"),
                        ({**signed(), "epoch": 8}, "signature does not verify"),
                        (status.model_dump(mode="json"), "not signed (the auditor runs an older release)"),
                        (signed(body=NetworkStatus(model_shares=shares)), "no endpoint is scored yet"),
                        (ConnectionError("refused"), "not available (ConnectionError: refused)")]:
        served["doc"] = doc
        m, why = read()
        assert m is None and reason in why
    assert read(Chain(owner.ss58_address, served=False)) == (None, "the subnet owner's validator serves no axon")


def test_the_capacity_measured_before_is_read_per_gpu(tmp_path) -> None:
    MeasurementStore(tmp_path / STORE_FILE).save({
        "gpu0": Measurement("gpu0", "f", 20.0, 16, (20.0,) * 5, 1.0, "kev-9b"),
        "gpu1": Measurement("gpu1", "f", 10.0, 16, (10.0,) * 5, 1.0),  # a store of an older release
        "mac": Measurement("mac", "f", 5.0, 4, (5.0,) * 5, 1.0, "kev-4b")})  # an endpoint the setup did not install
    engines = {"gpu0": {"gpu": 0, "model": "kev-4b", "port": 8008}, "gpu1": {"gpu": 1, "model": "kev-4b", "port": 8009}}
    assert setup.measured_capacity(tmp_path, engines, 0.95) == {0: ("kev-9b", pytest.approx(19.0)),
                                                                1: ("kev-4b", pytest.approx(9.5))}
    (tmp_path / STORE_FILE).write_text("{")
    assert setup.measured_capacity(tmp_path, engines, 0.95) == {}
    assert setup.measured_capacity(tmp_path / "none", engines, 0.95) == {}


class Terminal:
    """Answers the setup's questions as given."""

    interactive = True

    def __init__(self, answers: list[str]) -> None:
        self.answers, self.asked = answers, []

    def choose(self, question: str, options: list[str], default: str) -> str:
        self.asked.append((question, options, default))
        return self.answers.pop(0) or default


def test_on_a_terminal_each_kind_of_gpu_is_asked_and_an_answer_is_a_choice_by_hand() -> None:
    gpus = [a100(0), a100(1)]
    rule = {"market": market({"kev-4b": 0.5, "kev-9b": 0.5}, {"kev-4b": 100.0, "kev-9b": 80.0}), "relative": SPEED}
    ui = Terminal([""])
    chosen = setup.choose_models(ui, gpus, KEV, NEED, [], {}, set(KEV), **rule)
    assert models(chosen) == {0: "kev-9b", 1: "kev-4b"}
    (question, options, default), = ui.asked
    assert default == plan.AUTO and options == ["kev-4b", "kev-9b", plan.AUTO]
    assert question.endswith("(auto: kev-9b on GPU 0, kev-4b on GPU 1)")
    chosen = setup.choose_models(Terminal(["kev-4b"]), gpus, KEV, NEED, [], {}, set(KEV), **rule)
    assert models(chosen) == {0: "kev-4b", 1: "kev-4b"} and {c.reason for c in chosen.values()} == {plan.ANSWERED}
