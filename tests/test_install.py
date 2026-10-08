"""The guided setup's decisions: which models fit a GPU, where references run, ports, the configuration written,
and the service definitions."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import pytest

from tests.lineups import two_active_and_a_candidate
from verathos import catalog
from verathos.engine import code_version as engine_code_version
from verathos.install import common, engines, plan, services
from verathos.install.host import GiB, Gpu, parse_gpus
from verathos.install.miner import miner_config
from verathos.install.validator import validator_config
from verathos.miner.config import MinerConfig
from verathos.validator.config import ValidatorConfig

KEV = {"kev-4b": 4_659_865_088, "kev-9b": 9_653_104_368}  # parameters of the Kev models' base models
MINER_NEED = {m: plan.miner_memory(p) for m, p in KEV.items()}
ROOT = Path(__file__).resolve().parents[1]


def gpu(index: int, gib: float, name: str = "NVIDIA A100 80GB PCIe", free: float | None = None) -> Gpu:
    return Gpu(index, name, int(gib * GiB), int((gib if free is None else free) * GiB))


def test_gpus_are_read_from_nvidia_smi() -> None:
    gpus = parse_gpus("0, NVIDIA A100 80GB PCIe, 81920, 81037\n1, NVIDIA GeForce RTX 5090, 32607, 32100\n")
    assert [g.index for g in gpus] == [0, 1]
    assert gpus[0].memory_total == 81920 * (1 << 20) and gpus[1].name == "NVIDIA GeForce RTX 5090"
    assert parse_gpus("2, NVIDIA H100 80GB HBM3, [N/A], [N/A]\n")[0].memory_free == 0


def test_the_kev_models_fit_the_gpus_they_are_known_to_run_on() -> None:
    rtx_4090, rtx_5090, a100 = gpu(0, 23.99), gpu(0, 31.84), gpu(0, 80)
    assert plan.fitting(["kev-4b", "kev-9b"], MINER_NEED, rtx_4090) == ["kev-4b"]
    assert plan.fitting(["kev-4b", "kev-9b"], MINER_NEED, rtx_5090) == ["kev-4b", "kev-9b"]
    assert plan.fitting(["kev-4b", "kev-9b"], MINER_NEED, a100) == ["kev-4b", "kev-9b"]
    # float32 references, as measured on an A100 host: Kev-4B holds 26.2 GiB on the GPU, Kev-9B 33 GiB of RAM
    assert plan.reference_memory(KEV["kev-4b"], "gpu") > 26.2 * GiB
    assert 33 * GiB < plan.reference_memory(KEV["kev-9b"], "cpu") < 44 * GiB


def models(choices: dict[int, plan.Choice]) -> dict[int, str]:
    return {i: c.model for i, c in choices.items()}


def test_without_the_networks_numbers_each_gpu_gets_the_requested_model_or_keeps_its_model_or_the_first_that_fits(
) -> None:
    offered = ["kev-4b", "kev-9b"]
    gpus = [gpu(0, 80), gpu(1, 24)]
    assert models(plan.choose_models(gpus, offered, MINER_NEED, [], {})) == {0: "kev-4b", 1: "kev-4b"}
    kept = plan.choose_models(gpus, offered, MINER_NEED, [], {0: "kev-9b", 1: "kev-9b"})
    assert models(kept) == {0: "kev-9b", 1: "kev-4b"} and [c.reason for c in kept.values()] == [plan.BEFORE, plan.FIRST]
    assert models(plan.choose_models(gpus[:1], offered, MINER_NEED, ["kev-9b"], {})) == {0: "kev-9b"}
    assert models(plan.choose_models(gpus, offered, MINER_NEED, ["kev-9b", "kev-4b"], {})) == {0: "kev-9b",
                                                                                              1: "kev-4b"}
    assert models(plan.choose_models(gpus, offered, MINER_NEED, ["kev-9b", plan.AUTO], {})) == {0: "kev-9b",
                                                                                               1: "kev-4b"}
    with pytest.raises(plan.PlanError, match="needs about"):
        plan.choose_models(gpus, offered, MINER_NEED, ["kev-9b"], {})
    with pytest.raises(plan.PlanError, match="not a catalog model this setup can install"):
        plan.choose_models(gpus, offered, MINER_NEED, ["kev-27b"], {})
    with pytest.raises(plan.PlanError, match="one per GPU"):
        plan.choose_models(gpus, offered, MINER_NEED, ["kev-4b", "kev-4b", "kev-4b"], {})
    with pytest.raises(plan.PlanError, match="no active catalog model fits"):
        plan.choose_models([gpu(0, 12)], offered, MINER_NEED, [], {})


CLEF = 9_409_813_744  # parameters of clef-flash
NEED = {**MINER_NEED, "clef-flash": plan.miner_memory(CLEF, "clef")}
OFFERED, SCORED = ["kev-4b", "kev-9b", "clef-flash"], {"kev-4b", "kev-9b"}


def test_the_setups_offer_active_models_then_candidates_and_never_retired_ones() -> None:
    cat = two_active_and_a_candidate()
    statuses = {"kev-4b": "candidate", "kev-9b": "retired", "clef-flash": "active"}
    changed = cat.model_copy(update={"models": [m.model_copy(update={"status": statuses[m.model_id]})
                                                for m in cat.models]})
    assert [m.model_id for m in plan.servable_models(changed)] == ["clef-flash", "kev-4b"]
    assert [m.model_id for m in plan.servable_models(cat)] == ["kev-4b", "kev-9b", "clef-flash"]


def test_a_candidate_is_served_only_when_requested_or_chosen_by_hand_before() -> None:
    a100, rtx_4090 = gpu(0, 80), gpu(1, 23.99, "NVIDIA GeForce RTX 4090")
    assert models(plan.choose_models([a100, rtx_4090], OFFERED, NEED, [], {}, SCORED)) == {0: "kev-4b", 1: "kev-4b"}
    assert models(plan.choose_models([a100, rtx_4090], OFFERED, NEED, ["kev-4b", "clef-flash"], {}, SCORED)) == {
        0: "kev-4b", 1: "clef-flash"}
    kept = plan.choose_models([a100], OFFERED, NEED, [], {0: "clef-flash"}, SCORED, manual={0})
    assert models(kept) == {0: "clef-flash"} and kept[0].reason == plan.MANUAL
    assert models(plan.choose_models([a100], OFFERED, NEED, [plan.AUTO], {0: "clef-flash"}, SCORED,
                                     manual={0})) == {0: "kev-4b"}
    # the setup never keeps a candidate it did not get by hand, and one no longer offered (retired) gives way
    assert models(plan.choose_models([a100], OFFERED, NEED, [], {0: "clef-flash"}, SCORED)) == {0: "kev-4b"}
    assert models(plan.choose_models([a100], ["kev-4b", "kev-9b"], NEED, [], {0: "clef-flash"}, SCORED,
                                     manual={0})) == {0: "kev-4b"}
    with pytest.raises(plan.PlanError, match=r"with --model it can serve clef-flash \(candidate"):
        plan.choose_models([gpu(0, 23.99)], ["kev-9b", "clef-flash"], NEED, [], {}, {"kev-9b"})


def test_the_plan_says_why_each_gpu_serves_its_model_what_else_fits_and_how_to_choose() -> None:
    a100, rtx_4090 = gpu(0, 80), gpu(1, 23.99, "NVIDIA GeForce RTX 4090")
    offer = plan.offer_line(OFFERED, NEED, SCORED)
    assert offer.startswith("models: kev-4b (about 15 GiB), kev-9b (about 26 GiB), clef-flash (about 21 GiB; candidate")
    first, how = plan.choice_lines([a100], {0: plan.Choice("kev-4b", plan.FIRST)}, OFFERED, NEED, SCORED)
    assert first == "GPU 0: kev-4b, the first active model that fits. Also fit: kev-9b, clef-flash (candidate)"
    assert how.endswith("one per GPU in order, as in --model kev-9b")
    chosen = {0: plan.Choice("kev-4b", plan.REQUESTED), 1: plan.Choice("clef-flash", plan.REQUESTED)}
    first, second, how = plan.choice_lines([a100, rtx_4090], chosen, OFFERED, NEED, SCORED)
    assert first == "GPU 0: kev-4b, as --model asks. Also fit: kev-9b, clef-flash (candidate)"
    assert second == "GPU 1: clef-flash (candidate), as --model asks. Also fit: kev-4b"
    assert how.endswith("--model kev-9b,clef-flash")
    first, = plan.choice_lines([gpu(0, 20)], {0: plan.Choice("kev-4b", plan.FIRST)}, OFFERED, NEED, SCORED)
    assert first == "GPU 0: kev-4b, the first active model that fits. No other catalog model fits"
    first, _ = plan.choice_lines([a100], {0: plan.Choice("clef-flash", plan.MANUAL)}, OFFERED, NEED, SCORED)
    assert first.startswith("GPU 0: clef-flash (candidate), as chosen by hand before (--model auto lets the setup "
                            "choose)")


def test_candidate_references_take_the_memory_the_active_ones_leave_or_are_left_out() -> None:
    parameters = {**KEV, "clef-flash": CLEF}
    assert plan.place_references(parameters, {0: 80 * GiB}, 64 * GiB, "auto", {}, optional={"clef-flash"}) == {
        "kev-4b": "cuda:0", "kev-9b": "cuda:0", "clef-flash": "cpu"}
    # room for the active models only: the candidate is left out, not an error
    assert plan.place_references(parameters, {}, 70 * GiB, "cpu", {}, optional={"clef-flash"}) == {
        "kev-4b": "cpu", "kev-9b": "cpu"}
    # the candidate has fewer parameters than kev-9b but comes after it, so it cannot take kev-9b's place
    assert plan.place_references(parameters, {0: 80 * GiB}, 0, "auto", {}, optional={"clef-flash"}) == {
        "kev-4b": "cuda:0", "kev-9b": "cuda:0"}
    with pytest.raises(plan.PlanError, match="kev-9b reference needs"):
        plan.place_references(parameters, {}, 50 * GiB, "cpu", {}, optional={"clef-flash"})


def test_references_go_on_a_gpu_smallest_first_and_the_rest_in_ram() -> None:
    # An A100 with a miner's engine on it: room for the Kev-4B reference, not for both.
    assert plan.place_references(KEV, {0: 60 * GiB}, 200 * GiB, "auto", {}) == {"kev-4b": "cuda:0", "kev-9b": "cpu"}
    assert plan.place_references(KEV, {0: 79 * GiB}, 8 * GiB, "auto", {}) == {"kev-4b": "cuda:0", "kev-9b": "cuda:0"}
    # Several GPUs: each reference on the one with the most free memory left.
    assert plan.place_references(KEV, {0: 50 * GiB, 1: 50 * GiB}, 0, "gpu", {}) == {"kev-4b": "cuda:0",
                                                                                   "kev-9b": "cuda:1"}
    assert plan.place_references(KEV, {0: 80 * GiB}, 100 * GiB, "cpu", {}) == {"kev-4b": "cpu", "kev-9b": "cpu"}
    with pytest.raises(plan.PlanError, match="kev-9b reference needs about .* of RAM"):
        plan.place_references(KEV, {}, 50 * GiB, "cpu", {})
    with pytest.raises(plan.PlanError, match="on a GPU"):
        plan.place_references(KEV, {0: 30 * GiB}, 500 * GiB, "gpu", {})


def test_a_reference_placed_before_stays_where_it_is() -> None:
    """Its memory is taken by itself, so it is not counted again; a placement the device option excludes moves."""
    keep = {"kev-4b": "cuda:0", "kev-9b": "cpu"}
    assert plan.place_references(KEV, {0: 1 * GiB}, 1 * GiB, "auto", keep) == keep
    assert plan.place_references(KEV, {0: 1 * GiB}, 100 * GiB, "cpu", keep) == {"kev-4b": "cpu", "kev-9b": "cpu"}
    assert plan.place_references(KEV, {}, 100 * GiB, "auto", {"kev-4b": "cuda:3"})["kev-4b"] == "cpu"


def test_ports_are_kept_and_new_ones_skip_busy_and_taken_ports() -> None:
    busy = {8008, 8010}
    assert plan.pick_ports(["gpu0", "gpu1", "gpu2"], {"gpu1": 8009}, 8008, lambda p: p not in busy) == {
        "gpu0": 8011, "gpu1": 8009, "gpu2": 8012}


def test_the_miner_configuration_keeps_what_the_setup_does_not_manage(tmp_path) -> None:
    existing = {"network": "finney", "netuid": 96, "wallet_name": "w", "hotkey_name": "h", "public_host": "203.0.113.5",
                "port": 8091, "gateway_hotkeys": ["5F..."], "endpoints": [
                    {"endpoint_id": "gpu0", "model_id": "kev-4b", "engine_url": "http://127.0.0.1:8008",
                     "max_concurrency": 64},
                    {"endpoint_id": "gpu1", "model_id": "kev-4b", "engine_url": "http://127.0.0.1:8009"},
                    {"endpoint_id": "mac", "model_id": "kev-4b", "engine_url": "http://10.0.0.2:8008"}]}
    config = miner_config(existing, network="finney", netuid=96, wallet_name="w", hotkey_name="h",
                          public_ip="198.51.100.7", port=20003, endpoints={"gpu0": ("kev-9b", 8008)},
                          managed_before={"gpu0", "gpu1"}, home=tmp_path)
    assert [e["endpoint_id"] for e in config["endpoints"]] == ["gpu0", "mac"]
    assert config["endpoints"][0] == {"endpoint_id": "gpu0", "model_id": "kev-9b",
                                      "engine_url": "http://127.0.0.1:8008", "max_concurrency": 64}
    assert config["gateway_hotkeys"] == ["5F..."] and config["public_host"] == "198.51.100.7"
    assert config["state_dir"] == str(tmp_path / "miner")
    MinerConfig.model_validate(config)


def test_the_role_comes_from_the_options_or_the_configuration(tmp_path) -> None:
    """Fully answered on the command line: --worker on a worker, --peers on the primary; a run with --yes alone
    keeps the role configured before."""
    from verathos.install.miner import choose_role

    ui = common.Prompter(assume_yes=True)

    def options(**given) -> argparse.Namespace:
        return argparse.Namespace(**{"worker": False, "peers": None, **given})

    assert choose_role(ui, options(), {}) == "primary"
    assert choose_role(ui, options(), {"role": "worker"}) == "worker"
    assert choose_role(ui, options(worker=True), {}) == "worker"
    assert choose_role(ui, options(peers="9.9.9.9:20017"), {"role": "worker"}) == "primary"
    with pytest.raises(SystemExit, match="not both"):
        choose_role(ui, options(worker=True, peers="9.9.9.9:20017"), {})


def test_a_worker_and_its_primary_are_configured_for_one_hotkey(tmp_path) -> None:
    """A worker's endpoint ids are its own (its name, or its IP address) and it has no peers; the primary lists the
    workers' public addresses, kept on the next run unless --peers replaces them."""
    from verathos.install.miner import endpoint_prefix, parse_peers

    prefix = endpoint_prefix("worker", None, "9.9.9.9", [])
    assert prefix == "9-9-9-9-"
    assert endpoint_prefix("worker", None, "8.8.8.8", ["9-9-9-9-gpu0", "9-9-9-9-gpu1"]) == prefix
    assert endpoint_prefix("worker", "a100-sxm", "8.8.8.8", [prefix + "gpu0"]) == "a100-sxm-"
    assert endpoint_prefix("worker", None, "2606:4700::1111", []).startswith("v6-")
    assert endpoint_prefix("primary", None, "8.8.8.8", [prefix + "gpu0"]) == ""
    with pytest.raises(SystemExit, match="--name"):
        endpoint_prefix("worker", "A100_SXM", "8.8.8.8", [])

    def configure(existing: dict, role: str, ip: str, ids: list[str], peers: list[str] | None) -> dict:
        config = miner_config(existing, network="test", netuid=405, wallet_name="decide", hotkey_name="sybil",
                              public_ip=ip, port=20017, endpoints={e: ("kev-4b", 8008 + i) for i, e in enumerate(ids)},
                              managed_before={e["endpoint_id"] for e in existing.get("endpoints", [])},
                              home=tmp_path, role=role,
                              peers=peers)
        MinerConfig.model_validate(config)
        return config

    worker = configure({}, "worker", "9.9.9.9", [prefix + "gpu0", prefix + "gpu1"], None)
    assert worker["role"] == "worker" and "peers" not in worker
    assert [e["endpoint_id"] for e in worker["endpoints"]] == ["9-9-9-9-gpu0", "9-9-9-9-gpu1"]

    peers = parse_peers("9.9.9.9:20017, [2606:4700::1111]:8091")
    assert peers == ["http://9.9.9.9:20017", "http://[2606:4700::1111]:8091"]
    for bad in ("10.0.0.1:8091", "9.9.9.9", "localhost:8091", "[::ffff:8.8.8.8]:80"):
        with pytest.raises(SystemExit, match="--peers"):
            parse_peers(bad)
    primary = configure({}, "primary", "1.0.0.1", ["gpu0"], peers)
    assert primary["role"] == "primary" and primary["peers"] == peers
    assert configure(primary, "primary", "1.0.0.1", ["gpu0"], None)["peers"] == peers  # --yes keeps them
    assert "peers" not in configure(primary, "primary", "1.0.0.1", ["gpu0"], [])  # --peers ''
    assert "peers" not in configure(primary, "worker", "1.0.0.1", ["x-gpu0"], None)


def test_the_validator_configuration_keeps_remote_references_and_settings(tmp_path) -> None:
    existing = {"references": {"kev-4b": "http://127.0.0.1:8101", "clef-flash": {"url": "https://ref.example/x",
                                                                                 "api_key": "k"}},
                "state_dir": "/data/validator", "public_host": "203.0.113.9"}
    config = validator_config(existing, network="test", netuid=405, wallet_name="v", hotkey_name="default",
                              ports={"kev-4b": 8101, "kev-9b": 8102}, dry_run=True, managed_before={"kev-4b"},
                              home=tmp_path)
    assert config["references"] == {"clef-flash": {"url": "https://ref.example/x", "api_key": "k"},
                                    "kev-4b": "http://127.0.0.1:8101", "kev-9b": "http://127.0.0.1:8102"}
    assert config["state_dir"] == "/data/validator" and config["dry_run"] is True
    assert ValidatorConfig.model_validate(config).public_host == "203.0.113.9"


def service(**changes) -> services.Service:
    fields = dict(name="verathos-engine-gpu0", description="Verathos engine gpu0", directory=Path("/opt/kev"),
                  user="miner", command=["/opt/kev/.venv/bin/python", "-m", "kev.serve", "--run", "a/b@c", "--port",
                                         "8008"],
                  environment={"CUDA_VISIBLE_DEVICES": "", "HF_HOME": "/home/miner/.verathos/hf"},
                  after=("verathos-engine-gpu1",))
    return services.Service(**{**fields, **changes})


def test_systemd_units_restart_the_service_and_quote_what_needs_it() -> None:
    unit = services.Systemd.render(service())
    assert "ExecStart=/opt/kev/.venv/bin/python -m kev.serve --run a/b@c --port 8008\n" in unit
    assert "Environment=CUDA_VISIBLE_DEVICES=\n" in unit and "Restart=always" in unit and "User=miner" in unit
    assert "After=network-online.target verathos-engine-gpu1.service" in unit
    assert "WantedBy=multi-user.target" in unit
    assert services.systemd_quote("a b") == '"a b"' and services.systemd_quote("50%") == "50%%"
    # $ expands variables in ExecStart= but is literal in Environment=
    unit = services.Systemd.render(service(command=["/bin/x", "$HOME"], environment={"KEY": "a$b c"}))
    assert "ExecStart=/bin/x $$HOME\n" in unit and 'Environment="KEY=a$b c"\n' in unit


def test_supervisor_programs_restart_the_service_and_quote_what_needs_it() -> None:
    conf = services.Supervisor.render(service())
    assert "[program:verathos-engine-gpu0]" in conf and "autorestart=true" in conf and "user=miner" in conf
    assert 'environment=CUDA_VISIBLE_DEVICES="",HF_HOME=/home/miner/.verathos/hf' in conf
    assert "stdout_logfile=/var/log/verathos/verathos-engine-gpu0.log" in conf
    assert services.supervisor_quote("a,b") == '"a,b"' and services.supervisor_quote("50%") == "50%%"


def test_supervisor_logs_are_followed_from_the_mark(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(services.Supervisor, "log_dir", tmp_path)
    log = tmp_path / "verathos-miner.log"
    log.write_text("old line\n")
    manager = services.Supervisor()
    manager.mark("verathos-miner")
    with log.open("a") as f:
        f.write("measuring\nendpoint gpu0: 19.9 requests/s\npartial")
    assert manager.new_lines("verathos-miner") == ["measuring", "endpoint gpu0: 19.9 requests/s"]
    with log.open("a") as f:
        f.write(" line\n")
    assert manager.new_lines("verathos-miner") == ["partial line"]
    assert manager.new_lines("verathos-miner") == []
    log.write_text("rotated\n")
    assert manager.new_lines("verathos-miner") == ["rotated"]


class FakeManager(services.Logs):
    """A service whose log and process ids are given in advance."""

    def __init__(self, batches: list[list[str]], pids: list[int]) -> None:
        super().__init__()
        self.batches, self.pids = batches, pids

    def new_lines(self, name: str) -> list[str]:
        return self.batches.pop(0) if self.batches else []

    def pid(self, name: str) -> int:
        return self.pids.pop(0) if len(self.pids) > 1 else self.pids[0]

    def log_command(self, name: str) -> str:
        return f"tail -f {name}.log"


def test_following_a_log_leaves_the_lines_after_the_match_for_the_next_wait(monkeypatch) -> None:
    monkeypatch.setattr(common.time, "sleep", lambda s: None)
    manager = FakeManager([["starting", "started", "epoch 7: 3 miners"]], [101])
    assert common.follow(manager, "v", done=lambda line: line == "started", failed=lambda line: False, timeout_s=5)
    assert common.follow(manager, "v", done=lambda line: line.startswith("epoch"), failed=lambda line: False,
                         timeout_s=5)
    assert manager.lines("v") == []
    with pytest.raises(SystemExit, match="failed"):
        common.follow(FakeManager([["no endpoint could be measured"]], [101]), "m", done=lambda line: False,
                      failed=lambda line: "no endpoint" in line, timeout_s=5)


def test_a_service_that_keeps_restarting_or_stays_down_is_failing(monkeypatch) -> None:
    restarting = common.Health(FakeManager([], [101, 0, 102, 0, 103]), "e")
    assert [restarting.failing() for _ in range(5)] == [False, False, False, False, True]
    healthy = common.Health(FakeManager([], [101]), "e")
    assert not any(healthy.failing() for _ in range(10))
    monkeypatch.setattr(common.Health, "DOWN_S", 0.0)
    down = common.Health(FakeManager([], [0]), "e")
    assert [down.failing() for _ in range(2)] == [False, True]


def test_engine_commands_pin_the_revisions_and_the_precision(tmp_path) -> None:
    model = catalog.load().get("kev-4b")
    engine = engines.engine_for(tmp_path, model)
    assert engine.directory == tmp_path / "engines" / f"kev-{model.engine.revision[:12]}"
    command = engine.command("miner", model, 8008)
    assert command[command.index("--run") + 1] == f"{model.hf_repo}@{model.hf_revision}"
    assert command[command.index("--host") + 1] == "127.0.0.1"
    miner_env = engine.environment_variables("miner", "cuda:2", tmp_path / "hf")
    assert miner_env["CUDA_VISIBLE_DEVICES"] == "2" and miner_env["HF_HUB_OFFLINE"] == "1"
    assert "KEV_DTYPE" not in miner_env
    reference_env = engine.environment_variables("reference", "cpu", tmp_path / "hf")
    assert reference_env["CUDA_VISIBLE_DEVICES"] == "" and reference_env["KEV_DTYPE"] == "fp32"
    assert engine.python("reference") != engine.python("miner")
    assert engines.FLA_VERSION.search('x = 1\nFLA_VERSION = "0.5.2"   # pinned\n')[1] == "0.5.2"


def test_clef_models_install_verathos_engine_and_others_are_refused(tmp_path) -> None:
    clef = catalog.load().get("clef-flash")
    assert plan.installable(clef)
    engine = engines.engine_for(tmp_path, clef)
    assert isinstance(engine, engines.ClefEngine) and engine.directory == tmp_path / "engines" / "verathos-clef"
    miner = engine.command("miner", clef, 8010)
    assert miner[1:4] == ["-m", "verathos.engine", "clef"]
    assert miner[miner.index("--revision") + 1] == clef.hf_revision
    assert miner[-4:] == ["--device", "cuda", "--dtype", "bfloat16"]
    assert engine.command("reference", clef, 8103)[-4:] == ["--device", "auto", "--dtype", "float32"]
    assert engines.FLA_VERSION.search(engines.QWEN35.read_text())  # the miner environment's kernel pin
    env = engine.environment_variables("miner", "cuda:1", tmp_path / "hf")
    assert env["CUDA_VISIBLE_DEVICES"] == "1" and env["HF_HUB_OFFLINE"] == "1"
    # new engine code changes the service, so the engine restarts on an update
    assert re.fullmatch(r"[0-9a-f]{12}", env["VERATHOS_ENGINE_CODE"])
    assert env["VERATHOS_ENGINE_CODE"] == engine_code_version()
    other = clef.model_copy(update={"engine": clef.engine.model_copy(update={"name": "other"})})
    assert not plan.installable(other)
    with pytest.raises(plan.PlanError, match="does not install"):
        engines.engine_for(tmp_path, other)


def test_a_9b_clef_model_fits_a_24_gb_gpu_but_not_a_20_gb_one() -> None:
    need = plan.miner_memory(9_400_000_000, "clef")
    assert 20 * GiB < need < 24 * GiB
    assert plan.miner_memory(9_400_000_000, "kev") > need


def test_the_record_round_trips_and_defaults_to_nothing_installed(tmp_path) -> None:
    assert common.Record.load(tmp_path, "miner") == common.Record(role="miner")
    record = common.Record(role="validator", manager="supervisor", services=["verathos-validator"],
                           config="/x/validator.json", engines={"kev-4b": {"device": "cpu", "port": 8101}},
                           paths=["/x/engines/kev-5e42a7a03f28"])
    record.save(tmp_path)
    assert common.Record.load(tmp_path, "validator") == record


def test_network_and_netuid_defaults() -> None:
    args = argparse.Namespace(network=None, netuid=None)
    assert common.network_and_netuid(args, {}) == ("finney", 96)
    assert common.network_and_netuid(argparse.Namespace(network="test", netuid=None), {}) == ("test", 405)
    assert common.network_and_netuid(args, {"network": "ws://10.0.0.1:9944", "netuid": 405}) == (
        "ws://10.0.0.1:9944", 405)
    assert common.network_and_netuid(argparse.Namespace(network="finney", netuid=None),
                                     {"network": "test", "netuid": 405}) == ("finney", 96)


def test_without_a_terminal_every_question_takes_its_default() -> None:
    ui = common.Prompter(assume_yes=True)
    assert ui.ask("Port", "8091") == "8091"
    assert ui.choose("Model", ["kev-4b", "kev-9b"], "kev-9b") == "kev-9b"
    assert ui.confirm("Go ahead?")
    with pytest.raises(SystemExit, match="in use"):
        ui.ask("Port", "8091", lambda v: "port 8091 is in use")


def test_install_script_parses_and_documents_its_options() -> None:
    script = ROOT / "scripts" / "install.sh"
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0
    usage = subprocess.run(["bash", str(script), "--help"], capture_output=True, text=True)
    assert usage.returncode == 0 and "--source" in usage.stdout and "uninstall" in usage.stdout
    assert "--worker" in usage.stdout and "--peers" in usage.stdout
    assert "--light" in usage.stdout and "--full" in usage.stdout and "--references" in usage.stdout


def test_a_validator_is_light_unless_full_is_asked_for_or_set_up_before_and_the_owners_is_full(tmp_path) -> None:
    """--yes keeps the mode configured; a configuration from before modes existed is a full validator's."""
    from verathos.install.validator import choose_mode, mode_lines, parser

    ui = common.Prompter(assume_yes=True)
    assert choose_mode(ui, None, {}, owner=False) == "light"  # a new validator
    assert choose_mode(ui, None, {"references": {"kev-4b": "http://127.0.0.1:8101"}}, owner=False) == "full"
    assert choose_mode(ui, None, {"mode": "light"}, owner=False) == "light"
    assert choose_mode(ui, None, {"mode": "full"}, owner=False) == "full"
    assert choose_mode(ui, "full", {"mode": "light"}, owner=False) == "full"
    assert choose_mode(ui, "light", {"mode": "full"}, owner=False) == "light"
    assert choose_mode(ui, None, {"mode": "light"}, owner=True) == "full"  # the auditor
    with pytest.raises(SystemExit, match="subnet owner's hotkey"):
        choose_mode(ui, "light", {}, owner=True)
    assert (parser().parse_args(["--light"]).mode, parser().parse_args(["--full"]).mode) == ("light", "full")
    assert parser().parse_args([]).mode is None
    with pytest.raises(SystemExit):
        parser().parse_args(["--light", "--full"])

    cat = catalog.load()
    assert "mode light" in mode_lines("light", cat, False)[0] and "no GPU" in mode_lines("light", cat, False)[1]
    assert "10,000 stake" in mode_lines("full", cat, False)[1] and "auditor" in mode_lines("full", cat, True)[0]

    def configure(existing: dict, mode: str, ports: dict[str, int], managed_before: set[str]) -> dict:
        config = validator_config(existing, network="finney", netuid=96, wallet_name="v", hotkey_name="default",
                                  ports=ports, dry_run=False, managed_before=managed_before, home=tmp_path, mode=mode)
        assert ValidatorConfig.model_validate(config).mode == mode
        return config

    remote = {"url": "https://ref.example/x", "api_key": "k"}
    full = configure({"references": {"clef-flash": remote}}, "full", {"kev-4b": 8101}, set())
    light = configure(full, choose_mode(ui, "light", full, owner=False), {}, {"kev-4b"})
    assert light["mode"] == "light" and light["references"] == {"clef-flash": remote}  # set up by hand: kept
    again = configure(light, choose_mode(ui, None, light, owner=False), {}, set())  # bash install.sh validator --yes
    assert again == light
    assert configure(again, "full", {"kev-4b": 8101}, set())["references"]["kev-4b"] == "http://127.0.0.1:8101"


class ServiceManager(FakeManager):
    """A service manager that records what the setup installs and removes."""

    name = "systemd"

    def __init__(self, batches: list[list[str]]) -> None:
        super().__init__(batches, [101])
        self.applied: list[services.Service] = []
        self.removed: list[str] = []

    def mark(self, name: str) -> None:
        pass

    def apply(self, applied: list[services.Service], restart: set[str]) -> set[str]:
        self.applied += applied
        return {s.name for s in applied}

    def remove(self, names: list[str]) -> None:
        self.removed += names

    def status_command(self, names: list[str]) -> str:
        return "systemctl status " + " ".join(names)

    def restart_command(self, names: list[str]) -> str:
        return "systemctl restart " + " ".join(names)


def test_a_light_setup_runs_the_validator_alone_and_stops_the_references_of_a_full_one(tmp_path, monkeypatch) -> None:
    from verathos.install.validator import VALIDATOR, init_light

    monkeypatch.setattr(common.time, "sleep", lambda s: None)
    home, config_path = tmp_path, tmp_path / "validator.json"
    record = common.Record(role="validator", manager="systemd", config=str(config_path),
                           services=["verathos-reference-kev-4b", VALIDATOR],
                           engines={"kev-4b": {"model": "kev-4b", "device": "cpu", "port": 8101}},
                           paths=[str(tmp_path / "engines" / "kev-x")])
    existing = {"network": "finney", "netuid": 96, "wallet_name": "v", "hotkey_name": "default", "mode": "full",
                "references": {"kev-4b": "http://127.0.0.1:8101"}}
    manager = ServiceManager([["light mode: setting the weights the subnet owner's validator publishes each epoch"]])
    init_light(common.Prompter(assume_yes=True), manager, record, existing, config_path, home, False,
               cat=catalog.load(), network="finney", netuid=96, wallet_name="v", hotkey_name="default")
    config = ValidatorConfig.model_validate(common.read_json(config_path))
    assert config.mode == "light" and config.references == {}
    assert manager.removed == ["verathos-reference-kev-4b"]
    (validator,) = manager.applied
    assert validator.name == VALIDATOR and validator.after == () and validator.command[-2:] == ["--config",
                                                                                                str(config_path)]
    saved = common.Record.load(home, "validator")
    assert (saved.services, saved.engines, saved.paths) == ([VALIDATOR], {}, record.paths)  # uninstall removes them

    with pytest.raises(SystemExit, match="failed"):  # the owner's hotkey: the validator refuses light mode
        init_light(common.Prompter(assume_yes=True), ServiceManager([[
            "SystemExit: this is the subnet owner's hotkey: its validator is the auditor"]]), saved,
            common.read_json(config_path), config_path, home, False, cat=catalog.load(), network="finney", netuid=96,
            wallet_name="v", hotkey_name="default")


def test_references_run_one_at_a_time_when_the_active_ones_do_not_all_fit() -> None:
    """The setup keeps them resident while the active models' references all fit in the free memory, else runs them
    one at a time, where each alone needs to fit; --references forces either, and one at a time stays once set up."""
    from verathos.install.validator import place, reference_line

    def decide(ram_gib: float, gpu_gib: float | None = None, references: str | None = None, existing=None,
               record=None, owner: bool = False, device: str = "auto"):
        args = argparse.Namespace(device=device, references=references)
        gpus = {} if gpu_gib is None else {0: int(gpu_gib * GiB)}
        return place(KEV, gpus, int(ram_gib * GiB), args, existing or {}, record or common.Record(role="validator"),
                     owner, [])

    mode, why, placement = decide(128)
    assert (mode, placement) == ("resident", {"kev-4b": "cpu", "kev-9b": "cpu"}) and "fit" in why
    assert reference_line(mode, placement, KEV) == "all references resident (RAM about 63 GiB)"
    mode, why, placement = decide(50)
    assert (mode, placement) == ("sequential", {"kev-4b": "cpu", "kev-9b": "cpu"})
    assert why == "the active models' references do not all fit in the free memory at once"
    assert reference_line(mode, placement, KEV) == "references one at a time (RAM for the largest: about 42 GiB)"
    with pytest.raises(plan.PlanError, match="kev-9b reference needs about 42 GiB of RAM, even one at a time"):
        decide(30, device="cpu")
    # a GPU that holds each alone, not both: one at a time on the GPU
    mode, _, placement = decide(30, gpu_gib=50)
    assert (mode, placement) == ("sequential", {"kev-4b": "cuda:0", "kev-9b": "cuda:0"})
    assert reference_line(mode, placement, KEV) == "references one at a time (GPU 0 for the largest: about 48 GiB)"
    # forced, kept, and the auditor's
    assert decide(128, references="sequential")[:2] == ("sequential", "as --references sequential asks")
    mode, why, _ = decide(128, existing={"reference_mode": "sequential"})
    assert mode == "sequential" and "--references resident runs them all at once, which fits now" in why
    assert decide(128, existing={"reference_mode": "sequential"}, references="resident")[0] == "resident"
    with pytest.raises(plan.PlanError, match="kev-9b reference needs"):
        decide(50, references="resident")
    assert decide(50, existing={"reference_mode": "resident"})[0] == "sequential"  # no longer fits
    assert decide(128, owner=True)[0] == "resident"
    with pytest.raises(SystemExit, match="auditor .* stay resident"):
        decide(50, owner=True, references="sequential")
    # resident references set up before hold their memory: they stay resident, and one at a time it is theirs again
    before = common.Record(role="validator", services=["verathos-reference-kev-4b", "verathos-reference-kev-9b",
                                                       "verathos-validator"],
                           engines={"kev-4b": {"device": "cpu", "port": 8101},
                                    "kev-9b": {"device": "cpu", "port": 8102}})
    assert decide(4, record=before)[:3:2] == ("resident", {"kev-4b": "cpu", "kev-9b": "cpu"})
    assert decide(4, record=before, references="sequential")[:3:2] == ("sequential", {"kev-4b": "cpu",
                                                                                      "kev-9b": "cpu"})
    # one at a time, a candidate that fits alone is checked too; one that fits nowhere is left out
    parameters = {**KEV, "clef-flash": CLEF}
    assert plan.place_one_at_a_time(parameters, {}, 45 * GiB, "auto", {}, optional={"clef-flash"}) == {
        "kev-4b": "cpu", "kev-9b": "cpu", "clef-flash": "cpu"}
    assert plan.place_one_at_a_time({"kev-4b": KEV["kev-4b"], "clef-flash": CLEF}, {}, 30 * GiB, "auto", {},
                                    optional={"clef-flash"}) == {"kev-4b": "cpu"}


def test_the_configuration_of_references_one_at_a_time_holds_how_to_start_them(tmp_path) -> None:
    launch = {"kev-4b": {"command": ["/e/.venv-reference/bin/python", "-m", "kev.serve", "--port", "8101"],
                         "directory": "/e", "environment": {"KEV_DTYPE": "fp32", "CUDA_VISIBLE_DEVICES": ""}}}
    config = validator_config({}, network="finney", netuid=96, wallet_name="v", hotkey_name="default",
                              ports={"kev-4b": 8101}, dry_run=False, managed_before=set(), home=tmp_path,
                              reference_mode="sequential", launch=launch)
    assert config["reference_mode"] == "sequential"
    assert config["references"] == {"kev-4b": {"url": "http://127.0.0.1:8101", **launch["kev-4b"]}}
    parsed = ValidatorConfig.model_validate(config)
    assert parsed.reference_mode == "sequential" and parsed.references["kev-4b"].command[-1] == "8101"
    resident = validator_config(config, network="finney", netuid=96, wallet_name="v", hotkey_name="default",
                                ports={"kev-4b": 8101}, dry_run=False, managed_before={"kev-4b"}, home=tmp_path,
                                reference_mode="resident")
    assert resident["references"] == {"kev-4b": "http://127.0.0.1:8101"} and resident["reference_mode"] == "resident"


def test_a_full_setup_runs_the_references_one_at_a_time_or_as_services_by_the_memory_it_finds(
        tmp_path, monkeypatch, capsys) -> None:
    """With too little memory for both active references at once, no reference services: the validator starts each
    from its configuration. With enough, a service each, as before; a later run with --references sequential removes
    them."""
    from types import SimpleNamespace

    from verathos.install import host
    from verathos.install import validator as setup

    class Manager(ServiceManager):
        def __init__(self) -> None:
            super().__init__([["throughput tests are run by 5Owner; applying its verdicts"],
                              ["epoch 7: 3 miners, 3 open endpoints"]])
            self.stopped: list[str] = []

        def stop(self, names: list[str]) -> None:
            self.stopped += names

        def log_command(self, name: str) -> str:
            return f"journalctl -u {name} -f"

    chain = SimpleNamespace(
        get_neuron_for_pubkey_and_subnet=lambda hotkey, netuid: SimpleNamespace(is_null=False, uid=5,
                                                                                validator_permit=True),
        get_subnet_owner_hotkey=lambda netuid: "5Owner")
    managers: list[Manager] = []
    monkeypatch.setattr(common.time, "sleep", lambda s: None)
    monkeypatch.setattr(common, "manager_for", lambda args, record: managers[-1])
    monkeypatch.setattr(common, "load_hotkey", lambda w, h: SimpleNamespace(hotkey=SimpleNamespace(
        ss58_address="5Validator")))
    monkeypatch.setattr(common, "connect", lambda network: chain)
    monkeypatch.setattr(common, "wait_engines", lambda manager, ports, timeout_s: None)
    monkeypatch.setattr(setup, "current_catalog", lambda *args: two_active_and_a_candidate())
    monkeypatch.setattr(host, "wallets", lambda: [])
    monkeypatch.setattr(host, "gpus", lambda: [])
    monkeypatch.setattr(host, "port_free", lambda port, address="0.0.0.0": True)
    monkeypatch.setattr(engines, "parameters", lambda m: {**KEV, "clef-flash": CLEF}[m.model_id])
    monkeypatch.setattr(engines.KevEngine, "install", lambda self, purpose: None)
    monkeypatch.setattr(engines, "download", lambda python, m, hf_home: None)
    argv = ["--full", "--yes", "--wallet", "v", "--hotkey", "default", "--home", str(tmp_path)]
    config_path = tmp_path / "validator.json"

    def run(free_gib: float, *extra: str) -> tuple[Manager, dict, str]:
        managers.append(Manager())
        monkeypatch.setattr(host, "memory_available", lambda: (int(64 * GiB), int(free_gib * GiB)))
        setup.init([*argv, *extra])
        return managers[-1], common.read_json(config_path), capsys.readouterr().out

    manager, config, out = run(50)
    assert "references one at a time (RAM for the largest: about 42 GiB): the active models' references do not " \
           "all fit in the free memory at once" in out
    assert "kev-9b: float32 reference on CPU (about 42 GiB), 127.0.0.1:8102, started by the validator" in out
    assert config["reference_mode"] == "sequential"
    kev9 = ValidatorConfig.model_validate(config).references["kev-9b"]
    assert kev9.url == "http://127.0.0.1:8102" and kev9.command[1:4] == ["-m", "kev.serve", "--run"]
    assert kev9.environment["KEV_DTYPE"] == "fp32" and kev9.environment["HF_HUB_OFFLINE"] == "1"
    (validator,) = manager.applied
    assert validator.name == setup.VALIDATOR and validator.after == ()
    assert common.Record.load(tmp_path, "validator").services == [setup.VALIDATOR]

    manager, config, out = run(50)  # an update keeps it, whatever the memory then
    assert config["reference_mode"] == "sequential" and [s.name for s in manager.applied] == [setup.VALIDATOR]

    manager, config, out = run(100, "--references", "resident")
    assert "all references resident (RAM about 63 GiB): as --references resident asks" in out
    assert config["reference_mode"] == "resident" and config["references"]["kev-4b"] == "http://127.0.0.1:8101"
    assert [s.name for s in manager.applied] == ["verathos-reference-kev-4b", "verathos-reference-kev-9b",
                                                 setup.VALIDATOR]
    assert manager.stopped == [setup.VALIDATOR]  # its own reference ends before the services take the memory

    manager, config, out = run(10, "--references", "sequential")
    assert manager.removed == ["verathos-reference-kev-4b", "verathos-reference-kev-9b"]
    assert config["reference_mode"] == "sequential" and [s.name for s in manager.applied] == [setup.VALIDATOR]
