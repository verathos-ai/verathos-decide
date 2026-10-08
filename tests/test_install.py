"""The guided setup's decisions: which models fit a GPU, where references run, ports, the configuration written,
and the service definitions."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pytest

from verathos import catalog
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


def test_each_gpu_gets_the_requested_model_or_keeps_its_model_or_the_first_that_fits() -> None:
    offered = ["kev-4b", "kev-9b"]
    gpus = [gpu(0, 80), gpu(1, 24)]
    assert plan.choose_models(gpus, offered, MINER_NEED, [], {}) == {0: "kev-4b", 1: "kev-4b"}
    assert plan.choose_models(gpus, offered, MINER_NEED, [], {0: "kev-9b", 1: "kev-9b"}) == {0: "kev-9b",
                                                                                            1: "kev-4b"}
    assert plan.choose_models(gpus[:1], offered, MINER_NEED, ["kev-9b"], {}) == {0: "kev-9b"}
    assert plan.choose_models(gpus, offered, MINER_NEED, ["kev-9b", "kev-4b"], {}) == {0: "kev-9b", 1: "kev-4b"}
    with pytest.raises(plan.PlanError, match="needs about"):
        plan.choose_models(gpus, offered, MINER_NEED, ["kev-9b"], {})
    with pytest.raises(plan.PlanError, match="not an active catalog model"):
        plan.choose_models(gpus, offered, MINER_NEED, ["kev-27b"], {})
    with pytest.raises(plan.PlanError, match="one per GPU"):
        plan.choose_models(gpus, offered, MINER_NEED, ["kev-4b", "kev-4b", "kev-4b"], {})
    with pytest.raises(plan.PlanError, match="no active catalog model fits"):
        plan.choose_models([gpu(0, 12)], offered, MINER_NEED, [], {})


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


def test_models_on_other_engines_are_not_installed(tmp_path) -> None:
    clef = catalog.load().get("clef-flash")
    assert not plan.installable(clef)
    with pytest.raises(plan.PlanError, match="does not install"):
        engines.engine_for(tmp_path, clef)


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
