"""The Docker path (``docker/``): the image files against the installer's engines and the shipped catalog, the compose
files' mounts and variables, the build script, and ``verathos-container``, which sets each role up from its
environment and runs it."""

from __future__ import annotations

import importlib.util
import inspect
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
import tomllib
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

import verathos
from tests.test_install import CLEF, KEV, gpu
from tests.test_reference import engines_file
from tests.tlsfake import free_port, serving
from verathos import catalog
from verathos.install import common, container, engines, host, plan
from verathos.install import miner as miner_setup
from verathos.install import validator as validator_setup
from verathos.install.host import GiB
from verathos.miner.config import MinerConfig
from verathos.protocol import tls
from verathos.reference.config import ReferenceConfig, api_key
from verathos.throughput import loadgen
from verathos.validator.config import ValidatorConfig

ROOT = Path(__file__).resolve().parents[1]
DOCKER = ROOT / "docker"
COMPOSE = {"miner": "compose.miner.yml", "validator": "compose.validator.yml",
           "validator-full": "compose.validator-full.yml", "reference": "compose.reference.yml"}
PARAMETERS = {**KEV, "clef-flash": CLEF}
WALLET_TARGETS = ("/root/.bittensor/wallets/${WALLET}/hotkeys/${HOTKEY:-default}",
                  "/root/.bittensor/wallets/${WALLET}/coldkeypub.txt")
REGISTRY = "ghcr.io/verathos-ai"


def compose(name: str) -> dict:
    return yaml.safe_load((DOCKER / name).read_text())


def service(name: str) -> dict:
    (only,) = compose(name)["services"].values()
    return only


def example_env() -> dict[str, str]:
    lines = (DOCKER / ".env.example").read_text().splitlines()
    return dict(line.split("=", 1) for line in lines if line and not line.startswith("#"))


# ---- the images ----

def load_pins():
    spec = importlib.util.spec_from_file_location("pins", DOCKER / "pins.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_engines_image_first_fills_the_environments_the_installer_then_installs(tmp_path) -> None:
    """docker/pins.py reads from the release's files alone what the engines' layer installs: the Kev engines of the
    shipped catalog's active and candidate models, as the installer chooses them, and verathos-engine's requirements,
    as its clef extra names them."""
    pins = load_pins()
    pins.main([str(ROOT / "pyproject.toml"), str(ROOT / "src/verathos/catalog/catalog.json"), str(tmp_path)])
    kev = [tuple(line.split(" ")) for line in (tmp_path / "kev-engines.txt").read_text().splitlines()]

    def installed(cat: catalog.Catalog) -> set[tuple[str, str]]:
        return {(m.engine.source, m.engine.revision) for m in plan.servable_models(cat) if plan.installable(m)
                and isinstance(engines.engine_for(Path("/opt"), m), engines.KevEngine)}

    assert set(kev) == installed(catalog.load()) and len(kev) == len(set(kev))
    document = catalog.load().document()
    models = [{**m, "engine": {**m["engine"], "revision": f"{i}" * 40}} for i, m in enumerate(document["models"])]
    models[0]["status"], models[1]["status"] = "retired", "candidate"
    other = catalog.Catalog.parse({**document, "models": models})
    assert set(pins.kev_engines(other.document())) == installed(other)
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert (tmp_path / "clef-requirements.txt").read_text().splitlines() == [
        *project["dependencies"], *project["optional-dependencies"]["clef"]]


def test_the_dockerfile_fills_each_environment_where_the_installer_runs_it() -> None:
    text = (DOCKER / "Dockerfile").read_text()
    home = container.ENGINES
    kev = engines.engine_for(home, catalog.load().get("kev-4b"))
    clef = engines.engine_for(home, catalog.load().get("clef-flash"))
    assert kev.directory == home / "engines" / f"kev-{kev.engine.revision[:12]}"
    assert 'dir="/opt/engines/kev-$(printf \'%.12s\' "${revision}")"' in text  # the same directory
    for purpose, sync in (("reference", "uv sync --quiet --extra serve"),
                          ("miner", "uv sync --quiet --inexact --extra serve")):  # as KevEngine.install syncs them
        assert (f'UV_PROJECT_ENVIRONMENT="${{dir}}/{kev.environment(purpose).name}" uv --directory "${{dir}}" '
                f'{sync.removeprefix("uv ")} -p {engines.PYTHON}') in text
    assert '"${dir}/kev/fused_qwen35.py"' in text and "flash-linear-attention==${fla}" in text
    assert clef.directory == Path("/opt/engines/verathos-clef")
    assert {clef.environment(p).name for p in ("miner", "reference")} == {".venv", ".venv-reference"}
    assert 'uv venv --quiet -p 3.12 "/opt/engines/verathos-clef/${environment}"' in text
    assert "for environment in .venv-reference .venv; do" in text
    # the code as install.sh installs it, with its Python
    installer = (ROOT / "scripts" / "install.sh").read_text()
    assert re.search(r'^PYTHON_VERSION="3\.12"$', installer, re.M) and engines.PYTHON == "3.12"
    assert 'uv pip install --quiet -e "$dir[server,chain]"' in installer
    assert text.count('uv pip install --quiet -e "/opt/verathos[server,chain]"') == 2
    assert text.count("--extra server --extra chain") == 2
    assert "&& verathos-container engines" in text  # then the installer's code installs every engine
    # the auditor's load generator where verathos-build-loadgen puts it, in both images
    built = "/opt/verathos/" + loadgen.BINARY_PATH.relative_to(ROOT).as_posix()
    assert text.count(f"COPY --from=loadgen /verathos-loadgen {built}") == 2
    toolchain = tomllib.loads((ROOT / "native" / "loadgen" / "rust-toolchain.toml").read_text())["toolchain"]
    assert f"ARG RUST=rust:{toolchain['channel']}-slim-bookworm" in text
    assert re.search(r"^ARG UV=ghcr\.io/astral-sh/uv:\d+\.\d+\.\d+$", text, re.M)  # a pinned uv
    assert 'ENTRYPOINT ["/usr/bin/tini", "--", "verathos-container"]' in text
    stages = re.findall(r"^FROM \S+ AS (\w+)$", text, re.M)
    assert stages == ["uv", "loadgen", "base", "pins", "engines", "node"]
    engines_stage, node_stage = text.split("FROM base AS engines")[1].split("FROM base AS node")
    assert 'CMD ["miner"]' in engines_stage and 'CMD ["validator"]' in node_stage
    assert "verathos-container engines" not in node_stage and "nftables" in engines_stage


def test_the_images_take_the_code_and_nothing_else_of_the_repository() -> None:
    lines = [line for line in (DOCKER / "Dockerfile.dockerignore").read_text().splitlines()
             if line and not line.startswith("#")]
    assert lines[0] == "**"  # everything is left out, then the code is let in
    let_in = {line[1:].rstrip("/") for line in lines if line.startswith("!")}
    assert let_in == {"pyproject.toml", "LICENSE", "README.md", "src", "scripts", "release", "native/loadgen",
                      "docker/pins.py"}
    assert {"native/loadgen/target/", "**/__pycache__/"} <= set(lines)
    text = (DOCKER / "Dockerfile").read_text()
    for sources in re.findall(r"^COPY (?!--from)(.+) \S+$", text, re.M):
        for source in sources.split():
            assert source == "." or any(source.rstrip("/") == p or source.startswith(p + "/") for p in let_in), source


# ---- the compose files ----

def test_the_containers_get_the_hotkey_and_coldkeypub_read_only_and_never_a_coldkey() -> None:
    """Each container that needs a wallet mounts exactly two files of it, read-only and only if they exist: the
    hotkey and coldkeypub.txt. No mount names the wallets directory, a wallet's directory or a coldkey."""
    for role, name in COMPOSE.items():
        wallet_mounts = []
        for volume in service(name)["volumes"]:
            if isinstance(volume, str):
                assert ".bittensor" not in volume and "wallet" not in volume.lower() and "coldkey" not in volume
                continue
            source, target = volume["source"], volume["target"]
            if ".bittensor" in source + target or "WALLET" in source + target:
                wallet_mounts.append(target)
                assert volume["type"] == "bind" and volume["read_only"] is True
                assert volume["bind"] == {"create_host_path": False}
                assert source.startswith("${WALLETS_DIR:-~/.bittensor/wallets}/${WALLET}/")
                assert source.removeprefix("${WALLETS_DIR:-~/.bittensor/wallets}") == target.removeprefix(
                    "/root/.bittensor/wallets")
            assert "coldkey" not in (source + target).replace("coldkeypub.txt", "")
        assert sorted(wallet_mounts) == ([] if role == "reference" else sorted(WALLET_TARGETS)), role


def test_a_hotkey_loads_from_the_read_only_files_the_containers_mount(tmp_path, monkeypatch) -> None:
    """The wallet's two files and directories read-only, as the bind mounts are; the rest of ~/.bittensor writable, as a
    container's own filesystem is."""
    import bittensor_wallet as bw

    monkeypatch.setenv("HOME", str(tmp_path))
    wallet = bw.Wallet(name="w", hotkey="h", path=str(tmp_path / ".bittensor" / "wallets"))
    wallet.create_new_hotkey(n_words=12, use_password=False, overwrite=True, suppress=True)
    coldkey = bw.Keypair.create_from_mnemonic(bw.Keypair.generate_mnemonic())  # only its public key is kept
    wallet.set_coldkeypub(coldkey, overwrite=True)
    root = tmp_path / ".bittensor" / "wallets" / "w"
    paths = sorted(root.rglob("*"), reverse=True) + [root]
    try:
        for path in paths:
            path.chmod(0o500 if path.is_dir() else 0o400)
        loaded = common.load_hotkey("w", "h")
        assert loaded.hotkey.ss58_address == wallet.hotkey.ss58_address
        assert loaded.coldkeypub.ss58_address == coldkey.ss58_address
    finally:
        for path in paths:
            path.chmod(0o700 if path.is_dir() else 0o600)


@pytest.mark.parametrize("role, image, gpus, network, caps", [
    ("miner", "verathos-engines", True, "host", ["NET_ADMIN"]),
    ("validator", "verathos", False, None, None),
    ("validator-full", "verathos", False, "host", None),
    ("reference", "verathos-engines", True, "host", None)])
def test_each_compose_file_runs_its_role_from_the_image_that_carries_it(role, image, gpus, network, caps) -> None:
    s = service(COMPOSE[role])
    command = role.removesuffix("-full")
    assert s["image"] == f"${{IMAGE_REGISTRY:-{REGISTRY}}}/{image}:${{VERATHOS_VERSION:?set VERATHOS_VERSION in .env}}"
    assert s["command"] == [command] and s["restart"] == "unless-stopped"
    assert s.get("network_mode") == network and "ports" not in s
    assert s.get("cap_add") == caps and s["cap_drop"] == ["NET_RAW"]
    assert s["security_opt"] == ["no-new-privileges:true"]
    assert "${DATA_DIR:-./data}:/data" in s["volumes"]
    assert s["logging"] == {"driver": "json-file", "options": {"max-size": "50m", "max-file": "5"}}
    devices = s.get("deploy", {}).get("resources", {}).get("reservations", {}).get("devices")
    assert devices == ([{"driver": "nvidia", "count": "all", "capabilities": ["gpu"]}] if gpus else None)
    if "healthcheck" in s:
        assert s["healthcheck"]["test"] == ["CMD", "verathos-container", "health", command]
    assert ("healthcheck" in s) == (role != "validator")  # a light validator's process is the container's
    if command == "validator":
        assert s["environment"]["MODE"] == ("full" if role == "validator-full" else "light")
        assert compose(COMPOSE[role])["name"] == "verathos-validator"  # either replaces the other's container


def test_the_example_env_names_every_variable_of_the_compose_files_and_this_release() -> None:
    used = {v for name in COMPOSE.values() for v in re.findall(r"\$\{([A-Z_]+)", (DOCKER / name).read_text())}
    example = example_env()
    assert used == set(example) - {"COMPOSE_FILE"}
    assert example["VERATHOS_VERSION"] == verathos.__version__  # the images of this release
    assert example["IMAGE_REGISTRY"] == REGISTRY
    assert f'REGISTRY="{REGISTRY}"' in (DOCKER / "build.sh").read_text()  # where the build script pushes
    assert example["COMPOSE_FILE"] in COMPOSE.values()
    assert all(example[k] == "" for k in ("WALLET", "HF_TOKEN", "ALLOW", "PEERS", "NO_PUBLISH", "DRY_RUN"))


def test_the_containers_read_the_variables_their_compose_files_pass() -> None:
    read = set(re.findall(r'(?:value|flag|number|port_number|listed)\(env, "([A-Z_]+)"', inspect.getsource(container)))
    passed = {k for name in COMPOSE.values() for k in service(name).get("environment", {})}
    assert passed - read == {"HF_TOKEN"}  # the downloads take it from the environment
    assert read <= passed


def docker_compose(env: dict[str, str] | None = None) -> bool:
    try:
        return subprocess.run(["docker", "compose", "version"], capture_output=True, timeout=30,
                              env=env).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


@pytest.mark.skipif(not docker_compose(), reason="needs docker compose")
def test_docker_compose_renders_each_file_with_the_example_env(tmp_path) -> None:
    home = {**os.environ, "HOME": str(tmp_path)}  # ~ in the compose files is the operator's home
    if not docker_compose(home):
        pytest.skip("docker compose is installed in the user's home")
    for name in COMPOSE.values():
        shutil.copy(DOCKER / name, tmp_path / name)
    env = (DOCKER / ".env.example").read_text().replace("\nWALLET=\n", "\nWALLET=w\n")
    (tmp_path / ".env").write_text(env.replace("\nALLOW=\n", "\nALLOW=198.51.100.9\n"))
    for role, name in COMPOSE.items():
        result = subprocess.run(["docker", "compose", "-f", name, "config", "--format", "json"], cwd=tmp_path,
                                capture_output=True, text=True, timeout=60, env=home)
        assert result.returncode == 0, result.stderr
        (s,) = json.loads(result.stdout)["services"].values()
        assert s["image"].startswith(f"{REGISTRY}/verathos") and s["image"].endswith(f":{verathos.__version__}")
        binds = {v["target"]: v for v in s["volumes"]}
        assert binds["/data"]["source"] == str(tmp_path / "data")
        wallets = {t: v for t, v in binds.items() if t.startswith("/root/.bittensor")}
        assert {t: (v["source"], v["read_only"]) for t, v in wallets.items()} == ({} if role == "reference" else {
            "/root/.bittensor/wallets/w/hotkeys/default": (str(tmp_path / ".bittensor/wallets/w/hotkeys/default"),
                                                           True),
            "/root/.bittensor/wallets/w/coldkeypub.txt": (str(tmp_path / ".bittensor/wallets/w/coldkeypub.txt"), True)})
    missing = subprocess.run(["docker", "compose", "-f", COMPOSE["miner"], "config"], cwd=tmp_path, text=True,
                             capture_output=True, env={**home, "WALLET": ""}, timeout=60)
    assert missing.returncode != 0 and "set WALLET in .env" in missing.stderr
    local = subprocess.run(["docker", "compose", "-f", COMPOSE["miner"], "config", "--format", "json"], cwd=tmp_path,
                           text=True, capture_output=True, env={**home, "IMAGE_REGISTRY": "local"}, timeout=60)
    (s,) = json.loads(local.stdout)["services"].values()  # the images docker/build.sh built on this machine
    assert s["image"] == f"local/verathos-engines:{verathos.__version__}"


# ---- the build script ----

def build(*args: str, script: Path = DOCKER / "build.sh") -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(script), *args], capture_output=True, text=True, timeout=120)


def test_the_build_script_builds_both_images_of_this_checkout_tagged_with_its_version() -> None:
    assert subprocess.run(["bash", "-n", str(DOCKER / "build.sh")]).returncode == 0
    usage = build("--help")
    assert usage.returncode == 0
    for option in ("--registry", "--push", "--only", "--jobs", "--dry-run", "--working-tree"):
        assert option in usage.stdout
    assert build("--working-tree", "--latest", "--dry-run").returncode == 2  # version tags only
    dry = build("--working-tree", "--dry-run", "--jobs", "2")
    assert dry.returncode == 0, dry.stderr
    for target, name in (("node", "verathos"), ("engines", "verathos-engines")):
        assert (f"docker build -f {DOCKER / 'Dockerfile'} --target {target} -t local/{name}:{verathos.__version__}-dev "
                f"--build-arg VERSION={verathos.__version__}") in dry.stdout
    assert dry.stdout.count("--build-arg JOBS=2") == 2
    only = build("--working-tree", "--dry-run", "--only", "engines", "--registry", "ghcr.io/owner")
    assert "--target node" not in only.stdout and "-t ghcr.io/owner/verathos-engines:" in only.stdout
    for args, why in ((("--working-tree", "--push", "--dry-run"), "--push is for a release"),
                      (("--working-tree", "--push", "--registry", "local", "--dry-run"), "--push needs a registry"),
                      (("--working-tree", "--only", "gpu", "--dry-run"), "--only takes node or engines"),
                      (("--working-tree", "--registry", "ghcr.io/<org>", "--dry-run"), "--registry: a registry"),
                      (("v9.9.9", "--dry-run"), "no such tag, branch or commit")):
        refused = build(*args)
        assert refused.returncode != 0 and why in refused.stderr, (args, refused.stderr)


def release_repo(tmp_path: Path, signers: str | None = None) -> tuple[Path, Callable[..., str]]:
    """A repository holding the build script and the Dockerfile at version 1.2.3, its release keys ``signers`` (default:
    the shipped placeholder), committed; -> the repository and its git."""
    repo = tmp_path / "repo"
    for path in ("docker/build.sh", "docker/Dockerfile", "release/allowed_signers"):
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / path, repo / path)
    if signers is not None:
        (repo / "release" / "allowed_signers").write_text(signers)
    (repo / "src" / "verathos").mkdir(parents=True)
    (repo / "src" / "verathos" / "__init__.py").write_text('__version__ = "1.2.3"  # the release\n')

    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.org",
                               "-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false",
                               "-c", "core.hooksPath=/dev/null", *args], check=True, capture_output=True, text=True,
                              env={**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1"},
                              ).stdout.strip()

    git("init", "--quiet")
    git("add", ".")
    git("commit", "--quiet", "-m", "release")
    return repo, git


def test_the_build_script_builds_a_commit_as_it_is_and_a_release_only_when_its_tag_verifies(tmp_path) -> None:
    repo, git = release_repo(tmp_path)
    commit = git("rev-parse", "HEAD")
    script = repo / "docker" / "build.sh"
    built = build(commit, "--dry-run", script=script)
    assert built.returncode == 0, built.stderr
    assert "is not a release tag; it is built as it is, unverified" in built.stdout
    assert f"-t local/verathos-engines:1.2.3-{commit[:12]} " in built.stdout
    assert f"--build-arg REVISION={commit}" in built.stdout and str(repo) + " " not in built.stdout  # an export
    git("tag", "v1.2.3")  # a release tag the shipped placeholder key cannot verify
    refused = build("v1.2.3", "--dry-run", script=script)
    assert refused.returncode != 0 and "lists no release signing key yet" in refused.stderr
    git("tag", "v1.2.4")
    assert "lists no release signing key yet" in build("v1.2.4", "--dry-run", script=script).stderr


def test_a_release_builds_images_for_this_machine_and_only_push_publishes_them(tmp_path) -> None:
    """A release whose tag a release key signed: version tags only, local by default; --push tags them for the
    registry and pushes them."""
    found = re.search(r"(\d+)\.(\d+)", subprocess.run(["git", "--version"], capture_output=True, text=True).stdout)
    if shutil.which("ssh-keygen") is None or found is None or (int(found[1]), int(found[2])) < (2, 34):
        pytest.skip("git here cannot sign or verify SSH signatures (git 2.34 and ssh-keygen needed)")
    key = tmp_path / "release-key"
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "release", "-f", str(key)], check=True)
    public = " ".join(key.with_name("release-key.pub").read_text().split()[:2])
    repo, git = release_repo(tmp_path, f'verathos-release namespaces="git" {public}\n')
    git("-c", "gpg.format=ssh", "-c", "gpg.ssh.program=ssh-keygen", "-c", f"user.signingkey={key}", "tag", "-s",
        "-m", "v1.2.3", "v1.2.3")
    script = repo / "docker" / "build.sh"
    local = build("v1.2.3", "--dry-run", script=script)
    assert local.returncode == 0, local.stderr
    assert 'v1.2.3: Good "git" signature for verathos-release' in local.stdout
    assert "-t local/verathos:1.2.3 " in local.stdout and "-t local/verathos-engines:1.2.3 " in local.stdout
    assert "docker push" not in local.stdout
    pushed = build("v1.2.3", "--dry-run", "--push", script=script)
    assert pushed.returncode == 0, pushed.stderr
    for name in ("verathos", "verathos-engines"):
        assert f"-t {REGISTRY}/{name}:1.2.3 " in pushed.stdout
    assert [line for line in pushed.stdout.splitlines() if line.startswith("docker push")] == [
        f"docker push {REGISTRY}/verathos:1.2.3", f"docker push {REGISTRY}/verathos-engines:1.2.3"]
    assert "latest" not in local.stdout + pushed.stdout


# ---- verathos-container: the environment ----

def test_the_messages_of_the_installers_code_name_the_variables_set_here() -> None:
    assert container.named("kev-9b would pay 1.2x: --model kev-9b to switch") == (
        "kev-9b would pay 1.2x: MODEL=kev-9b to switch")
    assert container.named("another choice: --model <id> for every GPU, as in --model kev-4b,clef-flash") == (
        "another choice: MODEL=<id> for every GPU, as in MODEL=kev-4b,clef-flash")
    assert container.named("(--model auto lets the setup choose)") == "(MODEL=auto lets the setup choose)"
    assert container.named("as --model asks; with --model it can serve x") == "as MODEL asks; with MODEL it can serve x"
    assert container.named("--gpus takes GPU numbers") == "GPUS takes GPU numbers"
    assert container.named("--peers: x; --name: y; --allow: z") == "PEERS: x; NAME: y; ALLOW: z"


def test_variables_are_read_as_set_and_refused_with_their_names() -> None:
    env = {"A": " on ", "B": "0", "C": "", "N": "405", "P": "70000", "L": " 203.0.113.1, ,198.51.100.2 "}
    assert container.flag(env, "A", False) and not container.flag(env, "B", True) and container.flag(env, "C", True)
    assert container.flag(env, "missing", False) is False
    assert container.number(env, "N") == 405 and container.number(env, "C") is None
    assert container.listed(env, "L") == ["203.0.113.1", "198.51.100.2"]
    with pytest.raises(SystemExit, match="A2='maybe': on or off"):
        container.flag({"A2": "maybe"}, "A2", True)
    with pytest.raises(SystemExit, match="N2='x': a number"):
        container.number({"N2": "x"}, "N2")
    with pytest.raises(SystemExit, match="P=70000: a TCP port number"):
        container.port_number(env, "P")
    with pytest.raises(SystemExit, match="WALLET is not set"):
        container.wallet_names({})
    assert container.wallet_names({"WALLET": "w"}) == ("w", "default")


# ---- verathos-container: the processes ----

def python(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def test_the_supervisor_starts_a_process_that_exits_again_and_stops_them_all(tmp_path) -> None:
    out, now = io.StringIO(), [0.0]
    quick = container.Process("quick", python("print('hello'); raise SystemExit(3)"), tmp_path, {"X": "1"})
    slow = container.Process("slow", python("import time; time.sleep(60)"), tmp_path)
    missing = container.Process("missing", [str(tmp_path / "nothing-here")], tmp_path)
    supervisor = container.Supervisor([quick, slow, missing], out=out, clock=lambda: now[0])
    for p in supervisor.processes:
        supervisor.start(p)
    first = supervisor.children["quick"].pid
    supervisor.children["quick"].wait(timeout=30)
    supervisor.round()  # quick exited: it starts again in RESTART_S
    assert "quick" not in supervisor.children and supervisor.due["quick"] == supervisor.RESTART_S
    now[0] = supervisor.RESTART_S
    supervisor.round()
    assert supervisor.children["quick"].pid != first and "missing" not in supervisor.children
    supervisor.children["quick"].wait(timeout=30)
    sleeper = supervisor.children["slow"]
    started = time.monotonic()
    supervisor.stop()
    assert sleeper.returncode is not None and time.monotonic() - started < 30  # SIGTERM, not the SIGKILL deadline
    log = out.getvalue()
    assert log.count("quick | hello\n") == 2
    assert "quick | exited with code 3; starting it again in 10 s" in log
    assert "missing | did not start (" in log


def test_a_setup_that_stops_says_why_and_waits_before_the_container_exits(tmp_path, monkeypatch, capsys) -> None:
    slept = []

    def stopped(env, data, home):
        raise SystemExit("hotkey 5X is not registered on netuid 96")

    monkeypatch.setattr(container, "miner", stopped)
    assert container.run("miner", {}, tmp_path, tmp_path, tmp_path, sleep=slept.append) == 1
    assert slept == [container.SETUP_RETRY_S]
    assert "verathos-container miner: hotkey 5X is not registered on netuid 96" in capsys.readouterr().err

    def failed(env, data, home):
        raise RuntimeError("the chain closed the connection")

    monkeypatch.setattr(container, "reference", failed)
    assert container.run("reference", {}, tmp_path, tmp_path, tmp_path, sleep=slept.append) == 1
    err = capsys.readouterr().err
    assert "RuntimeError: the chain closed the connection" in err and "the setup failed; trying again in 60 s" in err
    with pytest.raises(SystemExit):
        container.main(["health"])  # takes the role


# ---- verathos-container: the roles ----

def image_engines(home: Path, cat: catalog.Catalog) -> None:
    """The engines an image holds under ``home``: each model's engine for both purposes, the Kev engine's checkout at
    its revision, as install_engines leaves them."""
    for m in cat.models:
        engine = engines.engine_for(home, m)
        for purpose in ("miner", "reference"):
            engine.python(purpose).parent.mkdir(parents=True, exist_ok=True)
            engine.python(purpose).touch()
        if isinstance(engine, engines.KevEngine):
            (engine.directory / ".git").mkdir(exist_ok=True)
            (engine.directory / ".git" / "HEAD").write_text(m.engine.revision + "\n")


@pytest.fixture()
def box(tmp_path, monkeypatch, lab_network):
    """A container's view: the image's engines under ``home``, the volume at ``data``, its GPUs, the chain and the
    downloads stood in for."""
    data, home = tmp_path / "data", tmp_path / "opt"
    data.mkdir()
    image_engines(home, catalog.load())
    state = SimpleNamespace(data=data, home=home, remote=tmp_path / "remote", gpus=[gpu(0, 80), gpu(1, 24)],
                            catalog=catalog.load(), registered={"5Miner": 7, "5Validator": 5}, owner="5Owner",
                            hotkey="5Miner", downloads=[])
    chain = SimpleNamespace(
        get_uid_for_hotkey_on_subnet=lambda hotkey, netuid: state.registered.get(hotkey),
        get_neuron_for_pubkey_and_subnet=lambda hotkey, netuid: SimpleNamespace(
            is_null=hotkey not in state.registered, uid=state.registered.get(hotkey), validator_permit=True),
        get_subnet_owner_hotkey=lambda netuid: state.owner)
    monkeypatch.setattr(common, "connect", lambda network: chain)
    monkeypatch.setattr(common, "load_hotkey", lambda w, h: SimpleNamespace(
        hotkey=SimpleNamespace(ss58_address=state.hotkey)))
    monkeypatch.setattr("verathos.subnet.chain.current_catalog", lambda *args: state.catalog)
    monkeypatch.setattr(host, "gpus", lambda: state.gpus)
    monkeypatch.setattr(host, "driver_cuda_version", lambda: (12, 8))
    monkeypatch.setattr(host, "port_free", lambda port, address="0.0.0.0": True)
    monkeypatch.setattr(host, "detect_public_ip", lambda: "203.0.113.7")
    monkeypatch.setattr(host, "memory_available", lambda: (216 * GiB, 210 * GiB))
    monkeypatch.setattr(engines, "parameters", lambda m: PARAMETERS.get(m.model_id, KEV["kev-9b"]))
    monkeypatch.setattr(engines, "download", lambda python, m, hf_home: state.downloads.append(
        (Path(python), m.model_id, hf_home)))
    monkeypatch.setattr(miner_setup, "read_market", lambda *args, **kwargs: (None, "no network status"))

    async def remote_lines(references, screens, cat):
        return [f"{m}: checked" for m in sorted(references)] + [f"{m} screen: checked" for m in sorted(screens)]

    monkeypatch.setattr(validator_setup, "remote_lines", remote_lines)
    return state


def test_a_miner_container_runs_an_engine_per_gpu_and_the_miner_as_the_installer_sets_them_up(box, capsys) -> None:
    env = {"WALLET": "miner", "NETWORK": "test", "PORT": "20091", "MODEL": "kev-9b,kev-4b"}
    processes = container.miner(env, box.data, box.home)
    assert [p.name for p in processes] == ["engine-gpu0", "engine-gpu1", "miner"]
    kev9, kev4 = (engines.engine_for(box.home, box.catalog.get(m)) for m in ("kev-9b", "kev-4b"))
    gpu0, gpu1, miner = processes
    assert gpu0.command[0] == str(kev9.python("miner")) and gpu1.command[0] == str(kev4.python("miner"))
    assert gpu0.command[gpu0.command.index("--run") + 1].startswith("jaredpalmer/kev-9b@")
    assert gpu0.environment["CUDA_VISIBLE_DEVICES"] == "0" and gpu1.environment["CUDA_VISIBLE_DEVICES"] == "1"
    assert gpu0.environment["HF_HOME"] == str(box.data / "hf") and gpu0.environment["HF_HUB_OFFLINE"] == "1"
    assert miner.command == [str(Path(sys.prefix) / "bin" / "verathos-miner"), "--config", str(box.data / "miner.json")]
    assert miner.directory == box.data
    config = MinerConfig.load(box.data / "miner.json")
    assert (config.network, config.netuid, config.public_host, config.port) == ("test", 405, "203.0.113.7", 20091)
    assert (config.auto_update, config.firewall, config.role) == (False, True, "primary")
    assert (config.state_dir, config.catalog_dir) == (box.data / "miner", box.data / "catalog")
    assert [(e.endpoint_id, e.model_id) for e in config.endpoints] == [("gpu0", "kev-9b"), ("gpu1", "kev-4b")]
    assert gpu0.command[gpu0.command.index("--port") + 1] == config.endpoints[0].engine_url.rsplit(":", 1)[1]
    assert sorted((p.name, m, h) for p, m, h in box.downloads) == [("python", "kev-4b", box.data / "hf"),
                                                                   ("python", "kev-9b", box.data / "hf")]
    assert tls.stored_pin(box.data / "miner") is not None
    out = capsys.readouterr().out
    assert "self-update off: a release is a new image" in out
    assert "GPU 0: kev-9b, as MODEL asks" in out and "another choice: MODEL=<id> for every GPU" in out
    assert "--model" not in out  # the installer's option, as it is set here

    # again: hand-made settings stay, the endpoints keep their engine ports, the variables decide the rest
    document = json.loads((box.data / "miner.json").read_text())
    document["endpoints"][0]["max_concurrency"] = 64
    document["state_dir"] = "/root/.verathos/miner"  # anywhere outside the volume is lost on recreation
    common.write_json(box.data / "miner.json", document)
    again = container.miner({**env, "FIREWALL": "off", "PEERS": "203.0.113.8:8091"}, box.data, box.home)
    config = MinerConfig.load(box.data / "miner.json")
    assert config.endpoints[0].max_concurrency == 64 and config.state_dir == box.data / "miner"
    assert config.firewall is False and config.peers == ["https://203.0.113.8:8091"]
    assert [p.command for p in again[:2]] == [p.command for p in processes[:2]]
    assert common.Record.load(box.data, "miner").engines["gpu0"] == {"gpu": 0, "model": "kev-9b",
                                                                     "port": int(config.endpoints[0].engine_url
                                                                                 .rsplit(":", 1)[1]),
                                                                     "manual": True}


def test_a_miner_container_tries_the_machine_unregistered_only_with_no_publish(box) -> None:
    box.registered.pop("5Miner")
    with pytest.raises(SystemExit, match=r"(?s)not registered on netuid 96.*set NO_PUBLISH=1"):
        container.miner({"WALLET": "miner"}, box.data, box.home)
    box.catalog = None  # the shipped catalog, not the chain's
    processes = container.miner({"WALLET": "miner", "NO_PUBLISH": "1"}, box.data, box.home)
    assert processes[-1].command[-1] == "--no-publish"


def test_a_worker_container_and_the_variables_it_refuses(box) -> None:
    worker = container.miner({"WALLET": "miner", "WORKER": "1", "NAME": "box-a"}, box.data, box.home)
    config = MinerConfig.load(box.data / "miner.json")
    assert config.role == "worker" and [e.endpoint_id for e in config.endpoints] == ["box-a-gpu0", "box-a-gpu1"]
    assert [p.name for p in worker] == ["engine-gpu0", "engine-gpu1", "miner"] and config.peers == []
    for env, why in (({"WORKER": "1", "PEERS": "203.0.113.8:8091"}, "not both"),
                     ({"NAME": "box-a"}, "NAME names a worker's endpoints"),
                     ({"WORKER": "1", "NAME": "Box A"}, "NAME: 1-24 characters"),
                     ({"GPUS": "7"}, "no GPU 7"), ({"GPUS": "x"}, "GPUS takes GPU numbers"),
                     ({"MODEL": "kev-27b"}, "kev-27b is not a catalog model"),
                     ({"MODEL": "kev-4b,kev-4b,kev-4b"}, "MODEL takes one model for all GPUs"),
                     ({"PEERS": "192.168.1.5:8091"}, "PEERS: 192.168.1.5:8091 is not a public address"),
                     ({"PUBLIC_IP": "192.168.1.5"}, "PUBLIC_IP: 192.168.1.5 is not a public IP address"),
                     ({"FIREWALL": "maybe"}, "FIREWALL='maybe'"), ({"PORT": "0"}, "PORT=0: a TCP port number"),
                     ({"WALLET": ""}, "WALLET is not set")):
        with pytest.raises(SystemExit, match=why):
            container.miner({"WALLET": "miner", **env}, box.data, box.home)
    box.gpus = []
    with pytest.raises(SystemExit, match="no NVIDIA GPU in the container"):
        container.miner({"WALLET": "miner"}, box.data, box.home)


def test_an_engine_revision_the_image_lacks_is_served_with_the_one_it_carries_by_miners_only(box, capsys) -> None:
    """Validators check a miner's answers, not its engine: a miner serves with the Kev engine the image carries until
    the image of the release that ships the new revision runs. A reference machine serves exactly what the catalog
    pins, and leaves the model out."""
    document = catalog.load().document()
    moved = [{**m, "engine": {**m["engine"], "revision": "f" * 40}} if m["model_id"] == "kev-4b" else m
             for m in document["models"]]
    new = {**moved[0], "model_id": "kev-14b", "engine": {**moved[0]["engine"], "revision": "e" * 40}}
    box.catalog = catalog.Catalog.parse({**document, "models": [*moved, new]})
    processes = container.miner({"WALLET": "miner", "MODEL": "kev-4b"}, box.data, box.home)
    shipped = engines.engine_for(box.home, catalog.load().get("kev-4b"))
    assert processes[0].command[0] == str(shipped.python("miner"))
    out = capsys.readouterr().out
    assert "kev-4b: the catalog pins the kev engine at ffffffffffff; this image carries 5e42a7a03f28" in out
    assert "kev-14b runs on the kev engine at eeeeeeeeeeee, which this image does not carry" in out
    with pytest.raises(SystemExit, match="kev-14b is not a catalog model this setup can install"):
        container.miner({"WALLET": "miner", "MODEL": "kev-14b"}, box.data, box.home)

    names = [p.name for p in container.reference({"ALLOW": "198.51.100.9"}, box.data, box.home)]
    assert "reference-kev-4b" not in names and "reference-kev-9b" in names
    assert "kev-4b is not served: the catalog pins the kev engine at ffffffffffff" in capsys.readouterr().out
    assert container.carried(engines.engine_for(box.home, box.catalog.get("kev-9b")), "reference")
    (box.home / "engines" / "verathos-clef" / ".venv-reference" / "bin" / "python").unlink()
    assert not container.carried(engines.engine_for(box.home, box.catalog.get("clef-flash")), "reference")


def test_a_light_validator_container_sets_the_auditors_weights_with_the_hotkey_alone(box) -> None:
    box.hotkey = "5Validator"
    p = container.validator({"WALLET": "v", "MODE": "light", "NETWORK": "test"}, box.data, box.remote)
    assert p.command == [str(Path(sys.prefix) / "bin" / "verathos-validator"), "--config",
                         str(box.data / "validator.json")]
    config = ValidatorConfig.load(box.data / "validator.json")
    assert (config.mode, config.auto_update, config.netuid, config.references) == ("light", False, 405, {})
    assert (config.state_dir, config.catalog_dir) == (box.data / "validator", box.data / "catalog")
    box.owner = "5Validator"
    with pytest.raises(SystemExit, match="the subnet owner's hotkey: its validator is the auditor and runs full"):
        container.validator({"WALLET": "v", "MODE": "light"}, box.data, box.remote)
    box.owner, box.registered = "5Owner", {}
    with pytest.raises(SystemExit, match="is not registered on netuid 405"):  # the network configured before
        container.validator({"WALLET": "v"}, box.data, box.remote)
    container.validator({"WALLET": "v", "DRY_RUN": "1"}, box.data, box.remote)  # a dry run tries it first
    assert ValidatorConfig.load(box.data / "validator.json").dry_run
    with pytest.raises(SystemExit, match="MODE='half': light or full"):
        container.validator({"WALLET": "v", "MODE": "half"}, box.data, box.remote)


def test_a_full_validator_container_takes_its_references_from_the_reference_machines_files(box, capsys,
                                                                                         monkeypatch) -> None:
    box.hotkey = "5Validator"
    env = {"WALLET": "v", "MODE": "full"}
    with pytest.raises(SystemExit, match="put the file a reference machine writes"):
        container.validator(env, box.data, box.remote)
    box.remote.mkdir()
    pin, key = "ab" * 32, "k" * 43
    engines_file(box.remote / "a.json", pin, key, models=("kev-4b", "kev-9b"))
    with pytest.raises(SystemExit, match="no reference for the active models clef-flash"):
        container.validator(env, box.data, box.remote)
    engines_file(box.remote / "b.json", "cd" * 32, "j" * 43, models=("clef-flash",), screens=(), host_="203.0.113.8")
    common.write_json(box.data / "validator.json", {"references": {"kev-4b": "http://127.0.0.1:8101"},
                                                    "reference_mode": "sequential"})  # from a machine's own engines
    container.validator(env, box.data, box.remote)
    config = ValidatorConfig.load(box.data / "validator.json")
    assert {m: (e.url, e.tls_pin, e.api_key) for m, e in config.references.items()} == {
        "kev-4b": ("https://203.0.113.7:20004/reference/kev-4b", pin, key),
        "kev-9b": ("https://203.0.113.7:20004/reference/kev-9b", pin, key),
        "clef-flash": ("https://203.0.113.8:20004/reference/clef-flash", "cd" * 32, "j" * 43)}
    assert (config.mode, config.reference_mode, config.audit_screens, config.public_host) == (
        "full", "resident", {}, None)
    assert config.egress_ips == ["203.0.113.7"] and config.auto_update is False
    assert (box.data / "validator.json").stat().st_mode & 0o777 == 0o600  # it holds the keys
    assert "the screens of kev-4b are the auditor's: not configured here" in capsys.readouterr().out
    container.validator({**env, "PUBLIC_IP": "203.0.113.9", "PORT": "20099", "EGRESS_IPS": "203.0.113.9"}, box.data,
                        box.remote)
    config = ValidatorConfig.load(box.data / "validator.json")
    assert (config.public_host, config.port, config.egress_ips) == ("203.0.113.9", 20099, ["203.0.113.9"])

    box.owner = "5Validator"  # the auditor: the screens, its service and its load generator
    monkeypatch.setattr(loadgen, "BINARY_PATH", box.data / "verathos-loadgen")
    with pytest.raises(SystemExit, match="verathos-loadgen is missing"):
        container.validator({"WALLET": "v", "MODE": "full"}, box.data, box.remote)
    (box.data / "verathos-loadgen").touch()
    (box.data / "validator.json").unlink()
    container.validator({"WALLET": "v", "MODE": "full"}, box.data, box.remote)
    config = ValidatorConfig.load(box.data / "validator.json")
    assert list(config.audit_screens) == ["kev-4b"] and (config.public_host, config.port) == ("203.0.113.7", 8091)


def test_a_reference_container_runs_each_engine_and_the_front_and_writes_the_validators_file(box, capsys) -> None:
    with pytest.raises(SystemExit, match="ALLOW is not set"):
        container.reference({}, box.data, box.home)
    with pytest.raises(SystemExit, match="ALLOW: '10.0.0.300' is not an IP address"):
        container.reference({"ALLOW": "10.0.0.300"}, box.data, box.home)
    box.gpus = [gpu(0, 80)]
    processes = container.reference({"ALLOW": "198.51.100.9", "NETWORK": "test", "PORT": "20004"}, box.data, box.home)
    assert [p.name for p in processes] == ["reference-clef-flash", "reference-kev-4b", "reference-kev-9b",
                                           "screen-clef-flash", "screen-kev-4b", "screen-kev-9b", "front"]
    by_name = {p.name: p for p in processes}
    fp32, bf16 = by_name["reference-kev-4b"], by_name["screen-kev-4b"]
    assert fp32.environment["KEV_DTYPE"] == "fp32" and fp32.command[0].endswith(".venv-reference/bin/python")
    assert bf16.environment["CUDA_VISIBLE_DEVICES"] == "0" and bf16.command[0].endswith("/.venv/bin/python")
    assert by_name["front"].command[-2:] == ["--config", str(box.data / "reference.json")]
    config = ReferenceConfig.load(box.data / "reference.json")
    assert (config.public_host, config.port, config.allow, config.state_dir) == (
        "203.0.113.7", 20004, ["198.51.100.9"], box.data / "reference")
    document = json.loads((box.data / common.ENGINES_FILE).read_text())
    assert (box.data / common.ENGINES_FILE).stat().st_mode & 0o777 == 0o600
    assert document["references"]["kev-4b"] == {"url": "https://203.0.113.7:20004/reference/kev-4b",
                                                "tls_pin": tls.stored_pin(box.data / "reference"),
                                                "api_key": api_key(box.data / "reference")}
    assert "into the REMOTE_ENGINES_DIR of each validator machine" in capsys.readouterr().out
    without = container.reference({"ALLOW": "198.51.100.9", "SCREENS": "off", "MODEL": "kev-4b"}, box.data, box.home)
    assert [p.name for p in without] == ["reference-kev-4b", "front"]
    again = json.loads((box.data / common.ENGINES_FILE).read_text())
    assert again["references"]["kev-4b"] == document["references"]["kev-4b"] and again["audit_screens"] == {}
    config = ReferenceConfig.load(box.data / "reference.json")  # the network set before stays, as the installer's
    assert (config.network, config.netuid) == ("test", 405)


def test_the_engines_of_the_shipped_catalog_are_installed_for_references_and_miners(tmp_path, monkeypatch) -> None:
    installed = []

    def install(self, purpose: str) -> None:
        installed.append((self.engine.name, purpose))
        image_engines(tmp_path, catalog.load())

    for family in (engines.KevEngine, engines.ClefEngine):
        monkeypatch.setattr(family, "install", install)
    container.install_engines(tmp_path)
    assert installed == [("kev", "reference"), ("kev", "miner"), ("clef", "reference"), ("clef", "miner")]
    for family in (engines.KevEngine, engines.ClefEngine):
        monkeypatch.setattr(family, "install", lambda self, purpose: None)
    shutil.rmtree(tmp_path / "engines" / "verathos-clef")
    with pytest.raises(SystemExit, match="verathos-clef: no reference environment"):
        container.install_engines(tmp_path)


# ---- the health check ----

def test_the_health_check_asks_the_roles_service_on_loopback(tmp_path) -> None:
    data, identity = tmp_path / "data", tls.identity(tmp_path / "identity")
    data.mkdir()
    assert not container.health("miner", data)  # not set up yet
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"hotkey": "5Miner", "endpoints": ["gpu0"]}

    @app.get("/{kind}/{model}/v1/models")
    async def models(kind: str, model: str, request: Request):
        if request.headers.get("authorization") != "Bearer " + "k" * 43:
            return JSONResponse({"error": "key"}, status_code=401)
        return {"data": [{"id": model}]}

    @app.get("/v1/network")
    async def network():
        return JSONResponse({"error": "no epoch yet"}, status_code=503)

    with serving(app, identity) as url:
        port = int(url.rsplit(":", 1)[1])
        common.write_json(data / "miner.json", {"port": port})
        assert container.health("miner", data)
        state = data / "reference"
        state.mkdir()
        (state / "api.key").write_text("k" * 43 + "\n")
        common.write_json(data / "reference.json", {"port": port, "state_dir": str(state), "engines": {
            "kev-4b": {"reference": "http://127.0.0.1:1", "screen": "http://127.0.0.1:2"}}})
        assert container.health("reference", data)
        (state / "api.key").write_text("j" * 43 + "\n")
        assert not container.health("reference", data)
        common.write_json(data / "validator.json", {"mode": "full", "public_host": "203.0.113.9", "port": port})
        assert container.health("validator", data)  # its service answers, whatever it answers
    common.write_json(data / "miner.json", {"port": free_port()})
    assert not container.health("miner", data)
    common.write_json(data / "validator.json", {"mode": "full", "public_host": "203.0.113.9", "port": free_port()})
    assert not container.health("validator", data)
    common.write_json(data / "validator.json", {"mode": "light"})
    assert container.health("validator", data)


def test_the_container_command_is_installed_with_the_code() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["scripts"]["verathos-container"] == "verathos.install.container:main"
