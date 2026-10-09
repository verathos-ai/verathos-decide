"""``verathos-container``: a miner, a validator or a reference machine as a Docker container (``docker/``).

A release is two images: ``verathos``, its code (a light validator; a full validator or the auditor whose engines run
on a reference machine, with the auditor's load generator built in), and ``verathos-engines``, its code and the engines
of its shipped catalog at their pinned revisions, installed by the installer's own code (``install_engines``, run when
the image is built). At each start the container sets its role up from its environment, which the compose file takes
from the operator's ``.env``, as ``verathos-<role> init --yes`` sets up a machine: the catalog the subnet owner
commits, the models and where their engines run, the pinned weights (downloaded once, into ``<data>/hf``) and the
configuration (``<data>/<role>.json``, which keeps what the operator added to it). A miner or a reference machine
then runs its engines and its service as children of this process (``Supervisor``): each started again ``RESTART_S``
after it exits, their output in the container's log behind their names. A validator replaces this process.
Everything that persists is under ``<data>``: the configuration, the TLS identity (so the pin stays), measurements,
scores and caches. The container never updates its code: a release is a new image.

    verathos-container miner | validator | reference   set the role up and run it
    verathos-container health miner | validator | reference
    verathos-container engines                          install the shipped catalog's engines (image build)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

DATA = Path("/data")  # the volume: configuration, state, weights, caches
ENGINES = Path("/opt")  # the image's engines, under <ENGINES>/engines as under a setup's home
REMOTE_ENGINES = Path("/etc/verathos/remote-engines")  # a full validator's files of reference machines (*.json)
ROLES = ("miner", "validator", "reference")
SETUP_RETRY_S = 60.0  # a setup that stops waits this long before the container exits and Docker starts it again
TRUE, FALSE = ("1", "true", "yes", "on"), ("0", "false", "no", "off")
# The installer's options in the messages of its code, as they are set here: variables of the .env. "--model x"
# becomes MODEL=x where x is a value (a model, a list, <id> or auto).
VARIABLES = ((re.compile(r"--model (?=auto\b|<|[\w.]*[-\d,])"), "MODEL="), (re.compile(r"--model\b"), "MODEL"),
             (re.compile(r"--gpus\b"), "GPUS"), (re.compile(r"--peers\b"), "PEERS"), (re.compile(r"--name\b"), "NAME"),
             (re.compile(r"--public-ip\b"), "PUBLIC_IP"), (re.compile(r"--allow\b"), "ALLOW"),
             (re.compile(r"--egress-ip\b"), "EGRESS_IPS"), (re.compile(r"--remote-engines\b"), "REMOTE_ENGINES_DIR"))


def named(text: str) -> str:
    """``text``, a message of the installer's code, with its options as the variables that set them here."""
    for pattern, variable in VARIABLES:
        text = pattern.sub(variable, text)
    return text


# ---- the environment ----

def value(env: Mapping[str, str], name: str) -> str | None:
    """A variable's value; None when it is unset or empty (compose passes an unset variable as empty)."""
    text = (env.get(name) or "").strip()
    return text or None


def flag(env: Mapping[str, str], name: str, default: bool) -> bool:
    text = (value(env, name) or "").lower()
    if not text:
        return default
    if text in TRUE:
        return True
    if text in FALSE:
        return False
    raise SystemExit(f"{name}={env[name]!r}: on or off (1, true, yes; 0, false, no)")


def number(env: Mapping[str, str], name: str) -> int | None:
    text = value(env, name)
    if text is None:
        return None
    if not text.isdigit():
        raise SystemExit(f"{name}={text!r}: a number")
    return int(text)


def port_number(env: Mapping[str, str], name: str) -> int | None:
    port = number(env, name)
    if port is not None and not 0 < port < 65536:
        raise SystemExit(f"{name}={port}: a TCP port number")
    return port


def listed(env: Mapping[str, str], name: str) -> list[str]:
    """A comma-separated variable's items."""
    return [v.strip() for v in (value(env, name) or "").split(",") if v.strip()]


def chain_options(env: Mapping[str, str]) -> argparse.Namespace:
    """``NETWORK`` and ``NETUID`` as the installer's options (``common.network_and_netuid``)."""
    return argparse.Namespace(network=value(env, "NETWORK"), netuid=number(env, "NETUID"))


def wallet_names(env: Mapping[str, str]) -> tuple[str, str]:
    wallet = value(env, "WALLET")
    if wallet is None:
        raise SystemExit("WALLET is not set: the wallet whose hotkey this machine uses (.env)")
    return wallet, value(env, "HOTKEY") or "default"


def from_options[T](call: Callable[[], T]) -> T:
    """``call()``, its stop (SystemExit) said with the variables in place of the installer's options."""
    try:
        return call()
    except SystemExit as exc:
        raise SystemExit(named(str(exc.code)) if isinstance(exc.code, str) else exc.code) from None


# ---- the processes of a role ----

@dataclass(frozen=True)
class Process:
    """A process of the role, an engine or a service, as the installer's services run it."""

    name: str
    command: list[str]
    directory: Path
    environment: dict[str, str] = field(default_factory=dict)


class Supervisor:
    """Runs ``processes`` as children, each in a session of its own, their output to ``out`` line by line behind their
    names, and starts each again ``RESTART_S`` after it exits, as systemd restarts the installer's services. On SIGTERM
    or SIGINT it stops them (SIGTERM, then SIGKILL after ``STOP_S``) and returns."""

    RESTART_S = 10.0
    STOP_S = 50.0  # within the compose files' stop_grace_period
    TICK_S = 1.0

    def __init__(self, processes: list[Process], out: IO[str] | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.processes, self.out, self.clock = processes, out or sys.stdout, clock
        self.stopping = threading.Event()
        self.children: dict[str, subprocess.Popen] = {}
        self.due: dict[str, float] = {}  # name -> when it starts again
        self._lock = threading.Lock()
        self._copiers: list[threading.Thread] = []

    def say(self, name: str, line: str) -> None:
        with self._lock:
            self.out.write(f"{name} | {line.rstrip()}\n")
            self.out.flush()

    def _copy(self, name: str, stream: IO[bytes]) -> None:
        for raw in iter(stream.readline, b""):
            self.say(name, raw.decode(errors="replace"))
        stream.close()

    def start(self, p: Process) -> None:
        try:
            child = subprocess.Popen(p.command, cwd=p.directory, env={**os.environ, **p.environment},
                                     stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        except OSError as exc:
            self.say(p.name, f"did not start ({exc}); trying again in {self.RESTART_S:.0f} s")
            self.due[p.name] = self.clock() + self.RESTART_S
            return
        self.children[p.name] = child
        self.due.pop(p.name, None)
        copier = threading.Thread(target=self._copy, args=(p.name, child.stdout), name=f"log-{p.name}", daemon=True)
        copier.start()
        self._copiers = [c for c in self._copiers if c.is_alive()] + [copier]

    def round(self) -> None:
        """Note what exited, and start again what is due."""
        now = self.clock()
        for p in self.processes:
            child = self.children.get(p.name)
            if child is not None and child.poll() is not None:
                del self.children[p.name]
                self.say(p.name, f"exited with code {child.returncode}; starting it again in {self.RESTART_S:.0f} s")
                self.due[p.name] = now + self.RESTART_S
            elif child is None and now >= self.due.get(p.name, now):
                self.start(p)

    def stop(self) -> None:
        for child in self.children.values():
            _signal(child, signal.SIGTERM)
        deadline = self.clock() + self.STOP_S
        for name, child in self.children.items():
            try:
                child.wait(timeout=max(deadline - self.clock(), 0.1))
            except subprocess.TimeoutExpired:
                self.say(name, f"did not stop within {self.STOP_S:.0f} s; killed")
                _signal(child, signal.SIGKILL)
                child.wait()
        for copier in self._copiers:
            copier.join(timeout=5)

    def run(self) -> int:
        def stop(signum: int, frame: object) -> None:
            self.stopping.set()

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        for p in self.processes:
            self.start(p)
        while not self.stopping.wait(self.TICK_S):
            self.round()
        self.stop()
        return 0


def _signal(child: subprocess.Popen, signum: int) -> None:
    """Signal the child's session: it may have started processes of its own."""
    try:
        os.killpg(child.pid, signum)
    except (ProcessLookupError, PermissionError):
        pass


# ---- the engines in the image ----

def checked_out(directory: Path) -> str | None:
    """The commit of a checkout's detached HEAD, as ``KevEngine.install`` leaves it; read without git."""
    try:
        return (directory / ".git" / "HEAD").read_text().strip()
    except OSError:
        return None


def carried(engine, purpose: str) -> bool:
    """Whether the image holds ``engine`` (``engines.engine_for``) for ``purpose`` ("miner" or "reference"): its
    environment, and for the Kev engine its checkout at the revision it is for."""
    from verathos.install import engines

    if not engine.python(purpose).exists():
        return False
    return not isinstance(engine, engines.KevEngine) or checked_out(engine.directory) == engine.engine.revision


def miner_engine(home: Path, model, shipped) -> tuple[object | None, str]:
    """The engine this image serves ``model`` with on a miner, and a line on it: the engine the catalog pins, if the
    image holds it; else, for a model of the release's shipped catalog (``shipped``) on the same engine, the revision
    that catalog pins, with a warning (a miner may serve any engine whose answers pass: validators check answers, not
    engines); else none, and why."""
    from verathos.install import engines, plan

    if not plan.installable(model):
        return None, f"{model.model_id} runs on the {model.engine.name} engine, which this image does not carry"
    engine = engines.engine_for(home, model)
    if carried(engine, "miner"):
        return engine, ""
    before = shipped.get(model.model_id)
    if before is not None and before.engine.name == model.engine.name and plan.installable(before):
        fallback = engines.engine_for(home, before)
        if carried(fallback, "miner"):
            return fallback, (f"{model.model_id}: the catalog pins the {model.engine.name} engine at "
                              f"{model.engine.revision[:12]}; this image carries {before.engine.revision[:12]} and "
                              "serves with it (validators check answers, not engines); the image of the release that "
                              "ships the new revision runs it")
    return None, (f"{model.model_id} runs on the {model.engine.name} engine at {model.engine.revision[:12]}, which "
                  "this image does not carry (the image of a newer release may)")


def install_engines(home: Path = ENGINES) -> None:
    """``verathos-container engines``, when the image is built: the engine of each active and candidate model of the
    shipped catalog, installed by the installer (``engines``) under ``home``, for references (float32) and for miners
    (bfloat16, with the fused kernels), as a reference machine installs both."""
    from verathos import catalog
    from verathos.install import engines, plan
    from verathos.install.common import info, step

    step("Engines of the shipped catalog")
    installs = {}
    for model in plan.servable_models(catalog.load()):
        if not plan.installable(model):
            info(f"{model.model_id}: this release does not install the {model.engine.name} engine")
            continue
        engine = engines.engine_for(home, model)
        installs.setdefault(engine.directory, engine)
    for engine in installs.values():
        info(f"{engine.engine.name} at {engine.engine.revision[:12]} in {engine.directory}")
        for purpose in ("reference", "miner"):
            engine.install(purpose)
            if not carried(engine, purpose):
                raise SystemExit(f"{engine.directory}: no {purpose} environment where it runs from")


# ---- the roles ----

def force_data(config: dict, data: Path, state: str, catalog_dir: bool = True) -> dict:
    """The state (and catalog) directory under ``data``: anything else in a container is lost when it is recreated."""
    config["state_dir"] = str(data / state)
    if catalog_dir:
        config["catalog_dir"] = str(data / "catalog")
    return config


def miner(env: Mapping[str, str], data: Path = DATA, home: Path = ENGINES) -> list[Process]:
    """Set the miner up as ``verathos-miner init --yes`` does, from ``env``; -> its engines, one per GPU, and the
    miner."""
    from pydantic import ValidationError

    from verathos import catalog
    from verathos.install import common, engines, host, plan
    from verathos.install import miner as setup
    from verathos.install.common import info, step, warn
    from verathos.miner.config import MinerConfig
    from verathos.protocol import tls
    from verathos.protocol.manifest import ENDPOINTS_PATH, axon_base, is_public
    from verathos.subnet.chain import current_catalog

    config_path = data / "miner.json"
    existing = common.read_json(config_path)
    record = common.Record.load(data, "miner")
    no_publish = flag(env, "NO_PUBLISH", False)
    role = "worker" if flag(env, "WORKER", False) else "primary"
    name, peers_value = value(env, "NAME"), value(env, "PEERS")
    if role == "worker" and peers_value:
        raise SystemExit("PEERS is set on the primary, WORKER on a worker: not both")
    if name and role != "worker":
        raise SystemExit("NAME names a worker's endpoints (WORKER=1)")
    firewall = flag(env, "FIREWALL", True)

    step("Wallet and chain")
    network, netuid = common.network_and_netuid(chain_options(env), existing)
    info(f"network {network}, netuid {netuid}")
    wallet_name, hotkey_name = wallet_names(env)
    hotkey = common.load_hotkey(wallet_name, hotkey_name).hotkey.ss58_address
    st = common.connect(network)
    uid = st.get_uid_for_hotkey_on_subnet(hotkey, netuid)
    if uid is not None:
        info(f"registered on netuid {netuid} as UID {uid}")
    elif no_publish:
        warn(f"not registered on netuid {netuid}; with NO_PUBLISH the miner serves but publishes nothing")
    else:
        raise SystemExit(f"hotkey {hotkey} is not registered on netuid {netuid}. Register it on the machine holding "
                         f"the coldkey:\n  {common.register_command(network, netuid, wallet_name, hotkey_name)}\nTo "
                         "test this machine first, without registering, set NO_PUBLISH=1.")
    shipped = catalog.load()
    try:
        cat = shipped if no_publish else current_catalog(st, netuid, catalog.CatalogStore(data / "catalog"),
                                                         existing.get("catalog_mirrors", []))
    except catalog.CatalogUnavailable as exc:
        raise SystemExit(f"the subnet's catalog is not available: {exc}") from None
    info(f"catalog {cat.revision} ({cat.digest()[:12]}){' as shipped' if no_publish else ''}")

    step("GPUs and models")
    gpus = from_options(lambda: setup.select_gpus(host.gpus(), value(env, "GPUS")))
    if not gpus:
        raise SystemExit("no NVIDIA GPU in the container (nvidia-smi): start it with the GPUs (the compose file's "
                         "deploy.resources) on a host with the NVIDIA driver and the NVIDIA Container Toolkit")
    driver = host.driver_cuda_version()
    if driver is not None and driver < setup.MIN_DRIVER_CUDA:
        warn(f"the NVIDIA driver supports CUDA {driver[0]}.{driver[1]}; the engines need "
             f"{setup.MIN_DRIVER_CUDA[0]}.{setup.MIN_DRIVER_CUDA[1]} or newer: update the host's driver")
    for g in gpus:
        info(g.describe())
    served = plan.servable_models(cat)
    serving = {}
    for m in served:
        engine, line = miner_engine(home, m, shipped)
        if engine is not None:
            serving[m.model_id] = engine
        if line:
            (warn if engine is not None else info)(line)
    models = {m.model_id: m for m in served if m.model_id in serving}
    if not models:
        raise SystemExit("no catalog model this image serves: run the image of the newest release")
    scored = {m for m in models if models[m].status == "active"}
    try:
        need = {m: plan.miner_memory(engines.parameters(models[m]), models[m].engine.name) for m in models}
    except plan.PlanError as exc:
        raise SystemExit(str(exc)) from None
    info(plan.offer_line(list(models), need, scored))
    previous = {e["gpu"]: e["model"] for e in record.engines.values()}
    manual = {e["gpu"] for e in record.engines.values() if e.get("manual", e["model"] not in scored)}
    market, why = setup.read_market(st, netuid, hotkey, set(record.engines), cat)
    info(plan.market_line(market) if market else f"choosing models without the network's numbers: {why}; each GPU "
         "keeps the model it served before, else takes the first active model that fits")
    state_dir = data / "miner"
    measured = setup.measured_capacity(state_dir, record.engines, cat.audit_policy().capacity_margin)
    choices = from_options(lambda: setup.choose_models(
        common.Prompter(assume_yes=True), gpus, list(models), need, plan.parse_models(value(env, "MODEL")), previous,
        scored, manual=manual, market=market, measured=measured,
        relative={m: models[m].relative_speed for m in models}))
    chosen = {g: c.model for g, c in choices.items()}

    step("Network")
    public_ip = value(env, "PUBLIC_IP") or existing.get("public_host") or host.detect_public_ip() or ""
    if not is_public(public_ip):
        raise SystemExit(f"PUBLIC_IP: {public_ip or 'none detected'} is not a public IP address; set the address "
                         "validators reach this machine at")
    port = port_number(env, "PORT") or existing.get("port") or setup.DEFAULT_PORT
    if not host.port_free(port):
        raise SystemExit(f"port {port} is in use on this machine: choose another PORT")
    peers = from_options(lambda: setup.parse_peers(peers_value or "")) if role == "primary" else []
    prefix = from_options(lambda: setup.endpoint_prefix(role, name, public_ip, list(record.engines)))
    endpoint_ids = {g.index: f"{prefix}gpu{g.index}" for g in gpus}
    engine_ports = plan.pick_ports(list(endpoint_ids.values()), {k: v["port"] for k, v in record.engines.items()},
                                   setup.ENGINE_PORT, lambda p: p != port and host.port_free(p, "127.0.0.1"))

    step("Plan")
    for g in gpus:
        info(f"{g.describe()}: {chosen[g.index]}, endpoint {endpoint_ids[g.index]}, engine on "
             f"127.0.0.1:{engine_ports[endpoint_ids[g.index]]}")
    for line in plan.choice_lines(gpus, choices, list(models), need, scored, market.shares if market else None):
        info(named(line))
    info(f"miner at {public_ip}:{port} with hotkey {wallet_name}/{hotkey_name}, "
         + ("a worker: its primary lists its endpoints" if role == "worker" else "the primary")
         + (" (testing: nothing written on chain)" if no_publish else ""))
    if peers:
        info(f"workers whose endpoints the manifest lists: {', '.join(peers)}")
    info("firewall on: the miner's port answers only the network's callers and this hotkey's machines once the "
         "catalog has miners enforce it; the miner says at its start whether the kernel drops the others too (host "
         "network and NET_ADMIN) or its own process alone" if firewall else
         "firewall off: anyone reaches the miner's port (FIREWALL=on turns it on)")
    info("self-update off: a release is a new image")

    step("Weights")
    hf_home = data / "hf"
    for model_id in sorted(set(chosen.values())):
        engines.download(serving[model_id].python("miner"), models[model_id], hf_home)

    step("Configuration")
    config = setup.miner_config(existing, network=network, netuid=netuid, wallet_name=wallet_name,
                                hotkey_name=hotkey_name, public_ip=public_ip, port=port,
                                endpoints={eid: (chosen[gpu], engine_ports[eid]) for gpu, eid in endpoint_ids.items()},
                                managed_before=set(record.engines), home=data, role=role, peers=peers,
                                auto_update=False, firewall=firewall)
    force_data(config, data, "miner")
    try:
        MinerConfig.model_validate(config)
    except ValidationError as exc:
        raise common.invalid(config_path, exc) from None
    common.write_json(config_path, config)
    info(f"wrote {config_path}")
    identity = tls.identity(state_dir)
    info(f"TLS certificate {identity.cert_path}, pin {identity.pin}: the manifest names it, and validators, the "
         f"auditor and gateways accept this machine by it (it stays while {state_dir} does)")
    record.config, record.paths = str(config_path), []
    record.engines = {eid: {"gpu": g, "model": chosen[g], "port": engine_ports[eid],
                            "manual": choices[g].reason in plan.BY_HAND} for g, eid in endpoint_ids.items()}
    record.save(data)

    processes = []
    for g in gpus:
        model, engine = models[chosen[g.index]], serving[chosen[g.index]]
        processes.append(Process(f"engine-gpu{g.index}",
                                 engine.command("miner", model, engine_ports[endpoint_ids[g.index]]),
                                 engine.directory, engine.environment_variables("miner", f"cuda:{g.index}", hf_home)))
    processes.append(Process("miner", [str(Path(sys.prefix) / "bin" / "verathos-miner"), "--config", str(config_path)]
                             + (["--no-publish"] if no_publish else []), data))

    step("Running")
    address = axon_base(public_ip, port)
    info(", ".join(f"engine-gpu{g.index} {chosen[g.index]}" for g in gpus) + f"; miner {address}: it waits for its "
         "engines, measures its capacity (a few minutes), then "
         + ("serves its endpoints to its primary" if role == "worker" else "publishes its manifest"))
    info(f"check from another machine: curl -sk --pinnedpubkey '{tls.curl_pin(identity.pin)}' {address}"
         + (ENDPOINTS_PATH if role == "worker" else "/v1/manifest"))
    return processes


def validator(env: Mapping[str, str], data: Path = DATA, remote_dir: Path = REMOTE_ENGINES) -> Process:
    """Set the validator up as ``verathos-validator init --yes`` does, from ``env``: light (``MODE=light``), or full
    with its references on reference machines (their files in ``remote_dir``); -> the validator."""
    from pydantic import ValidationError

    from verathos import catalog
    from verathos.install import common, host
    from verathos.install import validator as setup
    from verathos.install.common import info, step, warn
    from verathos.subnet.chain import current_catalog
    from verathos.throughput import loadgen
    from verathos.validator.config import ValidatorConfig

    config_path = data / "validator.json"
    existing = common.read_json(config_path)
    mode = (value(env, "MODE") or "light").lower()
    if mode not in ("light", "full"):
        raise SystemExit(f"MODE={mode!r}: light or full")
    dry_run = flag(env, "DRY_RUN", False)

    step("Wallet and chain")
    network, netuid = common.network_and_netuid(chain_options(env), existing)
    info(f"network {network}, netuid {netuid}")
    wallet_name, hotkey_name = wallet_names(env)
    hotkey = common.load_hotkey(wallet_name, hotkey_name).hotkey.ss58_address
    st = common.connect(network)
    neuron = st.get_neuron_for_pubkey_and_subnet(hotkey, netuid)
    if neuron is None or neuron.is_null:
        message = (f"hotkey {hotkey} is not registered on netuid {netuid}. Register it on the machine holding the "
                   f"coldkey:\n  {common.register_command(network, netuid, wallet_name, hotkey_name)}")
        if not dry_run:
            raise SystemExit(message)
        warn(message.replace("\n", " "))
    elif not neuron.validator_permit:
        warn(f"UID {neuron.uid} has no validator permit yet: a validator sets weights, and miners answer it, only "
             "with one (it follows the stake)")
    else:
        info(f"registered on netuid {netuid} as UID {neuron.uid}, with a validator permit")
    owner = hotkey == st.get_subnet_owner_hotkey(netuid)
    if owner and mode == "light":
        raise SystemExit("this is the subnet owner's hotkey: its validator is the auditor and runs full "
                         "(compose.validator-full.yml)")
    try:
        cat = current_catalog(st, netuid, catalog.CatalogStore(data / "catalog"), existing.get("catalog_mirrors", []))
    except catalog.CatalogUnavailable as exc:
        raise SystemExit(f"the subnet's catalog is not available: {exc}") from None
    info(f"catalog {cat.revision} ({cat.digest()[:12]})")

    step("Plan")
    for line in setup.mode_lines(mode, cat, owner):
        info(line)
    remote = screens = egress = None
    if mode == "full":
        files = sorted(remote_dir.glob("*.json")) if remote_dir.is_dir() else []
        if not files:
            raise SystemExit("a full validator here takes its references from reference machines: put the file a "
                             f"reference machine writes (reference-engines.json) into {remote_dir} (REMOTE_ENGINES_DIR "
                             "in .env)")
        remote, screens = from_options(lambda: setup.read_remote_engines([str(p) for p in files]))
        if screens and not owner:
            info(f"the screens of {', '.join(sorted(screens))} are the auditor's: not configured here")
            screens = {}
        if missing := sorted(m.model_id for m in cat.models if m.status == "active" and m.model_id not in remote):
            raise SystemExit(f"no reference for the active models {', '.join(missing)} in {remote_dir}: serve them on "
                             "a reference machine and put its file there (references on this machine: the installer)")
        for line in asyncio.run(setup.remote_lines(remote, screens, cat)):
            info(line)
        egress = from_options(lambda: setup.egress_ips(argparse.Namespace(egress_ip=listed(env, "EGRESS_IPS") or None),
                                                       existing))
        if not egress:
            warn("no public egress address detected (set EGRESS_IPS): miners that admit callers by address would not "
                 "admit this validator")
        info(f"egress address(es) {', '.join(egress) or 'none'}: "
             + ("the catalog's auditor.egress names them" if owner else "published in this validator's chain "
                "commitment, for miners to admit its requests"))
        if owner and not loadgen.BINARY_PATH.exists():
            raise SystemExit(f"{loadgen.BINARY_PATH} is missing: the auditor runs an image with its load generator")
    info(f"validator with hotkey {wallet_name}/{hotkey_name}" + (", dry run: no weights set" if dry_run else ""))
    info("self-update off: a release is a new image")

    step("Configuration")
    config = setup.validator_config(existing, network=network, netuid=netuid, wallet_name=wallet_name,
                                    hotkey_name=hotkey_name, ports={}, dry_run=dry_run, managed_before=set(),
                                    home=data, mode=mode, reference_mode="resident" if mode == "full" else None,
                                    auto_update=False, egress_ips=egress, remote=remote, screens=screens)
    for section in ("references", "audit_screens"):  # this image runs no engine: the engines on other machines only
        config[section] = {m: e for m, e in config.get(section, {}).items() if setup.remote_entry(e)}
    if not config["audit_screens"]:
        del config["audit_screens"]
    public_ip = value(env, "PUBLIC_IP") or existing.get("public_host")
    if mode == "full" and owner and not public_ip:
        public_ip = host.detect_public_ip()
        if not public_ip:
            raise SystemExit("the auditor serves the catalog and its verdicts: set PUBLIC_IP, its public address")
    if mode == "full" and public_ip:
        config["public_host"] = public_ip
        config["port"] = port_number(env, "PORT") or existing.get("port") or 8091
    force_data(config, data, "validator")
    try:
        ValidatorConfig.model_validate(config)
    except ValidationError as exc:
        raise common.invalid(config_path, exc) from None
    keys = any(isinstance(e, dict) and e.get("api_key") for section in ("references", "audit_screens")
               for e in config.get(section, {}).values())
    common.write_json(config_path, config, private=keys)  # the keys of the engines on other machines
    info(f"wrote {config_path}" + (f"; its service at {public_ip}:{config['port']}" if mode == "full" and public_ip
                                   else ""))
    return Process("validator", [str(Path(sys.prefix) / "bin" / "verathos-validator"), "--config", str(config_path)],
                   data)


def reference(env: Mapping[str, str], data: Path = DATA, home: Path = ENGINES) -> list[Process]:
    """Set a reference machine up as ``verathos-reference init --yes`` does, from ``env``; -> its engines (each model's
    float32 reference and bfloat16 screen) and the front. Writes ``<data>/reference-engines.json``, the validators'
    file."""
    from pydantic import ValidationError

    from verathos import catalog
    from verathos.install import common, engines, host, plan
    from verathos.install import reference as setup
    from verathos.install.common import ENGINES_FILE, info, step, warn
    from verathos.protocol import tls
    from verathos.reference.config import REFERENCE, SCREEN, ReferenceConfig, api_key, caller_address, engines_document
    from verathos.subnet.chain import current_catalog

    config_path = data / "reference.json"
    existing = common.read_json(config_path)
    record = common.Record.load(data, setup.ROLE)
    if value(env, "ALLOW") is None:
        raise SystemExit("ALLOW is not set: the addresses of the validator machines that use these engines (.env)")
    callers = from_options(lambda: setup.parse_callers(value(env, "ALLOW")))
    screens = flag(env, "SCREENS", True)
    device = (value(env, "DEVICE") or "auto").lower()

    step("Catalog")
    network, netuid = common.network_and_netuid(chain_options(env), existing)
    info(f"network {network}, netuid {netuid}")
    st = common.connect(network)
    try:
        cat = current_catalog(st, netuid, catalog.CatalogStore(data / "catalog"), [])
    except catalog.CatalogUnavailable as exc:
        raise SystemExit(f"the subnet's catalog is not available: {exc}") from None
    info(f"catalog {cat.revision} ({cat.digest()[:12]}): the engines serve exactly the revisions it pins")

    step("Models and memory")
    models = {}
    for m in from_options(lambda: setup.choose_models(cat, plan.parse_models(value(env, "MODEL")), None)):
        engine = engines.engine_for(home, cat.get(m))
        if all(carried(engine, purpose) for purpose in (("reference", "miner") if screens else ("reference",))):
            models[m] = cat.get(m)
        else:  # references and screens run exactly what the catalog pins
            warn(f"{m} is not served: the catalog pins the {engine.engine.name} engine at "
                 f"{engine.engine.revision[:12]}, which this image does not carry; run the image of the release that "
                 "ships it")
    if not models:
        raise SystemExit("no model this image serves as the catalog pins it: run the image of the newest release")
    gpus = host.gpus()
    ram_total, ram_free = host.memory_available()
    info(f"RAM {ram_total / host.GiB:.0f} GiB ({ram_free / host.GiB:.0f} free)"
         + "".join(f"; {g.describe()} ({g.memory_free / host.GiB:.0f} free)" for g in gpus))
    if not gpus:
        warn("no NVIDIA GPU in the container (nvidia-smi): the references run in RAM, and there are no screens")
    try:
        parameters = {m: engines.parameters(models[m]) for m in models}
        # nothing runs yet in a new container: every engine is placed in the memory free now
        placement = plan.place_engines(parameters, {m: models[m].engine.name for m in models},
                                       list(models) if screens else [], {g.index: g.memory_free for g in gpus},
                                       ram_free, device, {})
    except plan.PlanError as exc:
        raise SystemExit(named(str(exc))) from None

    step("Network")
    try:
        public_ip = caller_address(value(env, "PUBLIC_IP") or existing.get("public_host") or host.detect_public_ip()
                                   or "")
    except ValueError as exc:
        raise SystemExit(f"PUBLIC_IP: {exc}; set the address the validators reach this machine at") from None
    port = port_number(env, "PORT") or existing.get("port") or setup.DEFAULT_PORT
    if not host.port_free(port):
        raise SystemExit(f"port {port} is in use on this machine: choose another PORT")
    ports = plan.pick_ports(sorted(placement), {i: e["port"] for i, e in record.engines.items()}, setup.ENGINE_PORT,
                            lambda p: p != port and host.port_free(p, "127.0.0.1"))

    step("Plan")
    for item in sorted(placement):
        kind, model = item.split("/", 1)
        need = (plan.reference_memory(parameters[model], "cpu" if placement[item] == "cpu" else "gpu")
                if kind == REFERENCE else plan.miner_memory(parameters[model], models[model].engine.name))
        info(f"{model} {kind}: {setup.PRECISION[kind]} on {setup.describe(placement[item])} (about "
             f"{need / host.GiB:.0f} GiB), 127.0.0.1:{ports[item]}")
    for model in models:
        if screens and plan.engine_key(SCREEN, model) not in placement:
            info(f"{model} screen: no GPU has room left; the reference recomputes every check of the model itself")
    info(f"front https://{'[' + public_ip + ']' if ':' in public_ip else public_ip}:{port} (TLS), answering only "
         f"{', '.join(callers)} (every other address is dropped) and only with the key")

    step("Weights")
    hf_home = data / "hf"
    for model in sorted(models):
        engines.download(engines.engine_for(home, models[model]).python("reference"), models[model], hf_home)

    step("Configuration")
    config = force_data(setup.reference_config(existing, network=network, netuid=netuid, public_ip=public_ip, port=port,
                                               callers=callers, ports=ports, home=data), data, setup.ROLE,
                        catalog_dir=False)
    try:
        parsed = ReferenceConfig.model_validate(config)
    except ValidationError as exc:
        raise common.invalid(config_path, exc) from None
    common.write_json(config_path, config)
    info(f"wrote {config_path}")
    identity, key = tls.identity(parsed.state_dir), api_key(parsed.state_dir)
    document = engines_document(parsed, identity.pin, key)
    engines_path = data / ENGINES_FILE
    common.write_json(engines_path, document, private=True)
    record.config, record.paths = str(config_path), []
    record.engines = {item: {"model": item.split("/", 1)[1], "kind": item.split("/", 1)[0],
                             "device": placement[item], "port": ports[item]} for item in placement}
    record.options = {"screens": screens}
    record.save(data)

    processes = []
    for item in sorted(placement):
        kind, model = item.split("/", 1)
        engine, purpose = engines.engine_for(home, models[model]), setup.PURPOSE[kind]
        processes.append(Process(f"{kind}-{model}", engine.command(purpose, models[model], ports[item]),
                                 engine.directory, engine.environment_variables(purpose, placement[item], hf_home)))
    processes.append(Process("front", [str(Path(sys.prefix) / "bin" / setup.FRONT), "--config", str(config_path)],
                             data))

    step("For the validators")
    for line in setup.engines_lines(document):
        info(line)
    info(f"TLS pin {identity.pin}. {engines_path} holds the engines' URLs, the pin and the key (keep it private): copy "
         "it into the REMOTE_ENGINES_DIR of each validator machine, then start the validator there "
         "(compose.validator-full.yml)")
    return processes


# ---- the health check ----

def health(role: str, data: Path = DATA, timeout_s: float = 10.0) -> bool:
    """The container's health: its service answers on loopback, which the firewalls always pass. A miner its
    ``GET /health``; a reference machine every engine through its front, with the callers' key; a validator its
    service if it runs one (otherwise the validator's process is the container's)."""
    import httpx

    def get(url: str, headers: dict[str, str] | None = None) -> httpx.Response:
        return httpx.get(url, headers=headers, verify=False, timeout=timeout_s)  # loopback: no pin to check

    try:
        config = json.loads((data / f"{role}.json").read_text())
        if role == "miner":
            return get(f"https://127.0.0.1:{config['port']}/health").status_code == 200
        if role == "reference":
            key = (Path(config.get("state_dir", data / "reference")) / "api.key").read_text().strip()
            urls = [f"https://127.0.0.1:{config['port']}/{kind}/{model}/v1/models"
                    for model, kinds in config["engines"].items() for kind in kinds]
            return all(get(url, {"Authorization": f"Bearer {key}"}).status_code == 200 for url in urls)
        if config.get("mode") == "full" and config.get("public_host"):
            get(f"https://127.0.0.1:{config.get('port', 8091)}/v1/network")  # any answer: the service runs
        return True
    except (OSError, ValueError, KeyError, TypeError, httpx.HTTPError):
        return False


# ---- running ----

def run(role: str, env: Mapping[str, str], data: Path, engines_home: Path, remote_dir: Path,
        sleep: Callable[[float], None] = time.sleep) -> int:
    """Set the role up and run it. A setup that stops says why and waits ``SETUP_RETRY_S`` before the container exits,
    so that Docker's restart tries it again about once a minute."""
    try:
        if role == "validator":
            p = validator(env, data, remote_dir)
            sys.stdout.flush()
            os.chdir(p.directory)
            os.execv(p.command[0], p.command)
        processes = miner(env, data, engines_home) if role == "miner" else reference(env, data, engines_home)
    except SystemExit as exc:
        if not isinstance(exc.code, str):
            raise
        print(f"\nverathos-container {role}: {exc.code}", file=sys.stderr, flush=True)
        sleep(SETUP_RETRY_S)
        return 1
    except Exception:
        traceback.print_exc()
        print(f"\nverathos-container {role}: the setup failed; trying again in {SETUP_RETRY_S:.0f} s", file=sys.stderr,
              flush=True)
        sleep(SETUP_RETRY_S)
        return 1
    sys.stdout.flush()
    return Supervisor(processes).run()


def main(argv: list[str] | None = None, environ: Mapping[str, str] | None = None) -> None:
    env = os.environ if environ is None else environ
    parser = argparse.ArgumentParser(prog="verathos-container", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=[*ROLES, "health", "engines"])
    parser.add_argument("role", nargs="?", choices=ROLES, help="health: the role whose health to check")
    args = parser.parse_args(argv)
    data = Path(env.get("VERATHOS_DATA") or DATA)
    engines_home = Path(env.get("VERATHOS_ENGINES") or ENGINES)
    if args.command == "health":
        if args.role is None:
            parser.error("health takes the role")
        sys.exit(0 if health(args.role, data) else 1)
    if args.command == "engines":
        install_engines(engines_home)
        return
    data.mkdir(parents=True, exist_ok=True)
    sys.exit(run(args.command, env, data, engines_home, Path(env.get("REMOTE_ENGINES") or REMOTE_ENGINES)))


if __name__ == "__main__":
    main()
