"""``verathos-reference init``: set up this machine as a reference machine, which lends its GPUs to validators on other
machines.

Per model, its float32 reference and its bfloat16 screen (the miners' engine), placed as a validator places its
references (``plan.place_engines``), each a service on this machine's loopback; and the front (``verathos.reference``),
which serves them over TLS to the callers' addresses only, with a bearer key. The setup ends with the file a
validator's setup takes (``verathos-validator init --remote-engines``): per model the engines' URLs, the front's pin
and the key. Run it again to update: the front keeps its certificate (so its pin) and its key.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
import time
from pathlib import Path

from pydantic import ValidationError

from verathos import catalog
from verathos.install import common, engines, host, plan
from verathos.install.common import ENGINES_FILE, Prompter, info, step, warn
from verathos.install.services import Service
from verathos.miner.engine import HttpEngine
from verathos.protocol import tls
from verathos.reference.config import REFERENCE, SCREEN, ReferenceConfig, api_key, caller_address, engines_document
from verathos.subnet.chain import current_catalog
from verathos.validator.references import served_code, served_revisions

ROLE = "reference"
FRONT = "verathos-reference"
DEFAULT_PORT = 8443
ENGINE_PORT = 8101
READY_S = 2700.0  # float32 on CPU loads for several minutes
FRONT_READY_S = 120.0
PRECISION = {REFERENCE: "float32", SCREEN: "bfloat16"}
PURPOSE = {REFERENCE: "reference", SCREEN: "miner"}  # the screen is the miners' engine


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="verathos-reference init", description="Set up this machine to serve float32 "
                                "reference engines and bfloat16 screens to validators on other machines, through one "
                                "TLS front that answers only their addresses and their key. Run it again to update; "
                                "the previous answers are the defaults.")
    common.common_arguments(p, wallet=False)
    p.add_argument("--allow", help="The callers' IP addresses, comma-separated: every validator machine that uses "
                   "these engines (replaces the list; required the first time)")
    p.add_argument("--model", help="The models to serve, comma-separated (default: the catalog's active models, or the "
                   f"ones named before; {plan.AUTO}: the active models again)")
    p.add_argument("--screens", action=argparse.BooleanOptionalAction, default=None, help="Also run each model's "
                   "bfloat16 screen, which the auditor checks answers on first, where a GPU has room (default: on; a "
                   "later run keeps the choice)")
    p.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto", help="Where the references run: on "
                   "GPUs while they fit, the smallest engines first, else in RAM (auto); only in RAM, leaving the GPUs "
                   "to the screens (cpu); only on GPUs, before the screens (gpu)")
    p.add_argument("--public-ip", help="IP address the callers reach this machine at (default: detected)")
    p.add_argument("--port", type=int, help=f"TCP port of the front (default {DEFAULT_PORT}; open it for inbound TCP)")
    p.add_argument("--config", type=Path, help="Configuration (default <home>/reference.json)")
    return p


def engine_service(item: str) -> str:
    """The service of an engine (``plan.engine_key``): ``verathos-reference-<model>`` or ``verathos-screen-<model>``."""
    kind, model = item.split("/", 1)
    return f"verathos-{kind}-" + re.sub(r"[^A-Za-z0-9-]", "-", model)


def describe(where: str) -> str:
    return "CPU" if where == "cpu" else f"GPU {where.split(':', 1)[1]}"


def parse_callers(value: str) -> list[str]:
    """``--allow``: IP addresses, comma-separated."""
    try:
        return list(dict.fromkeys(caller_address(v) for v in value.split(",") if v.strip()))
    except ValueError as exc:
        raise SystemExit(f"--allow: {exc}") from None


def choose_models(cat: catalog.Catalog, requested: list[str], before: list[str] | None) -> list[str]:
    """The models to serve: as requested, else as named before, else the catalog's active models."""
    if requested == [plan.AUTO]:
        requested = [m.model_id for m in cat.models if m.status == "active"]
    names = requested or before or [m.model_id for m in cat.models if m.status == "active"]
    for name in names:
        model = cat.get(name)
        if model is None or model.status not in plan.SERVED:
            raise SystemExit(f"{name} is not an active or candidate model of the catalog; choose from "
                             + ", ".join(m.model_id for m in plan.servable_models(cat)))
        if not plan.installable(model):
            raise SystemExit(f"{name} runs on the {model.engine.name} engine, which this setup does not install")
    return list(dict.fromkeys(names))


def reference_config(existing: dict, *, public_ip: str, port: int, callers: list[str], ports: dict[str, int],
                     home: Path) -> dict:
    """The configuration to write: ``existing`` with this setup's address, callers and engines (``plan.engine_key`` ->
    port on this machine's loopback)."""
    served: dict[str, dict[str, str]] = {}
    for item, engine_port in sorted(ports.items()):
        kind, model = item.split("/", 1)
        served.setdefault(model, {})[kind] = f"http://127.0.0.1:{engine_port}"
    config = {**existing, "public_host": public_ip, "port": port, "allow": callers, "engines": served}
    if any(":" in c for c in callers):  # an IPv6 caller: a socket for both families
        config["listen_host"] = "::"
    config.setdefault("state_dir", str(home / ROLE))
    return config


async def through_front(config: ReferenceConfig, pin: str, key: str, cat: catalog.Catalog,
                        timeout_s: float) -> dict[str, str]:
    """Per engine (``plan.engine_key``), what it serves through the front, asked as a caller asks (on this machine's
    loopback, which the front always answers): its revision, or why it does not answer, until each answers or
    ``timeout_s`` passes."""
    base = f"https://127.0.0.1:{config.port}"
    out, deadline = {}, time.monotonic() + timeout_s
    for kind, model in config.served():
        engine = HttpEngine(f"{base}/{kind}/{model}", key, timeout_s=30.0, tls_pin=pin)
        try:
            while (why := await engine.problem()) is not None and time.monotonic() < deadline:
                await asyncio.sleep(2.0)
            models = await engine.info() if why is None else None
        finally:
            await engine.close()
        pinned = cat.get(model).hf_revision if cat.get(model) is not None else ""
        out[plan.engine_key(kind, model)] = common.engine_state(why, served_revisions(models), pinned,
                                                                served_code(models))
    return out


def engines_lines(document: dict) -> list[str]:
    """Per model, the URLs of its engines on the front."""
    models = sorted(set(document["references"]) | set(document["audit_screens"]))
    return [f"{m}: " + ", ".join(f"{name} {section[m]['url']}" for name, section in
                                 (("reference", document["references"]), ("screen", document["audit_screens"]))
                                 if m in section) for m in models]


def init(argv: list[str]) -> None:
    args = parser().parse_args(argv)
    ui = Prompter(args.yes)
    home = args.home.expanduser().absolute()
    record = common.Record.load(home, ROLE)
    config_path = (args.config or Path(record.config or home / "reference.json")).expanduser().absolute()
    existing = common.read_json(config_path)
    if common.Record.load(home, "validator").services:
        raise SystemExit("this machine runs a validator: a reference machine lends its GPUs to validators on other "
                         "machines and runs none itself. Remove the validator first (verathos-validator uninstall), or "
                         "set this up on another machine")
    manager = common.manager_for(args, record)

    step("Catalog")
    network, netuid = common.network_and_netuid(args, record.options)
    info(f"network {network}, netuid {netuid}")
    st = common.connect(network)
    try:
        cat = current_catalog(st, netuid, catalog.CatalogStore(home / "catalog"), [])
    except catalog.CatalogUnavailable as exc:
        raise SystemExit(f"the subnet's catalog is not available: {exc}") from None
    info(f"catalog {cat.revision} ({cat.digest()[:12]}): the engines serve exactly the revisions it pins")

    step("Models and memory")
    requested = plan.parse_models(args.model)
    named = None if requested == [plan.AUTO] else requested or record.options.get("models")
    chosen = choose_models(cat, requested, record.options.get("models"))
    models = {m: cat.get(m) for m in chosen}
    screens = args.screens if args.screens is not None else bool(record.options.get("screens", True))
    wanted = {plan.engine_key(REFERENCE, m) for m in models} | (
        {plan.engine_key(SCREEN, m) for m in models} if screens else set())
    stale = [item for item in record.engines if item not in wanted]
    if stale and manager is not None:
        manager.remove([engine_service(item) for item in stale])  # their memory is free for the others
        info(f"removed {', '.join(engine_service(item) for item in stale)}, which this setup no longer runs")
        record.engines = {item: e for item, e in record.engines.items() if item not in stale}
        record.services = [s for s in record.services if s not in {engine_service(item) for item in stale}]
        record.save(home)
    gpus = host.gpus()
    ram_total, ram_free = host.memory_available()
    info(f"RAM {ram_total / host.GiB:.0f} GiB ({ram_free / host.GiB:.0f} free)"
         + "".join(f"; {g.describe()} ({g.memory_free / host.GiB:.0f} free)" for g in gpus))
    if not gpus:
        warn("no NVIDIA GPU found (nvidia-smi): the references run in RAM, and there are no screens")
    try:
        parameters = {m: engines.parameters(models[m]) for m in models}
        placement = plan.place_engines(parameters, {m: models[m].engine.name for m in models},
                                       list(models) if screens else [], {g.index: g.memory_free for g in gpus},
                                       ram_free, args.device, {i: e["device"] for i, e in record.engines.items()})
    except plan.PlanError as exc:
        raise SystemExit(str(exc)) from None

    step("Network")
    detected = args.public_ip or existing.get("public_host") or host.detect_public_ip() or ""

    def address_problem(value: str) -> str | None:
        try:
            caller_address(value)
        except ValueError as exc:
            return f"{exc} (pass --public-ip)"
        return None

    public_ip = caller_address(ui.ask("IP address the callers reach this machine at", detected, address_problem))
    front_running = manager is not None and FRONT in record.services and manager.active(FRONT)

    def port_problem(value: str) -> str | None:
        if not value.isdigit() or not 0 < int(value) < 65536:
            return "a TCP port number"
        if int(value) == existing.get("port") and front_running:
            return None
        return None if host.port_free(int(value)) else f"port {value} is in use on this machine"

    port = int(ui.ask("TCP port of the front (open it for inbound TCP)",
                      str(args.port or existing.get("port") or DEFAULT_PORT), port_problem))
    callers = parse_callers(args.allow) if args.allow is not None else list(existing.get("allow", []))
    if not callers:
        def callers_problem(value: str) -> str | None:
            try:
                return None if [caller_address(v) for v in value.split(",") if v.strip()] else \
                    "give each validator machine's address (--allow <ip>[,<ip>...])"
            except ValueError as exc:
                return str(exc)

        callers = parse_callers(ui.ask("The callers' IP addresses, comma-separated", "", callers_problem))
    ports = plan.pick_ports(sorted(placement), {i: e["port"] for i, e in record.engines.items()}, ENGINE_PORT,
                            lambda p: p != port and host.port_free(p, "127.0.0.1"))

    step("Plan")
    for item in sorted(placement):
        kind, model = item.split("/", 1)
        need = (plan.reference_memory(parameters[model], "cpu" if placement[item] == "cpu" else "gpu") if kind ==
                REFERENCE else plan.miner_memory(parameters[model], models[model].engine.name))
        info(f"{model} {kind}: {PRECISION[kind]} on {describe(placement[item])} (about {need / host.GiB:.0f} GiB), "
             f"127.0.0.1:{ports[item]}")
    for model in models:
        if screens and plan.engine_key(SCREEN, model) not in placement:
            info(f"{model} screen: no GPU has room left (it needs about "
                 f"{plan.miner_memory(parameters[model], models[model].engine.name) / host.GiB:.0f} GiB); the "
                 "reference recomputes every check of the model itself")
    if not screens:
        info("no screens (--screens runs them where a GPU has room)")
    info(f"front https://{'[' + public_ip + ']' if ':' in public_ip else public_ip}:{port} (TLS), answering only "
         f"{', '.join(callers)} (every other address is dropped) and only with the key")
    info("this machine holds no wallet and runs no validator: the validators at those addresses use its engines")
    info(f"configuration {config_path}; engines and weights in {home}; update it with the installer")
    if not ui.confirm("Go ahead?"):
        raise SystemExit("stopped; nothing changed")

    step("Engines")
    installs: dict[Path, tuple[engines.KevEngine | engines.ClefEngine, set[str]]] = {}
    for item in sorted(placement):
        kind, model = item.split("/", 1)
        engine = engines.engine_for(home, models[model])
        installs.setdefault(engine.directory, (engine, set()))[1].add(PURPOSE[kind])
    for engine, purposes in installs.values():
        info(f"{engine.engine.name} at {engine.engine.revision[:12]} in {engine.directory}")
        for purpose in sorted(purposes, reverse=True):  # reference, then miner
            engine.install(purpose)

    step("Weights")
    hf_home = home / "hf"
    for model in sorted(models):
        engines.download(engines.engine_for(home, models[model]).python("reference"), models[model], hf_home)

    step("Configuration")
    config = reference_config(existing, public_ip=public_ip, port=port, callers=callers, ports=ports, home=home)
    try:
        parsed = ReferenceConfig.model_validate(config)
    except ValidationError as exc:
        raise SystemExit(f"{config_path} would not be valid; fix or remove it:\n{exc}") from None
    common.write_json(config_path, config)
    info(f"wrote {config_path}")
    identity, key = tls.identity(parsed.state_dir), api_key(parsed.state_dir)
    info(f"TLS certificate {identity.cert_path}, pin {identity.pin}; the callers' key in "
         f"{parsed.state_dir.expanduser() / 'api.key'} (delete tls.key and tls.crt, or api.key, there to replace them)")
    document = engines_document(parsed, identity.pin, key)
    engines_path = home / ENGINES_FILE
    common.write_json(engines_path, document, private=True)
    info(f"wrote {engines_path}, for the validator machine")

    engine_services = {}
    for item in sorted(placement):
        kind, model = item.split("/", 1)
        engine = engines.engine_for(home, models[model])
        engine_services[item] = Service(
            name=engine_service(item), description=f"Verathos {model} {kind} ({PRECISION[kind]}, "
            f"{describe(placement[item])})", command=engine.command(PURPOSE[kind], models[model], ports[item]),
            directory=engine.directory, user=common.operator(),
            environment={"HOME": str(Path.home()),
                         **engine.environment_variables(PURPOSE[kind], placement[item], hf_home)})
    front = Service(name=FRONT, description="Verathos reference machine: the front of its engines",
                    command=[str(Path(sys.prefix) / "bin" / FRONT), "--config", str(config_path)], directory=home,
                    user=common.operator(), environment={"HOME": str(Path.home())},
                    after=tuple(s.name for s in engine_services.values()))
    run = [*engine_services.values(), front]
    record.config = str(config_path)
    record.engines = {}
    for item, service in engine_services.items():
        kind, model = item.split("/", 1)
        # the code it runs: verathos-engine this release's (its service restarts when that changes), the Kev engine
        # the revision the catalog pins
        code = service.environment.get("VERATHOS_ENGINE_CODE") or f"kev {models[model].engine.revision}"
        record.engines[item] = {"model": model, "kind": kind, "device": placement[item], "port": ports[item],
                                "code": code}
    record.paths = sorted(str(d) for d in installs)
    record.options = {"network": network, "netuid": netuid, "screens": screens, **({"models": named} if named else {})}

    if manager is None:
        previous = [s for s in record.services if s not in {r.name for r in run}]
        record.manager, record.services = None, [s.name for s in run]
        record.save(home)
        common.show_run("each engine and the front", run)
        if previous:
            info(f"stop the engines no longer run (pm2: pm2 delete {' '.join(previous)})")
        handover(document, engines_path, identity.pin, key)
        for line in common.script_commands(ROLE, home, " --no-service"):
            info(line)
        return

    step("Services")
    for name in (s.name for s in run):
        manager.mark(name)
    record.manager, record.services = manager.name, [s.name for s in run]
    record.save(home)  # before the services exist, so uninstall finds them whatever happens next
    changed = manager.apply(run, restart={FRONT})
    info(f"{manager.name}: {', '.join(record.services)}" + (f" (new or changed: {', '.join(sorted(changed))})"
                                                             if changed else ""))
    info("waiting for the engines to load their models (float32 on CPU takes several minutes)")
    common.wait_engines(manager, {engine_services[i].name: ports[i] for i in engine_services}, READY_S)

    step("Front")
    for item, line in asyncio.run(through_front(parsed, identity.pin, key, cat, FRONT_READY_S)).items():
        kind, model = item.split("/", 1)
        info(f"{model} {kind} through the front: {line}")

    step("Done")
    handover(document, engines_path, identity.pin, key)
    info(f"status:  {manager.status_command(record.services)}")
    info(f"logs:    {manager.log_command(FRONT)} (the front; each engine under its own name)")
    for line in common.script_commands(ROLE, home):
        info(line)


def handover(document: dict, path: Path, pin: str, key: str) -> None:
    """What the validator machine needs, and how it takes it."""
    for line in engines_lines(document):
        info(line)
    info(f"TLS pin {pin}, key {key}")
    info(f"all of it is in {path} (it holds the key: keep it private). Copy it to each validator machine, then run "
         "the installer there with:")
    print(f"      validator --full --remote-engines {path.name}", flush=True)
