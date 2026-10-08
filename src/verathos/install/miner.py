"""``verathos-miner init``: set up this machine as a miner.

One engine per GPU serving a catalog model, the miner service in front of them on one public port, both as system
services. The miner measures its endpoints' capacity when it starts (or reuses the stored measurement) and then
publishes its manifest; the setup shows that, and how to follow it afterwards.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from verathos import catalog
from verathos.install import common, engines, host, plan
from verathos.install.common import Prompter, info, step, warn
from verathos.install.services import Service
from verathos.miner.config import MinerConfig
from verathos.subnet.chain import current_catalog

DEFAULT_PORT = 8091
ENGINE_PORT = 8008
MINER = "verathos-miner"
ENGINE_READY_S = 1800.0
MEASURE_S = 2400.0
MIN_DRIVER_CUDA = (12, 8)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="verathos-miner init", description="Set up this machine as a miner: one engine "
                                "per GPU, the miner service, and system services for both. Run it again to update; "
                                "the previous answers are the defaults.")
    common.common_arguments(p)
    p.add_argument("--public-ip", help="Public IP address validators reach the miner at (default: detected)")
    p.add_argument("--port", type=int, help=f"Public TCP port of the miner (default {DEFAULT_PORT})")
    p.add_argument("--gpus", help="GPUs to serve on, comma-separated nvidia-smi indices (default: all)")
    p.add_argument("--model", help="Model for every GPU, or one per GPU comma-separated (default: the first active "
                   "catalog model that fits)")
    p.add_argument("--no-publish", action="store_true", help="Test the machine: the miner uses the shipped catalog "
                   "and writes nothing on chain, so the hotkey need not be registered")
    p.add_argument("--config", type=Path, help="Miner configuration (default <home>/miner.json)")
    return p


def select_gpus(gpus: list[host.Gpu], value: str | None) -> list[host.Gpu]:
    if not value:
        return gpus
    wanted = {int(v) for v in value.split(",") if v.strip()}
    unknown = wanted - {g.index for g in gpus}
    if unknown:
        raise SystemExit(f"no GPU {', '.join(map(str, sorted(unknown)))}; found "
                         f"{', '.join(str(g.index) for g in gpus)}")
    return [g for g in gpus if g.index in wanted]


def choose_models(ui: Prompter, gpus: list[host.Gpu], offered: list[str], need: dict[str, int], requested: list[str],
                  previous: dict[int, str]) -> dict[int, str]:
    """The model per GPU: the defaults (or ``requested``), asked once per kind of GPU on a terminal."""
    try:
        chosen = plan.choose_models(gpus, offered, need, requested, previous)
    except plan.PlanError as exc:
        raise SystemExit(str(exc)) from None
    if requested or not ui.interactive:
        return chosen
    kinds: dict[tuple[str, int], list[host.Gpu]] = {}
    for g in gpus:
        kinds.setdefault((g.name, g.memory_total), []).append(g)
    for group in kinds.values():
        label = group[0].describe() if len(group) == 1 else \
            f"the {len(group)} {group[0].name} GPUs ({', '.join(str(g.index) for g in group)})"
        model = ui.choose(f"Model for {label}", plan.fitting(offered, need, group[0]), chosen[group[0].index])
        for g in group:
            chosen[g.index] = model
    return chosen


def miner_config(existing: dict, *, network: str, netuid: int, wallet_name: str, hotkey_name: str, public_ip: str,
                 port: int, endpoints: dict[str, tuple[str, int]], managed_before: set[str], home: Path) -> dict:
    """The configuration to write: ``existing`` with this setup's answers and its endpoints (endpoint id -> model,
    engine port). Endpoints added by hand and settings the setup does not ask about are kept; endpoints it set up
    before and no longer serves are dropped."""
    old = {e["endpoint_id"]: e for e in existing.get("endpoints", [])}
    listed = [{**old.get(eid, {}), "endpoint_id": eid, "model_id": model, "engine_url": f"http://127.0.0.1:{p}"}
              for eid, (model, p) in endpoints.items()]
    listed += [e for eid, e in old.items() if eid not in endpoints and eid not in managed_before]
    config = {**existing, "network": network, "netuid": netuid, "wallet_name": wallet_name,
              "hotkey_name": hotkey_name, "public_host": public_ip, "port": port, "endpoints": listed}
    config.setdefault("state_dir", str(home / "miner"))
    config.setdefault("catalog_dir", str(home / "catalog"))
    return config


def init(argv: list[str]) -> None:
    args = parser().parse_args(argv)
    ui = Prompter(args.yes)
    home = args.home.expanduser()
    record = common.Record.load(home, "miner")
    config_path = (args.config or Path(record.config or home / "miner.json")).expanduser().absolute()
    existing = common.read_json(config_path)
    manager = common.manager_for(args, record)

    step("Wallet and chain")
    network, netuid = common.network_and_netuid(args, existing)
    info(f"network {network}, netuid {netuid}")
    wallet_name, hotkey_name = common.choose_wallet(ui, args, existing)
    wallet = common.load_hotkey(wallet_name, hotkey_name)
    st = common.connect(network)
    uid = st.get_uid_for_hotkey_on_subnet(wallet.hotkey.ss58_address, netuid)
    if uid is not None:
        info(f"registered on netuid {netuid} as UID {uid}")
    elif args.no_publish:
        warn(f"not registered on netuid {netuid}; with --no-publish the miner serves but publishes nothing")
    else:
        raise SystemExit(
            f"hotkey {wallet.hotkey.ss58_address} is not registered on netuid {netuid}. Register it on the machine "
            f"holding the coldkey:\n  {common.register_command(network, netuid, wallet_name, hotkey_name)}\nthen run "
            "this again. To test this machine first, without registering, add --no-publish.")
    store = catalog.CatalogStore(Path(existing.get("catalog_dir", home / "catalog")))
    try:
        cat = catalog.load() if args.no_publish else current_catalog(st, netuid, store,
                                                                      existing.get("catalog_mirrors", []))
    except catalog.CatalogUnavailable as exc:
        raise SystemExit(f"the subnet's catalog is not available: {exc}") from None
    info(f"catalog {cat.revision} ({cat.digest()[:12]}){' as shipped' if args.no_publish else ''}")

    step("GPUs and models")
    gpus = select_gpus(host.gpus(), args.gpus)
    if not gpus:
        raise SystemExit("no NVIDIA GPU found (nvidia-smi): a miner serves its models on GPUs; install the NVIDIA "
                         "driver first")
    driver = host.driver_cuda_version()
    if driver is not None and driver < MIN_DRIVER_CUDA:
        warn(f"the NVIDIA driver supports CUDA {driver[0]}.{driver[1]}; the engines need {MIN_DRIVER_CUDA[0]}."
             f"{MIN_DRIVER_CUDA[1]} or newer: update the driver")
    for g in gpus:
        info(g.describe())
    active = plan.servable_models(cat)
    models = {m.model_id: m for m in active if plan.installable(m)}
    for m in active:
        if m.model_id not in models:
            info(f"{m.model_id} runs on the {m.engine.name} engine, which this setup does not install")
    if not models:
        raise SystemExit("no active catalog model this setup can install")
    try:
        need = {m: plan.miner_memory(engines.parameters(models[m])) for m in models}
    except plan.PlanError as exc:
        raise SystemExit(str(exc)) from None
    previous = {e["gpu"]: e["model"] for e in record.engines.values()}
    chosen = choose_models(ui, gpus, list(models), need, plan.parse_models(args.model), previous)

    step("Network")
    detected = args.public_ip or existing.get("public_host") or host.detect_public_ip() or ""
    public_ip = ui.ask("Public IP address of this machine", detected,
                       lambda v: None if host.is_public(v) else "not a public IP address (pass --public-ip)")
    miner_running = manager is not None and MINER in record.services and manager.active(MINER)

    def port_problem(value: str) -> str | None:
        if not value.isdigit() or not 0 < int(value) < 65536:
            return "a TCP port number"
        if int(value) == existing.get("port") and miner_running:
            return None
        return None if host.port_free(int(value)) else f"port {value} is in use on this machine"

    port = int(ui.ask("Public TCP port of the miner (open it for inbound TCP)",
                      str(args.port or existing.get("port") or DEFAULT_PORT), port_problem))
    endpoint_ids = {g.index: f"gpu{g.index}" for g in gpus}
    engine_ports = plan.pick_ports(list(endpoint_ids.values()),
                                   {k: v["port"] for k, v in record.engines.items()}, ENGINE_PORT,
                                   lambda p: p != port and host.port_free(p, "127.0.0.1"))

    step("Plan")
    for g in gpus:
        info(f"{g.describe()}: {chosen[g.index]}, engine on 127.0.0.1:{engine_ports[endpoint_ids[g.index]]}")
    info(f"miner at {public_ip}:{port} with hotkey {wallet_name}/{hotkey_name}"
         + (" (testing: nothing written on chain)" if args.no_publish else ""))
    info(f"configuration {config_path}; engines and weights in {home}")
    if not ui.confirm("Go ahead?"):
        raise SystemExit("stopped; nothing changed")

    step("Engines")
    kev = {}
    for model_id in sorted(set(chosen.values())):
        engine = engines.engine_for(home, models[model_id])
        kev.setdefault(engine.directory, engine)
    for engine in kev.values():
        info(f"{engine.engine.name} at {engine.engine.revision[:12]} in {engine.directory}")
        engine.install("miner")

    step("Weights")
    hf_home = home / "hf"
    for model_id in sorted(set(chosen.values())):
        engines.download(engines.engine_for(home, models[model_id]).python("miner"), models[model_id], hf_home)

    step("Configuration")
    config = miner_config(existing, network=network, netuid=netuid, wallet_name=wallet_name,
                          hotkey_name=hotkey_name, public_ip=public_ip, port=port,
                          endpoints={eid: (chosen[gpu], engine_ports[eid]) for gpu, eid in endpoint_ids.items()},
                          managed_before=set(record.engines), home=home)
    MinerConfig.model_validate(config)
    common.write_json(config_path, config)
    info(f"wrote {config_path}")

    engine_services = {}
    for g in gpus:
        eid = endpoint_ids[g.index]
        model = models[chosen[g.index]]
        engine = engines.engine_for(home, model)
        engine_services[eid] = Service(
            name=f"verathos-engine-{eid}", description=f"Verathos engine {eid}: {model.model_id} on GPU {g.index}",
            command=engine.command("miner", model, engine_ports[eid]), directory=engine.directory,
            user=common.operator(),
            environment={"HOME": str(Path.home()),
                         **engine.environment_variables("miner", f"cuda:{g.index}", hf_home)})
    miner_bin = Path(sys.prefix) / "bin" / "verathos-miner"
    miner = Service(name=MINER, description="Verathos miner", command=[str(miner_bin), "--config", str(config_path)]
                    + (["--no-publish"] if args.no_publish else []), directory=home, user=common.operator(),
                    environment={"HOME": str(Path.home())},
                    after=tuple(s.name for s in engine_services.values()))
    stale = [f"verathos-engine-{eid}" for eid in record.engines if eid not in engine_ports]

    record.config = str(config_path)
    record.engines = {eid: {"gpu": g, "model": chosen[g], "port": engine_ports[eid]}
                      for g, eid in endpoint_ids.items()}
    record.paths = sorted(str(d) for d in kev)

    if manager is None:
        record.save(home)
        step("Run (no services installed)")
        info("start each engine, then the miner, under your supervisor:")
        for s in [*engine_services.values(), miner]:
            env = " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted(s.environment.items()) if k != "HOME")
            print(f"      cd {shlex.quote(str(s.directory))} && {env + ' ' if env else ''}{shlex.join(s.command)}")
        info("the miner measures its capacity when it starts (a few minutes), then publishes its manifest")
        return

    step("Services")
    names = [s.name for s in engine_services.values()]
    for name in [*names, MINER]:
        manager.mark(name)
    if stale:
        manager.remove(stale)
    changed = manager.apply([*engine_services.values(), miner], restart={MINER})
    record.manager, record.services = manager.name, [*names, MINER]
    record.save(home)
    info(f"{manager.name}: {', '.join(names + [MINER])}" + (f" (new or changed: {', '.join(sorted(changed))})"
                                                             if changed else ""))
    info("waiting for the engines to load their models")
    common.wait_engines(manager, {f"verathos-engine-{eid}": p for eid, p in engine_ports.items()}, ENGINE_READY_S)

    step("Capacity")
    info("the miner measures its endpoints' capacity (or reuses the stored measurement), then serves")
    common.follow(manager, MINER, done=lambda line: " endpoints on port " in line and "serving" in line,
                  failed=lambda line: "no endpoint could be measured" in line or "no configured endpoint" in line,
                  timeout_s=MEASURE_S)

    step("Done")
    for g in gpus:
        eid = endpoint_ids[g.index]
        info(f"endpoint {eid}: {chosen[g.index]} on GPU {g.index}, service verathos-engine-{eid}")
    info(f"miner: http://{public_ip}:{port}, service {MINER}"
         + (" (testing: nothing published on chain)" if args.no_publish else ""))
    info(f"configuration: {config_path}; state: {config['state_dir']}")
    info(f"status:  {manager.status_command(record.services)}")
    info(f"logs:    {manager.log_command(MINER)}")
    info(f"after editing the configuration: {manager.restart_command([MINER])}")
    info(f"check from another machine: curl http://{public_ip}:{port}/v1/manifest")
    script = Path(catalog.__file__).parents[3] / "scripts" / "install.sh"
    if script.exists():
        again = " --no-publish" if args.no_publish else ""
        info(f"update: bash {script} miner --yes{again}      uninstall: bash {script} uninstall")
