"""``verathos-miner init``: set up this machine as a miner.

One engine per GPU serving a catalog model, the miner service in front of them on one public port, both as system
services. Each GPU serves the model that pays best on it by the auditor's signed network status
(``plan.choose_models``), unless ``--model`` names one. The miner measures its endpoints' capacity when it starts (or
reuses the stored measurement) and then publishes its manifest; the setup shows that, and how to follow it afterwards.
With GPUs on several machines, each is set up the same way: the workers (``--worker``) first, then the primary with
the workers' addresses (``--peers``).
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import re
import sys
import time
from collections.abc import Callable, Collection
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import ValidationError

from verathos import catalog
from verathos.install import common, engines, host, plan
from verathos.install import firewall as firewall_setup
from verathos.install.common import Prompter, info, step, warn
from verathos.install.services import Service
from verathos.miner.config import MinerConfig, peer_url
from verathos.protocol import tls
from verathos.protocol.http import get_json
from verathos.protocol.manifest import ENDPOINTS_PATH, axon_base, is_public
from verathos.protocol.status import MAX_NETWORK_BYTES, NETWORK_DOMAIN, NETWORK_PATH, NetworkStatus, unverified
from verathos.subnet.chain import axon_url, current_catalog, one_line
from verathos.throughput.measure import STORE_FILE, MeasurementStore
from verathos.throughput.policy import BLOCK_TIME_S

DEFAULT_PORT = 8091
ENGINE_PORT = 8008
MINER = "verathos-miner"
ENGINE_READY_S = 1800.0
MEASURE_S = 2400.0
MIN_DRIVER_CUDA = (12, 8)
NETWORK_MAX_AGE_EPOCHS = 2  # the auditor signs its network status once per epoch


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="verathos-miner init", description="Set up this machine as a miner: one engine "
                                "per GPU, the miner service, and system services for both. Run it again to update; "
                                "the previous answers are the defaults.")
    common.common_arguments(p)
    p.add_argument("--public-ip", help="Public IP address validators reach the miner at (default: detected)")
    p.add_argument("--port", type=int, help=f"Public TCP port of the miner (default {DEFAULT_PORT})")
    p.add_argument("--gpus", help="GPUs to serve on, comma-separated nvidia-smi indices (default: all)")
    p.add_argument("--model", help=f"Model for every GPU, or one per GPU in order, comma-separated; {plan.AUTO}: the "
                   "setup chooses (default: the active model that pays best on each GPU by the network's numbers, "
                   "or one named before; candidate models are served only when named here)")
    p.add_argument("--no-publish", action="store_true", help="Test the machine: the miner uses the shipped catalog "
                   "and writes nothing on chain, so the hotkey need not be registered")
    p.add_argument("--worker", action="store_true", help="This machine serves its GPUs for the hotkey's primary "
                   "machine, which lists them (its --peers); it writes nothing on chain")
    p.add_argument("--name", help="Worker only: the start of its endpoint ids, unique among the hotkey's machines "
                   "(default: from its public IP address)")
    p.add_argument("--peers", help="Primary only: the workers' public ip:port, comma-separated; replaces the list "
                   "('' for none)")
    p.add_argument("--firewall", action=argparse.BooleanOptionalAction, default=None, help="Drop connections to the "
                   "miner's port from anyone but the network's callers and the hotkey's machines (default: on; "
                   "--no-firewall turns it off and removes it; a later run keeps the choice)")
    p.add_argument("--config", type=Path, help="Miner configuration (default <home>/miner.json)")
    return p


def choose_role(ui: Prompter, args: argparse.Namespace, existing: dict) -> str:
    """primary or worker: as given (``--worker``; ``--peers`` names a primary), as configured before, or asked."""
    if args.worker and args.peers is not None:
        raise SystemExit("--peers is set on the primary, --worker on a worker: not both")
    if args.worker:
        return "worker"
    if args.peers is not None:
        return "primary"
    if existing.get("role"):
        return existing["role"]
    return ui.choose("Role of this machine (primary: publishes the hotkey's manifest; worker: serves its GPUs for a "
                     "primary)", ["primary", "worker"], "primary")


def parse_peers(value: str) -> list[str]:
    """``--peers``: public ip:port, comma-separated, as the miner's configuration writes them."""
    peers = []
    for item in (v for v in value.split(",") if v.strip()):
        try:
            peer = peer_url(item)
        except ValueError as exc:
            raise SystemExit(f"--peers: {exc}") from None
        if not is_public(urlsplit(peer).hostname or ""):
            raise SystemExit(f"--peers: {item.strip()} is not a public address; validators reach every machine of "
                             "the hotkey at its public address")
        peers.append(peer)
    return peers


def endpoint_prefix(role: str, name: str | None, public_ip: str, previous: list[str]) -> str:
    """What this machine's endpoint ids start with: nothing on the primary (``gpu0``, ...); on a worker its name, the
    one it had (``previous``: its endpoint ids before), or one from its public IP address, so ids stay unique among the
    hotkey's machines and the same across runs."""
    if role == "primary":
        return ""
    if name:
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,23}", name):
            raise SystemExit("--name: 1-24 characters of a-z, 0-9 and '-'")
        return name + "-"
    before = {m[1] for e in previous if (m := re.fullmatch(r"(.+-)gpu\d+", e))}
    if len(before) == 1:
        return before.pop()
    address = ipaddress.ip_address(public_ip)
    if address.version == 4:
        return str(address).replace(".", "-") + "-"
    return "v6-" + hashlib.sha256(str(address).encode()).hexdigest()[:8] + "-"


def select_gpus(gpus: list[host.Gpu], value: str | None) -> list[host.Gpu]:
    if not value:
        return gpus
    try:
        wanted = {int(v) for v in value.split(",") if v.strip()}
    except ValueError:
        raise SystemExit(f"--gpus takes GPU numbers as nvidia-smi shows them, like 0,1; not {value!r}") from None
    unknown = wanted - {g.index for g in gpus}
    if unknown:
        raise SystemExit(f"no GPU {', '.join(map(str, sorted(unknown)))}; found "
                         f"{', '.join(str(g.index) for g in gpus)}")
    return [g for g in gpus if g.index in wanted]


def read_market(st, netuid: int, hotkey: str, own: Collection[str], cat: catalog.Catalog,
                clock: Callable[[], float] = time.time) -> tuple[plan.Market | None, str]:
    """The market in the auditor's network status (``GET /v1/network`` on the subnet owner's axon) once it verifies:
    signed by the subnet owner hotkey for ``netuid`` within ``NETWORK_MAX_AGE_EPOCHS`` epochs of this machine's clock.
    Otherwise None and why. ``own``: the endpoint ids this machine served, left out of the capacity it competes
    with."""
    try:
        auditor = st.get_subnet_owner_hotkey(netuid)
        base = axon_url(st, netuid, auditor) if auditor else None
        if base is None:
            return None, "the subnet owner's validator serves no axon"
        document = get_json(base + NETWORK_PATH, limit=MAX_NETWORK_BYTES, timeout=15, tls=cat.auditor_pin())
    except Exception as exc:  # whatever stops the reading, the setup chooses without the network
        return None, f"the auditor's network status is not available ({one_line(exc)})"
    why = unverified(document, domain=NETWORK_DOMAIN, auditor=auditor, netuid=netuid, now_ms=int(clock() * 1000),
                     max_age_s=NETWORK_MAX_AGE_EPOCHS * cat.epoch_blocks * BLOCK_TIME_S)
    if why is None:
        try:
            return plan.network_market(NetworkStatus.model_validate(document), hotkey=hotkey, own=own,
                                       active=[m.model_id for m in cat.models if m.status == "active"],
                                       relative={m.model_id: m.relative_speed for m in cat.models}), ""
        except ValueError as exc:
            why = str(exc).splitlines()[0]
    return None, f"the auditor's network status is not used: {why}"


def measured_capacity(state_dir: Path, engines: dict[str, dict], margin: float) -> dict[int, tuple[str, float]]:
    """Per GPU index, the model this machine's miner measured on it and the requests/s it declares from that (its
    stored measurements, ``margin`` of them), for the endpoints the setup installed (``engines``)."""
    try:
        stored = MeasurementStore(state_dir / STORE_FILE).load()
    except (OSError, ValueError, TypeError):
        return {}
    return {engines[eid]["gpu"]: (m.model_id or engines[eid]["model"], m.measured_rps * margin)
            for eid, m in stored.items() if eid in engines}


def choose_models(ui: Prompter, gpus: list[host.Gpu], offered: list[str], need: dict[str, int], requested: list[str],
                  previous: dict[int, str], scored: set[str], **rule) -> dict[int, plan.Choice]:
    """The model per GPU (``plan.choose_models``, with ``rule``'s inputs): the setup's choice or ``requested``, asked
    once per kind of GPU on a terminal."""
    try:
        chosen = plan.choose_models(gpus, offered, need, requested, previous, scored, **rule)
        if requested or not ui.interactive:
            return chosen
        kinds: dict[tuple[str, int], list[host.Gpu]] = {}
        for g in gpus:
            kinds.setdefault((g.name, g.memory_total), []).append(g)
        answered = {}
        for group in kinds.values():
            label = group[0].describe() if len(group) == 1 else \
                f"the {len(group)} {group[0].name} GPUs ({', '.join(str(g.index) for g in group)})"
            setup = {chosen[g.index].model for g in group}
            default = setup.pop() if len(setup) == 1 else plan.AUTO
            if default == plan.AUTO:
                label += f" ({plan.AUTO}: {', '.join(f'{chosen[g.index].model} on GPU {g.index}' for g in group)})"
            options = plan.fitting(offered, need, group[0]) + ([plan.AUTO] if default == plan.AUTO else [])
            model = ui.choose(f"Model for {label}", options, default)
            if model != default:
                answered.update((g.index, model) for g in group)
        return plan.choose_models(gpus, offered, need, requested, previous, scored, answered=answered,
                                  **rule) if answered else chosen
    except plan.PlanError as exc:
        raise SystemExit(str(exc)) from None


def miner_config(existing: dict, *, network: str, netuid: int, wallet_name: str, hotkey_name: str, public_ip: str,
                 port: int, endpoints: dict[str, tuple[str, int]], managed_before: set[str], home: Path,
                 role: str = "primary", peers: list[str] | None = None, auto_update: bool | None = None,
                 firewall: bool | None = None) -> dict:
    """The configuration to write: ``existing`` with this setup's answers and its endpoints (endpoint id -> model,
    engine port). Endpoints added by hand and settings the setup does not ask about are kept; endpoints it set up
    before and no longer serves are dropped. ``peers`` (a primary's) replace the configured ones; None keeps them."""
    old = {e["endpoint_id"]: e for e in existing.get("endpoints", [])}
    listed = [{**old.get(eid, {}), "endpoint_id": eid, "model_id": model, "engine_url": f"http://127.0.0.1:{p}"}
              for eid, (model, p) in endpoints.items()]
    listed += [e for eid, e in old.items() if eid not in endpoints and eid not in managed_before]
    config = {**existing, "network": network, "netuid": netuid, "wallet_name": wallet_name,
              "hotkey_name": hotkey_name, "public_host": public_ip, "port": port, "role": role, "endpoints": listed}
    config["peers"] = (existing.get("peers", []) if peers is None else peers) if role == "primary" else []
    if not config["peers"]:
        del config["peers"]
    if auto_update is not None:
        config["auto_update"] = auto_update
    if firewall is not None:
        config["firewall"] = firewall
    config.setdefault("state_dir", str(home / "miner"))
    config.setdefault("catalog_dir", str(home / "catalog"))
    return config


def firewall_line(on: bool, root: bool) -> str:
    if not on:
        return "firewall off: anyone reaches the miner's port (--firewall turns it on)"
    return ("firewall on: the miner's port answers only the network's callers and this hotkey's machines once the "
            "catalog has miners enforce it (--no-firewall turns it off)"
            + ("" if root else "; without root it filters in the miner process"))


def set_up_firewall(on: bool, port: int, before: bool) -> bool:
    """Set up the kernel layer of the firewall (``install.firewall``), or remove it when the firewall is off; ->
    whether it is set up."""
    if not on:
        if before or firewall_setup.installed():
            removed = firewall_setup.remove()
            info("firewall off" + (f": removed {', '.join(removed)}" if removed else ""))
        return False
    try:
        firewall_setup.setup(common.operator(), port)
    except firewall_setup.Unavailable as exc:
        info(f"the kernel layer is off ({exc}); connections are filtered in the miner process")
        return False
    info(f"kernel layer: helper {firewall_setup.HELPER}"
         + ("" if common.operator() == "root" else f", sudo rule {firewall_setup.SUDOERS}")
         + f"; the miner applies it to port {port} whenever it enforces")
    return True


def init(argv: list[str]) -> None:
    args = parser().parse_args(argv)
    ui = Prompter(args.yes)
    home = args.home.expanduser().absolute()
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
    served = plan.servable_models(cat)
    models = {m.model_id: m for m in served if plan.installable(m)}
    for m in served:
        if m.model_id not in models:
            info(f"{m.model_id} runs on the {m.engine.name} engine, which this setup does not install")
    if not models:
        raise SystemExit("no catalog model this setup can install")
    scored = {m for m in models if models[m].status == "active"}
    try:
        need = {m: plan.miner_memory(engines.parameters(models[m]), models[m].engine.name) for m in models}
    except plan.PlanError as exc:
        raise SystemExit(str(exc)) from None
    info(plan.offer_line(list(models), need, scored))
    previous = {e["gpu"]: e["model"] for e in record.engines.values()}
    # records of earlier releases do not say; there only a candidate was ever chosen by hand
    manual = {e["gpu"] for e in record.engines.values() if e.get("manual", e["model"] not in scored)}
    market, why = read_market(st, netuid, wallet.hotkey.ss58_address, set(record.engines), cat)
    info(plan.market_line(market) if market else f"choosing models without the network's numbers: {why}; each GPU "
         "keeps the model it served before, else one of the active models that fit, spread by the hotkey")
    measured = measured_capacity(Path(existing.get("state_dir", home / "miner")).expanduser(), record.engines,
                                 cat.audit_policy().capacity_margin)
    requested = plan.parse_models(args.model)
    choices = choose_models(ui, gpus, list(models), need, requested, previous, scored, manual=manual, market=market,
                            measured=measured, relative={m: models[m].relative_speed for m in models},
                            seed=wallet.hotkey.ss58_address)
    chosen = {g: c.model for g, c in choices.items()}

    step("Network")
    role = choose_role(ui, args, existing)
    if args.name and role != "worker":
        raise SystemExit("--name names a worker's endpoints")
    peers = parse_peers(args.peers) if args.peers is not None else None
    detected = args.public_ip or existing.get("public_host") or host.detect_public_ip() or ""
    public_ip = ui.ask("Public IP address of this machine", detected,
                       lambda v: None if is_public(v) else "not a public IP address (pass --public-ip)")
    miner_running = manager is not None and MINER in record.services and manager.active(MINER)

    def port_problem(value: str) -> str | None:
        if not value.isdigit() or not 0 < int(value) < 65536:
            return "a TCP port number"
        if int(value) == existing.get("port") and miner_running:
            return None
        return None if host.port_free(int(value)) else f"port {value} is in use on this machine"

    port = int(ui.ask("Public TCP port of the miner (open it for inbound TCP)",
                      str(args.port or existing.get("port") or DEFAULT_PORT), port_problem))
    prefix = endpoint_prefix(role, args.name, public_ip, list(record.engines))
    endpoint_ids = {g.index: f"{prefix}gpu{g.index}" for g in gpus}
    engine_ports = plan.pick_ports(list(endpoint_ids.values()),
                                   {k: v["port"] for k, v in record.engines.items()}, ENGINE_PORT,
                                   lambda p: p != port and host.port_free(p, "127.0.0.1"))
    workers = (existing.get("peers", []) if peers is None else peers) if role == "primary" else []

    step("Plan")
    for g in gpus:
        info(f"{g.describe()}: {chosen[g.index]}, endpoint {endpoint_ids[g.index]}, engine on "
             f"127.0.0.1:{engine_ports[endpoint_ids[g.index]]}")
    for line in plan.choice_lines(gpus, choices, list(models), need, scored, market.shares if market else None):
        info(line)
    info(f"miner at {public_ip}:{port} with hotkey {wallet_name}/{hotkey_name}, "
         + ("a worker: its primary lists its endpoints" if role == "worker" else "the primary")
         + (" (testing: nothing written on chain)" if args.no_publish else ""))
    if workers:
        info(f"workers whose endpoints the manifest lists: {', '.join(workers)}")
    auto_update = common.auto_update(args, existing)
    info(common.update_line(auto_update))
    firewall = args.firewall if args.firewall is not None else bool(existing.get("firewall", True))
    info(firewall_line(firewall, manager is not None))
    info(f"configuration {config_path}; engines and weights in {home}")
    if not ui.confirm("Go ahead?"):
        raise SystemExit("stopped; nothing changed")

    step("Engines")
    installs = {}
    for model_id in sorted(set(chosen.values())):
        engine = engines.engine_for(home, models[model_id])
        installs.setdefault(engine.directory, engine)
    for engine in installs.values():
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
                          managed_before=set(record.engines), home=home, role=role, peers=peers,
                          auto_update=auto_update, firewall=firewall)
    try:
        MinerConfig.model_validate(config)
    except ValidationError as exc:
        raise common.invalid(config_path, exc) from None
    common.write_json(config_path, config)
    info(f"wrote {config_path}")
    identity = tls.identity(Path(config["state_dir"]))
    info(f"TLS certificate {identity.cert_path}, pin {identity.pin}: the manifest names it, and validators, the "
         "auditor and gateways accept this machine by it (delete tls.key and tls.crt there to replace it)")

    engine_services = {}
    for g in gpus:
        eid = endpoint_ids[g.index]
        model = models[chosen[g.index]]
        engine = engines.engine_for(home, model)
        engine_services[eid] = Service(
            name=f"verathos-engine-gpu{g.index}",
            description=f"Verathos engine of endpoint {eid}: {model.model_id} on GPU {g.index}",
            command=engine.command("miner", model, engine_ports[eid]), directory=engine.directory,
            user=common.operator(),
            environment={"HOME": str(Path.home()),
                         **engine.environment_variables("miner", f"cuda:{g.index}", hf_home)})
    miner_bin = Path(sys.prefix) / "bin" / "verathos-miner"
    miner = Service(name=MINER, description="Verathos miner", command=[str(miner_bin), "--config", str(config_path)]
                    + (["--no-publish"] if args.no_publish else []), directory=home, user=common.operator(),
                    environment={"HOME": str(Path.home())},
                    after=tuple(s.name for s in engine_services.values()))
    names = [s.name for s in engine_services.values()]
    stale = [s for s in record.services if s.startswith("verathos-engine-") and s not in names]

    record.config = str(config_path)
    record.engines = {eid: {"gpu": g, "model": chosen[g], "port": engine_ports[eid],
                            "manual": choices[g].reason in plan.BY_HAND} for g, eid in endpoint_ids.items()}
    record.paths = sorted(str(d) for d in installs)

    if manager is None:
        previous = [s for s in record.services if s not in names and s != MINER]
        record.manager, record.services = None, [*names, MINER]
        record.save(home)
        common.show_run("each engine and the miner", [*engine_services.values(), miner], auto_update)
        if previous:
            info(f"stop the engines no longer served (pm2: pm2 delete {' '.join(previous)})")
        info("the miner waits for its engines, measures its capacity (a few minutes), then "
             + ("serves its endpoints to its primary" if role == "worker" else "publishes its manifest"))
        if firewall and not firewall_setup.installed():
            info("the firewall filters in the miner process; its kernel layer takes root once: verathos-miner "
                 f"firewall install --config {config_path} (as a user with sudo), then restart the miner")
        for line in common.script_commands("miner", home, " --no-service"
                                           + (" --no-publish" if args.no_publish else "")):
            info(line)
        return

    step("Firewall")
    record.firewall = set_up_firewall(firewall, port, record.firewall)

    step("Services")
    for name in [*names, MINER]:
        manager.mark(name)
    if stale:
        manager.remove(stale)
    record.manager, record.services = manager.name, [*names, MINER]
    record.save(home)  # before the services exist, so uninstall finds them whatever happens next
    changed = manager.apply([*engine_services.values(), miner], restart={MINER})
    info(f"{manager.name}: {', '.join(names + [MINER])}" + (f" (new or changed: {', '.join(sorted(changed))})"
                                                             if changed else ""))
    info("waiting for the engines to load their models")
    common.wait_engines(manager, {engine_services[eid].name: p for eid, p in engine_ports.items()}, ENGINE_READY_S)

    step("Capacity")
    info("the miner measures its endpoints' capacity (or reuses the stored measurement), then serves")
    # The miner probes its engines through its own service while they load; those failures are expected.
    common.follow(manager, MINER, done=lambda line: " endpoints on port " in line and "serving" in line,
                  failed=lambda line: "no endpoint could be measured" in line or "no configured endpoint" in line,
                  timeout_s=MEASURE_S,
                  hide=lambda line: "engine unreachable" in line or "waiting for the engine" in line)
    if firewall:
        info(firewall_setup.summary(firewall_setup.miner_state(config_path), firewall_setup.kernel_status(), port))

    step("Done")
    for g in gpus:
        eid = endpoint_ids[g.index]
        info(f"endpoint {eid}: {chosen[g.index]} on GPU {g.index}, service {engine_services[eid].name}")
    info(f"miner: https://{public_ip}:{port}, service {MINER}"
         + (" (testing: nothing published on chain)" if args.no_publish else ""))
    info(common.update_line(auto_update))
    info(f"configuration: {config_path}; state: {config['state_dir']}")
    info(f"status:  {manager.status_command(record.services)}")
    if start := manager.start_line():
        info(start)
    info(f"logs:    {manager.log_command(MINER)}")
    info(f"after editing the configuration: {manager.restart_command([MINER])}")
    address = urlsplit(axon_base(public_ip, port)).netloc
    check = f"curl -sk --pinnedpubkey '{tls.curl_pin(identity.pin)}' https://{address}"  # the pin is the check
    where = "from another machine (once the firewall enforces, only from the network's callers)" if firewall \
        else "from another machine"
    if role == "worker":
        info(f"check {where}: {check}{ENDPOINTS_PATH}")
        info("list this machine on the primary, with every other worker (comma-separated):")
        print(f"      bash {common.INSTALLER} miner --yes --peers {address}", flush=True)
        info(f'or add "{address}" to "peers" in the primary\'s miner.json and restart its miner')
    else:
        info(f"check {where}: {check}/v1/manifest")
    for line in common.script_commands("miner", home, " --no-publish" if args.no_publish else ""):
        info(line)
