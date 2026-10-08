"""``verathos-validator init``: set up this machine as a validator.

A float32 reference engine for every active catalog model, on a GPU where one has room and in RAM otherwise, and the
validator, all as system services. The setup shows the validator's first epoch start, and how to follow it.
"""

from __future__ import annotations

import argparse
import re
import shlex
import sys
from pathlib import Path

from pydantic import ValidationError

from verathos import catalog
from verathos.install import common, engines, host, plan
from verathos.install.common import Prompter, info, step, warn
from verathos.install.services import Service
from verathos.subnet.chain import current_catalog
from verathos.validator.config import ValidatorConfig

REFERENCE_PORT = 8101
VALIDATOR = "verathos-validator"
REFERENCE_READY_S = 2700.0
STARTUP_S = 600.0
FIRST_EPOCH_S = 120.0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="verathos-validator init", description="Set up this machine as a validator: a "
                                "float32 reference engine per active catalog model, the validator, and system "
                                "services for them. Run it again to update; the previous answers are the defaults.")
    common.common_arguments(p)
    p.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto", help="Where the references run: on "
                   "GPUs while they fit, smallest model first, else in RAM (auto); only in RAM (cpu); only on GPUs "
                   "(gpu)")
    p.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=None, help="Compute weights without "
                   "setting them (kept until --no-dry-run)")
    p.add_argument("--config", type=Path, help="Validator configuration (default <home>/validator.json)")
    return p


def service_name(model_id: str) -> str:
    return "verathos-reference-" + re.sub(r"[^A-Za-z0-9-]", "-", model_id)


def describe(device: str) -> str:
    return "CPU" if device == "cpu" else f"GPU {device.split(':', 1)[1]}"


def validator_config(existing: dict, *, network: str, netuid: int, wallet_name: str, hotkey_name: str,
                     ports: dict[str, int], dry_run: bool, managed_before: set[str], home: Path) -> dict:
    """The configuration to write: ``existing`` with this setup's answers and a local reference per model (model ->
    port). References configured by hand (a remote engine) and other settings are kept."""
    references = {m: v for m, v in existing.get("references", {}).items()
                  if m not in ports and m not in managed_before}
    references.update({m: f"http://127.0.0.1:{p}" for m, p in ports.items()})
    config = {**existing, "network": network, "netuid": netuid, "wallet_name": wallet_name,
              "hotkey_name": hotkey_name, "references": references, "dry_run": dry_run}
    config.setdefault("state_dir", str(home / "validator"))
    config.setdefault("catalog_dir", str(home / "catalog"))
    return config


def init(argv: list[str]) -> None:
    args = parser().parse_args(argv)
    ui = Prompter(args.yes)
    home = args.home.expanduser().absolute()
    record = common.Record.load(home, "validator")
    config_path = (args.config or Path(record.config or home / "validator.json")).expanduser().absolute()
    existing = common.read_json(config_path)
    manager = common.manager_for(args, record)
    dry_run = existing.get("dry_run", False) if args.dry_run is None else args.dry_run

    step("Wallet and chain")
    network, netuid = common.network_and_netuid(args, existing)
    info(f"network {network}, netuid {netuid}")
    wallet_name, hotkey_name = common.choose_wallet(ui, args, existing)
    wallet = common.load_hotkey(wallet_name, hotkey_name)
    hotkey = wallet.hotkey.ss58_address
    st = common.connect(network)
    neuron = st.get_neuron_for_pubkey_and_subnet(hotkey, netuid)
    if neuron is None or neuron.is_null:
        message = (f"hotkey {hotkey} is not registered on netuid {netuid}. Register it on the machine holding the "
                   f"coldkey:\n  {common.register_command(network, netuid, wallet_name, hotkey_name)}")
        if not dry_run:
            raise SystemExit(message + "\nthen run this again.")
        warn(message.replace("\n", " "))
    elif not neuron.validator_permit:
        warn(f"UID {neuron.uid} has no validator permit yet: miners answer only validators with one (it follows "
             "the stake)")
    else:
        info(f"registered on netuid {netuid} as UID {neuron.uid}, with a validator permit")
    if hotkey == st.get_subnet_owner_hotkey(netuid):
        warn("this is the subnet owner's hotkey: its validator also serves the catalog and runs the throughput "
             "audit, which needs public_host, port and the load generator (docs/VALIDATING.md)")
    store = catalog.CatalogStore(Path(existing.get("catalog_dir", home / "catalog")))
    try:
        cat = current_catalog(st, netuid, store, existing.get("catalog_mirrors", []))
    except catalog.CatalogUnavailable as exc:
        raise SystemExit(f"the subnet's catalog is not available: {exc}") from None
    info(f"catalog {cat.revision} ({cat.digest()[:12]})")

    step("References")
    active = plan.servable_models(cat)
    for m in active:
        if not plan.installable(m):
            raise SystemExit(f"{m.model_id} is active and runs on the {m.engine.name} engine, which this setup does "
                             "not install; set up its reference by hand (docs/VALIDATING.md)")
    models = {m.model_id: m for m in active}
    try:
        parameters = {m: engines.parameters(models[m]) for m in models}
    except plan.PlanError as exc:
        raise SystemExit(str(exc)) from None
    stale = [m for m in record.engines if m not in models]
    if stale and manager is not None:
        # Their memory is free for the active models' references.
        manager.remove([service_name(m) for m in stale])
        info(f"removed the references of {', '.join(stale)}, which the catalog no longer lists as active")
        record.engines = {m: e for m, e in record.engines.items() if m in models}
        record.services = [s for s in record.services if s not in {service_name(m) for m in stale}]
        record.save(home)
    gpus = host.gpus()
    ram_total, ram_free = host.memory_available()
    info(f"RAM {ram_total / host.GiB:.0f} GiB ({ram_free / host.GiB:.0f} free)"
         + "".join(f"; {g.describe()} ({g.memory_free / host.GiB:.0f} free)" for g in gpus))
    keep = {m: e["device"] for m, e in record.engines.items()}
    try:
        placement = plan.place_references(parameters, {g.index: g.memory_free for g in gpus}, ram_free, args.device,
                                          keep)
    except plan.PlanError as exc:
        raise SystemExit(str(exc)) from None
    ports = plan.pick_ports(list(models), {m: e["port"] for m, e in record.engines.items()}, REFERENCE_PORT,
                            lambda p: host.port_free(p, "127.0.0.1"))

    step("Plan")
    for m in models:
        need = plan.reference_memory(parameters[m], "cpu" if placement[m] == "cpu" else "gpu")
        info(f"{m}: float32 reference on {describe(placement[m])} (about {need / host.GiB:.0f} GiB), "
             f"127.0.0.1:{ports[m]}")
    info(f"validator with hotkey {wallet_name}/{hotkey_name}" + (", dry run: no weights set" if dry_run else ""))
    info(f"configuration {config_path}; engines and weights in {home}")
    if not ui.confirm("Go ahead?"):
        raise SystemExit("stopped; nothing changed")

    step("Engines")
    kev = {}
    for m in sorted(models):
        engine = engines.engine_for(home, models[m])
        kev.setdefault(engine.directory, engine)
    for engine in kev.values():
        info(f"{engine.engine.name} at {engine.engine.revision[:12]} in {engine.directory}")
        engine.install("reference")

    step("Weights")
    hf_home = home / "hf"
    for m in sorted(models):
        engines.download(engines.engine_for(home, models[m]).python("reference"), models[m], hf_home)

    step("Configuration")
    config = validator_config(existing, network=network, netuid=netuid, wallet_name=wallet_name,
                              hotkey_name=hotkey_name, ports=ports, dry_run=dry_run,
                              managed_before=set(record.engines), home=home)
    try:
        ValidatorConfig.model_validate(config)
    except ValidationError as exc:
        raise SystemExit(f"{config_path} would not be valid; fix or remove it:\n{exc}") from None
    common.write_json(config_path, config)
    info(f"wrote {config_path}")

    reference_services = {}
    for m in models:
        engine = engines.engine_for(home, models[m])
        reference_services[m] = Service(
            name=service_name(m), description=f"Verathos {m} reference (float32, {describe(placement[m])})",
            command=engine.command("reference", models[m], ports[m]), directory=engine.directory,
            user=common.operator(),
            environment={"HOME": str(Path.home()),
                         **engine.environment_variables("reference", placement[m], hf_home)})
    validator_bin = Path(sys.prefix) / "bin" / "verathos-validator"
    validator = Service(name=VALIDATOR, description="Verathos validator",
                        command=[str(validator_bin), "--config", str(config_path)], directory=home,
                        user=common.operator(), environment={"HOME": str(Path.home())},
                        after=tuple(s.name for s in reference_services.values()))

    record.config = str(config_path)
    record.engines = {m: {"model": m, "device": placement[m], "port": ports[m]} for m in models}
    record.paths = sorted(str(d) for d in kev)

    if manager is None:
        record.save(home)
        step("Run (no services installed)")
        info("start each reference engine, then the validator, under your supervisor:")
        for s in [*reference_services.values(), validator]:
            env = " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted(s.environment.items()) if k != "HOME")
            print(f"      cd {shlex.quote(str(s.directory))} && {env + ' ' if env else ''}{shlex.join(s.command)}")
        return

    step("Services")
    names = [s.name for s in reference_services.values()]
    for name in [*names, VALIDATOR]:
        manager.mark(name)
    record.manager, record.services = manager.name, [*names, VALIDATOR]
    record.save(home)  # before the services exist, so uninstall finds them whatever happens next
    changed = manager.apply([*reference_services.values(), validator], restart={VALIDATOR})
    info(f"{manager.name}: {', '.join(record.services)}" + (f" (new or changed: {', '.join(sorted(changed))})"
                                                             if changed else ""))
    info("waiting for the references to load their models (float32 on CPU takes several minutes)")
    common.wait_engines(manager, {service_name(m): ports[m] for m in models}, REFERENCE_READY_S)

    step("Validator")

    def failed(line: str) -> bool:
        return "no reference engine configured" in line or "; stopping" in line

    common.follow(manager, VALIDATOR, done=lambda line: "throughput tests are run by" in line
                  or "throughput auditor running" in line, failed=failed, timeout_s=STARTUP_S)
    if not common.follow(manager, VALIDATOR, done=lambda line: re.search(r"epoch \d+: \d+ miners", line) is not None,
                         failed=failed, timeout_s=FIRST_EPOCH_S, required=False):
        info("this epoch was scored before the restart; the validator starts the next one at its first block")

    step("Done")
    for m in models:
        info(f"{m}: reference on {describe(placement[m])}, service {service_name(m)}")
    info(f"validator: service {VALIDATOR}" + (" (dry run: no weights set)" if dry_run else ""))
    info(f"configuration: {config_path}; state: {config['state_dir']} (keep it: canary secret and score history)")
    info(f"status:  {manager.status_command(record.services)}")
    info(f"logs:    {manager.log_command(VALIDATOR)}")
    info("each epoch logs its miners and endpoints, one line per endpoint checked, and the weights set")
    info(f"after editing the configuration: {manager.restart_command([VALIDATOR])}")
    for line in common.script_commands("validator", home):
        info(line)
