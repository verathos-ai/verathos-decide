"""``verathos-validator init``: set up this machine as a full validator, or a light one (``init_light``, the default).

A float32 reference engine for every active catalog model (and, with ``--candidates``, for every candidate model that
fits in the memory left), on a GPU where one has room and in RAM otherwise, and the validator. The references are
resident, each a system service, while they all fit in the free memory; otherwise (or with ``--references
sequential``) the validator runs them one at a time, each started for its model's canaries and stopped after, so the
memory needs only the largest. The setup shows the validator's first epoch start, and how to follow it.
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
from verathos.install.services import Manager, Service
from verathos.subnet.chain import current_catalog
from verathos.validator.config import ValidatorConfig

REFERENCE_PORT = 8101
VALIDATOR = "verathos-validator"
REFERENCE_READY_S = 2700.0
STARTUP_S = 600.0
FIRST_EPOCH_S = 60.0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="verathos-validator init", description="Set up this machine as a validator, "
                                "light (the validator alone) or full (also a float32 reference engine per active "
                                "catalog model), as system services. Run it again to update; the previous answers are "
                                "the defaults.")
    common.common_arguments(p)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--light", dest="mode", action="store_const", const="light", help="Set the weights the subnet "
                      "owner's validator publishes each epoch, once their signature verifies: no GPU, no reference "
                      "engines, no requests to miners (the default for a new validator, except the subnet owner's)")
    mode.add_argument("--full", dest="mode", action="store_const", const="full", help="Check every endpoint's "
                      "answers on float32 reference engines and score them (the subnet owner's validator always does)")
    p.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto", help="Where the references run: on "
                   "GPUs while they fit, smallest model first, else in RAM (auto); only in RAM (cpu); only on GPUs "
                   "(gpu)")
    p.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=None, help="Compute weights without "
                   "setting them (kept until --no-dry-run)")
    p.add_argument("--references", choices=("resident", "sequential"), help="How the references run: all at once, "
                   "each a service (resident), or one at a time, each started by the validator for its model's "
                   "canaries and stopped after, so the memory needs only the largest (sequential). Default: as set up "
                   "before, else resident while the active models' references all fit in the free memory")
    p.add_argument("--candidates", action=argparse.BooleanOptionalAction, default=None, help="Also run references for "
                   "the catalog's candidate models (checked, scored zero) where memory is left after the active ones; "
                   "a later run keeps them, --no-candidates removes them")
    p.add_argument("--config", type=Path, help="Validator configuration (default <home>/validator.json)")
    return p


def service_name(model_id: str) -> str:
    return "verathos-reference-" + re.sub(r"[^A-Za-z0-9-]", "-", model_id)


def describe(device: str) -> str:
    return "CPU" if device == "cpu" else f"GPU {device.split(':', 1)[1]}"


def choose_mode(ui: Prompter, requested: str | None, existing: dict, owner: bool) -> str:
    """``light`` or ``full``: as requested, else as configured before (a configuration without a mode is a full
    validator's), else light. The subnet owner's validator is the auditor and always full."""
    if owner:
        if requested == "light":
            raise SystemExit("this is the subnet owner's hotkey: its validator is the auditor, whose results light "
                             "validators set, and runs in full mode; drop --light")
        return "full"
    if requested:
        return requested
    default = existing.get("mode") or ("full" if existing else "light")
    return ui.choose("Mode (light: set the auditor's results; full: check answers on references)", ["light", "full"],
                     default)


def choose_reference_mode(requested: str | None, existing: dict, resident_fits: bool, owner: bool) -> tuple[str, str]:
    """``resident`` or ``sequential``, and why: as requested; else one at a time if set up so before; else resident
    while the active models' references all fit in the free memory, and one at a time when they do not. The subnet
    owner's validator (the auditor) keeps its references resident."""
    if owner:
        if requested == "sequential":
            raise SystemExit("this is the subnet owner's hotkey: the auditor checks test answers at any time and every "
                             "model each epoch, so its references stay resident; drop --references sequential")
        return "resident", "the auditor's references stay resident"
    if requested:
        return requested, f"as --references {requested} asks"
    if existing.get("reference_mode") == "sequential":
        return "sequential", "as set up before" + ("; --references resident runs them all at once, which fits now"
                                                   if resident_fits else "")
    if resident_fits:
        return "resident", "they fit in the free memory"
    return "sequential", "the active models' references do not all fit in the free memory at once"


def reference_line(reference_mode: str, placement: dict[str, str], parameters: dict[str, int]) -> str:
    """The plan's line on how the references run and the memory they take."""
    def where(place: str) -> str:
        return "RAM" if place == "cpu" else f"GPU {place.split(':', 1)[1]}"

    needs = plan.reference_needs(placement, parameters)
    if reference_mode == "resident":
        return "all references resident (" + ", ".join(
            f"{where(w)} about {sum(n for _, n in models) / host.GiB:.0f} GiB" for w, models in needs.items()) + ")"
    return "references one at a time (" + ", ".join(
        f"{where(w)} for the largest: about {models[0][1] / host.GiB:.0f} GiB" for w, models in needs.items()) + ")"


def validator_config(existing: dict, *, network: str, netuid: int, wallet_name: str, hotkey_name: str,
                     ports: dict[str, int], dry_run: bool, managed_before: set[str], home: Path,
                     mode: str = "full", reference_mode: str | None = None,
                     launch: dict[str, dict] | None = None) -> dict:
    """The configuration to write: ``existing`` with this setup's answers and a local reference per model (model ->
    port; none in light mode), with the command that serves it where the validator starts it (``launch``: model ->
    command, directory, environment). References configured by hand (a remote engine) and other settings are kept."""
    references = {m: v for m, v in existing.get("references", {}).items()
                  if m not in ports and m not in managed_before}
    for m, port in ports.items():
        url = f"http://127.0.0.1:{port}"
        references[m] = {"url": url, **launch[m]} if launch and m in launch else url
    config = {**existing, "network": network, "netuid": netuid, "wallet_name": wallet_name,
              "hotkey_name": hotkey_name, "mode": mode, "references": references, "dry_run": dry_run}
    if reference_mode is not None:
        config["reference_mode"] = reference_mode
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
        warn(f"UID {neuron.uid} has no validator permit yet: a validator sets weights, and miners answer it, only "
             "with one (it follows the stake)")
    else:
        info(f"registered on netuid {netuid} as UID {neuron.uid}, with a validator permit")
    owner = hotkey == st.get_subnet_owner_hotkey(netuid)
    if owner:
        warn("this is the subnet owner's hotkey: its validator also serves the catalog and runs the throughput "
             "audit, which needs public_host, port and the load generator (docs/VALIDATING.md)")
    mode = choose_mode(ui, args.mode, existing, owner)
    store = catalog.CatalogStore(Path(existing.get("catalog_dir", home / "catalog")))
    try:
        cat = current_catalog(st, netuid, store, existing.get("catalog_mirrors", []))
    except catalog.CatalogUnavailable as exc:
        raise SystemExit(f"the subnet's catalog is not available: {exc}") from None
    info(f"catalog {cat.revision} ({cat.digest()[:12]})")
    if mode == "light":
        init_light(ui, manager, record, existing, config_path, home, dry_run, cat=cat, network=network, netuid=netuid,
                   wallet_name=wallet_name, hotkey_name=hotkey_name)
        return

    step("References")
    served = plan.servable_models(cat)
    for m in served:
        if m.status == "active" and not plan.installable(m):
            raise SystemExit(f"{m.model_id} is active and runs on the {m.engine.name} engine, which this setup does "
                             "not install; set up its reference by hand (docs/VALIDATING.md)")
    candidates = sorted(m.model_id for m in served if m.status == "candidate")
    with_candidates = args.candidates if args.candidates is not None else any(m in record.engines for m in candidates)
    models = {m.model_id: m for m in served
              if m.status == "active" or (with_candidates and plan.installable(m))}
    try:
        parameters = {m: engines.parameters(models[m]) for m in models}
    except plan.PlanError as exc:
        raise SystemExit(str(exc)) from None
    stale = [m for m in record.engines if m not in models]
    if stale and manager is not None:
        # Their memory is free for the other references.
        manager.remove([service_name(m) for m in stale])
        info(f"removed the references of {', '.join(stale)}, which this setup no longer runs (not an active model"
             + (", nor a candidate)" if with_candidates else "; candidates only with --candidates)"))
        record.engines = {m: e for m, e in record.engines.items() if m in models}
        record.services = [s for s in record.services if s not in {service_name(m) for m in stale}]
        record.save(home)
    gpus = host.gpus()
    ram_total, ram_free = host.memory_available()
    info(f"RAM {ram_total / host.GiB:.0f} GiB ({ram_free / host.GiB:.0f} free)"
         + "".join(f"; {g.describe()} ({g.memory_free / host.GiB:.0f} free)" for g in gpus))
    try:
        reference_mode, why_mode, placement = place(parameters, {g.index: g.memory_free for g in gpus}, ram_free,
                                                    args, existing, record, owner, candidates)
    except plan.PlanError as exc:
        raise SystemExit(str(exc)) from None
    sequential = reference_mode == "sequential"
    left_out = [m for m in models if m not in placement]
    models = {m: model for m, model in models.items() if m in placement}
    ports = plan.pick_ports(list(models), {m: e["port"] for m, e in record.engines.items()}, REFERENCE_PORT,
                            lambda p: host.port_free(p, "127.0.0.1"))

    step("Plan")
    for line in mode_lines("full", cat, owner):
        info(line)
    info(f"{reference_line(reference_mode, placement, {m: parameters[m] for m in models})}: {why_mode}")
    if sequential:
        info("each epoch the validator checks the models whose references fit in its reference time, drawn in turn; "
             "a model it does not check that epoch takes the results of "
             + (f"the validator {existing['source_hotkey']}" if existing.get("source_hotkey")
                else "the subnet owner's validator"))
    for m in models:
        need = plan.reference_memory(parameters[m], "cpu" if placement[m] == "cpu" else "gpu")
        info(f"{m}{' (' + plan.CANDIDATE + ')' if m in candidates else ''}: float32 reference on "
             f"{describe(placement[m])} (about {need / host.GiB:.0f} GiB), 127.0.0.1:{ports[m]}"
             + (", started by the validator for its canaries" if sequential else ""))
    for m in left_out:
        info(f"{m} ({plan.CANDIDATE}): no reference, not enough memory left (it needs about "
             f"{plan.reference_memory(parameters[m], 'cpu') / host.GiB:.0f} GiB of RAM or "
             f"{plan.reference_memory(parameters[m], 'gpu') / host.GiB:.0f} GiB on a GPU); this validator does not "
             "check its endpoints")
    if candidates and not with_candidates:
        info(f"not checked by this validator: {', '.join(candidates)} ({plan.CANDIDATE}); --candidates runs "
             "their references too, where memory is left")
    for m in served:
        if with_candidates and m.status == "candidate" and not plan.installable(m):
            info(f"{m.model_id} ({plan.CANDIDATE}) runs on the {m.engine.name} engine, which this setup does not "
                 "install")
    info(f"validator with hotkey {wallet_name}/{hotkey_name}" + (", dry run: no weights set" if dry_run else ""))
    info(f"configuration {config_path}; engines and weights in {home}")
    if not ui.confirm("Go ahead?"):
        raise SystemExit("stopped; nothing changed")

    step("Engines")
    installs = {}
    for m in sorted(models):
        engine = engines.engine_for(home, models[m])
        installs.setdefault(engine.directory, engine)
    for engine in installs.values():
        info(f"{engine.engine.name} at {engine.engine.revision[:12]} in {engine.directory}")
        engine.install("reference")

    step("Weights")
    hf_home = home / "hf"
    for m in sorted(models):
        engines.download(engines.engine_for(home, models[m]).python("reference"), models[m], hf_home)

    step("Configuration")
    reference_services = {}
    for m in models:
        engine = engines.engine_for(home, models[m])
        reference_services[m] = Service(
            name=service_name(m), description=f"Verathos {m} reference (float32, {describe(placement[m])})",
            command=engine.command("reference", models[m], ports[m]), directory=engine.directory,
            user=common.operator(),
            environment={"HOME": str(Path.home()),
                         **engine.environment_variables("reference", placement[m], hf_home)})
    launch = {m: {"command": s.command, "directory": str(s.directory),
                  "environment": {k: v for k, v in s.environment.items() if k != "HOME"}}
              for m, s in reference_services.items()} if sequential else None
    config = validator_config(existing, network=network, netuid=netuid, wallet_name=wallet_name,
                              hotkey_name=hotkey_name, ports=ports, dry_run=dry_run,
                              managed_before=set(record.engines), home=home, reference_mode=reference_mode,
                              launch=launch)
    try:
        ValidatorConfig.model_validate(config)
    except ValidationError as exc:
        raise SystemExit(f"{config_path} would not be valid; fix or remove it:\n{exc}") from None
    common.write_json(config_path, config)
    info(f"wrote {config_path}")
    if sequential:  # the validator starts them itself
        reference_services = {}
    validator_bin = Path(sys.prefix) / "bin" / "verathos-validator"
    validator = Service(name=VALIDATOR, description="Verathos validator",
                        command=[str(validator_bin), "--config", str(config_path)], directory=home,
                        user=common.operator(), environment={"HOME": str(Path.home())},
                        after=tuple(s.name for s in reference_services.values()))

    record.config = str(config_path)
    record.engines = {m: {"model": m, "device": placement[m], "port": ports[m]} for m in models}
    record.paths = sorted(str(d) for d in installs)

    if manager is None:
        record.save(home)
        step("Run (no services installed)")
        info("start each reference engine, then the validator, under your supervisor:" if reference_services else
             "start the validator under your supervisor (it starts the references itself):")
        for s in [*reference_services.values(), validator]:
            env = " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted(s.environment.items()) if k != "HOME")
            print(f"      cd {shlex.quote(str(s.directory))} && {env + ' ' if env else ''}{shlex.join(s.command)}")
        return

    step("Services")
    if sequential and (resident := [service_name(m) for m in sorted(set(record.engines) | set(models))
                                    if service_name(m) in record.services]):
        manager.remove(resident)  # their memory is the references' one at a time
        info(f"removed the reference services {', '.join(resident)}: the validator starts the references itself")
    if not sequential and existing.get("reference_mode") == "sequential" and VALIDATOR in record.services:
        manager.stop([VALIDATOR])  # with the reference it may run, before the resident ones take the memory
    names = [s.name for s in reference_services.values()]
    for name in [*names, VALIDATOR]:
        manager.mark(name)
    record.manager, record.services = manager.name, [*names, VALIDATOR]
    record.save(home)  # before the services exist, so uninstall finds them whatever happens next
    changed = manager.apply([*reference_services.values(), validator], restart={VALIDATOR})
    info(f"{manager.name}: {', '.join(record.services)}" + (f" (new or changed: {', '.join(sorted(changed))})"
                                                             if changed else ""))
    if not sequential:
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
        if sequential:
            info(f"{m}: reference on {describe(placement[m])}, started by the validator when it checks {m}; log "
                 f"{Path(config['state_dir']).expanduser() / 'references' / (m + '.log')}")
        else:
            info(f"{m}: reference on {describe(placement[m])}, service {service_name(m)}")
    info(f"validator: service {VALIDATOR}" + (" (dry run: no weights set)" if dry_run else ""))
    info(f"configuration: {config_path}; state: {config['state_dir']} (keep it: canary secret and score history)")
    info(f"status:  {manager.status_command(record.services)}")
    info(f"logs:    {manager.log_command(VALIDATOR)}")
    info("each epoch logs its miners and endpoints, one line per endpoint checked, and the weights set")
    info(f"after editing the configuration: {manager.restart_command([VALIDATOR])}")
    for line in common.script_commands("validator", home):
        info(line)


def place(parameters: dict[str, int], gpu_free: dict[int, int], ram_free: int, args: argparse.Namespace,
          existing: dict, record: common.Record, owner: bool,
          candidates: list[str]) -> tuple[str, str, dict[str, str]]:
    """How the references run (``choose_reference_mode``), why, and where each runs. Resident references placed
    before keep their place and memory; one at a time, the memory of the resident references that stop is free for
    each in turn. Raises PlanError."""
    resident_before = {m: e["device"] for m, e in record.engines.items() if service_name(m) in record.services}
    try:
        resident = plan.place_references(parameters, gpu_free, ram_free, args.device, resident_before,
                                         optional=candidates)
        error = None
    except plan.PlanError as exc:
        resident, error = None, str(exc)
    reference_mode, why = choose_reference_mode(args.references, existing, resident is not None, owner)
    if reference_mode == "resident":
        if resident is None:
            raise plan.PlanError(error)
        return reference_mode, why, resident
    freed_gpu: dict[int, int] = {}
    freed_ram = 0
    for m, where in resident_before.items():
        if m in parameters and where == "cpu":
            freed_ram += plan.reference_memory(parameters[m], "cpu")
        elif m in parameters and where.startswith("cuda:"):
            index = int(where.split(":", 1)[1])
            freed_gpu[index] = freed_gpu.get(index, 0) + plan.reference_memory(parameters[m], "gpu")
    placement = plan.place_one_at_a_time(
        parameters, {i: free + freed_gpu.get(i, 0) for i, free in gpu_free.items()}, ram_free + freed_ram,
        args.device, {m: e["device"] for m, e in record.engines.items()}, optional=candidates)
    return reference_mode, why, placement


def mode_lines(mode: str, cat: catalog.Catalog, owner: bool) -> list[str]:
    """What the plan says about the validator's mode and what it needs."""
    if mode == "light":
        return ["mode light: sets the weights the subnet owner's validator publishes each epoch, once their signature "
                "verifies; it follows the auditor's results, where a full validator (--full) checks answers itself",
                "needs: a validator permit to set weights; no GPU, no reference engines and no stake beyond the "
                "permit; it sends nothing to miners"]
    if owner:
        return ["mode full: the subnet owner's validator is the auditor; it checks every endpoint's answers on its "
                "references, runs the throughput tests and publishes the results light validators set",
                "needs: public_host, port and the load generator (docs/VALIDATING.md), and the references below"]
    return ["mode full: checks every endpoint's answers on float32 references and scores them (--light sets the subnet "
            "owner's validator's results instead)",
            f"needs: a validator permit and {cat.scoring.validator_min_stake:,.0f} stake (the metagraph's S) for "
            "miners to answer it, and the references below"]


def init_light(ui: Prompter, manager: Manager | None, record: common.Record, existing: dict, config_path: Path,
               home: Path, dry_run: bool, *, cat: catalog.Catalog, network: str, netuid: int, wallet_name: str,
               hotkey_name: str) -> None:
    """The light validator: the validator alone, as a service. References a full setup placed before stop."""
    stale = sorted(record.engines)
    step("Plan")
    for line in mode_lines("light", cat, owner=False):
        info(line)
    if stale:
        info(f"the references of {', '.join(stale)} stop (their engines and weights stay until uninstall)")
    info(f"validator with hotkey {wallet_name}/{hotkey_name}" + (", dry run: no weights set" if dry_run else ""))
    info(f"configuration {config_path}")
    if not ui.confirm("Go ahead?"):
        raise SystemExit("stopped; nothing changed")

    step("Configuration")
    config = validator_config(existing, network=network, netuid=netuid, wallet_name=wallet_name,
                              hotkey_name=hotkey_name, ports={}, dry_run=dry_run, managed_before=set(stale),
                              home=home, mode="light")
    try:
        ValidatorConfig.model_validate(config)
    except ValidationError as exc:
        raise SystemExit(f"{config_path} would not be valid; fix or remove it:\n{exc}") from None
    common.write_json(config_path, config)
    info(f"wrote {config_path}")
    validator = Service(name=VALIDATOR, description="Verathos validator (light)",
                        command=[str(Path(sys.prefix) / "bin" / "verathos-validator"), "--config", str(config_path)],
                        directory=home, user=common.operator(), environment={"HOME": str(Path.home())})
    record.config, record.engines = str(config_path), {}

    if manager is None:
        record.save(home)
        step("Run (no services installed)")
        info("start the validator under your supervisor" + (" (the references it ran with are not needed)"
                                                            if stale else "") + ":")
        print(f"      cd {shlex.quote(str(validator.directory))} && {shlex.join(validator.command)}")
        return

    step("Services")
    if stale:
        manager.remove([service_name(m) for m in stale])
        info(f"removed the services of the references of {', '.join(stale)}")
    manager.mark(VALIDATOR)
    record.manager, record.services = manager.name, [VALIDATOR]
    record.save(home)  # before the service exists, so uninstall finds it whatever happens next
    changed = manager.apply([validator], restart={VALIDATOR})
    info(f"{manager.name}: {VALIDATOR}" + (" (new or changed)" if changed else ""))

    step("Validator")
    common.follow(manager, VALIDATOR, done=lambda line: "light mode:" in line,
                  failed=lambda line: "subnet owner's hotkey" in line or "source_hotkey" in line, timeout_s=STARTUP_S)

    step("Done")
    info(f"validator: service {VALIDATOR}, light mode" + (" (dry run: no weights set)" if dry_run else ""))
    info(f"configuration: {config_path}; state: {config['state_dir']}")
    info(f"status:  {manager.status_command(record.services)}")
    info(f"logs:    {manager.log_command(VALIDATOR)}")
    info("each epoch, once the subnet owner's validator has published its results (from about 60% into the epoch), "
         "logs the weights set: the number of UIDs and the highest weights")
    info(f"after editing the configuration: {manager.restart_command([VALIDATOR])}")
    for line in common.script_commands("validator", home):
        info(line)
