"""The setup's decisions, free of side effects: which models a GPU can serve and which one it serves, where each
reference engine runs (all resident, or one at a time) and which ports the engines take."""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass, field, replace
from statistics import median

from verathos import catalog
from verathos.install.host import GiB, Gpu
from verathos.protocol.status import NetworkStatus

# A serving engine holds the bfloat16 weights plus batches, CUDA graphs and its state cache; a reference holds the
# float32 weights plus the activations of a request, and on a GPU its CUDA graphs and state cache too.
MINER_WEIGHTS_FACTOR = 1.25
MINER_FIXED = 4 * GiB
# The Clef engine references the bfloat16 weights without copies and its CUDA graphs share one pool: Clef-flash (9.4B
# parameters, 17.5 GiB) peaked at 19.1-20.3 GB allocated on an A100, an H100 and an RTX 4090 with 525-token requests.
CLEF_MINER_FIXED = 3 * GiB
REFERENCE_WEIGHTS_FACTOR = 1.1
REFERENCE_FIXED = {"cpu": 2 * GiB, "gpu": 8 * GiB}
INSTALLABLE_ENGINES = ("kev", "clef")


class PlanError(ValueError):
    pass


def miner_memory(parameters: int, engine: str = "kev") -> int:
    """GPU memory the ``engine`` family needs to serve a model of ``parameters`` in bfloat16."""
    if engine == "clef":
        return parameters * 2 + CLEF_MINER_FIXED
    return int(parameters * 2 * MINER_WEIGHTS_FACTOR) + MINER_FIXED


def reference_memory(parameters: int, device: str) -> int:
    """Memory a reference engine needs for a model of ``parameters`` in float32: RAM (``device`` "cpu") or GPU
    memory ("gpu")."""
    return int(parameters * 4 * REFERENCE_WEIGHTS_FACTOR) + REFERENCE_FIXED[device]


SERVED = ("active", "candidate")  # active models are scored; candidates are checked by validators and score zero
CANDIDATE = "candidate: checked, scores zero"


def servable_models(cat: catalog.Catalog) -> list[catalog.CatalogModel]:
    """The catalog's active models, then its candidates, each in catalog order."""
    return sorted((m for m in cat.models if m.status in SERVED), key=lambda m: SERVED.index(m.status))


def installable(model: catalog.CatalogModel) -> bool:
    return model.engine.name in INSTALLABLE_ENGINES


def fitting(models: Iterable[str], need: dict[str, int], gpu: Gpu) -> list[str]:
    """The models whose engine fits in the GPU's memory."""
    return [m for m in models if need[m] <= gpu.memory_total]


# ---- which model each GPU serves ----

AUTO = "auto"  # in --model: the setup chooses that GPU's model
SWITCH_RATIO = 1.2  # a GPU leaves the model the setup chose for it before only for one paying this many times as much
ADVICE_RATIO = 1.1  # the plan names a model paying this many times as much as the one a GPU serves
REQUESTED, ANSWERED, MANUAL, PAYS, KEPT, BEFORE, FIRST = (
    "requested", "answered", "manual", "pays", "kept", "before", "first")
BY_HAND = (REQUESTED, ANSWERED, MANUAL)
WHY = {REQUESTED: "as --model asks", ANSWERED: "as chosen",
       MANUAL: f"as chosen by hand before (--model {AUTO} lets the setup choose)", PAYS: "pays best",
       KEPT: "as set up before", BEFORE: "as set up before", FIRST: "the first active model that fits"}


@dataclass
class Market:
    """What the network pays for each active model, from the auditor's network status: the model's share of the
    emission, the capacity proven by the endpoints serving it (requests/s of its standard mix, as credited to scored
    endpoints; this machine's own left out, so a GPU does not compete with itself) and how much of that the hotkey
    holds, and the median scored endpoint's speed in requests/s of a model of relative speed 1."""

    shares: dict[str, float]
    capacity: dict[str, float]
    held: dict[str, float]
    median_speed: float
    epoch: int | None = None

    def pays(self, model: str, rate: float) -> float:
        """What an endpoint serving ``model`` at ``rate`` requests/s adds to the hotkey's share of the emission. The
        model's share splits among its endpoints by proven capacity, so it is the hotkey's part of the model's share
        with the endpoint, less its part without: for a hotkey holding none of it, share x rate / (capacity + rate)."""
        share, total, held = self.shares.get(model, 0.0), self.capacity.get(model, 0.0), self.held.get(model, 0.0)
        if rate <= 0:
            return 0.0
        return share * ((held + rate) / (total + rate) - (held / total if total > 0 else 0.0))

    def add(self, model: str, rate: float) -> None:
        """Count an endpoint of the hotkey serving ``model`` at ``rate``."""
        self.capacity[model] = self.capacity.get(model, 0.0) + rate
        self.held[model] = self.held.get(model, 0.0) + rate


def network_market(status: NetworkStatus, *, hotkey: str, own: Collection[str], active: Collection[str],
                   relative: Mapping[str, float]) -> Market:
    """The market of the auditor's network ``status`` for a machine of ``hotkey`` whose endpoints are ``own``
    (endpoint ids), with the catalog's ``active`` models and their ``relative`` speeds. Raises ValueError, saying
    why, if its shares are not for the active models or no endpoint is scored yet."""
    if not status.model_shares or set(status.model_shares) != set(active):
        raise ValueError("its emission shares are not for the catalog's active models")
    capacity, held, speeds = dict.fromkeys(active, 0.0), dict.fromkeys(active, 0.0), []
    for h in status.hotkeys:
        for e in h.endpoints:
            if e.state != "scored" or e.model_id not in capacity or e.credited_rps <= 0:
                continue
            speeds.append(e.credited_rps / relative.get(e.model_id, 1.0))
            if h.hotkey == hotkey and e.endpoint_id in own:
                continue
            capacity[e.model_id] += e.credited_rps
            if h.hotkey == hotkey:
                held[e.model_id] += e.credited_rps
    if not speeds:
        raise ValueError("no endpoint is scored yet")
    return Market(dict(status.model_shares), capacity, held, median(speeds), status.epoch)


def gpu_speeds(gpus: list[Gpu], measured: Mapping[int, tuple[str, float]], relative: Mapping[str, float],
               median_speed: float) -> dict[int, tuple[float, str]]:
    """Each GPU's speed in requests/s of a model of relative speed 1, and where it comes from: what this machine's
    miner measured on it (``measured``: GPU index -> model, requests/s), else the median of what it measured on GPUs
    of the same kind, else the network's median endpoint."""
    kind = {g.index: (g.name, g.memory_total) for g in gpus}
    known = {i: (rate / relative[model], model) for i, (model, rate) in measured.items()
             if model in relative and rate > 0}
    speeds = {}
    for g in gpus:
        if g.index in known:
            speed, model = known[g.index]
            speeds[g.index] = (speed, f"by its {model} measurement")
        elif same := [s for i, (s, _) in known.items() if kind.get(i) == kind[g.index]]:
            speeds[g.index] = (median(same), "as measured on its kind of GPU")
        else:
            speeds[g.index] = (median_speed, "as the network's median endpoint")
    return speeds


@dataclass(frozen=True)
class Choice:
    """The model a GPU serves and why (``WHY``); with a market, for each active model that fits, the requests/s the
    GPU would serve, what that would add to the hotkey's share of the emission and the capacity proven by the others
    serving it, and where the GPU's speed comes from."""

    model: str
    reason: str
    rates: dict[str, float] = field(default_factory=dict)
    pays: dict[str, float] = field(default_factory=dict)
    network: dict[str, float] = field(default_factory=dict)
    source: str = ""


def _named(model: str, offered: list[str], need: dict[str, int], gpu: Gpu) -> str:
    if model not in offered:
        raise PlanError(f"{model} is not a catalog model this setup can install; choose from "
                        f"{', '.join(offered)} or {AUTO}")
    if need[model] > gpu.memory_total:
        raise PlanError(f"{model} needs about {need[model] / GiB:.0f} GiB of GPU memory; {gpu.describe()} has less")
    return model


def choose_models(gpus: list[Gpu], offered: list[str], need: dict[str, int], requested: list[str],
                  previous: Mapping[int, str], scored: Collection[str] | None = None, *,
                  manual: Collection[int] = (), answered: Mapping[int, str] | None = None,
                  market: Market | None = None, measured: Mapping[int, tuple[str, float]] | None = None,
                  relative: Mapping[str, float] | None = None) -> dict[int, Choice]:
    """The model each GPU serves, and why.

    A model named for a GPU is served: ``requested`` (``--model``: one for every GPU, or one per GPU in order),
    ``answered`` (GPU index -> model, on a terminal), or chosen by hand before (``manual``: GPU indexes whose
    ``previous`` model was) unless ``requested`` says ``auto`` for it. The setup chooses the others' among the
    ``scored`` (active) models that fit. With the network's ``market``: the model that adds most to the hotkey's share
    of the emission (``Market.pays``), at the GPU's speed (``gpu_speeds``, from this machine's ``measured`` capacity)
    times the model's ``relative`` speed, counting the GPUs given a model before it (those named first, then in
    order), so a machine spreads over models where that pays more; a GPU keeps its ``previous`` model unless another
    pays ``SWITCH_RATIO`` times as much. Without a market: its previous model, else the first active model that fits.
    Candidates (offered, not scored) are served only when named."""
    scored = offered if scored is None else [m for m in offered if m in scored]
    if requested and len(requested) not in (1, len(gpus)):
        raise PlanError(f"--model takes one model for all GPUs or one per GPU ({len(gpus)}), not {len(requested)}")
    relative = {m: (relative or {}).get(m, 1.0) for m in offered}
    answered = answered or {}
    by_hand: dict[int, tuple[str, str]] = {}
    for i, gpu in enumerate(gpus):
        asked = (requested[0] if len(requested) == 1 else requested[i]) if requested else None
        if gpu.index in answered:
            by_hand[gpu.index] = (_named(answered[gpu.index], offered, need, gpu), ANSWERED)
        elif asked is not None and asked != AUTO:
            by_hand[gpu.index] = (_named(asked, offered, need, gpu), REQUESTED)
        elif asked is None and gpu.index in manual and previous.get(gpu.index) in fitting(offered, need, gpu):
            by_hand[gpu.index] = (previous[gpu.index], MANUAL)
    speeds = gpu_speeds(gpus, measured or {}, relative, market.median_speed) if market else {}
    if market is not None:
        market = replace(market, capacity=dict(market.capacity), held=dict(market.held))
    chosen: dict[int, Choice] = {}

    def options(gpu: Gpu) -> dict:
        if market is None:
            return {}
        speed, source = speeds[gpu.index]
        rates = {m: speed * relative[m] for m in fitting(scored, need, gpu)}
        return {"rates": rates, "pays": {m: market.pays(m, r) for m, r in rates.items()},
                "network": {m: market.capacity.get(m, 0.0) for m in rates}, "source": source}

    def place(gpu: Gpu, model: str, reason: str, found: dict) -> None:
        chosen[gpu.index] = Choice(model, reason, **found)
        if market is not None and model in found["rates"]:
            market.add(model, found["rates"][model])

    for gpu in gpus:
        if gpu.index in by_hand:
            place(gpu, *by_hand[gpu.index], options(gpu))
    for gpu in gpus:
        if gpu.index in by_hand:
            continue
        fits = fitting(offered, need, gpu)
        active = [m for m in fits if m in scored]
        if not active:
            others = [m for m in fits if m not in scored]
            raise PlanError(f"no active catalog model fits {gpu.describe()}; the smallest needs about "
                            f"{min(need[m] for m in scored or offered) / GiB:.0f} GiB"
                            + (f"; with --model it can serve {', '.join(others)} ({CANDIDATE})" if others else ""))
        before = previous.get(gpu.index) if previous.get(gpu.index) in active else None
        found = options(gpu)
        if market is None:
            model, reason = (before, BEFORE) if before else (active[0], FIRST)
        else:
            pays = found["pays"]
            best = max(active, key=lambda m: pays[m])
            if before is not None and best != before and pays[best] < SWITCH_RATIO * pays[before]:
                model, reason = before, KEPT
            else:
                model, reason = best, PAYS
        place(gpu, model, reason, found)
    return {g.index: chosen[g.index] for g in gpus}


def offer_line(offered: list[str], need: dict[str, int], scored: Collection[str]) -> str:
    """The models a miner can serve, with the GPU memory each needs."""
    return "models: " + ", ".join(f"{m} (about {need[m] / GiB:.0f} GiB{'' if m in scored else '; ' + CANDIDATE})"
                                  for m in offered)


def market_line(market: Market) -> str:
    """The network's numbers the setup chooses models by."""
    epoch = "" if market.epoch is None else f" in the auditor's epoch {market.epoch}"
    return (f"the network{epoch}: " + "; ".join(f"{m} {market.shares[m]:.1%} of the emission, "
                                                f"{market.capacity.get(m, 0.0):.1f} requests/s proven"
                                                for m in market.shares)
            + f"; the median endpoint {market.median_speed:.1f} requests/s at relative speed 1")


def choice_lines(gpus: list[Gpu], chosen: Mapping[int, Choice], offered: list[str], need: dict[str, int],
                 scored: Collection[str], shares: Mapping[str, float] | None = None) -> list[str]:
    """For the plan: per GPU why it serves its model, with the numbers when the network's ``shares`` are known, what
    else fits (and what it would pay), and a model that pays ``ADVICE_RATIO`` times as much with how to switch; then
    how to choose another."""
    def label(model: str) -> str:
        return model + ("" if model in scored else " (candidate)")

    def alternatives(gpu: Gpu) -> list[str]:
        return [m for m in fitting(offered, need, gpu) if m != chosen[gpu.index].model]

    def switch(gpu: Gpu, model: str) -> str:
        """``--model`` serving ``model`` on ``gpu`` and the others' models as now."""
        if len(gpus) == 1:
            return model
        return ",".join(model if g.index == gpu.index else chosen[g.index].model
                        if chosen[g.index].reason in BY_HAND else AUTO for g in gpus)

    lines = []
    for gpu in gpus:
        c = chosen[gpu.index]
        line = f"GPU {gpu.index}: {label(c.model)}, {WHY[c.reason]}"
        if shares is not None and c.model in c.pays:
            line += (f": about {c.pays[c.model]:.2%} of the emission (its share {shares[c.model]:.1%}, "
                     f"{c.network[c.model]:.1f} requests/s proven by other endpoints, about "
                     f"{c.rates[c.model]:.1f} on this GPU {c.source})")
        others = alternatives(gpu)
        line += (". Also fit: " + ", ".join(f"{m} {c.pays[m]:.2%}" if m in c.pays else label(m) for m in others)
                 if others else ". No other catalog model fits")
        here, best = c.pays.get(c.model, 0.0), max(c.pays, key=lambda m: c.pays[m], default=None)
        if best is not None and best != c.model and c.pays[best] > 0 and c.pays[best] >= ADVICE_RATIO * here:
            more = f"about {c.pays[best] / here:.2f}x as much" if here > 0 else f"about {c.pays[best]:.2%}"
            line += f". {best} would pay {more} on GPU {gpu.index}: --model {switch(gpu, best)} to switch"
        lines.append(line)
    if not any(alternatives(g) for g in gpus):
        return lines
    # the example changes one GPU, to a model no other GPU serves where one fits
    example, used = [chosen[g.index].model for g in gpus], {c.model for c in chosen.values()}
    position, model = next(((i, m) for i, g in reversed(list(enumerate(gpus))) for m in alternatives(g)
                            if m not in used), next((i, alternatives(g)[0]) for i, g in reversed(list(enumerate(gpus)))
                                                    if alternatives(g)))
    example[position] = model
    return lines + [f"another choice: --model <id> for every GPU, or one per GPU in order, as in --model "
                    f"{','.join(example)}"]


def place_references(parameters: dict[str, int], gpu_free: dict[int, int], ram_free: int, device: str,
                     keep: dict[str, str], optional: Collection[str] = ()) -> dict[str, str]:
    """Where the reference engine of each model (model -> parameters) runs: ``"cpu"`` or ``"cuda:<index>"``.

    A reference an earlier setup placed stays where it is if ``device`` allows it (its memory is already taken).
    The others go, smallest first, on the GPU with the most free memory while one has room (``device`` "auto" or
    "gpu"), else in RAM (``device`` "auto" or "cpu"). The ``optional`` models (candidates) come after the others, in
    what they leave, and are left out (absent from the result) where they do not fit. Raises PlanError if another
    model fits nowhere."""
    if device not in ("auto", "cpu", "gpu"):
        raise PlanError(f"unknown device {device!r}")
    placement: dict[str, str] = {}
    for model, where in keep.items():
        on_gpu = where.startswith("cuda:") and int(where.split(":", 1)[1]) in gpu_free
        if model in parameters and ((where == "cpu" and device != "gpu") or (on_gpu and device != "cpu")):
            placement[model] = where
    free, ram = dict(gpu_free), ram_free
    for model in sorted((m for m in parameters if m not in placement),
                        key=lambda m: (m in optional, parameters[m], m)):
        gpu_need, cpu_need = reference_memory(parameters[model], "gpu"), reference_memory(parameters[model], "cpu")
        if device != "cpu" and free:
            index = max(free, key=lambda i: (free[i], -i))
            if free[index] >= gpu_need:
                placement[model] = f"cuda:{index}"
                free[index] -= gpu_need
                continue
        if device != "gpu" and ram >= cpu_need:
            placement[model] = "cpu"
            ram -= cpu_need
            continue
        if model in optional:
            continue
        needs = {"auto": f"{gpu_need / GiB:.0f} GiB on a GPU or {cpu_need / GiB:.0f} GiB of RAM",
                 "cpu": f"{cpu_need / GiB:.0f} GiB of RAM", "gpu": f"{gpu_need / GiB:.0f} GiB on a GPU"}[device]
        raise PlanError(f"the {model} reference needs about {needs}; not enough is free (RAM {ram / GiB:.0f} GiB"
                        + "".join(f", GPU {i} {f / GiB:.0f} GiB" for i, f in sorted(free.items())) + ")")
    return placement


def place_one_at_a_time(parameters: dict[str, int], gpu_free: dict[int, int], ram_free: int, device: str,
                        keep: dict[str, str], optional: Collection[str] = ()) -> dict[str, str]:
    """Where each reference runs when the validator runs them one at a time (``reference_mode`` sequential): each
    alone in the free memory, so the memory needs only the largest. A placement an earlier setup chose stays where
    ``device`` allows it; the others go on the GPU with the most free memory if it holds them (``device`` "auto" or
    "gpu"), else in RAM ("auto" or "cpu"). The ``optional`` models (candidates) that fit nowhere are left out; another
    raises PlanError."""
    if device not in ("auto", "cpu", "gpu"):
        raise PlanError(f"unknown device {device!r}")
    placement: dict[str, str] = {}
    for model in sorted(parameters, key=lambda m: (m in optional, parameters[m], m)):
        where = keep.get(model, "")
        on_gpu = where.startswith("cuda:") and int(where.split(":", 1)[1]) in gpu_free
        if (where == "cpu" and device != "gpu") or (on_gpu and device != "cpu"):
            placement[model] = where
            continue
        gpu_need, cpu_need = reference_memory(parameters[model], "gpu"), reference_memory(parameters[model], "cpu")
        if device != "cpu" and gpu_free:
            index = max(gpu_free, key=lambda i: (gpu_free[i], -i))
            if gpu_free[index] >= gpu_need:
                placement[model] = f"cuda:{index}"
                continue
        if device != "gpu" and ram_free >= cpu_need:
            placement[model] = "cpu"
            continue
        if model in optional:
            continue
        needs = {"auto": f"{gpu_need / GiB:.0f} GiB on a GPU or {cpu_need / GiB:.0f} GiB of RAM",
                 "cpu": f"{cpu_need / GiB:.0f} GiB of RAM", "gpu": f"{gpu_need / GiB:.0f} GiB on a GPU"}[device]
        raise PlanError(f"the {model} reference needs about {needs}, even one at a time; not enough is free (RAM "
                        f"{ram_free / GiB:.0f} GiB" + "".join(f", GPU {i} {f / GiB:.0f} GiB"
                                                           for i, f in sorted(gpu_free.items())) + ")")
    return placement


def reference_needs(placement: dict[str, str], parameters: dict[str, int]) -> dict[str, list[tuple[str, int]]]:
    """Per place a reference runs ("cpu" or "cuda:<index>"), its models and the memory each needs there, largest
    first."""
    out: dict[str, list[tuple[str, int]]] = {}
    for model, where in placement.items():
        need = reference_memory(parameters[model], "cpu" if where == "cpu" else "gpu")
        out.setdefault(where, []).append((model, need))
    return {where: sorted(models, key=lambda item: (-item[1], item[0])) for where, models in sorted(out.items())}


def pick_ports(keys: list[str], previous: dict[str, int], start: int, is_free: Callable[[int], bool]) -> dict[str, int]:
    """A local port per key: the one it had before, else the next free one from ``start`` that no other key
    takes."""
    ports = {k: previous[k] for k in keys if k in previous}
    taken = set(ports.values())
    port = start
    for key in keys:
        if key in ports:
            continue
        while port in taken or not is_free(port):
            port += 1
        ports[key] = port
        taken.add(port)
    return ports


def parse_models(value: str | None) -> list[str]:
    return [m.strip() for m in value.split(",") if m.strip()] if value else []
