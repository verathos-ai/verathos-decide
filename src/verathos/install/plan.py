"""The setup's decisions, free of side effects: which models a GPU can serve, where each reference engine runs and
which ports the engines take."""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable

from verathos import catalog
from verathos.install.host import GiB, Gpu

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


def choose_models(gpus: list[Gpu], offered: list[str], need: dict[str, int], requested: list[str],
                  previous: dict[int, str], scored: Collection[str] | None = None) -> dict[int, str]:
    """The model each GPU serves: ``requested`` (one model for every GPU, or one per GPU in order), else what the
    GPU served before if it is still offered and fits, else the first ``scored`` (active) model that fits. Other
    offered models (candidates) are served only when requested or kept from before."""
    scored = offered if scored is None else [m for m in offered if m in scored]
    if requested and len(requested) not in (1, len(gpus)):
        raise PlanError(f"--model takes one model for all GPUs or one per GPU ({len(gpus)}), not {len(requested)}")
    chosen = {}
    for i, gpu in enumerate(gpus):
        fits = fitting(offered, need, gpu)
        if requested:
            model = requested[0] if len(requested) == 1 else requested[i]
            if model not in offered:
                raise PlanError(f"{model} is not a catalog model this setup can install; choose from "
                                f"{', '.join(offered)}")
            if model not in fits:
                raise PlanError(f"{model} needs about {need[model] / GiB:.0f} GiB of GPU memory; "
                                f"{gpu.describe()} has less")
        elif previous.get(gpu.index) in fits:
            model = previous[gpu.index]
        elif default := next((m for m in scored if m in fits), None):
            model = default
        else:
            others = [m for m in fits if m not in scored]
            raise PlanError(f"no active catalog model fits {gpu.describe()}; the smallest needs about "
                            f"{min(need[m] for m in scored or offered) / GiB:.0f} GiB"
                            + (f"; with --model it can serve {', '.join(others)} ({CANDIDATE})" if others else ""))
        chosen[gpu.index] = model
    return chosen


def _gpu_list(indexes: list[int]) -> str:
    return ("GPU " if len(indexes) == 1 else "GPUs ") + ", ".join(map(str, indexes))


def offer_line(offered: list[str], need: dict[str, int], scored: Collection[str]) -> str:
    """The models a miner can serve, with the GPU memory each needs."""
    return "models: " + ", ".join(f"{m} (about {need[m] / GiB:.0f} GiB{'' if m in scored else '; ' + CANDIDATE})"
                                  for m in offered)


def choice_lines(gpus: list[Gpu], chosen: dict[int, str], offered: list[str], need: dict[str, int],
                 scored: Collection[str], requested: list[str], previous: dict[int, str]) -> list[str]:
    """Two lines for the plan: why each GPU serves its model and what else fits, then how to choose another."""
    def why(gpu: Gpu) -> str:
        model = chosen[gpu.index]
        if requested:
            return "as --model asks"
        if previous.get(gpu.index) == model:
            return "as set up before"
        if model == next((m for m in offered if m in scored and need[m] <= gpu.memory_total), None):
            return "the first active model that fits"
        return "as chosen"

    def label(model: str, note: str = "") -> str:
        notes = [n for n in ("" if model in scored else "candidate", note) if n]
        return model + (f" ({'; '.join(notes)})" if notes else "")

    def alternatives(gpu: Gpu) -> list[str]:
        return [m for m in offered if need[m] <= gpu.memory_total and m != chosen[gpu.index]]

    reasons: dict[tuple[str, str], list[int]] = {}
    for gpu in gpus:
        reasons.setdefault((chosen[gpu.index], why(gpu)), []).append(gpu.index)
    parts = [label(m) + ("" if len(gpus) == 1 else " on " + _gpu_list(indexes)) + ", " + reason
             for (m, reason), indexes in reasons.items()]
    others = [label(m, "" if len(gpus) == 1 else _gpu_list(fit)) for m in offered
              if (fit := [g.index for g in gpus if m in alternatives(g)])]
    if not others:
        return ["; ".join(parts) + ". No other catalog model fits"]
    # the example changes one GPU, to a model no other GPU serves where one fits
    example, used = [chosen[g.index] for g in gpus], set(chosen.values())
    position, model = next(((i, m) for i, g in reversed(list(enumerate(gpus))) for m in alternatives(g)
                            if m not in used), next((i, alternatives(g)[0]) for i, g in reversed(list(enumerate(gpus)))
                                                    if alternatives(g)))
    example[position] = model
    return ["; ".join(parts) + ". Also fit: " + ", ".join(others),
            f"another choice: --model <id> for every GPU, or one per GPU in order, as in --model {','.join(example)}"]


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
