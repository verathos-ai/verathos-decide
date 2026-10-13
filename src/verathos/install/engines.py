"""Engines and weights: the engine a catalog model names, at its pinned revision, the pinned weights, and the
command that serves the model.

Each engine family is a class with the same interface (``install``, ``python``, ``command``,
``environment_variables``), chosen by the catalog's ``engine.name`` in ``engine_for``; a new family is one more class
there and in ``plan.INSTALLABLE_ENGINES``. Each installs one environment per purpose: miners serve bfloat16 on the
GPU, references compute float32.
- ``kev``: the Kev engine from its source repository at ``engine.revision`` (``uv sync``); miners add the fused
  kernels (``flash-linear-attention`` at the version the engine pins), references run without them.
- ``clef``: ``verathos-engine`` from this release's code with the ``clef`` extra, which runs the inference code the
  model repository ships at ``hf_revision``; miners add ``flash-linear-attention`` at the version the fast path pins
  (``verathos.engine.qwen35.FLA_VERSION``), references run the release's own code without it.
- ``readout``: the same for code-readout models (d3, pplx-decider: ``verathos.engine.readout``), in an environment
  of its own.
Weights are the catalog's ``hf_repo`` at ``hf_revision`` and, for an adapter, ``base_repo`` at ``base_revision``, kept
in the setup's Hugging Face cache; the engine then runs offline, so it serves exactly the pinned revisions.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import httpx

from verathos import catalog
from verathos.engine import code_version
from verathos.install.plan import PlanError

HF_API = "https://huggingface.co/api/models/{repo}/revision/{revision}"
FLA_VERSION = re.compile(r'^FLA_VERSION\s*=\s*"([^"]+)"', re.M)
PYTHON = "3.12"
DOWNLOAD = """
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], revision=sys.argv[2])
"""


def uv() -> str:
    found = shutil.which("uv") or next((str(p) for p in (Path("~/.local/bin/uv").expanduser(),
                                                         Path("~/.cargo/bin/uv").expanduser()) if p.exists()), None)
    if found is None:
        raise SystemExit("uv not found; install it with: curl -LsSf https://astral.sh/uv/install.sh | sh")
    return found


def run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    """Run a step, its output shown; a failure stops the setup."""
    sys.stdout.flush()
    result = subprocess.run(cmd, cwd=cwd, env={**os.environ, **(env or {})})
    if result.returncode != 0:
        raise SystemExit(f"failed ({result.returncode}): {' '.join(cmd)}")


def git_output(directory: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(directory), *args], capture_output=True, text=True).stdout.strip()


def parameters(model: catalog.CatalogModel) -> int:
    """The model's parameter count: its base model's for an adapter (the adapter adds little), from the Hugging
    Face API at the pinned revision."""
    repo, revision = ((model.base_repo, model.base_revision) if model.base_repo and model.base_revision
                      else (model.hf_repo, model.hf_revision))
    headers = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if os.environ.get("HF_TOKEN") else {}
    try:
        r = httpx.get(HF_API.format(repo=repo, revision=revision), headers=headers, timeout=30,
                      follow_redirects=True)
        r.raise_for_status()
        return int(r.json()["safetensors"]["total"])
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
        raise PlanError(f"cannot read the size of {repo}@{revision[:12]} from Hugging Face: {exc}") from None


class KevEngine:
    """The Kev engine (``python -m kev.serve``) at one source revision, under ``<home>/engines``."""

    def __init__(self, home: Path, engine: catalog.Engine) -> None:
        self.engine = engine
        self.directory = home / "engines" / f"kev-{engine.revision[:12]}"

    def environment(self, purpose: str) -> Path:
        return self.directory / (".venv" if purpose == "miner" else ".venv-reference")

    def python(self, purpose: str) -> Path:
        return self.environment(purpose) / "bin" / "python"

    def install(self, purpose: str) -> None:
        """Check out the pinned revision and sync the environment for ``purpose`` ("miner" or "reference")."""
        revision = self.engine.revision
        if not (self.directory / ".git").exists():
            partial = self.directory.with_name(self.directory.name + ".partial")
            shutil.rmtree(partial, ignore_errors=True)
            partial.parent.mkdir(parents=True, exist_ok=True)
            run(["git", "clone", "--quiet", self.engine.source, str(partial)])
            partial.rename(self.directory)
        if git_output(self.directory, "rev-parse", "HEAD") != revision:
            if not git_output(self.directory, "cat-file", "-t", revision):
                run(["git", "-C", str(self.directory), "fetch", "--quiet", "origin"])
            run(["git", "-C", str(self.directory), "checkout", "--quiet", "--force", "--detach", revision])
        env = {"UV_PROJECT_ENVIRONMENT": str(self.environment(purpose))}
        if purpose == "reference":
            # An exact sync: nothing beyond the engine's lock file, so no fused kernels in the reference.
            run([uv(), "sync", "--quiet", "--extra", "serve", "-p", PYTHON], cwd=self.directory, env=env)
            return
        run([uv(), "sync", "--quiet", "--inexact", "--extra", "serve", "-p", PYTHON], cwd=self.directory, env=env)
        pin = FLA_VERSION.search((self.directory / "kev" / "fused_qwen35.py").read_text(errors="replace")) \
            if (self.directory / "kev" / "fused_qwen35.py").exists() else None
        if pin:
            run([uv(), "pip", "install", "--quiet", "--python", str(self.python(purpose)),
                 f"flash-linear-attention=={pin[1]}"])

    def command(self, purpose: str, model: catalog.CatalogModel, port: int) -> list[str]:
        return [str(self.python(purpose)), "-m", "kev.serve", "--run", f"{model.hf_repo}@{model.hf_revision}",
                "--host", "127.0.0.1", "--port", str(port)]

    @staticmethod
    def environment_variables(purpose: str, device: str, hf_home: Path) -> dict[str, str]:
        """``device``: "cpu" or "cuda:<index>" (nvidia-smi's numbering)."""
        env = {"HF_HOME": str(hf_home), "HF_HUB_OFFLINE": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
               "CUDA_VISIBLE_DEVICES": "" if device == "cpu" else device.split(":", 1)[1]}
        if purpose == "reference":
            env["KEV_DTYPE"] = "fp32"
        return env


CODE = Path(__file__).resolve().parents[3]  # the checkout this release runs from (holds pyproject.toml)
QWEN35 = CODE / "src" / "verathos" / "engine" / "qwen35.py"


class ClefEngine:
    """``verathos-engine clef`` from this release's code, under ``<home>/engines/verathos-clef``. Its source is this
    release, so the directory is shared by every Clef model and follows the code when it is updated."""

    family = "clef"

    def __init__(self, home: Path, engine: catalog.Engine) -> None:
        self.engine = engine
        self.directory = home / "engines" / f"verathos-{self.family}"

    def environment(self, purpose: str) -> Path:
        return self.directory / (".venv" if purpose == "miner" else ".venv-reference")

    def python(self, purpose: str) -> Path:
        return self.environment(purpose) / "bin" / "python"

    def install(self, purpose: str) -> None:
        """An environment for ``purpose`` ("miner" or "reference") with this release installed with the clef extra."""
        if not (CODE / "pyproject.toml").exists():
            raise SystemExit(f"the release's code is not at {CODE}; reinstall it with scripts/install.sh")
        self.directory.mkdir(parents=True, exist_ok=True)
        if not self.python(purpose).exists():
            run([uv(), "venv", "--quiet", "-p", PYTHON, str(self.environment(purpose))])
        run([uv(), "pip", "install", "--quiet", "--python", str(self.python(purpose)), "-e", f"{CODE}[clef]"])
        if purpose == "miner":
            pin = FLA_VERSION.search(QWEN35.read_text(errors="replace"))
            if pin is None:
                raise SystemExit(f"{QWEN35} names no FLA_VERSION")
            run([uv(), "pip", "install", "--quiet", "--python", str(self.python(purpose)),
                 f"flash-linear-attention=={pin[1]}"])

    def command(self, purpose: str, model: catalog.CatalogModel, port: int) -> list[str]:
        # the reference's device is its CUDA_VISIBLE_DEVICES (environment_variables): a GPU when one is visible
        device = ["--device", "cuda", "--dtype", "bfloat16"] if purpose == "miner" else \
            ["--device", "auto", "--dtype", "float32"]
        return [str(self.python(purpose)), "-m", "verathos.engine", self.family, "--repo", model.hf_repo,
                "--revision", model.hf_revision, "--host", "127.0.0.1", "--port", str(port), *device]

    @staticmethod
    def environment_variables(purpose: str, device: str, hf_home: Path) -> dict[str, str]:
        """``device``: "cpu" or "cuda:<index>" (nvidia-smi's numbering). ``VERATHOS_ENGINE_CODE`` changes the service
        when an update changes the engine's code, so the engine restarts on it."""
        return {"HF_HOME": str(hf_home), "HF_HUB_OFFLINE": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": "" if device == "cpu" else device.split(":", 1)[1],
                "VERATHOS_ENGINE_CODE": code_version()}


class ReadoutEngine(ClefEngine):
    """``verathos-engine readout`` from this release's code, under ``<home>/engines/verathos-readout``, set up as the
    Clef engine is."""

    family = "readout"


ENGINES = {"kev": KevEngine, "clef": ClefEngine, "readout": ReadoutEngine}


def engine_for(home: Path, model: catalog.CatalogModel) -> KevEngine | ClefEngine:
    family = ENGINES.get(model.engine.name)
    if family is None:
        raise PlanError(f"{model.model_id} runs on the {model.engine.name} engine, which this release does not "
                        "install; set it up by hand (docs/MINING.md, docs/VALIDATING.md)")
    return family(home, model.engine)


def download(python: Path, model: catalog.CatalogModel, hf_home: Path) -> None:
    """The model's pinned weights (and its base model's, for an adapter) into the cache at ``hf_home``; a completed
    download is recorded and not checked again."""
    pins = [(model.hf_repo, model.hf_revision)]
    if model.base_repo and model.base_revision:
        pins.append((model.base_repo, model.base_revision))
    env = {"HF_HOME": str(hf_home)}
    if not sys.stdout.isatty():
        env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    for repo, revision in pins:
        done = hf_home / "verathos-complete" / f"{repo.replace('/', '--')}@{revision}"
        if done.exists():
            print(f"    {repo}@{revision[:12]}: already downloaded", flush=True)
            continue
        print(f"    {repo}@{revision[:12]}: downloading", flush=True)
        run([str(python), "-c", DOWNLOAD, repo, revision], env=env)
        done.parent.mkdir(parents=True, exist_ok=True)
        done.touch()
