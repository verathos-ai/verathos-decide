"""Build the native code of the capacity audit in place: ``verathos-build-native``.

- ``verathos/capacity/bench/_workspace_cuda``: capacity and FP64 workload kernels (CUDA; miners).
- ``verathos/capacity/zk/native/_block_prover``: Q-block sumcheck prover (C++, CPU; miners, and tests).
- ``verathos/capacity/zk/libverathos_gemm.so``: GEMM sumcheck prover and int8 MLE evaluator (Rust; miners, and the
  audit validator for fast verification), built from ``native/gemm``.

Builds run from a source checkout (``pip install -e``).

CUDA targets the GPUs visible at build time unless ``TORCH_CUDA_ARCH_LIST`` is set.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from verathos.capacity.zk.gemm_native import LIBRARY_PATH

PACKAGE = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE.parents[1]
GEMM_CRATE = SRC_ROOT.parent / "native" / "gemm"
BUILD_DIR = SRC_ROOT.parent / "build" / "native"

_SETUP = """
import json, sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
spec = json.loads(sys.argv[1])
ext = (CUDAExtension if spec["kind"] == "cuda" else CppExtension)(
    name=spec["name"], sources=spec["sources"], extra_compile_args={"cxx": spec["cxx"], "nvcc": spec["nvcc"]},
    extra_link_args=spec["link"])
setup(name=spec["name"], ext_modules=[ext], cmdclass={"build_ext": BuildExtension},
      script_args=["build_ext", "--inplace", "--build-temp", spec["temp"], "--build-lib", spec["temp"]])
"""


def _cuda_toolkit() -> bool:
    from torch.utils.cpp_extension import CUDA_HOME

    return CUDA_HOME is not None


def _build_extension(kind: str, name: str, sources: list[Path], cxx: list[str], nvcc: list[str],
                     link: list[str]) -> None:
    print(f"building {name} ({kind})", flush=True)
    spec = {"kind": kind, "name": name, "sources": [str(s) for s in sources], "cxx": cxx, "nvcc": nvcc, "link": link,
            "temp": str(BUILD_DIR / name)}
    subprocess.run([sys.executable, "-c", _SETUP, json.dumps(spec)], cwd=SRC_ROOT, check=True)


def build_workspace() -> None:
    bench = PACKAGE / "bench"
    _build_extension("cuda", "verathos.capacity.bench._workspace_cuda",
                     [bench / "binding.cpp", bench / "workspace_kernel.cu"], ["-O3"],
                     ["-O3", "-Xcompiler", "-fPIC", "--expt-relaxed-constexpr"], ["-lcublas"])


def build_block_prover() -> None:
    _build_extension("cpu", "verathos.capacity.zk.native._block_prover", [PACKAGE / "zk" / "native" / "binding.cpp"],
                     ["-O3"], [], ["-lcrypto"])


def build_gemm() -> None:
    if not (GEMM_CRATE / "Cargo.toml").exists():
        raise SystemExit(f"{GEMM_CRATE} not found; the GEMM library builds from a source checkout")
    print("building libverathos_gemm (cargo)", flush=True)
    subprocess.run(["cargo", "build", "--release", "--locked"], cwd=GEMM_CRATE, check=True)
    shutil.copy2(GEMM_CRATE / "target" / "release" / LIBRARY_PATH.name, LIBRARY_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--validator", action="store_true", help="skip the CUDA workload kernels (no GPU needed)")
    args = parser.parse_args()
    if not args.validator and not _cuda_toolkit():
        raise SystemExit("no CUDA toolkit found (CUDA_HOME); miners need it, validators pass --validator")
    build_gemm()
    build_block_prover()
    if not args.validator:
        build_workspace()


if __name__ == "__main__":
    main()
