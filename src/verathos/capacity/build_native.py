"""Build the native code of the capacity audit in place: ``verathos-build-native``.

- ``verathos/capacity/bench/_workspace_cuda``: capacity and FP64 workload kernels (CUDA; miners).
- ``verathos/capacity/zk/cuda/_native``: field arithmetic, sumcheck and BLAKE3 Merkle kernels (CUDA when a toolkit
  is present, CPU otherwise; miners, optional speed-up for the validator).
- ``verathos/capacity/zk/libverathos_pcs_v2.so``: polynomial-commitment GEMM prover and verifier (Rust; miners
  and the audit validator), built from ``native/pcs`` of a source checkout.

CUDA targets the GPUs visible at build time unless ``TORCH_CUDA_ARCH_LIST`` is set.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from verathos.capacity.zk.crypto.pcs_v2 import LIBRARY_PATH

PACKAGE = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE.parents[1]
PCS_CRATE = SRC_ROOT.parent / "native" / "pcs"
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
                     [bench / "binding.cpp", bench / "workspace_kernel.cu"], ["-O3", "-std=c++17"],
                     ["-O3", "-Xcompiler", "-fPIC", "--expt-relaxed-constexpr"], ["-lcublas", "-lcublasLt"])


def build_zk(cuda: bool) -> None:
    zk = PACKAGE / "zk" / "cuda"
    if cuda:
        _build_extension("cuda", "verathos.capacity.zk.cuda._native",
                         [zk / "binding.cpp", zk / "sumcheck_kernels.cu", zk / "blake3_merkle.cu"],
                         ["-O3", "-std=c++17", "-DWITH_CUDA", "-fopenmp"],
                         ["-O3", "-Xcompiler", "-fPIC", "-Xcompiler", "-fopenmp", "--expt-relaxed-constexpr",
                          "-DWITH_CUDA"], ["-fopenmp", "-lcrypto"])
    else:
        _build_extension("cpu", "verathos.capacity.zk.cuda._native", [zk / "binding.cpp"],
                         ["-O3", "-std=c++17", "-fopenmp"], [], ["-fopenmp", "-lcrypto"])


def build_pcs() -> None:
    if not (PCS_CRATE / "Cargo.toml").exists():
        raise SystemExit(f"{PCS_CRATE} not found; the PCS library builds from a source checkout")
    print("building libverathos_pcs (cargo)", flush=True)
    subprocess.run(["cargo", "build", "--release", "--locked"], cwd=PCS_CRATE, check=True)
    shutil.copy2(PCS_CRATE / "target" / "release" / LIBRARY_PATH.name, LIBRARY_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--validator", action="store_true",
                        help="build only what the audit validator needs (PCS library and CPU zk kernels)")
    args = parser.parse_args()
    cuda = not args.validator and _cuda_toolkit()
    if not args.validator and not cuda:
        raise SystemExit("no CUDA toolkit found (CUDA_HOME); miners need it, validators pass --validator")
    build_pcs()
    build_zk(cuda)
    if cuda:
        build_workspace()


if __name__ == "__main__":
    main()
