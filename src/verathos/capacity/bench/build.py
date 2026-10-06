#!/usr/bin/env python3
"""Build the experimental fixed-workspace hot-capacity CUDA extension."""

from __future__ import annotations

import os
import subprocess

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME


ROOT = os.path.dirname(os.path.abspath(__file__))
EXTENSION_NAME = os.environ.get("HOT_CAPACITY_EXTENSION_NAME", "hot_capacity_workspace_cuda")


def _arch_to_flags(arch: str) -> list[str]:
    wants_ptx = arch.endswith("+PTX")
    normalized = arch.replace("+PTX", "").strip()
    sm = normalized.replace(".", "")
    flags = [f"-gencode=arch=compute_{sm},code=sm_{sm}"]
    if wants_ptx:
        flags.append(f"-gencode=arch=compute_{sm},code=compute_{sm}")
    return flags


def _normalize_capability(major: int, minor: int) -> str:
    # CUDA 12.8 supports Blackwell base targets such as sm_100, but not every
    # reported datacenter minor target, e.g. B300 reports capability 10.3.
    # Forward-compatible PTX is diagnostic-only because it can change the
    # benchmark path enough to invalidate calibration pass counts.
    if major == 10 and minor > 0:
        cuda_major = _cuda_major()
        if cuda_major >= 13:
            return f"{major}.{minor}"
        if os.environ.get("VERATHOS_HOT_CAPACITY_ALLOW_PTX_FALLBACK") != "1":
            raise SystemExit(
                f"CUDA toolkit does not support exact capability {major}.{minor}; "
                "refusing calibration fallback. Use a native toolkit or set "
                "VERATHOS_HOT_CAPACITY_ALLOW_PTX_FALLBACK=1 for diagnostic-only runs."
            )
        return "10.0+PTX"
    return f"{major}.{minor}"


def _local_arch_flags() -> list[str]:
    env_archs = os.environ.get("TORCH_CUDA_ARCH_LIST", "").strip()
    if env_archs:
        flags = []
        for raw_arch in env_archs.replace(" ", "").split(";"):
            arch = raw_arch.strip()
            if not arch:
                continue
            flags.extend(_arch_to_flags(arch))
        return flags

    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        return _arch_to_flags(_normalize_capability(major, minor))

    return []


def _cuda_version() -> str:
    if CUDA_HOME is None:
        return "unknown"
    try:
        out = subprocess.check_output([os.path.join(CUDA_HOME, "bin", "nvcc"), "--version"], text=True)
        for line in out.splitlines():
            if "release" in line:
                return line.split("release", 1)[1].split(",", 1)[0].strip()
    except Exception:
        pass
    return "unknown"


def _cuda_major() -> int:
    version = _cuda_version()
    try:
        return int(version.split(".", 1)[0])
    except Exception:
        return 0


if CUDA_HOME is None:
    raise SystemExit("CUDA_HOME not found; cannot build hot_capacity_workspace_cuda")

print(f"CUDA toolkit: {_cuda_version()}")
print("GPU arch flags:")
for flag in _local_arch_flags():
    print(f"  {flag}")

setup(
    name="hot_capacity_workspace_cuda",
    ext_modules=[
        CUDAExtension(
            name=EXTENSION_NAME,
            sources=[
                os.path.join(ROOT, "binding.cpp"),
                os.path.join(ROOT, "workspace_kernel.cu"),
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-Xcompiler", "-fPIC",
                    "--expt-relaxed-constexpr",
                ] + _local_arch_flags(),
            },
            extra_link_args=["-lcublas", "-lcublasLt"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
