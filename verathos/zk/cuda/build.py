#!/usr/bin/env python3
"""
Build script for zkllm native/CUDA extension.

Compiles C++ code for fast field arithmetic and sumcheck operations.
Optionally includes CUDA kernels for GPU acceleration.
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME
import os
import sys
import subprocess

# Get the directory containing this script
this_dir = os.path.dirname(os.path.abspath(__file__))

# setuptools writes ``--inplace`` outputs relative to the current working
# directory.  Normalize it here so ``python zkllm/cuda/build.py`` and
# ``cd zkllm/cuda && python build.py`` both place the extension beside this
# loader's sources, where ``verathos.zk.cuda`` expects to find it.
os.chdir(this_dir)

# Check if CUDA is available
HAS_CUDA = CUDA_HOME is not None


def get_cuda_version():
    """Get CUDA toolkit major.minor version as a tuple, e.g. (12, 8)."""
    nvcc = os.path.join(CUDA_HOME, "bin", "nvcc")
    try:
        out = subprocess.check_output([nvcc, "--version"], text=True)
        # e.g. "Cuda compilation tools, release 12.8, V12.8.93"
        for line in out.splitlines():
            if "release" in line:
                ver = line.split("release")[-1].split(",")[0].strip()
                parts = ver.split(".")
                return (int(parts[0]), int(parts[1]))
    except Exception:
        pass
    return (0, 0)


def get_gencode_flags(cuda_ver):
    """Return gencode flags supported by the detected CUDA toolkit version."""
    # (min_cuda_version, gencode_flag, comment)
    # fmt: off
    archs = [
        # Volta
        ((10, 0), "-gencode=arch=compute_70,code=sm_70",   "V100 16/32 GB"),
        # Turing
        ((10, 0), "-gencode=arch=compute_75,code=sm_75",   "T4 16 GB, Quadro RTX 6000/8000, RTX 20xx"),
        # Ampere
        ((10, 0), "-gencode=arch=compute_80,code=sm_80",   "A100 40/80 GB, A30 24 GB"),
        ((11, 1), "-gencode=arch=compute_86,code=sm_86",   "A10 24 GB, A40 48 GB, RTX A5000 24 GB, RTX A6000 48 GB, RTX 30xx"),
        # Ada Lovelace
        ((11, 8), "-gencode=arch=compute_89,code=sm_89",   "L4 24 GB, L40/L40S 48 GB, RTX 4090 24 GB, RTX 6000 Ada 48 GB"),
        # Hopper
        ((12, 0), "-gencode=arch=compute_90,code=sm_90",   "H100 80 GB, H200 141 GB, GH200"),
        # Blackwell datacenter
        ((12, 8), "-gencode=arch=compute_100,code=sm_100", "B100, B200 192 GB, GB200"),
        ((12, 9), "-gencode=arch=compute_103,code=sm_103", "B300 288 GB, GB300"),
        # Blackwell consumer/pro
        ((12, 8), "-gencode=arch=compute_120,code=sm_120", "RTX 5090 32 GB, RTX PRO 6000 Blackwell 96 GB"),
    ]
    # fmt: on
    flags = []
    for min_ver, flag, comment in archs:
        if cuda_ver >= min_ver:
            flags.append(f"{flag}")  # {comment}
            print(f"  {flag:<52s} # {comment}")
        else:
            print(f"  SKIP {flag:<47s} # {comment} (needs CUDA {min_ver[0]}.{min_ver[1]}+)")
    return flags


def get_local_gpu_gencode():
    """Auto-detect local GPU and return gencode flag for just that arch.

    Returns None if detection fails (fall back to all-arch build).
    Respects TORCH_CUDA_ARCH_LIST env var if set.
    """
    # Honor explicit override (e.g. TORCH_CUDA_ARCH_LIST="8.6")
    env_archs = os.environ.get("TORCH_CUDA_ARCH_LIST", "").strip()
    if env_archs:
        flags = []
        for arch_str in env_archs.replace(" ", "").split(";"):
            if not arch_str:
                continue
            # "8.6" → "86", "8.6+PTX" → "86" with PTX
            parts = arch_str.replace("+PTX", "").split(".")
            sm = parts[0] + (parts[1] if len(parts) > 1 else "0")
            flag = f"-gencode=arch=compute_{sm},code=sm_{sm}"
            flags.append(flag)
            print(f"  {flag:<52s} # from TORCH_CUDA_ARCH_LIST")
        return flags if flags else None

    # Auto-detect from local GPU
    try:
        import torch
        if torch.cuda.is_available():
            cc = torch.cuda.get_device_capability(0)
            sm = f"{cc[0]}{cc[1]}"
            gpu_name = torch.cuda.get_device_properties(0).name
            flag = f"-gencode=arch=compute_{sm},code=sm_{sm}"
            print(f"  {flag:<52s} # {gpu_name} (auto-detected)")
            return [flag]
    except Exception:
        pass
    return None


if HAS_CUDA:
    print(f"CUDA found at: {CUDA_HOME}")
    cuda_ver = get_cuda_version()
    print(f"CUDA toolkit version: {cuda_ver[0]}.{cuda_ver[1]}")
    print("Building with CUDA + OpenMP support...")
    print("GPU architectures:")

    # Prefer local GPU arch (fast build) over all-arch (portable but slow)
    gencode_flags = get_local_gpu_gencode()
    if gencode_flags is None:
        print("  (no local GPU detected — building for all supported archs)")
        gencode_flags = get_gencode_flags(cuda_ver)

    ext_modules = [
        CUDAExtension(
            name="zkllm_native",
            sources=[
                os.path.join(this_dir, "binding.cpp"),
                os.path.join(this_dir, "sumcheck_kernels.cu"),
                os.path.join(this_dir, "blake3_merkle.cu"),
            ],
            define_macros=[("WITH_CUDA", None)],
            extra_compile_args={
                # Release wheels run on heterogeneous miner hosts.  Do not
                # encode build-host-only CPU instructions into the CUDA
                # extension's host code.
                "cxx": ["-O3", "-mtune=generic", "-std=c++17", "-DWITH_CUDA", "-fopenmp"],
                "nvcc": [
                    "-O3",
                    "-Xcompiler", "-fPIC",
                    "-Xcompiler", "-fopenmp",
                    "--expt-relaxed-constexpr",
                ] + gencode_flags,
            },
            extra_link_args=["-fopenmp", "-lcrypto"],
        )
    ]
else:
    if sys.platform == "darwin":
        print("CUDA not found. Building CPU-only version without OpenMP on Darwin...")
        # The CPU-only extension does not use OpenMP pragmas. Avoid requiring
        # Homebrew libomp on Apple Silicon workers just to enable the native
        # field/sumcheck helpers.
        cpu_cxx_flags = ["-O3", "-ffast-math", "-std=c++17"]
        cpu_link_args = []
    else:
        print("CUDA not found. Building CPU-only version with OpenMP...")
        cpu_cxx_flags = [
            "-O3",
            "-march=native",
            "-ffast-math",
            "-std=c++17",
            "-fopenmp",
        ]
        cpu_link_args = ["-fopenmp"]

    ext_modules = [
        CppExtension(
            name="zkllm_native",
            sources=[
                os.path.join(this_dir, "binding.cpp"),
            ],
            extra_compile_args={
                "cxx": cpu_cxx_flags,
            },
            extra_link_args=cpu_link_args,
        )
    ]

setup(
    name="zkllm_native",
    version="0.2.0",
    description="zkllm native extension: fast field arithmetic, sumcheck, MLE" +
                (" + CUDA" if HAS_CUDA else ""),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    script_args=["build_ext", "--inplace"],
)
