"""
Auto-detect torch version and load the correct zkllm_native CUDA extension.

When zkllm is installed as a wheel, exact Python, torch, and CUDA-runtime
variants are bundled. This module detects the installed runtime and loads the
matching extension at import time — no manual build/switch step.

When running from source (editable install), the existing .so file in this
directory is used directly.
"""
from __future__ import annotations

import importlib
import importlib.metadata as importlib_metadata
import importlib.machinery
import importlib.util
import logging
import os
import sys

log = logging.getLogger(__name__)

__all__ = ["zkllm_native", "HAS_CUDA"]


def _candidate_cuda_dirs(source_cuda_dir: str) -> tuple[str, ...]:
    """Return source plus the installed wheel's immutable CUDA directory."""

    directories = [os.path.abspath(source_cuda_dir)]
    try:
        distribution = importlib_metadata.distribution("zkllm")
        wheel_cuda_dir = os.path.abspath(
            os.fspath(distribution.locate_file("zkllm/cuda"))
        )
    except (importlib_metadata.PackageNotFoundError, OSError, TypeError):
        wheel_cuda_dir = ""
    if (
        wheel_cuda_dir
        and wheel_cuda_dir not in directories
        and os.path.isdir(wheel_cuda_dir)
    ):
        directories.append(wheel_cuda_dir)
    return tuple(directories)


def _torch_abi_tag(torch_version) -> str:
    """Return the exact qualified torch major/minor ABI tag."""

    value = str(torch_version or "")
    release = value.split("+", 1)[0]
    parts = release.split(".")
    try:
        major = int(parts[0])
        minor_text = parts[1]
        for marker in ("a", "b", "rc"):
            minor_text = minor_text.split(marker, 1)[0]
        minor = int(minor_text)
    except (IndexError, ValueError):
        raise ImportError(f"Unsupported torch version for zkllm_native: {value or 'none'}")
    if major != 2 or minor not in (10, 11, 12, 13):
        raise ImportError(f"Unsupported torch version for zkllm_native: {value}")
    return f"{major}{minor}"


def _get_torch_abi_tag() -> str:
    import torch

    return _torch_abi_tag(torch.__version__)


def _find_so_basename() -> str:
    """Detect the cpython SO name for this Python version and platform."""
    # e.g. "zkllm_native.cpython-310-x86_64-linux-gnu.so"
    import sysconfig
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    return f"zkllm_native{ext_suffix}"


def _cuda_runtime_tag(cuda_version) -> str:
    """Return the qualified CUDA runtime family for the loaded torch.

    A torch with NO CUDA runtime (macOS/Metal and CPU builds - the mac
    mesh workers run the CPU field-arithmetic extension) is a supported
    platform, not an error: it maps to the 'cpu' family so the tagged-so
    lookup stays exact per platform. Raising here broke every fresh Mac
    install at the loader before the on-box build could pair with its
    own torch (live 2026-08-18, ornith Mac roll)."""

    value = str(cuda_version or "")
    if not value:
        return "cpu"
    if value.startswith("12.8"):
        return "cu128"
    if value.startswith("13.0"):
        return "cu130"
    raise ImportError(
        "zkllm_native supports torch CUDA 12.8 or 13.0; "
        f"found {value}"
    )


def _get_cuda_runtime_tag() -> str:
    import torch

    return _cuda_runtime_tag(torch.version.cuda)


def _matching_tagged_so(
    cuda_dir: str,
    so_basename: str,
    tag: str,
    cuda_tag: str,
) -> str | None:
    """Return only the exact Python, torch and CUDA runtime binary."""

    candidate = os.path.join(
        cuda_dir,
        f"{so_basename}.torch{tag}.{cuda_tag}",
    )
    return candidate if os.path.isfile(candidate) else None


def _load_native():
    """Load zkllm_native, auto-selecting the correct .so for the torch version."""
    cuda_dir = os.path.dirname(os.path.abspath(__file__))
    so_basename = _find_so_basename()
    active_so = os.path.join(cuda_dir, so_basename)

    # Always check for a tagged version matching the current torch ABI.
    # The wheel may ship an untagged .so from a different build — the tagged
    # version is authoritative.
    tag = _get_torch_abi_tag()
    cuda_tag = _get_cuda_runtime_tag()
    binary_dirs = _candidate_cuda_dirs(cuda_dir)
    tagged_so = next(
        (
            candidate
            for directory in binary_dirs
            if (
                candidate := _matching_tagged_so(
                    directory,
                    so_basename,
                    tag,
                    cuda_tag,
                )
            )
        ),
        None,
    )

    if tagged_so is not None:
        # Load the immutable exact variant in place. Copying it over a shared
        # active filename creates a cross-process race when different runtime
        # stacks use the same release tree.
        loader = importlib.machinery.ExtensionFileLoader(
            "zkllm_native",
            tagged_so,
        )
        spec = importlib.util.spec_from_file_location(
            "zkllm_native",
            tagged_so,
            loader=loader,
        )
        if spec is None:
            raise ImportError(f"Unable to load exact zkllm_native variant: {tagged_so}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["zkllm_native"] = module
        try:
            loader.exec_module(module)
        except Exception:
            sys.modules.pop("zkllm_native", None)
            raise
        log.info(
            "zkllm_native: selected torch%s/%s build",
            tag,
            cuda_tag,
        )
        return module
    elif os.path.isfile(os.path.join(cuda_dir, "binding.cpp")) and os.path.isfile(
        active_so
    ):
        # Editable/source builds have one untagged extension compiled against
        # the active torch runtime. Release wheels remove binding.cpp and must
        # always carry the complete exact-tag matrix above.
        tagged_so = active_so
    else:
        raise ImportError(
            "No zkllm_native binary for "
            f"torch{tag}/{cuda_tag} and {so_basename} in {binary_dirs}"
        )

    if tagged_so == active_so and os.path.isfile(active_so):
        if cuda_dir not in sys.path:
            sys.path.insert(0, cuda_dir)
        return importlib.import_module("zkllm_native")

    raise ImportError(
        f"No zkllm_native .so found for torch{tag}/{cuda_tag} in {cuda_dir}"
    )


def _jit_compile():
    """JIT-compile the CUDA extension using torch.utils.cpp_extension.load().

    Fallback when the pre-built .so has an ABI mismatch (e.g. Python version
    mismatch from cross-compiled torch wheels).  JIT links against the running
    Python's torch libraries directly, avoiding the mismatch.
    """
    from torch.utils.cpp_extension import load as _load
    cuda_dir = os.path.dirname(os.path.abspath(__file__))
    sources = [
        os.path.join(cuda_dir, "binding.cpp"),
        os.path.join(cuda_dir, "blake3_merkle.cu"),
        os.path.join(cuda_dir, "sumcheck_kernels.cu"),
    ]
    if not all(os.path.isfile(s) for s in sources):
        raise ImportError("CUDA source files not found for JIT compilation")
    log.info("JIT-compiling zkllm CUDA extension (first load may take ~30s)...")
    return _load(
        name="zkllm_native",
        sources=sources,
        extra_cflags=["-O3", "-std=c++17", "-DWITH_CUDA", "-fopenmp"],
        extra_cuda_cflags=["-O3", "-Xcompiler", "-fPIC", "-Xcompiler", "-fopenmp",
                           "--expt-relaxed-constexpr"],
        extra_ldflags=["-fopenmp", "-lcrypto"],
        verbose=False,
    )


# Module-level singleton: load once at import time
zkllm_native = None
HAS_CUDA = False

try:
    zkllm_native = _load_native()
    HAS_CUDA = getattr(zkllm_native, "HAS_CUDA", False)
except Exception as e:
    log.debug("Pre-built zkllm_native failed: %s — trying JIT compilation", e)
    try:
        zkllm_native = _jit_compile()
        HAS_CUDA = getattr(zkllm_native, "HAS_CUDA", False)
    except Exception as e2:
        log.debug("JIT compilation also failed: %s", e2)
