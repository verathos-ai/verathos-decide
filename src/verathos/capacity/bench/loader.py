"""Runtime loader template for the hot-capacity native extension."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import sys
import sysconfig


def _torch_abi_tag(torch_version: object) -> str:
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
        raise ImportError(
            f"Unsupported torch version for hot-capacity runtime: {value or 'none'}"
        )
    if major != 2 or minor not in (10, 11):
        raise ImportError(
            f"Unsupported torch version for hot-capacity runtime: {value}"
        )
    return f"{major}{minor}"


def _get_torch_abi_tag() -> str:
    import torch

    return _torch_abi_tag(torch.__version__)


def _cuda_runtime_tag(cuda_version: object) -> str:
    """Return the exact qualified CUDA runtime family."""

    value = str(cuda_version or "")
    if value.startswith("12.8"):
        return "cu128"
    if value.startswith("13.0"):
        return "cu130"
    raise ImportError(
        "hot-capacity runtime supports torch CUDA 12.8 or 13.0; "
        f"found {value or 'none'}"
    )


def _get_cuda_runtime_tag() -> str:
    import torch

    return _cuda_runtime_tag(torch.version.cuda)


def _native_basename() -> str:
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    return f"_hot_capacity_workspace_cuda_native{ext_suffix}"


def _matching_tagged_so(
    native_dir: str,
    native_basename: str,
    torch_tag: str,
    cuda_tag: str,
) -> str | None:
    candidate = os.path.join(
        native_dir,
        f"{native_basename}.torch{torch_tag}.{cuda_tag}",
    )
    return candidate if os.path.isfile(candidate) else None


def _load_native():
    here = os.path.dirname(os.path.abspath(__file__))
    native_dir = os.path.join(here, "hot_capacity_workspace_cuda_native")
    torch_tag = _get_torch_abi_tag()
    cuda_tag = _get_cuda_runtime_tag()
    native_so = _matching_tagged_so(
        native_dir,
        _native_basename(),
        torch_tag,
        cuda_tag,
    )
    if native_so is None:
        raise ImportError(
            "No hot-capacity native extension for "
            f"torch{torch_tag}/{cuda_tag} in {native_dir}"
        )

    # The tagged binary is immutable release material. Loading it in place is
    # both supported by ExtensionFileLoader and safe when sibling miners import
    # concurrently. Never copy it to a shared active filename at runtime: one
    # process can otherwise map a partially replaced ELF while another copies.
    loader = importlib.machinery.ExtensionFileLoader(
        "_hot_capacity_workspace_cuda_native",
        native_so,
    )
    spec = importlib.util.spec_from_file_location(
        "_hot_capacity_workspace_cuda_native",
        native_so,
        loader=loader,
    )
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Cannot load hot-capacity native extension from {native_so}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules["_hot_capacity_workspace_cuda_native"] = module
    spec.loader.exec_module(module)
    return module


def _install_native_exports() -> None:
    native = _load_native()
    for name in dir(native):
        if not name.startswith("__"):
            globals()[name] = getattr(native, name)
    globals()["__all__"] = [name for name in globals() if not name.startswith("_")]


if __name__ == "hot_capacity_workspace_cuda":
    _install_native_exports()
