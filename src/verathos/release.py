"""This release: its version and the weights version key validators send.

The version is ``major.minor.patch``, ``verathos.__version__`` (the package version reads it); a release is the
commit tagged ``v<version>``. Validators send ``major * 1_000_000 + minor * 1_000 + patch`` as the weights version
key, 1,000,000 for 1.0.0. The chain refuses weights whose key is below the subnet's ``WeightsVersionKey``: raising
that retires the validators of older releases.
"""

from __future__ import annotations

import re

from verathos import __version__

VERSION = re.compile(r"(\d+)\.(\d+)\.(\d+)")
TAG = re.compile(r"v(\d+)\.(\d+)\.(\d+)")


def parse(version: str, pattern: re.Pattern[str] = VERSION) -> tuple[int, int, int]:
    """``major.minor.patch`` (or, with ``TAG``, ``vmajor.minor.patch``) as numbers; minor and patch below 1000, so
    the weights version key orders as the versions do. Raises ValueError."""
    match = pattern.fullmatch(version)
    if match is None:
        raise ValueError(f"{version!r} is not a release version (major.minor.patch)")
    major, minor, patch = (int(p) for p in match.groups())
    if minor >= 1000 or patch >= 1000:
        raise ValueError(f"{version!r}: minor and patch are below 1000")
    return major, minor, patch


def weights_version_key(version: str = __version__) -> int:
    major, minor, patch = parse(version)
    return major * 1_000_000 + minor * 1_000 + patch
