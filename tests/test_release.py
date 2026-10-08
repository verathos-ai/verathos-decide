"""The release version and the weights version key validators send."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

import verathos
from verathos import release

ROOT = Path(__file__).resolve().parents[1]
LEGACY_KEY = 2004  # the key of the last Verathos validators before this code


def test_the_weights_version_key_encodes_the_release_and_orders_as_the_versions_do() -> None:
    assert release.weights_version_key("1.0.0") == 1_000_000
    assert release.weights_version_key("2.13.7") == 2_013_007
    versions = ["1.0.0", "1.0.1", "1.0.999", "1.1.0", "1.999.0", "2.0.0", "10.0.0"]
    assert sorted(versions, key=release.weights_version_key) == versions
    assert [release.parse(v) for v in versions] == sorted(release.parse(v) for v in versions)
    assert release.weights_version_key() >= 1_000_000 > LEGACY_KEY  # raising the key to 1,000,000 retires them
    assert release.parse("v1.2.3", release.TAG) == (1, 2, 3)
    for bad in ("1.0", "1.0.0rc1", "v1.0.0", "1.1000.0", "1.0.1000", " 1.0.0"):
        with pytest.raises(ValueError):
            release.parse(bad)
    with pytest.raises(ValueError):
        release.parse("1.0.0", release.TAG)


def test_the_package_version_is_the_release() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert "version" in project["project"]["dynamic"] and "version" not in project["project"]
    assert project["tool"]["setuptools"]["dynamic"]["version"] == {"attr": "verathos.__version__"}
    release.parse(verathos.__version__)
