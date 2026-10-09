"""What the engines image installs before the release's code, read from its files with the standard library alone
(``docker/Dockerfile``, stage ``pins``), so that the layer holding the engines' environments stays the same from release
to release until one of these changes:

- ``kev-engines.txt``: the Kev engine's source and revision for the shipped catalog's active and candidate models, one
  ``<source> <revision>`` per line;
- ``clef-requirements.txt``: what ``verathos-engine``'s environment needs, the project's dependencies and its ``clef``
  extra, one per line.

The installer's own code then installs every engine as ``install.sh`` does (``verathos-container engines``) and finds
them in place.

    python pins.py <pyproject.toml> <catalog.json> <output directory>
"""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

SERVED = ("active", "candidate")


def kev_engines(catalog: dict) -> list[tuple[str, str]]:
    """The Kev engines (source, revision) of the catalog's active and candidate models, each once, in catalog order."""
    out: list[tuple[str, str]] = []
    for model in catalog["models"]:
        engine = model["engine"]
        if model["status"] in SERVED and engine["name"] == "kev" and (engine["source"], engine["revision"]) not in out:
            out.append((engine["source"], engine["revision"]))
    return out


def clef_requirements(pyproject: dict) -> list[str]:
    project = pyproject["project"]
    return [*project["dependencies"], *project["optional-dependencies"]["clef"]]


def main(argv: list[str]) -> None:
    pyproject_path, catalog_path, out = (Path(a) for a in argv)
    out.mkdir(parents=True, exist_ok=True)
    pyproject = tomllib.loads(pyproject_path.read_text())
    catalog = json.loads(catalog_path.read_text())
    (out / "kev-engines.txt").write_text("".join(f"{s} {r}\n" for s, r in kev_engines(catalog)))
    (out / "clef-requirements.txt").write_text("".join(f"{r}\n" for r in clef_requirements(pyproject)))


if __name__ == "__main__":
    main(sys.argv[1:])
