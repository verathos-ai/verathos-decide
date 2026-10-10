"""Configuration files: JSON read into a settings model, refused with where and what is wrong, never with the values
there (a configuration holds keys and passwords, and an error reaches the log)."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ValidationError


def load[Model: BaseModel](model: type[Model], path: Path) -> Model:
    """``path`` as ``model``. Raises SystemExit naming the file, and per problem its place and what is wrong."""
    try:
        document = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path} is not JSON: line {exc.lineno}, column {exc.colno}: {exc.msg}") from None
    try:
        return model.model_validate(document)
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(map(str, e['loc'])) or 'the document'}: {e['msg']}"
                             for e in exc.errors(include_url=False, include_input=False, include_context=False))
        raise SystemExit(f"{path}: {problems}") from None
