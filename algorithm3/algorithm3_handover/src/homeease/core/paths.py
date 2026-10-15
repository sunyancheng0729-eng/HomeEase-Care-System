"""Project-relative paths.

All runtime paths are resolved against the project root so that commands work
from any working directory. The root is taken from ``HOMEEASE_ROOT`` when set,
otherwise the first parent directory of this file that contains a
``pyproject.toml`` declaring the ``homeease`` project.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def project_root() -> Path:
    env = os.environ.get("HOMEEASE_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        pp = parent / "pyproject.toml"
        if pp.is_file() and 'name = "homeease"' in pp.read_text(encoding="utf-8"):
            return parent
    return Path.cwd().resolve()


def resolve(path: str | os.PathLike[str] | None) -> Path | None:
    """Resolve a config path relative to the project root (absolute paths kept)."""
    if path is None:
        return None
    p = Path(path).expanduser()
    return p if p.is_absolute() else (project_root() / p)


def var_dir() -> Path:
    d = project_root() / "var"
    d.mkdir(parents=True, exist_ok=True)
    return d


def secrets_dir() -> Path:
    d = var_dir() / "secrets"
    d.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(d, 0o700)
    except OSError:
        pass
    return d


def frontend_dir() -> Path:
    return project_root() / "frontend"


def data_dir() -> Path:
    return project_root() / "data"


def models_dir() -> Path:
    return project_root() / "models"


def artifacts_dir() -> Path:
    d = project_root() / "artifacts" / "experiments"
    d.mkdir(parents=True, exist_ok=True)
    return d
