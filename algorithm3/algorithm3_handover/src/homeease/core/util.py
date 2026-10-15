"""Small shared helpers: ids, hashing, structured logging, code version."""

from __future__ import annotations

import hashlib
import json
import logging
import logging.handlers
import math
import os
import secrets
import sys
import uuid
from pathlib import Path
from typing import Any

from homeease.core.paths import project_root, var_dir


def new_id(prefix: str = "") -> str:
    u = uuid.uuid4().hex[:20]
    return f"{prefix}{u}" if prefix else u


def token(nbytes: int = 32) -> str:
    return secrets.token_urlsafe(nbytes)


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file(path: str | os.PathLike[str], chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


_CODE_VERSION: str | None = None


def code_version() -> str:
    """Content hash of the Python sources (works without git commits).

    Computed once per process, so a long run records the code it actually loaded even if
    the files are edited while it runs."""
    global _CODE_VERSION
    if _CODE_VERSION is None:
        root = project_root() / "src" / "homeease"
        h = hashlib.sha256()
        for p in sorted(root.rglob("*.py")):
            h.update(str(p.relative_to(root)).encode())
            h.update(p.read_bytes())
        _CODE_VERSION = h.hexdigest()[:12]
    return _CODE_VERSION


def finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def json_default(o: Any) -> Any:
    import datetime as _dt

    import numpy as np

    if isinstance(o, (_dt.datetime, _dt.date)):
        return o.isoformat()
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, set):
        return sorted(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def dumps(o: Any, **kw: Any) -> str:
    return json.dumps(o, default=json_default, ensure_ascii=False, **kw)


class JsonFormatter(logging.Formatter):
    """One JSON object per line; extra fields passed via ``extra={"ctx": {...}}``."""

    REDACT = ("token", "password", "secret", "authorization", "cookie")

    def format(self, record: logging.LogRecord) -> str:
        out: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S") + f".{int(record.msecs):03d}",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        ctx = getattr(record, "ctx", None)
        if isinstance(ctx, dict):
            for k, v in ctx.items():
                out[k] = "[redacted]" if any(r in k.lower() for r in self.REDACT) else v
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return dumps(out)


_configured = False


def setup_logging(level: str = "INFO", to_file: bool = True) -> None:
    global _configured
    if _configured:
        return
    root = logging.getLogger()
    root.setLevel(level)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(sh)
    if to_file:
        logdir = var_dir() / "logs"
        logdir.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            logdir / "homeease.jsonl", maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        fh.setFormatter(JsonFormatter())
        root.addHandler(fh)
    for noisy in ("uvicorn.access", "httpx", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
