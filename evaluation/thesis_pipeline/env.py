"""Load a .env file (KEY=VALUE lines) into os.environ, without third-party packages.

    from thesis_pipeline.env import load_env
    load_env()                      # reads the .env one folder above thesis_pipeline/
    key = os.environ["LISA_API_KEY"]
"""

from __future__ import annotations

import os
from pathlib import Path


def load_env(path: str | Path | None = None, override: bool = False) -> dict[str, str]:
    """Read path (default: the .env one folder above thesis_pipeline/) into os.environ.

    Variables that are already set win unless override=True. Returns the keys that were set.
    Does nothing if the file does not exist.
    """
    p = Path(path) if path else Path(__file__).resolve().parent.parent / ".env"
    loaded: dict[str, str] = {}
    if not p.exists():
        return loaded
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and (override or key not in os.environ):
            os.environ[key] = val
            loaded[key] = val
    return loaded
