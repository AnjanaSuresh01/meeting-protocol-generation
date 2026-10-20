"""Shared test helpers.

Several folders contain modules with the same name (pipeline/, evaluation/ and
recognizers/ each have e.g. carry_tag.py). Tests load modules by explicit path and
keep exactly one folder on sys.path at a time, so two folders never mix their
imports within one pytest run.
"""
from __future__ import annotations

import contextlib
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@contextlib.contextmanager
def folder_on_path(folder: Path):
    folder = folder.resolve()
    for name in list(sys.modules):
        if name == "conftest" or name.startswith("test_"):
            continue
        mod = sys.modules[name]
        f = getattr(mod, "__file__", None)
        if f and str(ROOT) in f and str(folder) not in f:
            del sys.modules[name]
    sys.path.insert(0, str(folder))
    try:
        yield
    finally:
        sys.path.remove(str(folder))


def load(folder: str, name: str):
    """Import <folder>/<name>.py with its folder importable for sibling imports."""
    with folder_on_path(ROOT / folder):
        return importlib.import_module(name)
