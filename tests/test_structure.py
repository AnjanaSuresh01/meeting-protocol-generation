"""Every module must import cleanly, without a .env and without network access.

Two modules read a goldset file at import time, so without the corpus they fail
with FileNotFoundError. That failure is also their frozen no-argument behavior
(see test_cli_surface), so it is pinned below instead of fixed. Any new import
failure fails the test. The second test checks that generate_protocols imports
with no LISA_* variable set.
"""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import ROOT

os.environ.setdefault("LISA_BASE_URL", "http://localhost:0")
os.environ.setdefault("LISA_API_KEY", "dummy")

FOLDERS = ["pipeline", "evaluation", "recognizers"]

# file -> substring of the pinned failure reason
PINNED_IMPORT_FAILURES = {
    "evaluation/check_turbo_seeding.py": "FileNotFoundError",
    "recognizers/whisper_long_redecode.py": "FileNotFoundError",
}


def _purge(folder: Path):
    for name in list(sys.modules):
        if name == "conftest" or name.startswith("test_"):
            continue
        mod = sys.modules[name]
        f = getattr(mod, "__file__", None)
        if f and str(ROOT) in f and str(folder) not in f:
            del sys.modules[name]


def test_all_modules_import():
    failures: dict[str, str] = {}
    for folder in FOLDERS:
        fdir = ROOT / folder
        files = sorted(fdir.glob("*.py"))
        if (fdir / "thesis_pipeline").exists():
            files += sorted((fdir / "thesis_pipeline").glob("*.py"))
        for f in files:
            if f.name == "__init__.py":
                continue
            rel = f.relative_to(fdir).with_suffix("")
            name = ".".join(rel.parts)
            key = f"{folder}/{rel.as_posix()}.py"
            _purge(fdir)
            sys.path.insert(0, str(fdir))
            try:
                importlib.import_module(name)
            except Exception as e:                                  # noqa: BLE001
                failures[key] = f"{type(e).__name__}: {str(e)[:120]}"
            finally:
                sys.path.remove(str(fdir))

    new = [k for k in failures if k not in PINNED_IMPORT_FAILURES]
    assert new == [], f"new import failures: {[(k, failures[k]) for k in new]}"
    for k, reason in failures.items():
        assert PINNED_IMPORT_FAILURES[k] in reason, (k, reason)


@pytest.mark.parametrize("folder", ["pipeline", "evaluation"])
def test_generate_protocols_imports_without_env(folder):
    env = {k: v for k, v in os.environ.items() if not k.startswith("LISA_")}
    code = ("import sys; sys.path.insert(0, r'%s'); import generate_protocols; "
            "print('import ok')" % (ROOT / folder))
    p = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), env=env,
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=120)
    assert p.returncode == 0, f"import raised:\n{p.stderr[-1500:]}"
    assert "import ok" in p.stdout
