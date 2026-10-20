"""Mirrored module copies must stay byte-identical, and tools/copy_groups.json must
list them.

After editing a canonical file run:  python tools/sync_copies.py

The duplicate groups are recomputed from disk and compared with the manifest, so
the test catches both drifted copies and a stale manifest.
"""
from __future__ import annotations

import hashlib
import json

from conftest import ROOT


def _sha(p) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _rank(p):
    s = p.as_posix()
    return (0 if s.startswith("pipeline") else 1 if s.startswith("evaluation") else 2, s)


def test_copies_in_sync_and_manifest_current():
    groups: dict[str, list] = {}
    for f in sorted(ROOT.rglob("*.py")):
        if ".git" in f.parts:
            continue
        groups.setdefault(_sha(f), []).append(f.relative_to(ROOT))
    actual = {}
    for paths in groups.values():
        if len(paths) < 2:
            continue
        paths = sorted(paths, key=_rank)
        actual[paths[0].as_posix()] = sorted(p.as_posix() for p in paths[1:])

    manifest = json.loads((ROOT / "tools" / "copy_groups.json").read_text(encoding="utf-8"))
    expected = {g["canonical"]: sorted(g["copies"]) for g in manifest}

    assert actual == expected, (
        "mirrored copies drifted or manifest is stale — "
        "run: python tools/sync_copies.py, then update tools/copy_groups.json if needed\n"
        f"only on disk: { {k: v for k, v in actual.items() if expected.get(k) != v} }\n"
        f"only in manifest: { {k: v for k, v in expected.items() if actual.get(k) != v} }")
