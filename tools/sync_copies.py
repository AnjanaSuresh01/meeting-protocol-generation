"""Copy the canonical file of every mirrored module over its copies.

Several folders carry identical copies of the same modules so that each folder
runs on its own. After editing a canonical file, run
`python tools/sync_copies.py`; tests/test_copies_in_sync.py fails whenever the
copies drift.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = Path(__file__).resolve().parent / "copy_groups.json"


def main() -> int:
    groups = json.loads(MANIFEST.read_text(encoding="utf-8"))
    n = 0
    for g in groups:
        src = ROOT / g["canonical"]
        for c in g["copies"]:
            shutil.copyfile(src, ROOT / c)
            n += 1
            print(f"{g['canonical']} -> {c}")
    print(f"{n} copies synced")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
