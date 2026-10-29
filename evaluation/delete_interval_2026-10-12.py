# -*- coding: utf-8 -*-
"""Computes a 95% interval for the DELETE difference between the v10 draft and last week's protocol.

DELETE is the share of a draft's bullets that cover no scored reference bullet (postedit_effort.covered).
The meetings are loaded with strip_copies_rescore.load_meeting, and the interval uses that script's
two-stage bootstrap: meetings drawn with replacement, then the bullets of each drawn meeting, with the
same seed (SEED) and number of draws (N_BOOT); the matching ADD interval comes from strip_copies_rescore.py
itself. The output gives both pooled DELETE rates, their difference, the percentile interval of the
difference and the number of meetings in which the draft's DELETE rate is higher.

    python delete_interval_2026-10-12.py    -> goldset/delete_interval_2026-10-12.json
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.stdout.reconfigure(encoding="utf-8")
import postedit_effort as PE  # noqa: E402
import strip_copies_rescore as SR  # noqa: E402


def main() -> int:
    meetings = {mid: SR.load_meeting(mid) for mid in SR.REF}
    draft, copy = [], []
    for M in meetings.values():
        dc = [d["content"] for d in M["draft"]]
        pc = [d["content"] for d in M["prev"]]
        rb = [n["content"] for n in M["nodes"] if n["scored"]]
        draft.append(np.array([not PE.covered(h, rb) for h in dc], dtype=float))
        copy.append(np.array([not PE.covered(h, rb) for h in pc], dtype=float))
    pooled = (sum(d.sum() for d in draft) / sum(len(d) for d in draft),
              sum(c.sum() for c in copy) / sum(len(c) for c in copy))
    rng = np.random.default_rng(SR.SEED)
    m = len(draft)
    diffs = np.empty(SR.N_BOOT)
    for b in range(SR.N_BOOT):
        dd = dn = cd = cn = 0.0
        for j in rng.integers(0, m, m):
            d, c = draft[j], copy[j]
            if len(d):
                dd += d[rng.integers(0, len(d), len(d))].sum()
                dn += len(d)
            if len(c):
                cd += c[rng.integers(0, len(c), len(c))].sum()
                cn += len(c)
        diffs[b] = dd / dn - cd / cn
    v = np.sort(diffs)
    out = {"delete_draft": round(100 * pooled[0], 2), "delete_copy": round(100 * pooled[1], 2),
           "difference": round(100 * (pooled[0] - pooled[1]), 2),
           "ci95": [round(100 * float(v[int(0.025 * len(v))]), 1), round(100 * float(v[int(0.975 * len(v))]), 1)],
           "n_boot": SR.N_BOOT, "seed": SR.SEED,
           "per_meeting_draft_higher": int(sum(d.mean() > c.mean() for d, c in zip(draft, copy)))}
    print(out)
    (REPO / "goldset" / "delete_interval_2026-10-12.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
