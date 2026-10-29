# -*- coding: utf-8 -*-
"""Runs paired exact McNemar tests on the new reference bullets missed by v10, v10pro_rep and v10flash, and a
cross-meeting chance check of the 30% word rule.

A new reference bullet is a bullet of the reference protocol (Backlog removed) that the previous protocol does not
cover (postedit_effort.covered, 30% rule). Each pair of arms is tested on the bullets that only one of the two misses.
The chance check scores each meeting's new reference bullets against the v10 draft and the previous protocol of every
other meeting; the share counted as recorded there is the chance level. The scorers are those of postedit_effort and
score_editforward.

    python paired_and_null_2026-10-11.py [OUT.json]   -> OUT.json if given (a relative path is taken from the
                                                         script's folder); the result is always printed"""
import json
import sys
from math import comb
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import os
os.chdir(REPO)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import postedit_effort as PE
import score_editforward as SE
from evaluate_protocols import strip_backlog

MEET = ["march09", "april13", "april27", "may11", "may18", "june01", "june15", "june22"]
ARMS = ["v10", "v10pro_rep", "v10flash"]


def ref_bullets(mid):
    ref = SE.no_backlog(strip_backlog((REPO / SE.REF[mid]).read_text(encoding="utf-8"))[0])
    return PE.bullets(ref)


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n)


out = {"meetings": {}, "pairs": {}}
flags = {a: [] for a in ARMS}
prev_used = {}
for mid in MEET:
    rb = ref_bullets(mid)
    pb = PE.bullets(SE.previous_only(mid))
    prev_used[mid] = str(SE.PREV[mid]) if hasattr(SE, "PREV") else "?"
    hb = {a: PE.bullets(SE.arm_text(a, mid)) for a in ARMS}
    n_new = 0
    for r in rb:
        if PE.covered(r, pb):
            continue
        n_new += 1
        for a in ARMS:
            flags[a].append(not PE.covered(r, hb[a]))
    out["meetings"][mid] = {"new": n_new, **{a: len(hb[a]) for a in ARMS}}

for a in ARMS:
    out[a] = {"new": len(flags[a]), "missed": sum(flags[a])}
for x, y in [("v10", "v10pro_rep"), ("v10pro_rep", "v10flash"), ("v10", "v10flash")]:
    b = sum(1 for p, q in zip(flags[x], flags[y]) if p and not q)   # missed by x only
    c = sum(1 for p, q in zip(flags[x], flags[y]) if q and not p)   # missed by y only
    out["pairs"][f"{x} vs {y}"] = {"missed_by_first_only": b, "missed_by_second_only": c,
                                   "mcnemar_exact_p": round(mcnemar_exact(b, c), 4)}

# cross-meeting chance matches: share of a meeting's new reference bullets that the draft (v10) or
# the copy (previous protocol) of another meeting "records" under the 30% rule
same, cross_draft, cross_copy = [], [], []
newref = {}
for mid in MEET:
    pb = PE.bullets(SE.previous_only(mid))
    newref[mid] = [r for r in ref_bullets(mid) if not PE.covered(r, pb)]
for y in MEET:
    for x in MEET:
        hb = PE.bullets(SE.arm_text("v10", x))
        pbx = PE.bullets(SE.previous_only(x))
        rec = sum(PE.covered(r, hb) for r in newref[y])
        if x == y:
            same.append((rec, len(newref[y])))
        else:
            cross_draft.append((rec, len(newref[y])))
            cross_copy.append((sum(PE.covered(r, pbx) for r in newref[y]), len(newref[y])))
agg = lambda L: round(100 * sum(a for a, _ in L) / max(1, sum(n for _, n in L)), 1)
out["chance"] = {"same_meeting_draft_%": agg(same), "other_meeting_draft_%": agg(cross_draft),
                 "other_meeting_copy_%": agg(cross_copy), "pairs": len(cross_draft)}
out["previous_document"] = prev_used
print(json.dumps(out, indent=1, ensure_ascii=False))
Path(sys.argv[1]).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8") if len(sys.argv) > 1 else None
