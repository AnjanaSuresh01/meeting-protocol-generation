"""Compares the share of nested bullets in the reference protocols and in the v10 protocols, per meeting.

The reference is Meeting_<date>.md with its Backlog bullet and everything under it removed; only its
bullets at depth 2 or deeper count, and one is nested when its depth exceeds 2. In
outputs_deliverable/v10/<meeting>/protocol_pro.md every bullet counts, and one is nested when its depth
exceeds 0. Depth is the indentation divided by the file's indent unit
(evaluate_speaker_attribution.md_indent_unit, default 2; a tab counts as four spaces). april13 is
printed but left out of the totals, because its reference is a Wiki PDF export with an invalid
structure; the totals over the other seven meetings also give the number of meetings where v10 nests
more.

    python depth_compare_ch5.py

Prints to stdout only.
"""
import re, sys
from pathlib import Path
R = Path(__file__).resolve().parent
sys.path.insert(0, str(R))
from evaluate_speaker_attribution import md_indent_unit
BUL = re.compile(r"^(\s*)[*\-+]\s+(.*)$")
def items(lines):
    unit = md_indent_unit(lines) or 2
    return [(len(m.group(1)) // unit, m.group(2).strip()) for m in (BUL.match(l.replace("\t", "    ")) for l in lines) if m]
H = V = HS = VS = 0; higher = 0
for mid, date in [("march09","2026-03-09"),("april13","2026-04-13"),("april27","2026-04-27"),("may11","2026-05-11"),("may18","2026-05-18"),("june01","2026-06-01"),("june15","2026-06-15"),("june22","2026-06-22")]:
    keep, skip = [], None
    for d, b in items((R / f"Meeting_{date}.md").read_text(encoding="utf-8").splitlines()):
        if skip is not None and d <= skip: skip = None
        if skip is None and re.match(r"[*_]*backlog", b, re.I): skip = d; continue
        if skip is None: keep.append((d, b))
    hc = [d for d, b in keep if d >= 2]; hs = sum(1 for d in hc if d > 2)
    vc = [d for d, b in items((R / f"outputs_deliverable/v10/{mid}/protocol_pro.md").read_text(encoding="utf-8").splitlines())]
    vs = sum(1 for d in vc if d > 0)
    print(f"{mid:8s} ref {hs}/{len(hc)} = {hs/len(hc):.1%} | v10 {vs}/{len(vc)} = {vs/len(vc):.1%}")
    if mid != "april13":
        H += len(hc); HS += hs; V += len(vc); VS += vs; higher += vs/len(vc) > hs/len(hc)
print(f"SEVEN (april13 excluded): ref {HS}/{H} = {HS/H:.1%} | v10 {VS}/{V} = {VS/V:.1%} | v10 higher in {higher} of 7")
