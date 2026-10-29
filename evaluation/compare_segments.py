"""Tests whether one LISA call per report misses fewer new reference items than one call per meeting.

Arms (same code state, same v10 speaker names via run_cite_arm.py, nothing else different):
    v14ctrl   one LISA call for the whole meeting (the v10 method)
    v14seg    protokoll.py --segments: one call per report, parts merged per person and topic
v10 and v10pro_rep are printed beside them for scale; the mean absolute per-meeting difference
between these two runs of the same method on "new items missed", NOISE = 4.6 points, is the
run-to-run noise yardstick.

All counts use word overlap (postedit_effort) with the Backlog excluded: new items are reference
bullets that are not in the previous protocol, ADD counts reference bullets missing from the
draft, DELETE counts draft bullets that are not in the reference.

Decision rule, set before any result was read: adopt --segments only if, against v14ctrl,
    (a) the share of new items missed falls in at least 6 of 8 meetings, and
    (b) the pooled fall is larger than the noise yardstick (4.6 points), and
    (c) the structural self-check passes in all 8 meetings and no invented person name is left.
The change in DELETE is reported beside it: deleting a wrong line is cheap and noticing a
missing one is expensive, so the trade is stated as bullets to delete per item recovered.

    python compare_segments.py    -> goldset/compare_segments_2026-10-01.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
MEET = ["march09", "april13", "april27", "may11", "may18", "june01", "june15", "june22"]
ARMS = ["v10", "v10pro_rep", "v14ctrl", "v14seg"]
NOISE = 4.6


def per_meeting(arm: str) -> dict:
    import postedit_effort as PE
    import score_editforward as SE
    from evaluate_protocols import REF, strip_backlog
    out = {}
    for mid in MEET:
        t = SE.arm_text(arm, mid)
        if t is None or mid not in REF:
            continue
        ref = SE.no_backlog(strip_backlog((REPO / REF[mid]).read_text(encoding="utf-8"))[0])
        pb, rb, hb = PE.bullets(SE.previous_only(mid)), PE.bullets(ref), PE.bullets(t)
        new = [r for r in rb if not PE.covered(r, pb)]
        out[mid] = {"new": len(new), "new_missing": sum(not PE.covered(r, hb) for r in new),
                    "ref": len(rb), "add": sum(not PE.covered(r, hb) for r in rb),
                    "draft": len(hb), "delete": sum(not PE.covered(h, rb) for h in hb)}
    return out


def run_facts(arm: str, mid: str) -> dict:
    log = REPO / "outputs_deliverable" / arm / mid / "run_pro.log"
    if not log.exists():
        return {}
    s = log.read_text(encoding="utf-8", errors="replace")
    # the self-check is re-run on the current protocol file, not read from the run log, because
    # a later re-merge (e.g. the known_leaks repair) can change its result after the log was written
    import protokoll as P
    prot = REPO / "outputs_deliverable" / arm / mid / "protocol_pro.md"
    return {"self_check": not P.self_check(prot.read_text(encoding="utf-8")),
            "parts": len(re.findall(r"^\s+part \d+/\d+", s, re.M)),
            "seconds": int(m.group(1)) if (m := re.search(r"wall time: (\d+) s", s)) else None,
            "exit": int(m.group(1)) if (m := re.search(r"exit: (-?\d+)", s)) else None}


def invented(arm: str) -> int:
    import invented_names as IVN
    from carry_tag import read_prev
    from term_source import PREVIOUS
    n = 0
    for mid in MEET:
        p = REPO / f"outputs_deliverable/{arm}/{mid}/protocol_pro.md"
        if not p.exists():
            continue
        tx = (REPO / f"outputs_v11_final/canary_diar/{mid}/transcript_speakers.txt").read_text(encoding="utf-8")
        _, restored, flagged = IVN.apply(p.read_text(encoding="utf-8"), tx, read_prev(REPO / PREVIOUS[mid]))
        n += len(restored) + len(flagged)
    return n


def main() -> int:
    from scipy.stats import binomtest
    data = {a: per_meeting(a) for a in ARMS}
    both = [m for m in MEET if m in data["v14ctrl"] and m in data["v14seg"]]
    print(f"NEW ITEMS, ADD AND DELETE — word overlap (postedit_effort), Backlog excluded; "
          f"{len(both)} meetings with both new arms\n")
    hdr = f"  {'arm':11s}{'new missed':>18s}{'ADD':>16s}{'DELETE':>18s}{'bullets':>9s}"
    print(hdr)
    pooled = {}
    for a in ARMS:
        d = [data[a][m] for m in both if m in data[a]]
        if len(d) != len(both):
            continue
        n, nm = sum(x["new"] for x in d), sum(x["new_missing"] for x in d)
        r, ad = sum(x["ref"] for x in d), sum(x["add"] for x in d)
        g, de = sum(x["draft"] for x in d), sum(x["delete"] for x in d)
        pooled[a] = {"new": n, "new_missing": nm, "ref": r, "add": ad, "draft": g, "delete": de}
        print(f"  {a:11s}{nm:5d}/{n:<4d}{100 * nm / n:6.1f} %{ad:6d}/{r:<4d}{100 * ad / r:5.1f} %"
              f"{de:6d}/{g:<5d}{100 * de / g:5.1f} %{g:9d}")
    print()
    better = worse = 0
    print(f"  {'meeting':9s}{'ctrl missed':>13s}{'seg missed':>12s}{'ctrl delete':>13s}{'seg delete':>12s}")
    for m in both:
        c, s = data["v14ctrl"][m], data["v14seg"][m]
        cm, sm = 100 * c["new_missing"] / c["new"], 100 * s["new_missing"] / s["new"]
        better += sm < cm
        worse += sm > cm
        print(f"  {m:9s}{cm:12.1f}%{sm:11.1f}%{100 * c['delete'] / c['draft']:12.1f}%"
              f"{100 * s['delete'] / s['draft']:11.1f}%")
    p = binomtest(better, better + worse, 0.5).pvalue if better + worse else 1.0
    pc, ps = pooled.get("v14ctrl"), pooled.get("v14seg")
    drop = 100 * pc["new_missing"] / pc["new"] - 100 * ps["new_missing"] / ps["new"]
    rec = pc["new_missing"] - ps["new_missing"]
    extra_del = ps["delete"] - pc["delete"]
    checks = {a: [run_facts(a, m) for m in both] for a in ("v14ctrl", "v14seg")}
    sc = {a: sum(1 for f in checks[a] if f.get("self_check")) for a in checks}
    inv = {a: invented(a) for a in ("v14ctrl", "v14seg")}
    print(f"\n  segments better in {better}, worse in {worse} of {len(both)} meetings (sign test p = {p:.3f})")
    print(f"  pooled: new items missed {drop:+.1f} points in favour of segments (noise yardstick {NOISE})")
    print(f"  items recovered {rec}, extra bullets to delete {extra_del}"
          + (f" -> {extra_del / rec:.1f} deletions per recovered item" if rec > 0 else ""))
    print(f"  self-check passes: ctrl {sc['v14ctrl']}/{len(both)}, seg {sc['v14seg']}/{len(both)};"
          f" invented names left: ctrl {inv['v14ctrl']}, seg {inv['v14seg']}")
    secs = {a: sum(f.get("seconds") or 0 for f in checks[a]) for a in checks}
    print(f"  LISA wall time: ctrl {secs['v14ctrl']} s, seg {secs['v14seg']} s")
    adopt = (better >= 6 and drop > NOISE and sc["v14seg"] == len(both) and inv["v14seg"] == 0
             and len(both) == 8)
    print(f"\n  DECISION (rules fixed in advance): {'ADOPT --segments' if adopt else 'do NOT adopt'}")
    (REPO / "goldset" / "compare_segments_2026-10-01.json").write_text(json.dumps(
        {"per_meeting": data, "pooled": pooled, "better": better, "worse": worse, "sign_p": p,
         "drop_points": round(drop, 1), "recovered": rec, "extra_delete": extra_del,
         "self_check": sc, "invented": inv, "seconds": secs, "adopt": adopt}, indent=2,
        ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
