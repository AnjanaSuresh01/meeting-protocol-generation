"""Compares the name spellings of protocols generated without and with the binding-name rules 5a/5b.

Rules 5a and 5b of GLOSSARY_RULES:

    5a  a name in the list is binding; never replace it with a different name, however
        similar it sounds or however much more familiar it is
    5b  letter+digit product codes are copied character for character; an acronym in capitals
        stays an acronym and is not written out as a person or a word

The comparison uses the same 18 cells (6 recognizers x 3 meetings, LISA Pro): the protocols without the
rules in goldset/name_ab_before/<system>__<meeting>.md and those with the rules in
outputs_v10/<system>/<meeting>/protocol_pro_fixed_carry.md. For each name of the meeting in
goldset/name_reachability.json it counts, per cell:

    HIT    the correct spelling appears in the protocol
    MISS   only a wrong form appears
    ABSENT neither appears; this is not a spelling failure and is left out of the comparison

The per-cell change is (hits - misses with the rules) - (hits - misses without them). The cells are
compared in pairs, because they differ widely in how many of these names they mention: a sign test over
the cells that changed and a Wilcoxon test over all cells. The two sets of protocols come from separate
runs of a nondeterministic model, so part of any change is run-to-run noise.

    python name_ab_score.py    -> goldset/name_ab_score.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import entity_correct as E                                      # noqa: E402

BEFORE = REPO / "goldset" / "name_ab_before"
TREE = REPO / "outputs_v10"
MEETINGS = ["june01", "june15", "june22"]
SYSTEMS = ["canary_diar", "cohere", "granite41",
           "whisper_large_v3", "whisper_turbo", "whisper_turbo_de"]


def tokens(text: str) -> set[str]:
    return {E.squash(t) for t in text.split()}


def score(text: str, names: list[dict]) -> tuple[int, int, int]:
    """(hits, misses, absent) for the names belonging to this meeting."""
    tok = tokens(text)
    hit = miss = absent = 0
    for n in names:
        good = E.squash(n["reference"]) in tok
        bad = any(E.squash(h) in tok for h in n["heard_as"])
        if good:
            hit += 1
        elif bad:
            miss += 1
        else:
            absent += 1
    return hit, miss, absent


def main() -> int:
    rk = json.loads((REPO / "goldset" / "name_reachability.json").read_text(encoding="utf-8"))
    by_meeting: dict[str, list[dict]] = {}
    for r in rk["rows"]:
        by_meeting.setdefault(r["meeting"], []).append(r)

    print("BINDING-NAME RULES 5a/5b — measured, same 18 cells, one variable changed")
    print("=" * 92)
    print(f"{'system':18}{'meeting':9}{'before H/M':>13}{'after H/M':>12}{'change':>9}")
    print("-" * 92)

    rows, better, worse, same = [], 0, 0, 0
    tb_h = tb_m = ta_h = ta_m = 0
    for s in SYSTEMS:
        for mid in MEETINGS:
            names = by_meeting.get(mid, [])
            bp, ap = BEFORE / f"{s}__{mid}.md", TREE / s / mid / "protocol_pro_fixed_carry.md"
            if not names or not bp.exists() or not ap.exists():
                continue
            bh, bm, _ = score(bp.read_text(encoding="utf-8", errors="replace"), names)
            ah, am, _ = score(ap.read_text(encoding="utf-8", errors="replace"), names)
            tb_h += bh; tb_m += bm; ta_h += ah; ta_m += am
            d = (ah - am) - (bh - bm)          # net correct-minus-wrong, paired
            rows.append(d)
            better += d > 0
            worse += d < 0
            same += d == 0
            print(f"{s:18}{mid:9}{f'{bh}/{bm}':>13}{f'{ah}/{am}':>12}{d:>+9}")

    if not rows:
        print("\nnothing to compare — is goldset/name_ab_before populated?")
        return 1

    print("-" * 92)
    print(f"{'POOLED':27}{f'{tb_h}/{tb_m}':>13}{f'{ta_h}/{ta_m}':>12}"
          f"{(ta_h - ta_m) - (tb_h - tb_m):>+9}")
    print(f"\n  cells improved {better}, unchanged {same}, worse {worse}  (n = {len(rows)})")

    from scipy.stats import binomtest, wilcoxon
    moved = better + worse
    if moved:
        p = binomtest(better, moved).pvalue
        print(f"  sign test over the {moved} cells that moved: p = {p:.4f}"
              f"{'  SIGNIFICANT' if p < 0.05 else '  not significant'}")
    try:
        _, pw = wilcoxon(rows)
        print(f"  Wilcoxon over all {len(rows)} paired cells:      p = {pw:.4f}"
              f"{'  SIGNIFICANT' if pw < 0.05 else '  not significant'}")
    except ValueError:
        print("  Wilcoxon: all differences zero — no effect at all")

    print("\n  Before and after were generated in different sessions, so part of any movement is")
    print("  run-to-run noise. Treat a change of one or two names as nothing. If the sign test")
    print("  does not separate, the honest report is that the rules did not measurably help.")

    (REPO / "goldset" / "name_ab_score.json").write_text(
        json.dumps({"per_cell_delta": rows, "better": better, "worse": worse, "same": same,
                    "pooled": {"before_hit": tb_h, "before_miss": tb_m,
                               "after_hit": ta_h, "after_miss": ta_m}},
                   indent=1), encoding="utf-8")
    print("\nwrote -> goldset/name_ab_score.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
