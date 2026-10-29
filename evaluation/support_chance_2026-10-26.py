"""Measures the chance level of the "support in the transcript" rule for the missed new reference bullets.

Under the rule, a bullet has support when at least 30% of its content words occur in the evaluated
transcript. For the 34 new reference bullets that the v14ctrl arm missed in the blind reading, the
rule is applied to the transcript of the bullet's own meeting and to the transcript of every other
meeting, where the content cannot have been said. The pass rate on the other meetings is the chance
level of the rule; the pass rate on earlier meetings only is reported as well. A narrower rule (one
transcript turn shares at least 30%) is scored the same way.

Inputs
  goldset/missed_bullets_categories_2026-10-11.json   the 34 missed bullets (blind reading, v14ctrl)
  outputs_v11_final/canary_diar/<meeting>/transcript_speakers.txt   transcript of the evaluated runs
Rule
  postedit_effort.content() and MATCH_TH (0.30).

    python support_chance_2026-10-26.py   -> goldset/support_chance_2026-10-26.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from postedit_effort import content, MATCH_TH

REPO = Path(__file__).resolve().parent
MEET = ["march09", "april13", "april27", "may11", "may18", "june01", "june15", "june22"]
TURN = re.compile(r"^\[\d\d:\d\d:\d\d\]\s+[^:]+:\s*(.*)$")


def turns(mid: str) -> list[set[str]]:
    p = REPO / "outputs_v11_final" / "canary_diar" / mid / "transcript_speakers.txt"
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        m = TURN.match(line.strip())
        if m:
            c = content(m.group(1))
            if c:
                out.append(c)
    return out


def whole(item: set[str], tr: list[set[str]]) -> bool:
    pool = set().union(*tr)
    return len(item & pool) / len(item) >= MATCH_TH


def one_turn(item: set[str], tr: list[set[str]]) -> bool:
    return any(len(item & t) / len(item) >= MATCH_TH for t in tr)


def main() -> int:
    cats = json.loads((REPO / "goldset" / "missed_bullets_categories_2026-10-11.json")
                      .read_text(encoding="utf-8"))
    items = [(x["meeting"], x["text"]) for v in cats["items"].values() for x in v]
    tr = {m: turns(m) for m in MEET}
    res = {"own_whole": 0, "own_turn": 0, "n": 0,
           "other_whole": [0, 0], "other_turn": [0, 0],
           "earlier_whole": [0, 0], "earlier_turn": [0, 0], "rows": []}
    for mid, text in items:
        c = content(text)
        if not c:
            res["rows"].append({"meeting": mid, "text": text, "content_words": 0})
            continue
        res["n"] += 1
        ow, ot = whole(c, tr[mid]), one_turn(c, tr[mid])
        res["own_whole"] += ow
        res["own_turn"] += ot
        row = {"meeting": mid, "text": text, "own_whole": ow, "own_turn": ot, "other": {}}
        for m in MEET:
            if m == mid:
                continue
            w, t = whole(c, tr[m]), one_turn(c, tr[m])
            row["other"][m] = [w, t]
            res["other_whole"][0] += w
            res["other_whole"][1] += 1
            res["other_turn"][0] += t
            res["other_turn"][1] += 1
            if MEET.index(m) < MEET.index(mid):
                res["earlier_whole"][0] += w
                res["earlier_whole"][1] += 1
                res["earlier_turn"][0] += t
                res["earlier_turn"][1] += 1
        res["rows"].append(row)
    out = REPO / "goldset" / "support_chance_2026-10-26.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    pct = lambda a: f"{a[0]}/{a[1]} = {100 * a[0] / a[1]:.1f}%"
    print(f"bullets with content words: {res['n']} of {len(items)}")
    print(f"own meeting, whole transcript: {res['own_whole']}   one turn: {res['own_turn']}")
    print(f"other meetings, whole: {pct(res['other_whole'])}   one turn: {pct(res['other_turn'])}")
    print(f"earlier meetings, whole: {pct(res['earlier_whole'])}   one turn: {pct(res['earlier_turn'])}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
