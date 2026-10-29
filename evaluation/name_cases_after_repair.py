"""Scores the protocols in outputs_deliverable/ on two sets of name cases, drawn before and after the long segments
were re-decoded with Whisper.

Old cases: goldset/name_reachability.json, drawn from the decodes before the long-segment re-decode.
New cases: name_check_eval.mangled() on the current decodes, for the same six meetings.
Each case is scored as right, wrong, absent or undecidable with the logic of score_editforward.names(), applied to a
given list of rows. Every arm with all protocols present is printed; the JSON file holds the cases dropped and added
and the counts for v10 and v9.

    python name_cases_after_repair.py   -> goldset/name_cases_after_repair_2026-10-29.json
"""
import json, os, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
import score_editforward as S
import name_ab_score as N
import name_check_eval as NCE


def score(rows, arm, ok_words):
    h = m = a = u = 0
    for r in rows:
        p = ROOT / "outputs_deliverable" / arm / r["meeting"] / "protocol_pro.md"
        if not p.exists():
            return None
        raw = p.read_text(encoding="utf-8", errors="replace")
        tok = N.tokens(raw)
        tok |= {part for t in tok for part in re.split(r"[-/]", t) if part}
        named = {N.E.squash(w) for w in re.findall(r"[\wÄÖÜäöüß-]+", raw) if not w.islower()}
        named |= {N.E.squash(x) for w in named for x in re.split(r"[-/]", w) if x}
        variants = [x for x in r["heard_as"] if N.E.squash(x) not in ok_words and N.E.squash(x) in named]
        dropped = len([x for x in r["heard_as"] if N.E.squash(x) not in variants])
        if N.E.squash(r["reference"]) in tok:
            h += 1
        elif any(N.E.squash(x) in tok for x in variants):
            m += 1
        elif dropped and any(N.E.squash(x) in tok for x in r["heard_as"]):
            u += 1
        else:
            a += 1
    return (h, m, a, u)


old = json.load(open(ROOT / "goldset" / "name_reachability.json", encoding="utf-8"))["rows"]
six = sorted({r["meeting"] for r in old})
new = [{"meeting": mid, **x} for mid in six for x in NCE.mangled(mid)]
ok = S.legitimate()
arms = sorted(d.name for d in (ROOT / "outputs_deliverable").iterdir() if d.is_dir())
for arm in arms:
    o = score(old, arm, ok)
    if o is None:
        continue
    n = score(new, arm, ok)
    print("%-28s old 31: right %d wrong %d absent %d undecidable %d | new %d: right %d wrong %d absent %d undecidable %d"
          % ((arm,) + o + (len(new),) + n))
res = {"note": "old = goldset/name_reachability.json (24 Sep, before the 25 Sep Whisper redecode); new = name_check_eval.mangled() on current decodes, same six meetings",
       "old_cases": len(old), "new_cases": len(new),
       "dropped": sorted({(r["meeting"], r["reference"]) for r in old} - {(r["meeting"], r["reference"]) for r in new}),
       "added": sorted({(r["meeting"], r["reference"]) for r in new} - {(r["meeting"], r["reference"]) for r in old}),
       "evaluated_runs_v10": dict(zip(["old_right", "old_wrong", "old_absent", "old_undecidable"], score(old, "v10", ok))) | dict(zip(["new_right", "new_wrong", "new_absent", "new_undecidable"], score(new, "v10", ok))),
       "previous_protocol_list_v9": dict(zip(["old_right", "old_wrong", "old_absent", "old_undecidable"], score(old, "v9", ok))) | dict(zip(["new_right", "new_wrong", "new_absent", "new_undecidable"], score(new, "v9", ok)))}
(ROOT / "goldset" / "name_cases_after_repair_2026-10-29.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False)[:600])
