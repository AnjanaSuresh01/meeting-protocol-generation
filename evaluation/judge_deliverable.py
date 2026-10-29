"""Measures, with a LISA judge, how much of the new content of each meeting a deliverable protocol captures.

The deliverable carries last week's protocol forward (carry_forward.edit_forward), so reference
bullets that were already in last week's protocol are covered by construction and say nothing
about the system. Only the new bullets are scored: reference bullets (judge_coverage.REF) that
word overlap does not find in the previous protocol (term_source.PREVIOUS). Word overlap misses a
bullet that the draft records in other wording, so each new bullet is also given to an LLM judge
(lisa-pro-03-2026, prompt judge_coverage.SYSTEM), which reports the draft bullet that covers it.

A judge citation counts only if it passes a mechanical test:

    1. it cites a draft bullet by number, and that number exists (judge_coverage's rule);
    2. the cited bullet shares at least one content word with the reference bullet.

Rule 2 is deliberately weak: it cannot confirm that the cited bullet says the same thing, it
only catches citations made at random. Citations that fail it are reported as `unevidenced`,
citations with an invalid number as `discarded`, and word-overlap coverage is reported beside
the judge's. The judge and the protocol writer are both LISA models, so the judge column is an
upper bound and the word-overlap column a lower one.

Each (meeting, arm) cell is appended to goldset/judge_deliverable.json as soon as it is judged,
and a cell already in that file is skipped, so an interrupted run resumes where it stopped.

    python judge_deliverable.py v6 v5               # default arm: v6
    python judge_deliverable.py --fresh v6          # ignore and overwrite earlier cells
"""
from __future__ import annotations

import json
import time
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import judge_coverage as J                                       # noqa: E402
from postedit_effort import content                              # noqa: E402
from term_source import PREVIOUS                                 # noqa: E402
from carry_tag import read_prev                                  # noqa: E402

OUT = REPO / "goldset" / "judge_deliverable.json"


def previous_bullets(mid: str) -> list[str]:
    p = REPO / PREVIOUS[mid]
    if p.suffix == ".docx":
        return [l.strip() for l in read_prev(p).splitlines() if len(l.split()) >= 3]
    return J.ref_bullets(p)


def judge_with_evidence(ref: list[str], draft: list[str], model_id: str) -> tuple[set, int, int]:
    """(1-based indices of covered ref bullets, citations with no content word in common,
    citations with an invalid number)."""
    import re
    import generate_protocols as G
    a = "\n".join(f"{i}. {t}" for i, t in enumerate(ref, 1))
    b = "\n".join(f"{i}. {t}" for i, t in enumerate(draft, 1))
    user = (f"=== A) REFERENZPROTOKOLL ({len(ref)} Punkte) ===\n{a}\n\n"
            f"=== B) GENERIERTES PROTOKOLL ({len(draft)} Punkte) ===\n{b}\n=== ENDE ===")
    raw = G.lisa(model_id, J.SYSTEM, user, timeout=600, retries=1, lang_gate=False)
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return set(), 0, 0
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return set(), 0, 0
    ok, unevidenced, bad = set(), 0, 0
    for e in data.get("covered", []):
        r, by = e.get("ref"), e.get("by")
        if not (isinstance(r, int) and 1 <= r <= len(ref)
                and isinstance(by, int) and 1 <= by <= len(draft)):
            bad += 1
            continue
        if content(ref[r - 1]) & content(draft[by - 1]):
            ok.add(r)
        else:
            unevidenced += 1
    return ok, unevidenced, bad


def main(argv: list[str]) -> int:
    fresh = "--fresh" in argv
    arms = [a for a in argv if not a.startswith("--")] or ["v6"]
    lisa = "lisa-pro-03-2026"
    rows = [] if fresh or not OUT.exists() else json.loads(OUT.read_text(encoding="utf-8"))
    done = {(r["meeting"], r["arm"]) for r in rows}
    for mid, ref_file in J.REF.items():
        his = J.ref_bullets(REPO / ref_file)
        carried = J.lexical_covered(his, previous_bullets(mid))
        new = [t for i, t in enumerate(his, 1) if i not in carried]
        if not new:
            continue
        for arm in arms:
            p = REPO / "outputs_deliverable" / arm / mid / "protocol_pro.md"
            if not p.exists() or (mid, arm) in done:
                continue
            draft = J.draft_bullets(p)
            lex = J.lexical_covered(new, draft)
            # The judge prompt (the new reference bullets plus every bullet of the draft) is
            # large, and the LISA endpoint can refuse large requests when it is busy, so a cell
            # is tried up to four times, with a pause of 60, 120, 180 and 240 s after each
            # failure, before it is given up.
            ok = None
            for attempt in range(4):
                try:
                    ok, unev, bad = judge_with_evidence(new, draft, lisa)
                    break
                except Exception as e:
                    print(f"  {mid}/{arm}: LISA refused ({type(e).__name__}), attempt "
                          f"{attempt + 1}/4", flush=True)
                    time.sleep(60 * (attempt + 1))
            if ok is None:
                print(f"  {mid}/{arm}: giving up on this cell; {len(rows)} saved, rerun to "
                      f"continue", flush=True)
                continue
            rows.append({"meeting": mid, "arm": arm, "new": len(new), "judge": len(ok),
                         "lexical": len(lex), "unevidenced": unev, "discarded": bad})
            OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"  {mid:8s} {arm:6s} ref NEW bullets {len(new):3d}  judge covers {len(ok):3d} "
                  f"({len(ok) / len(new):5.1%})  word-overlap {len(lex):3d} "
                  f"({len(lex) / len(new):5.1%})  citations rejected: no shared word {unev}, "
                  f"not a bullet {bad}", flush=True)
    print("\nPOOLED — of the reference bullets that were NOT in last week's protocol")
    for arm in arms:
        r = [x for x in rows if x["arm"] == arm]
        if not r:
            continue
        n = sum(x["new"] for x in r)
        j = sum(x["judge"] for x in r)
        l = sum(x["lexical"] for x in r)
        u = sum(x.get("unevidenced", 0) for x in r)
        print(f"  {arm:6s} n={n} over {len(r)} meeting(s)  judge {j} = {j / n:.1%} covered "
              f"(ADD {1 - j / n:.1%})   word-overlap {l} = {l / n:.1%} (ADD {1 - l / n:.1%})   "
              f"rejected citations {u}")
    print(f"\nwrote -> {OUT.relative_to(REPO)}")
    print("  LISA judges protocols LISA wrote, and a citation is only checked for one shared "
          "word, so the judge column is an UPPER bound and the word-overlap column a lower one.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
