"""Labels the previous-document name flags that reach a v10 protocol as right or wrong, to show which features could rank them.

name_check.uncertain_single returns its "klingt wie X aus dem vorherigen Dokument" flags in the
order in which the words first occur in the transcript. protokoll.py shows only a limited number of
these flags, so their order decides which ones the reader sees.

The label is objective and needs no knowledge of German (the same design as name_majority_gate.py).
Only flags whose heard word occurs in the v10 protocol (outputs_deliverable/v10) are labeled. A flag
"heard X, previous document has Y" is
    RIGHT        if the reference protocol of the same meeting writes Y and not X,
    WRONG        if the reference protocol writes X (then X is a real word of the meeting),
    undecidable  otherwise.
Each flag carries the features a ranking could use (position in the flag list, letter similarity of
X and Y, count of X in the transcript, the reason Y was extracted as a term); the table shows which
of them separate right from wrong flags.

    python near_flag_gate.py   -> stdout only
"""
from __future__ import annotations

import difflib
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
MEET = ["march09", "april13", "april27", "may11", "may18", "june01", "june15", "june22"]
DATE = {"march09": "2026-03-09", "april13": "2026-04-13", "april27": "2026-04-27",
        "may11": "2026-05-11", "may18": "2026-05-18", "june01": "2026-06-01",
        "june15": "2026-06-15", "june22": "2026-06-22"}


def history(mid: str, prev: Path) -> list[Path]:
    seen, out = set(), []
    for p in sorted(REPO.glob("Meeting_2026-*.md")):
        m = re.search(r"(\d{4}-\d{2}-\d{2})", p.name)
        if m and m.group(1) < DATE[mid] and p.name != prev.name and m.group(1) not in seen:
            seen.add(m.group(1))
            out.append(p)
    return out


def has(word: str, text: str) -> bool:
    return re.search(rf"(?<![\wÄÖÜäöüß]){re.escape(word)}(?![\wÄÖÜäöüß])", text, re.I) is not None


def rows() -> list[dict]:
    import entity_correct as E
    import name_check as NC
    from evaluate_speaker_attribution import HUMAN
    from term_source import PREVIOUS, extract
    out = []
    for mid in MEET:
        prev = REPO / PREVIOUS[mid]
        hist = history(mid, prev)
        tx = (REPO / f"outputs_v11_final/canary_diar/{mid}/transcript_speakers.txt").read_text(
            encoding="utf-8")
        prot = (REPO / f"outputs_deliverable/v10/{mid}/protocol_pro.md").read_text(encoding="utf-8")
        his = re.sub(r"<[^>]+>", "", (REPO / HUMAN[mid]).read_text(encoding="utf-8"))
        why = {}
        for d in [prev] + hist:
            for k, v in extract(d).items():
                why.setdefault(k, v)
        flags = NC.uncertain_single(tx, prev, hist)
        for pos, f in enumerate(flags):
            tok, can = f["majority"], f["canonical"]
            if not has(tok, prot):
                continue                        # not in the protocol, so the flag costs the reader nothing
            r_his, c_his = has(tok, his), has(can, his)
            verdict = ("WRONG" if r_his else "RIGHT" if c_his else "undecidable")
            sim = difflib.SequenceMatcher(None, E.squash(tok), E.squash(can)).ratio()
            out.append({"meeting": mid, "heard": tok, "suggest": can, "verdict": verdict,
                        "position": pos, "sim": round(sim, 2),
                        "marked": why.get(can, "?"),
                        "tok_in_tx": len(re.findall(rf"(?<!\w){re.escape(tok)}(?!\w)", tx))})
    return out


def main() -> int:
    rs = rows()
    from collections import Counter
    print(f"{len(rs)} previous-document flags reach a v10 protocol; labels: "
          f"{dict(Counter(r['verdict'] for r in rs))}")
    print(f"\n  {'meeting':8s} {'heard':16s} {'suggest':16s} {'label':12s} {'sim':>5s} {'x in tx':>8s}  why the suggestion is a term")
    for r in sorted(rs, key=lambda r: (r["verdict"], -r["sim"])):
        print(f"  {r['meeting']:8s} {r['heard'][:16]:16s} {r['suggest'][:16]:16s} {r['verdict']:12s} "
              f"{r['sim']:5.2f} {r['tok_in_tx']:8d}  {r['marked']}")
    print("\n  by the reason the SUGGESTED term was extracted:")
    by = {}
    for r in rs:
        by.setdefault(r["marked"], Counter())[r["verdict"]] += 1
    for k, c in sorted(by.items(), key=lambda kv: -sum(kv[1].values())):
        print(f"    {k:40s} {dict(c)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
