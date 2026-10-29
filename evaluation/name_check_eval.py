"""Measures how well the names-to-check list of name_check.py catches names the ASR got wrong.

The reference is the hand-written minutes of each meeting, which spell the names correctly and
owe nothing to the ASR. For every meeting with a previous document (term_source.PREVIOUS):

    for each proper name in the minutes, did the ASR write it correctly?
    if not, does the names-to-check list (name_check.list_for_meeting, canary_diar) flag it?

A name is a reference spelling when it appears in the minutes of that meeting and is marked as
a name by its spelling (see is_name). It is mangled when that exact spelling (after
entity_correct.squash) is absent from all ASR hypotheses while a heard token with the same
Kölner Phonetik code and a SequenceMatcher ratio of at least 0.6 is present. Both steps are
automatic and neither consults a name list. Recall is the share of mangled names whose heard form
is among the listed variants; precision is the share of listed rows that match a mangled name.

Limits:
  * the minutes are summaries, so a name that was never written down cannot be scored;
  * a name that every ASR system gets wrong the same way still counts as a target here, although
    name_check cannot flag it, so recall is measured in the strict reading;
  * "phonetically close" uses the same Kölner code as name_check, so the reference is not
    independent of the module for that one step.

    python name_check_eval.py            # at the shipped settings: ceiling 0.75, top 10
    python name_check_eval.py --sweep    # recall and precision over agreement ceilings and list lengths

Output goes to stdout only; no file is written.
"""

from __future__ import annotations

import argparse
import difflib
import re
from collections import Counter
from pathlib import Path

import entity_correct as E
import name_check as NC
from term_source import PREVIOUS, GERMAN_COMMON

REPO = Path(__file__).resolve().parent
V10 = REPO / "outputs_v10"
MINUTES = {
    "march09": "Meeting-Protocol-FG-MM-AI_2026-03-09.docx",
    "april13": "Meeting_2026-04-13.md",
    "april27": "Meeting_2026-04-27.md",
    "may11": "Meeting_2026-05-11.md",
    "may18": "Meeting_2026-05-18.md",
    "june01": "Meeting_2026-06-01.md",
    "june15": "Meeting_2026-06-15.md",
    "june22": "Meeting_2026-06-22.md",
}


def minute_names(path: Path) -> set[str]:
    """Capitalized tokens of at least four characters from the minutes, minus common German words
    and name_check.SKIP.

    Deliberately wide: this only collects candidate reference spellings and does not decide what
    a name is. A candidate that the ASR reproduces exactly is discarded in mangled(), which
    removes almost everything that is not a name without having to judge it.
    """
    if not path.exists():
        return set()
    text = path.read_text(encoding="utf-8", errors="replace")
    out = set()
    for tok in re.findall(r"\b[A-ZÄÖÜ][\wäöüßA-ZÄÖÜ-]{3,}\b", text):
        if tok.lower() in GERMAN_COMMON or tok.lower() in NC.SKIP:
            continue
        out.add(tok)
    return out


MARKED = re.compile(
    "[a-z" + chr(228) + chr(246) + chr(252) + chr(223) + "][A-Z"
    + chr(196) + chr(214) + chr(220) + "]"          # internal capital: OlaPh, GemmaX, WebUI
    "|^[A-Z" + chr(196) + chr(214) + chr(220) + "]{3,}$"   # all caps: HAICON, MOSS
    "|[A-Za-z][0-9]|[0-9][A-Za-z]"                  # letters+digits: M4-SKI
    "|[A-Z" + chr(196) + chr(214) + chr(220) + "][a-z"
    + chr(228) + chr(246) + chr(252) + chr(223) + "]*-[A-Z"
    + chr(196) + chr(214) + chr(220) + "]")         # hyphenated capitals: Graph-RAG


def is_name(tok: str) -> bool:
    """True if the token is certainly a name and not ordinary German.

    German capitalizes every noun, so a capitalized token alone does not mark a name: scored
    against every capitalized token, pairs such as "Mitarbeitern -> Mitarbeiterin" or
    "Eintrag -> Antrag" would count as name errors. Without a dictionary, only orthographically
    marked tokens (MARKED: internal capital, all capitals, letters next to digits, hyphenated
    capitals) can be classified. That excludes real names such as Qwen, so this subset
    understates how many name errors exist: it is a clean sample, not a complete one.
    """
    return bool(MARKED.search(tok))


def mangled(mid: str) -> list[dict]:
    """Names the minutes spell one way and the ASR systems spell another, as
    [{"reference": ..., "heard_as": [...]}]."""
    refs = minute_names(REPO / MINUTES.get(mid, ""))
    if not refs:
        return []
    hyps = E._hyps(mid)
    if not hyps:
        return []
    heard: Counter = Counter()
    for segs in hyps.values():
        for t in segs.values():
            for tok in NC.CAP.findall(t):
                heard[tok] += 1
    heard_sq = {E.squash(t) for t in heard}
    by_code: dict[str, list[str]] = {}
    for tok in heard:
        by_code.setdefault(E.koelner(E.squash(tok)), []).append(tok)

    out = []
    for ref in refs:
        if not is_name(ref):
            continue
        rsq = E.squash(ref)
        if rsq in heard_sq:
            continue                      # the ASR got it right somewhere — not a defect
        cand = by_code.get(E.koelner(rsq), [])
        cand = [c for c in cand
                if difflib.SequenceMatcher(None, E.squash(c), rsq).ratio() >= 0.6]
        if not cand:
            continue                      # nothing heard is close: the name was written, not spoken
        out.append({"reference": ref, "heard_as": sorted(set(cand))})
    return out


def evaluate(max_agreement: float, top: int, cluster: float) -> dict:
    tp = fn = 0
    listed = 0
    matched_rows = 0
    detail = []
    for mid in MINUTES:
        if mid not in PREVIOUS:
            continue
        truth = mangled(mid)
        if not truth:
            continue
        shown = NC.list_for_meeting(mid, "canary_diar", max_agreement, top)
        listed += len(shown)
        shown_forms = {E.squash(v) for r in shown for v in r["variants"]}
        for t in truth:
            got = any(E.squash(h) in shown_forms for h in t["heard_as"])
            tp += got
            fn += not got
            if got:
                detail.append((mid, t["reference"], t["heard_as"]))
        matched_rows += sum(
            1 for r in shown
            if any(E.squash(h) in {E.squash(v) for v in r["variants"]}
                   for t in truth for h in t["heard_as"]))
    return {"caught": tp, "missed": fn,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "listed": listed,
            "precision": matched_rows / listed if listed else 0.0,
            "detail": detail}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args()

    print("DOES THE NAMES-TO-CHECK LIST CATCH REAL NAME ERRORS?")
    print("=" * 88)
    print("Reference = the human minutes. A name the minutes spell one way and every ASR system")
    print("spells another is a defect the list ought to flag.\n")

    total = 0
    for mid in MINUTES:
        if mid not in PREVIOUS:
            continue
        m = mangled(mid)
        total += len(m)
        print(f"  {mid:<9} {len(m):>3} names mangled by the ASR", end="")
        if m:
            ex = ", ".join(f"{d['reference']}->{d['heard_as'][0]}" for d in m[:3])
            print(f"   e.g. {ex}")
        else:
            print()
    print(f"\n  {total} scoreable cases in total\n")

    if a.sweep:
        print(f"  {'ceiling':>9}{'top':>5}{'cluster':>9}{'caught':>8}{'missed':>8}"
              f"{'recall':>9}{'listed':>8}{'precision':>11}")
        for ceiling in (0.50, 0.67, 0.75, 0.90, 1.0):
            for top in (10, 20):
                r = evaluate(ceiling, top, 0.70)
                print(f"  {ceiling:>9.2f}{top:>5}{0.70:>9.2f}{r['caught']:>8}{r['missed']:>8}"
                      f"{r['recall']:>8.0%}{r['listed']:>8}{r['precision']:>10.0%}")
        print("\n  A ceiling tuned to one example is not a validated setting. Pick the row.")
    else:
        r = evaluate(0.75, 10, 0.70)
        print(f"  at the shipped settings (ceiling 0.75, top 10):")
        print(f"    caught    {r['caught']}")
        print(f"    missed    {r['missed']}")
        print(f"    recall    {r['recall']:.0%}")
        print(f"    listed    {r['listed']} names, of which {r['precision']:.0%} are real errors")
        print("\n  caught, with the spelling of the minutes and what the machines heard:")
        for mid, ref, heard in r["detail"][:15]:
            print(f"    {mid:<9} {ref:<22} heard as {', '.join(heard[:4])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
