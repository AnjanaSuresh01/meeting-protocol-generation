"""Tests whether a majority-vote name corrector would be safe on the v10 protocols.

The corrector under test: where the recognizers spell a word differently (name_check.flags with at
least three systems) and the majority form reaches an agreement of at least 0.6, a v10 spelling
that differs from the majority form would be replaced by it. For each such disagreement the script
takes the first non-majority variant that occurs in outputs_deliverable/v10/<meeting>/protocol_pro.md
(whole word, case-insensitive) and checks it against the human-written protocol of the same meeting
(evaluate_speaker_attribution.HUMAN):

  corrupt      the reference contains the v10 variant and not the majority form
  fix          the reference contains the majority form and not the v10 variant
  undecidable  otherwise

It also counts how often the majority form is a term of the previous protocol
(term_source.glossary_for_document) and prints the precision fix / (fix + corrupt).

    python name_majority_gate.py

Prints to stdout only.
"""
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import name_check as NC                                            # noqa: E402
from evaluate_speaker_attribution import HUMAN                     # noqa: E402
from term_source import PREVIOUS, glossary_for_document            # noqa: E402

MEET = ["march09", "april13", "april27", "may11", "may18", "june01", "june15", "june22"]


def has(word: str, text: str) -> bool:
    return re.search(rf"(?<![\wÄÖÜäöü]){re.escape(word)}(?![\wÄÖÜäöü])", text, re.I) is not None


def main() -> int:
    positions = corrob = fix = corrupt = undecidable = 0
    for mid in MEET:
        rows = NC.flags(mid, min_systems=3)
        prot = (REPO / f"outputs_deliverable/v10/{mid}/protocol_pro.md").read_text(encoding="utf-8")
        his = Path(HUMAN[mid]).read_text(encoding="utf-8", errors="replace")
        gl = set(glossary_for_document(REPO / PREVIOUS[mid]))
        for r in rows:
            maj, ag = r["majority"], r["agreement"]
            if ag < 0.6:
                continue
            for v in r["variants"]:
                if v == maj or not has(v, prot):
                    continue
                positions += 1
                corrob += maj in gl
                in_his, maj_in_his = has(v, his), has(maj, his)
                if in_his and not maj_in_his:
                    corrupt += 1
                elif maj_in_his and not in_his:
                    fix += 1
                else:
                    undecidable += 1
                break
    print("MAJORITY-VOTE NAME CORRECTOR — objective safety gate against the reference protocols")
    print(f"  positions where v10 uses a non-majority spelling : {positions}")
    print(f"  ... majority form also in the previous protocol  : {corrob}")
    print(f"  would corrupt a reference spelling              : {corrupt}")
    print(f"  would fix a word to the reference spelling       : {fix}")
    print(f"  undecidable                                      : {undecidable}")
    if fix + corrupt:
        print(f"  precision                                        : {fix}/{fix + corrupt} = "
              f"{100 * fix / (fix + corrupt):.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
