"""Classifies each misrecognized name by whether the previous protocol could have supplied its spelling.

For every meeting with reference minutes, name_check_eval.mangled() lists the names that the minutes spell
one way and the recognizers another. Each reference spelling is compared, after entity_correct.squash, with
the glossary that term_source.glossary_for_document extracts from the meeting's previous document and with
the full text of that document. Each name falls into one of three classes:

  offered               the spelling is in the glossary that is put into the prompt
  in_prev_not_offered   the spelling occurs in the previous document, but the glossary does not contain it
  unreachable           the spelling does not occur in the previous document, so no correction that relies
                        on the previous protocol alone can reach it

It prints one row per name and the count and share of each class.

    python name_reachability.py    -> goldset/name_reachability.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import entity_correct as E                                      # noqa: E402
from generate_protocols import PREV_PROTOCOL                    # noqa: E402
from name_check_eval import MINUTES, mangled                    # noqa: E402
from term_source import glossary_for_document                   # noqa: E402

OUT = REPO / "goldset" / "name_reachability.json"


def read(p: Path) -> str:
    if p.suffix.lower() == ".docx":
        try:
            from docx import Document
            return "\n".join(x.text for x in Document(str(p)).paragraphs)
        except Exception:
            return ""
    return p.read_text(encoding="utf-8", errors="replace")


def main() -> int:
    print("NAME REACHABILITY — of the names the ASR mangles, how many could be fixed at all?")
    print("=" * 96)
    print('Names are noisy and take a lot of time to correct.')
    print('Rule: every fix must rely on the previous protocol alone.\n')

    rows, tally = [], {"offered": 0, "in_prev_not_offered": 0, "unreachable": 0}
    print(f"{'meeting':10}{'ref spelling':22}{'ASR heard':26}{'class':24}")
    print("-" * 96)

    for mid in sorted(MINUTES):
        prev_name = PREV_PROTOCOL.get(mid)
        prev_path = REPO / prev_name if prev_name else None
        prev_text = read(prev_path) if prev_path and prev_path.exists() else ""
        prev_sq = {E.squash(t) for t in prev_text.split()}
        gloss = {E.squash(t) for t in (glossary_for_document(prev_path)
                                       if prev_path and prev_path.exists() else [])}

        for m in mangled(mid):
            ref, heard = m["reference"], m["heard_as"]
            rsq = E.squash(ref)
            if rsq in gloss:
                verdict, key = "OFFERED — real defect", "offered"
            elif rsq in prev_sq:
                verdict, key = "in prev, not extracted", "in_prev_not_offered"
            else:
                verdict, key = "unreachable (1st mention)", "unreachable"
            tally[key] += 1
            rows.append({"meeting": mid, "reference": ref, "heard_as": heard,
                         "verdict": key})
            print(f"{mid:10}{ref:22}{', '.join(heard)[:24]:26}{verdict:24}")

    total = sum(tally.values())
    if not total:
        print("\nno mangled names found — check that the ASR transcripts are present")
        return 1

    print("\n" + "=" * 96)
    print(f"{'bucket':34}{'n':>5}{'share':>9}   meaning")
    print("-" * 96)
    print(f"{'OFFERED and still wrong':34}{tally['offered']:>5}"
          f"{tally['offered']/total:>9.0%}   the model had the spelling and ignored it")
    print(f"{'in previous doc, not extracted':34}{tally['in_prev_not_offered']:>5}"
          f"{tally['in_prev_not_offered']/total:>9.0%}   term_source missed it — FIXABLE")
    print(f"{'unreachable (first mention)':34}{tally['unreachable']:>5}"
          f"{tally['unreachable']/total:>9.0%}   no rule that uses only the previous protocol can reach it")
    print("-" * 96)
    fixable = tally["offered"] + tally["in_prev_not_offered"]
    print(f"{'ADDRESSABLE AT ALL':34}{fixable:>5}{fixable/total:>9.0%}")
    print(f"{'STRUCTURALLY OUT OF REACH':34}{tally['unreachable']:>5}"
          f"{tally['unreachable']/total:>9.0%}")

    print("\n  Read this before quoting the accuracy score. A system cannot be marked down for")
    print("  the unreachable bucket without also marking down the rule that creates it, which is")
    print("  the rule above. The Senat case is different by construction: its invitation letter names")
    print("  the people and the agenda items in advance, so names that are first-mention here")
    print("  arrive pre-declared there.")

    OUT.write_text(json.dumps({"tally": tally, "total": total, "rows": rows},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwrote -> {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
