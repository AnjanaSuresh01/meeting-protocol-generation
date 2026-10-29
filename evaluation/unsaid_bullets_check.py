"""Estimates how many of the new reference bullets that the v10 draft misses were probably never said.

  * missed bullet = a new reference bullet (not covered by the previous protocol) with no counterpart in the v10
    draft, both under postedit_effort.covered (30% content-word rule), as in score_editforward
  * heard share  = fraction of its content words (postedit_effort.content) that occur anywhere in the meeting's
    transcript (outputs_v11_final/canary_diar, the transcript the evaluated runs received)
  * probably not said = heard share below the same 30% (postedit_effort.MATCH_TH)

The output gives the number of missed bullets, how many fall below 30%, how many share no content word with
the transcript, the counts below the thresholds 0.1 to 0.5, and the missed, below-30% and no-overlap counts
per meeting.

    python unsaid_bullets_check.py    -> goldset/unsaid_bullets_2026-10-10.json
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import postedit_effort as PE                      # noqa: E402
import score_editforward as S                     # noqa: E402
from evaluate_protocols import REF, strip_backlog  # noqa: E402

OUT = REPO / "goldset" / "unsaid_bullets_2026-10-10.json"


def main():
    shares, per = [], {}
    for mid, f in REF.items():
        ref = S.no_backlog(strip_backlog(Path(f).read_text(encoding="utf-8"))[0])
        rb, pb = PE.bullets(ref), PE.bullets(S.previous_only(mid))
        hb = PE.bullets(S.arm_text("v10", mid))
        tx = (REPO / "outputs_v11_final" / "canary_diar" / mid / "transcript_speakers.txt").read_text(
            encoding="utf-8", errors="replace")
        T = PE.content(tx)
        m = [len(r & T) / len(r) for r in rb if r and not PE.covered(r, pb) and not PE.covered(r, hb)]
        per[mid] = {"missed": len(m), "below_30": sum(x < PE.MATCH_TH for x in m), "zero": sum(x == 0 for x in m)}
        shares += m
    res = {"missed": len(shares), "below_30": sum(x < PE.MATCH_TH for x in shares),
           "zero_overlap": sum(x == 0 for x in shares),
           "by_threshold": {f"{t:.2f}": sum(x < t for x in shares) for t in (0.1, 0.2, 0.3, 0.4, 0.5)},
           "per_meeting": per}
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"missed {res['missed']}: below 30% heard {res['below_30']}, no content word heard {res['zero_overlap']}")


if __name__ == "__main__":
    main()
