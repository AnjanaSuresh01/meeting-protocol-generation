"""Computes the recorded share of the new reference bullets in each third of the meeting.

  * new bullet = a leaf of the reference protocol (Backlog removed) that the previous protocol does not cover
    (postedit_effort.covered, 30% content-word rule), the same base as score_editforward.postedit
  * recorded  = covered by a bullet of the v10 draft under the same 30% rule
  * position  = the time stamp of the one transcript turn (outputs_v11_final/canary_diar) that shares the largest
    fraction of the bullet's content words; a bullet is placed only if that fraction is at least PLACE_TH (30%);
    the meeting is split into thirds of the time of its last turn
  * test      = chi-square test of recorded against not recorded across the three thirds

    python thirds_recall.py      -> goldset/thirds_recall_2026-10-10.json
"""
import json
import re
import sys
from pathlib import Path

from scipy.stats import chi2_contingency

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import postedit_effort as PE                      # noqa: E402
import score_editforward as S                     # noqa: E402
from evaluate_protocols import REF, strip_backlog  # noqa: E402

ARM = "v10"
PLACE_TH = 0.30
TURN = re.compile(r"^\[(\d\d):(\d\d):(\d\d)\]\s+\S+:\s*(.*)$", re.M)
OUT = REPO / "goldset" / "thirds_recall_2026-10-10.json"


def turns_of(mid):
    tx = (REPO / "outputs_v11_final" / "canary_diar" / mid / "transcript_speakers.txt").read_text(
        encoding="utf-8", errors="replace")
    return [(int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]), PE.content(m[4])) for m in TURN.finditer(tx)]


def main():
    tot = {k: [0, 0] for k in ("first", "middle", "last")}
    unplaced = [0, 0]
    per_meeting = {}
    for mid, f in REF.items():
        ref = S.no_backlog(strip_backlog(Path(f).read_text(encoding="utf-8"))[0])
        rb, pb = PE.bullets(ref), PE.bullets(S.previous_only(mid))
        hb = PE.bullets(S.arm_text(ARM, mid))
        turns = turns_of(mid)
        dur = turns[-1][0]
        m_tot = {k: [0, 0] for k in tot}
        for r in rb:
            if PE.covered(r, pb):
                continue                                  # covered by last week's protocol, so not new
            rec = int(PE.covered(r, hb))
            best_t, best = None, 0.0
            for t, c in turns:
                if c:
                    sc = len(r & c) / len(r)
                    if sc > best:
                        best, best_t = sc, t
            if best < PLACE_TH:
                unplaced[0] += 1
                unplaced[1] += rec
                continue
            k = "first" if best_t < dur / 3 else ("middle" if best_t < 2 * dur / 3 else "last")
            for d in (tot, m_tot):
                d[k][0] += 1
                d[k][1] += rec
        per_meeting[mid] = m_tot
    chi2, p, dof, _ = chi2_contingency([[r, n - r] for n, r in tot.values()])
    res = {
        "arm": ARM,
        "placement_threshold": PLACE_TH,
        "new_bullets": sum(n for n, _ in tot.values()) + unplaced[0],
        "placed": sum(n for n, _ in tot.values()),
        "unplaced": {"n": unplaced[0], "recorded": unplaced[1]},
        "by_third": {k: {"n": n, "recorded": r, "share": round(r / n, 3)} for k, (n, r) in tot.items()},
        "chi_square": {"chi2": round(chi2, 2), "dof": dof, "p": round(p, 3)},
        "per_meeting": per_meeting,
    }
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    for k, v in res["by_third"].items():
        print(f"{k:6s} {v['recorded']}/{v['n']} = {v['share']:.0%}")
    print(f"placed {res['placed']} of {res['new_bullets']}; chi-square p = {res['chi_square']['p']}")


if __name__ == "__main__":
    main()
