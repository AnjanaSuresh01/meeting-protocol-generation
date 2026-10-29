"""Scores the arms of run_repair_rerun.py with the scorers used for the other arms.

Per arm: structure, invented names and report order (compare_v10_rerun.structure_and_names); voices named
correctly (score_deliverable.py, run as a subprocess); ADD, DELETE and new reference bullets missed
(score_editforward.postedit); recall of the reference leaves that are not word-for-word copies
(strip_copies_rescore, definition S1, leaf rule L). The pairs in PAIRS are compared with an exact McNemar
test on the new reference bullets each arm misses, as in paired_and_null_2026-10-11.py. An arm without a
protocol_pro.md for all eight meetings is skipped.

    python score_repair_rerun.py [arm ...]      default: v10 v12_evaltx v12_delivered v12_delivered_raw
        -> goldset/repair_rerun_scores_2026-10-11.json
"""
import json
import subprocess
import sys
from math import comb
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import compare_v10_rerun as CR  # noqa: E402
import postedit_effort as PE  # noqa: E402
import score_editforward as SE  # noqa: E402
import strip_copies_rescore as SR  # noqa: E402
from evaluate_protocols import strip_backlog  # noqa: E402

MEET = ["march09", "april13", "april27", "may11", "may18", "june01", "june15", "june22"]
PAIRS = [("v12_evaltx", "v12_delivered"), ("v12_delivered_raw", "v12_delivered"), ("v12_evaltx", "v12_delivered_raw")]


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n)


def naming(arm: str) -> str:
    r = subprocess.run([sys.executable, "score_deliverable.py", arm], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", cwd=REPO, env={**__import__("os").environ,
                                                                         "PYTHONIOENCODING": "utf-8"})
    line = [l.strip() for l in (r.stdout + r.stderr).splitlines() if "VOICE -> NAME" in l]
    return line[0] if line else "not found"


def stripped(arm: str) -> dict:
    SR.ARM = arm
    rows = [SR.evaluate(SR.load_meeting(m), "S1", "L")[0] for m in SR.REF]
    return SR.add_counts(rows)


def new_flags(arm: str) -> list[bool]:
    flags = []
    for mid in MEET:
        ref = SE.no_backlog(strip_backlog((REPO / SE.REF[mid]).read_text(encoding="utf-8"))[0])
        pb = PE.bullets(SE.previous_only(mid))
        hb = PE.bullets(SE.arm_text(arm, mid))
        flags += [not PE.covered(r, hb) for r in PE.bullets(ref) if not PE.covered(r, pb)]
    return flags


def main(arms: list[str]) -> int:
    out: dict = {"arms": {}, "pairs": {}}
    flags = {}
    for arm in arms:
        if not all((REPO / "outputs_deliverable" / arm / m / "protocol_pro.md").exists() for m in MEET):
            print(f"{arm}: incomplete, skipped")
            continue
        s = CR.structure_and_names(arm)
        pe = SE.postedit(arm)
        st = stripped(arm)
        flags[arm] = new_flags(arm)
        out["arms"][arm] = {"structure": s, "naming": naming(arm), "postedit": pe, "stripped_S1_L": st,
                            "new_missed": sum(flags[arm]), "new_total": len(flags[arm])}
        print(f"\n== {arm}\n  self-check {s['self_check']}/8, invented {s['invented']}, foreign {s['foreign_script']}, "
              f"order exact {s['order_exact']}/8, pairs {s['order_pairwise']} %\n  {out['arms'][arm]['naming']}\n"
              f"  postedit {json.dumps(pe, ensure_ascii=False)}\n  new bullets missed {sum(flags[arm])}/{len(flags[arm])}")
        print("  stripped (S1, leaves):", json.dumps(st, ensure_ascii=False)[:600])
    for x, y in PAIRS:
        if x in flags and y in flags:
            b = sum(1 for p, q in zip(flags[x], flags[y]) if p and not q)
            c = sum(1 for p, q in zip(flags[x], flags[y]) if q and not p)
            out["pairs"][f"{x} vs {y}"] = {"missed_by_first_only": b, "missed_by_second_only": c,
                                           "mcnemar_exact_p": round(mcnemar_exact(b, c), 4)}
            print(f"\n{x} vs {y}: missed only by first {b}, only by second {c}, exact McNemar p = {mcnemar_exact(b, c):.4f}")
    (REPO / "goldset" / "repair_rerun_scores_2026-10-11.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                                                         encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or ["v10", "v12_evaltx", "v12_delivered", "v12_delivered_raw"]))
