"""Compares two deliverable arms, by default the v10 run and a complete rerun of it, on the same measures.

For each arm it reads outputs_deliverable/<arm>/<meeting>/protocol_pro.md for the eight meetings and
counts the protocols that pass protokoll.self_check, the invented names (restored plus flagged by
invented_names.apply against the transcript in outputs_v11_final/canary_diar and the previous
protocol), the protocols that contain Cyrillic, CJK or Japanese kana characters, and the meetings
whose order of '## ' person sections matches the reference order exactly (over the persons in
both), with the mean pairwise order agreement (attribution_baselines.kendall_like). It then runs
score_deliverable.py, score_editforward.py, style_flag.py and qa_protocols.py on both arms and
prints a summary line from each (the full output of score_editforward.py). Every measure is
computed in the same way for both arms.

    python compare_v10_rerun.py [old] [new]    # defaults: v10 v10rerun

Prints to stdout. The qa_protocols.py runs write outputs_deliverable/qa_report.json; the old arm
runs last, so the file ends up holding the old arm's report.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
MEET = ["march09", "april13", "april27", "may11", "may18", "june01", "june15", "june22"]


def structure_and_names(arm: str) -> dict:
    import protokoll as P
    import invented_names as IVN
    from carry_tag import read_prev
    from term_source import PREVIOUS
    from attribution_baselines import human_order, gen_order, kendall_like
    ok = inv = foreign = exact = 0
    pw = []
    for mid in MEET:
        f = REPO / f"outputs_deliverable/{arm}/{mid}/protocol_pro.md"
        t = f.read_text(encoding="utf-8")
        ok += not P.self_check(t)
        tx = (REPO / f"outputs_v11_final/canary_diar/{mid}/transcript_speakers.txt").read_text(encoding="utf-8")
        _, restored, flagged = IVN.apply(t, tx, read_prev(REPO / PREVIOUS[mid]))
        inv += len(restored) + len(flagged)
        foreign += bool(re.search(r"[Ѐ-ӿ一-鿿぀-ヿ]", t))
        h, g = human_order(mid), gen_order(f)
        common = [x for x in g if x in h]
        exact += common == [x for x in h if x in common]
        pw.append(kendall_like(h, g))
    return {"self_check": ok, "invented": inv, "foreign_script": foreign,
            "order_exact": exact, "order_pairwise": round(100 * sum(pw) / len(pw), 1)}


def run(args: list[str]) -> str:
    r = subprocess.run([sys.executable] + args, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=REPO, env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
    return r.stdout + r.stderr


def main(old: str = "v10", new: str = "v10rerun") -> int:
    print(f"COMPARISON: {old} (shipped) vs {new} (complete rerun)\n")
    for arm in (old, new):
        s = structure_and_names(arm)
        print(f"  {arm:10s} self-check {s['self_check']}/8 | invented names {s['invented']} | "
              f"foreign script {s['foreign_script']} | order exact {s['order_exact']}/8, pairs {s['order_pairwise']} %")
    print()
    for arm in (old, new):
        out = run(["score_deliverable.py", arm])
        line = [l for l in out.splitlines() if "VOICE -> NAME" in l]
        print(f"  {arm:10s} {line[0].strip() if line else 'naming: not found'}")
    print("\n" + run(["score_editforward.py", old, new]))
    # new arm first, old arm last: qa_protocols.py writes outputs_deliverable/qa_report.json,
    # which must end up holding the old arm's report
    for arm in (new, old):
        out = run(["style_flag.py", arm])
        tot = [l for l in out.splitlines() if "flagged" in l.lower()]
        print(f"  {arm:10s} style: {tot[-1].strip() if tot else '?'}")
        qa = run(["qa_protocols.py", arm])
        hdr = [l for l in qa.splitlines() if l.startswith("ARM ")]
        print(f"  {arm:10s} quality gate: {hdr[0].strip() if hdr else '?'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:3]))
