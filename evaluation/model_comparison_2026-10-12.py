# -*- coding: utf-8 -*-
"""Compares the protocols of two language models on the same inputs, with the v10 arm as a third arm.

The pair is v10rerun (lisa-pro-03-2026, Qwen3.5-397B-A17B, reasoning off) and v12_evaltx (lisa-pro, listed as
GLM 5.3 Flash, reasoning effort low). Both read the evaluated transcript, name the same voices, draw names from
the same earlier protocols and use prompt examples that are not taken from the corpus; input_side() compares
their run-log lines up to the generation call per meeting. v10 used prompt examples taken from the corpus and
fewer name documents.

For v10, v10rerun and v12_evaltx the script reports the new reference bullets missed (30% word rule), ADD and
DELETE, recall and precision against the stripped reference (S1, leaf rule L), and the chance level of an
earlier meeting's draft on a later meeting's new bullets (computed as in paired_and_null_2026-10-11.py; for v10 it
must give 280 of 1,620). Arm pairs are compared with an exact McNemar test over the new bullets, and
v12_evaltx against v10rerun with a two-sided sign test over the meetings.

    python model_comparison_2026-10-12.py    -> goldset/model_comparison_2026-10-12.json
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import postedit_effort as PE  # noqa: E402
import score_editforward as SE  # noqa: E402
import score_repair_rerun as S  # noqa: E402
from evaluate_protocols import strip_backlog  # noqa: E402

MEET = ["march09", "april13", "april27", "may11", "may18", "june01", "june15", "june22"]
ARMS = ["v10", "v10rerun", "v12_evaltx"]


def new_refs(mid):
    ref = SE.no_backlog(strip_backlog((REPO / SE.REF[mid]).read_text(encoding="utf-8"))[0])
    pb = PE.bullets(SE.previous_only(mid))
    return [r for r in PE.bullets(ref) if not PE.covered(r, pb)]


def chance(arm, newref):
    """An earlier meeting's draft scored on a later meeting's new reference bullets."""
    rec = n = 0
    for j, y in enumerate(MEET):
        for x in MEET[:j]:
            hb = PE.bullets(SE.arm_text(arm, x))
            rec += sum(PE.covered(r, hb) for r in newref[y])
            n += len(newref[y])
    return {"recorded": rec, "of": n, "pct": round(100 * rec / n, 1)}


def input_side(arm, mid):
    """Log lines up to the generation call (transcript, voices, names) plus the name-list size."""
    log = (REPO / "outputs_deliverable" / arm / mid / "run_pro.log").read_text(encoding="utf-8").splitlines()
    start = next(i for i, l in enumerate(log) if l.startswith("protokoll.py"))
    stop = next(i for i, l in enumerate(log) if "writing the protocol" in l)
    skip = ("NOTE: no word", "four transcript repairs", "transcript_canary_ts", "diarization_hyp")
    lines = [l for l in log[start:stop] if not any(s in l for s in skip)]
    return lines + [l.strip() for l in log if "widened to" in l]


def main() -> int:
    newref = {m: new_refs(m) for m in MEET}
    out = {"arms": {}, "pairs": {},
           "inputs_identical_v10rerun_v12_evaltx": {m: input_side("v10rerun", m) == input_side("v12_evaltx", m)
                                                    for m in MEET}}
    print("inputs identical:", out["inputs_identical_v10rerun_v12_evaltx"])
    flags = {}
    for arm in ARMS:
        flags[arm] = S.new_flags(arm)
        pe = SE.postedit(arm)
        st = S.stripped(arm)
        out["arms"][arm] = {"new_missed": sum(flags[arm]), "new_total": len(flags[arm]),
                            "add": pe["add"], "ref": pe["ref"], "delete": pe["delete"], "draft": pe["draft"],
                            "stripped_leaves": st["recall_new_leaves"], "stripped_headlines": st["recall_headlines"],
                            "precision_model_items": st["precision_neu"],
                            "chance_earlier_draft": chance(arm, newref)}
        print(arm, json.dumps(out["arms"][arm], ensure_ascii=False))
    for x, y in [("v10", "v12_evaltx"), ("v10rerun", "v12_evaltx"), ("v10", "v10rerun")]:
        b = sum(1 for p, q in zip(flags[x], flags[y]) if p and not q)
        c = sum(1 for p, q in zip(flags[x], flags[y]) if q and not p)
        out["pairs"][f"{x} vs {y}"] = {"missed_by_first_only": b, "missed_by_second_only": c,
                                       "mcnemar_exact_p": S.mcnemar_exact(b, c)}
    # per meeting: new reference bullets missed by each model, and a two-sided sign test over the meetings
    per = {}
    k = 0
    for mid in MEET:
        n = len(newref[mid])
        per[mid] = {"new": n, **{a: sum(flags[a][k:k + n]) for a in ARMS}}
        k += n
    fewer = sum(1 for v in per.values() if v["v12_evaltx"] < v["v10rerun"])
    more = sum(1 for v in per.values() if v["v12_evaltx"] > v["v10rerun"])
    from math import comb
    m = fewer + more
    p_sign = min(1.0, 2 * sum(comb(m, i) for i in range(0, min(fewer, more) + 1)) / 2 ** m) if m else 1.0
    out["per_meeting"] = per
    out["sign_test_v12_evaltx_vs_v10rerun"] = {"meetings_fewer_missed": fewer, "meetings_more_missed": more,
                                               "ties": len(MEET) - m, "p_two_sided": p_sign}
    print(json.dumps(out["per_meeting"], indent=1), out["sign_test_v12_evaltx_vs_v10rerun"])
    print(json.dumps(out["pairs"], indent=1))
    (REPO / "goldset" / "model_comparison_2026-10-12.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
