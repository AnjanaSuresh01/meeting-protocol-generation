"""Recounts, in one run, the figures listed below and prints the full output of every step.

Each figure is either computed here or printed by a script that this file runs as a subprocess,
and every subprocess prints its whole output, so each number can be traced to a line of this run:
   0. corpus: reference protocols (Meeting_2026-*.md), meetings with audio, gold-set clips
      (total and scoreable)
   1. ASR word error rate: score_v10.py, which also writes goldset/score_v10_2026-09-30.json
   2. reliability of the reference transcripts: reference_reliability.py
   3. word ratio per ASR system: hypothesis words / reference words on the scoreable clips,
      pooled, and the number of clips below 0.8
   4. post-edit effort and names of the v10 protocols: score_editforward.py v10
   5. handover test: handover_test.py (printed for the record only)
   6. topic attribution over all topics, Backlog excluded: the v10 protocols against copying
      last week's owner, split by whether the owner changed, with an exact binomial test on the
      topics where only one of the two is right
   7. section order: the v10 protocols and last week's order against the reference order
      (exact matches and pairwise agreement)
   8. style: style_flag.py v10
   9. quality gate: qa_protocols.py v10
  10. invented person names left in the v10 protocols (invented_names.apply)

    python recount_numbers.py > recount_numbers.txt
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}
MEET = ["march09", "april13", "april27", "may11", "may18", "june01", "june15", "june22"]


def run(label: str, args: list[str], tail: int = 0) -> str:
    # tail=0 prints the whole output (out[-0:] is the full list); a positive tail prints only
    # the last `tail` lines. Every call here uses tail=0, so no figure is cut off.
    print("\n" + "=" * 96 + f"\n{label}\n   $ python {' '.join(args)}\n" + "=" * 96)
    r = subprocess.run([sys.executable] + args, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=ENV, cwd=REPO)
    out = (r.stdout + r.stderr).rstrip().splitlines()
    print("\n".join(out[-tail:]))
    return "\n".join(out)


def section(label: str) -> None:
    print("\n" + "=" * 96 + f"\n{label}\n" + "=" * 96)


def main() -> int:
    # ---------------- corpus ----------------
    section("0. CORPUS")
    his = sorted(REPO.glob("Meeting_2026-*.md"))
    audio = [p for p in REPO.glob("Team-Meeting_2026-*") if p.suffix in (".flac", ".mp3", ".wav")]
    # count meetings, not files: one meeting can exist as both .flac and .mp3
    dates = sorted({p.stem for p in audio})
    print(f"   reference protocols (markdown): {len(his)}")
    print(f"   meeting recordings      : {len(dates)} meetings ({len(audio)} audio files)")
    g = json.loads((REPO / "goldset/goldset_v10.json").read_text(encoding="utf-8"))["segments"]
    el = [s for s in g if s["scoring"].get("eligible")]
    print(f"   gold-set clips          : {len(g)} total, {len(el)} scoreable")

    # ---------------- ASR ----------------
    run("1. ASR WORD ERROR RATE — the frozen scorer", ["score_v10.py", "--json",
        str(REPO / "goldset" / "score_v10_2026-09-30.json")], tail=0)
    run("2. HOW RELIABLE IS THE REFERENCE ITSELF", ["reference_reliability.py"], tail=0)

    section("3. WORD RATIO — does a system silently drop words? (reference-free)")
    W = lambda t: len(re.findall(r"\w+", t or ""))
    for m in sorted({k for s in el for k in s["model_hypotheses"]}):
        ref = hyp = low = n = 0
        for s in el:
            h = s["model_hypotheses"].get(m)
            r = W(s["reference_text"])
            if h is None or r == 0:
                continue
            ref += r; hyp += W(h); n += 1; low += W(h) / r < 0.8
        print(f"   {m:18s} pooled {hyp / ref:.2f}   clips below 0.8: {low}/{n}")

    # ---------------- protocols ----------------
    run("4. POST-EDIT EFFORT AND NAMES — v10 against the reference protocols", ["score_editforward.py", "v10"],
        tail=0)
    run("5. HANDOVER TEST — result withdrawn after a reading item by item: printed for the "
        "record, NOT quotable", ["handover_test.py"], tail=0)

    section("6. ATTRIBUTION OVER ALL TOPICS, Backlog excluded — v10 against 'copy last week'")
    from attribution_baselines import (MIDS, PREV, bag, jac, read_any, read_human,
                                          read_generated, TH)
    from attribution_blocklevel import block_hits
    import handover_test as HT
    from scipy.stats import binomtest
    for arm in ("v10",):
        ch = un = chok = unok = b = c = 0
        for mid in MIDS:
            human = [h for h in read_human(mid) if h[0]]
            pb = [(p, bag(t)) for p, t in read_any(PREV[mid]) if p and not HT.is_backlog(t)]
            p = REPO / f"outputs_deliverable/{arm}/{mid}/protocol_pro.md"
            if not pb or not p.exists():
                continue
            raw = p.read_text(encoding="utf-8")
            hits = block_hits(human, read_generated(p)[0], TH)
            for i, (hp, ht) in enumerate(human):
                if HT.is_backlog(ht) or len(bag(ht)) < 3:
                    continue
                best_p, best_s = None, 0.0
                for pp, bb in pb:
                    s = jac(bag(ht), bb)
                    if s > best_s:
                        best_s, best_p = s, pp
                if best_s < 0.10:
                    continue
                who = hits.get(i) or HT._locate_by_keyword(ht, raw)
                if who is None:
                    continue
                ok, mem = who == hp, best_p == hp
                if best_p != hp:
                    ch += 1; chok += ok
                else:
                    un += 1; unok += ok
                b += ok and not mem
                c += mem and not ok
        n = ch + un
        pv = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
        print(f"   {arm}: owner changed {chok}/{ch}   owner same {unok}/{un} = {unok / un:.1%}"
              f"   ALL {chok + unok}/{n} = {(chok + unok) / n:.1%}")
        print(f"   copy last week: ALL {un}/{n} = {un / n:.1%}")
        print(f"   paired: pipeline right & memory wrong {b}, memory right & pipeline wrong {c}, "
              f"p = {pv:.3f}")

    section("7. SECTION ORDER — v10 against the reference order, and against copying last week's order")
    from attribution_baselines import human_order, gen_order, kendall_like
    ex_v = ex_m = 0; kv = []; km = []; n = 0
    for mid in MEET:
        try:
            h = human_order(mid)
        except Exception:
            continue
        p = REPO / f"outputs_deliverable/v10/{mid}/protocol_pro.md"
        prev_order = []
        for person, _ in read_any(PREV[mid]):
            if person and person not in prev_order:
                prev_order.append(person)
        g = gen_order(p) if p.exists() else []
        if len(h) < 2 or not g:
            continue
        n += 1
        common_v = [x for x in g if x in h]
        common_m = [x for x in prev_order if x in h]
        ex_v += common_v == [x for x in h if x in common_v]
        ex_m += common_m == [x for x in h if x in common_m]
        kv.append(kendall_like(h, g)); km.append(kendall_like(h, prev_order))
        print(f"   {mid:9s} ref {h}\n             v10 {g}\n             last week {prev_order}")
    if n:
        print(f"\n   exact order: v10 {ex_v}/{n}, copy last week {ex_m}/{n}")
        print(f"   pairwise agreement: v10 {sum(kv) / n:.1%}, copy last week {sum(km) / n:.1%}")

    run("8. STYLE — bare noun-phrase bullets", ["style_flag.py", "v10"], tail=0)
    run("9. QUALITY GATE", ["qa_protocols.py", "v10"], tail=0)

    section("10. INVENTED PERSON NAMES left in the final v10 protocols")
    import invented_names as IVN
    from carry_tag import read_prev
    from term_source import PREVIOUS
    tot = 0
    for mid in MEET:
        p = REPO / f"outputs_deliverable/v10/{mid}/protocol_pro.md"
        tx = (REPO / f"outputs_v11_final/canary_diar/{mid}/transcript_speakers.txt").read_text(
            encoding="utf-8")
        _, restored, flagged = IVN.apply(p.read_text(encoding="utf-8"), tx,
                                         read_prev(REPO / PREVIOUS[mid]))
        k = len(restored) + len(flagged)
        tot += k
        if k:
            print(f"   {mid}: restored {restored} flagged {[f for f, _ in flagged]}")
    print(f"   total invented names still in v10: {tot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
