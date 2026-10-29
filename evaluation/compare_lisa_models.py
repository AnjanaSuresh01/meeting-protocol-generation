"""Compares LISA Pro and LISA Flash as the protocol writer inside the v10 pipeline.

Three arms, each read from outputs_deliverable/<arm>/<meeting>/:
    v10         Pro   — the v10 protocols
    v10pro_rep  Pro   — generated again by run_cite_arm.py with v10's speaker names
    v10flash    Flash — the same rerun with lisa-flash-03-2026

The two reruns share their inputs. v10 itself was generated with fewer earlier protocols in its
prompt vocabulary and, for april13, an older previous protocol, so the three arms are not
exchangeable draws. The clean comparison is v10flash against v10pro_rep, where only the model
differs. The mean per-meeting gap |v10 - v10pro_rep| combines LISA's run-to-run randomness with
the input difference, so it is an upper bound on Pro's run-to-run noise: a Flash-Pro gap smaller
than it cannot be told apart from noise, and a larger one is a conservative signal. Before
scoring, the script reads each run's log and checks that the two reruns had identical inputs
(vocabulary width and speaker names).

Measures per meeting: post-edit ADD %, missed new items % and DELETE % (score_editforward,
postedit_effort); the share of new bullets that open with a noun phrase (style_flag, on
protocol_pro.neu.md); and an approximate automatic attribution score on topics that also occur
in the previous week (attribution_blocklevel.block_hits with handover_test's keyword fallback;
the matcher errs in both directions). For each measure, an exact two-sided sign test over the
meetings (ties dropped) compares Flash with the Pro rerun. Invented names (invented_names),
self-check passes, re-rolls, words written and wall time are listed per arm.

    python compare_lisa_models.py    -> stdout only
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

ARMS = {"v10": "Pro, 1st run", "v10pro_rep": "Pro, 2nd run", "v10flash": "Flash"}
MEET = ["march09", "april13", "april27", "may11", "may18", "june01", "june15", "june22"]
BUL = re.compile(r"(?m)^\s*[*-]\s+\S")


def exists(arm, mid):
    return (REPO / f"outputs_deliverable/{arm}/{mid}/protocol_pro.md").exists()


def postedit_mid(arm, mid):
    import postedit_effort as PE
    import score_editforward as SE
    from evaluate_protocols import strip_backlog
    ref = SE.no_backlog(strip_backlog((REPO / SE.REF[mid]).read_text(encoding="utf-8"))[0])
    t = SE.arm_text(arm, mid)
    pb, rb, hb = PE.bullets(SE.previous_only(mid)), PE.bullets(ref), PE.bullets(t)
    add = new = new_miss = 0
    for r in rb:
        miss = not PE.covered(r, hb)
        add += miss
        if not PE.covered(r, pb):
            new += 1
            new_miss += miss
    dele = sum(not PE.covered(h, rb) for h in hb)
    return {"ADD %": 100 * add / len(rb), "new missed %": 100 * new_miss / max(1, new),
            "DELETE %": 100 * dele / max(1, len(hb))}


def style_mid(arm, mid):
    import style_flag as SF
    t = (REPO / f"outputs_deliverable/{arm}/{mid}/protocol_pro.neu.md").read_text(encoding="utf-8")
    n = len(BUL.findall(t))
    return 100 * len(SF.scan(t)) / max(1, n)


def attribution_mid(arm, mid):
    from attribution_baselines import PREV, bag, jac, read_any, read_human, read_generated, TH
    from attribution_blocklevel import block_hits
    import handover_test as HT
    human = [h for h in read_human(mid) if h[0]]
    pb = [(p, bag(t)) for p, t in read_any(PREV[mid]) if p and not HT.is_backlog(t)]
    p = REPO / f"outputs_deliverable/{arm}/{mid}/protocol_pro.md"
    raw = p.read_text(encoding="utf-8")
    hits = block_hits(human, read_generated(p)[0], TH)
    ok = n = 0
    for i, (hp, ht) in enumerate(human):
        if HT.is_backlog(ht) or len(bag(ht)) < 3:
            continue
        if max((jac(bag(ht), b) for _, b in pb), default=0) < 0.10:
            continue
        who = hits.get(i) or HT._locate_by_keyword(ht, raw)
        if who is None:
            continue
        n += 1
        ok += who == hp
    return 100 * ok / max(1, n)


def order_mid(arm, mid):
    from attribution_baselines import human_order, gen_order, kendall_like
    return 100 * kendall_like(human_order(mid),
                              gen_order(REPO / f"outputs_deliverable/{arm}/{mid}/protocol_pro.md"))


def invented_mid(arm, mid):
    import invented_names as IVN
    from carry_tag import read_prev
    from term_source import PREVIOUS
    tx = (REPO / f"outputs_v11_final/canary_diar/{mid}/transcript_speakers.txt").read_text(
        encoding="utf-8")
    t = (REPO / f"outputs_deliverable/{arm}/{mid}/protocol_pro.md").read_text(encoding="utf-8")
    _, restored, flagged = IVN.apply(t, tx, read_prev(REPO / PREVIOUS[mid]))
    return len(restored) + len(flagged)


def log_facts(arm, mid):
    log = (REPO / f"outputs_deliverable/{arm}/{mid}/run_pro.log").read_text(
        encoding="utf-8", errors="replace")
    m = re.search(r"wall time: (\d+) s", log)
    return {"secs": int(m.group(1)) if m else None,
            "self-check ok": "passes every structural rule" in log,
            "re-rolls": log.count("[non-German output, retrying]") + log.count("[truncated, retrying]")}


# measure -> (function, lower_is_better)
MEASURES = {
    "ADD %": (lambda a, m: postedit_mid(a, m)["ADD %"], True),
    "new missed %": (lambda a, m: postedit_mid(a, m)["new missed %"], True),
    "DELETE %": (lambda a, m: postedit_mid(a, m)["DELETE %"], True),
    "noun-phrase %": (style_mid, True),
    "attribution % (approx.)": (attribution_mid, False),
    # Section order is not compared: order_by_audio derives it from the recording, so it is the
    # same for every model.
}


def inputs_line(arm, mid):
    """What the run says it was given: vocabulary width and speaker names (from its log)."""
    log = (REPO / f"outputs_deliverable/{arm}/{mid}/run_pro.log").read_text(
        encoding="utf-8", errors="replace")
    w = re.search(r"widened to (\d+) documents", log)
    n = re.search(r"speaker names \(voice, handover, topic\): (.*)", log)
    return (w.group(1) if w else "1", n.group(1).strip() if n else "")


def main() -> int:
    from scipy.stats import binomtest
    have = [m for m in MEET if all(exists(a, m) for a in ARMS)]
    print("=" * 100)
    print(f"LISA PRO vs LISA FLASH inside v10 — {len(have)} meetings with all three runs: {', '.join(have)}")
    print("=" * 100)
    if len(have) < 2:
        print("  not enough complete meetings yet")
        return 1
    print("\n  INPUT CHECK (from each run's own log): documents in the prompt vocabulary / names")
    same = True
    for m in have:
        a, b, c = (inputs_line(x, m) for x in ("v10", "v10pro_rep", "v10flash"))
        ok = b == c
        same &= ok
        print(f"  {m:9s} v10 {a[0]:>2s} docs | Pro rerun {b[0]:>2s} | Flash {c[0]:>2s} | "
              f"reruns identical: {'yes' if ok else 'NO'} | names as v10: "
              f"{'yes' if a[1] == b[1] == c[1] else 'NO'}")
    if not same:
        print("  !! the two reruns did not get identical inputs — the Flash-Pro comparison is void")

    vals = {meas: {a: {} for a in ARMS} for meas in MEASURES}
    for meas, (fn, _) in MEASURES.items():
        for a in ARMS:
            for m in have:
                vals[meas][a][m] = fn(a, m)

    print(f"\n  {'measure':26s}" + "".join(f"{ARMS[a]:>16s}" for a in ARMS)
          + f"{'|v10-Pro rerun|':>17s}{'Flash-Pro rerun':>17s}")
    for meas, (_, low) in MEASURES.items():
        mean = {a: sum(vals[meas][a].values()) / len(have) for a in ARMS}
        pp = sum(abs(vals[meas]["v10"][m] - vals[meas]["v10pro_rep"][m]) for m in have) / len(have)
        fp = sum(vals[meas]["v10flash"][m] - vals[meas]["v10pro_rep"][m] for m in have) / len(have)
        print(f"  {meas:26s}" + "".join(f"{mean[a]:16.1f}" for a in ARMS)
              + f"{pp:17.1f}{fp:+17.1f}")
    print("\n  '|v10-Pro rerun|' = mean per-meeting gap between the two Pro runs. Their inputs differ")
    print("  (see INPUT CHECK), so this is an UPPER bound on Pro's run-to-run noise.")
    print("  'Flash-Pro rerun' = mean per-meeting difference under IDENTICAL inputs.")

    print("\n  SIGN TEST — Flash vs Pro rerun, per meeting (identical inputs), ties dropped")
    for meas, (_, low) in MEASURES.items():
        better = worse = 0
        for m in have:
            d = vals[meas]["v10flash"][m] - vals[meas]["v10pro_rep"][m]
            if abs(d) < 1e-9:
                continue
            flash_better = d < 0 if low else d > 0
            better += flash_better
            worse += not flash_better
        n = better + worse
        p = binomtest(better, n, 0.5).pvalue if n else 1.0
        print(f"  {meas:26s} Flash better in {better}, worse in {worse} of {n} meetings   "
              f"(two-sided p = {p:.3f})")

    print("\n  OTHER FACTS")
    for a in ARMS:
        facts = [log_facts(a, m) for m in have]
        secs = [f["secs"] for f in facts if f["secs"]]
        inv = sum(invented_mid(a, m) for m in have)
        words = sum(len((REPO / f"outputs_deliverable/{a}/{m}/protocol_pro.neu.md")
                        .read_text(encoding="utf-8").split()) for m in have)
        print(f"  {ARMS[a]:14s} self-check {sum(f['self-check ok'] for f in facts)}/{len(have)}   "
              f"re-rolls {sum(f['re-rolls'] for f in facts)}   invented names {inv}   "
              f"words written {words}   "
              + (f"mean time/meeting {sum(secs) / len(secs):.0f} s" if secs else "time: not logged"))

    print("\n  PER MEETING (ADD % / new missed % / noun-phrase %)")
    for m in have:
        print(f"  {m:9s} " + "   ".join(
            f"{ARMS[a]}: {vals['ADD %'][a][m]:.1f} / {vals['new missed %'][a][m]:.1f} / "
            f"{vals['noun-phrase %'][a][m]:.1f}" for a in ARMS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
