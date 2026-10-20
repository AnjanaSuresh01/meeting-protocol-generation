"""Score the recognizers on the eligible clips of the gold set (goldset/goldset_v10.json).
It prints micro WER with a bootstrap interval and CER, paired bootstrap tests with Holm
correction, a per-meeting Wilcoxon test for contrast, and the reader-agreement floor from
goldset/reference_reliability.json. Systems that seeded the references are marked, and the
remaining systems are also scored and tested on their own.

    python score_v10.py
    python score_v10.py --json out.json
    python score_v10.py --gold other_goldset.json --json out.json
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import statistics
import sys

import jiwer

from thesis_pipeline.metrics import (bootstrap_ci, min_detectable_p,
                                     paired_bootstrap_holm, paired_wilcoxon_holm)
from thesis_pipeline.text_norm import NormConfig, normalize

REPO = pathlib.Path(__file__).resolve().parent
GOLD = REPO / "goldset" / "goldset_v10.json"

CFG = NormConfig(casefold=True)
SEEDING = {"canary_diar", "whisper_turbo"}  # seeded the references
SAME_FAMILY = {"whisper_turbo_de"}  # same family as a seeding system


def micro_wer(pairs: list[tuple[str, str]]) -> dict:
    """Return micro WER, CER and edit counts for (reference, hypothesis) pairs."""
    if not pairs:
        return {"n": 0, "wer": None, "cer": None, "ref_words": 0}
    refs = [normalize(r, CFG) for r, _ in pairs]
    hyps = [normalize(h, CFG) for _, h in pairs]
    m = jiwer.process_words(refs, hyps)
    # Cast to plain Python numbers so the report stays JSON-serializable.
    return {"n": len(pairs),
            "ref_words": sum(len(r.split()) for r in refs),
            "wer": round(float(m.wer), 4),
            "cer": round(float(jiwer.cer(refs, hyps)), 4),
            "sub": int(m.substitutions), "ins": int(m.insertions), "del": int(m.deletions)}


def per_segment(pairs: list[tuple[str, str]]) -> list[float]:
    return [jiwer.wer([normalize(r, CFG)], [normalize(h, CFG)]) for r, h in pairs]


def micro_ci(pairs: list[tuple[str, str]], n_boot: int = 10000, seed: int = 0,
             alpha: float = 0.05) -> tuple[float, float]:
    """Return the percentile bootstrap interval of the micro WER, resampling segments."""
    import random
    errs, words = [], []
    for r, h in pairs:
        rn, hn = normalize(r, CFG), normalize(h, CFG)
        m = jiwer.process_words([rn], [hn])
        errs.append(m.substitutions + m.insertions + m.deletions)
        words.append(len(rn.split()))
    k = len(pairs)
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        idx = [rng.randrange(k) for _ in range(k)]
        w = sum(words[i] for i in idx)
        boots.append(sum(errs[i] for i in idx) / w if w else 0.0)
    boots.sort()
    return round(boots[int((alpha / 2) * n_boot)], 4), round(boots[int((1 - alpha / 2) * n_boot)], 4)


def decile_weighted(rows: list[tuple[int, float]], drawn: dict[int, int]) -> float | None:
    """Average the per-segment WER per decile, then average the deciles with equal weight."""
    by = collections.defaultdict(list)
    for d, w in rows:
        by[d].append(w)
    if not by:
        return None
    return round(statistics.fmean([statistics.fmean(v) for v in by.values()]), 4)


def collect(segments: list[dict], system: str, subset) -> list[dict]:
    return [s for s in segments
            if s["scoring"]["eligible"] and system in s["model_hypotheses"] and subset(s)]


def table(segments: list[dict], systems: list[str], subset, title: str, note: str = "") -> dict:
    print(f"\n{title}")
    if note:
        print(f"  {note}")
    print(f"\n  {'system':<20}{'n':>5}{'words':>8}{'WER':>8}{'95% CI':>18}"
          f"{'CER':>8}{'decile-wt':>11}{'free 0s':>9}")
    out = {}
    for sysname in systems:
        recs = collect(segments, sysname, subset)
        if not recs:
            print(f"  {sysname:<20}{'—':>5}   no hypotheses on this subset")
            out[sysname] = None
            continue
        pairs = [(s["reference_text"], s["model_hypotheses"][sysname]) for s in recs]
        agg = micro_wer(pairs)
        ps = per_segment(pairs)
        mean, mlo, mhi = bootstrap_ci(ps)  # mean per-segment WER, JSON only
        lo, hi = micro_ci(pairs)  # interval of the micro WER
        dw = decile_weighted([(s["decile"], w) for s, w in zip(recs, ps)], {})
        free = sum(1 for s in recs if sysname in s["bias"]["identical_to"])
        mark = " *" if sysname in SEEDING else (" +" if sysname in SAME_FAMILY else "")
        print(f"  {sysname + mark:<20}{agg['n']:>5}{agg['ref_words']:>8}{agg['wer']:>8.3f}"
              f"{f'[{lo:.3f}, {hi:.3f}]':>18}{agg['cer']:>8.3f}{dw:>11.3f}{free:>9}")
        out[sysname] = {**agg, "per_segment_mean": mean, "ci95": [lo, hi],
                        "ci95_of": "micro WER (segment bootstrap)",
                        "per_segment_mean_ci95": [mlo, mhi],
                        "decile_weighted": dw, "free_zeros": free}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="also write the full report here")
    ap.add_argument("--gold", default=str(GOLD), help="gold set to score (default: %(default)s)")
    a = ap.parse_args()

    gold = pathlib.Path(a.gold)
    if not gold.exists():
        sys.exit(f"{gold} not found — run build_goldset_v10.py first.")
    doc = json.loads(gold.read_text(encoding="utf-8"))
    meta, segments = doc["meta"], doc["segments"]

    present = [m for m in meta["panel"]["systems"]
               if any(m in s["model_hypotheses"] for s in segments)]
    missing = meta["panel"]["missing"]

    print("=" * 92)
    print("WORD ERROR RATE — gold set v10")
    print("=" * 92)
    print(f"  normaliser   {CFG.as_dict()}")
    print(f"  metric       micro WER = (S+D+I)/ref words, unclipped; CER beside it")
    print(f"  interval     percentile bootstrap of the micro WER over segments, 10000 resamples, seed 0")
    print(f"  scope        scoring.eligible — {meta['counts']['eligible_now']} of "
          f"{meta['counts']['segments']} segments")

    c = meta["counts"]
    pend = c["by_state"].get("pending_second_opinion", 0)
    if pend:
        print(f"\n  ** PROVISIONAL — {pend} clips are still with a second reader. The eligible")
        print(f"     set is decile-skewed towards easy audio (decile 10 keeps "
              f"{c['per_decile_eligible']['10']} of {c['per_decile']['10']}), so every number")
        print(f"     below UNDERSTATES the true rate. No headline figure until this is zero.")

    rd = meta.get("reading_depth", {})
    if not pend and rd.get("eligible_one_reader"):
        print(f"\n  ** UNEVEN READING DEPTH — {rd['eligible_two_readers']} references were heard")
        print(f"     by two German readers, {rd['eligible_one_reader']} by one. Of the "
              f"{rd['second_reader_certified']} clips the second reader certified,")
        print(f"     that reader rewrote {rd['second_reader_rewrote']}. Those clips were "
              f"selected as the doubtful ones, so")
        print(f"     that rate does not transfer — but the one-reader half carries an unmeasured")
        print(f"     reference error. See table 3b before quoting any single number.")
    if missing:
        print(f"\n  ** INCOMPLETE PANEL — no hypotheses for: {', '.join(missing)}")
        print(f"     These are the systems that did NOT seed the reference, so until they run")
        print(f"     the seeding advantage cannot be separated from the segmentation gain.")

    all_sys = present
    unseeded = [m for m in present if m not in SEEDING and m not in SAME_FAMILY]

    rep = {"meta": {"normalisation": CFG.as_dict(), "scope": "scoring.eligible",
                    "eligible": c["eligible_now"], "pending": pend, "missing_systems": missing}}

    rep["all"] = table(segments, all_sys, lambda s: True,
                       "1. ALL ELIGIBLE CLIPS",
                       "* seeded the reference   + same family as a seeding system   "
                       "'free 0s' = clips where the reference IS this system's output")

    rep["edited"] = table(segments, all_sys, lambda s: s["bias"]["rater_edited"],
                          "2. ONLY CLIPS THE HUMAN ACTUALLY EDITED",
                          "NOT a clean seeding control — a rater edits a clip BECAUSE the "
                          "machine got it wrong, so this subset is harder for every system, "
                          "seeded or not. Read the GAP between systems, never the level.")
    rep["unedited"] = table(segments, all_sys, lambda s: not s["bias"]["rater_edited"],
                            "3. ONLY CLIPS RETURNED UNCHANGED FROM THE SEED",
                            "the seeding pair is scored here against its own text, and these "
                            "are also the easier clips; both effects push the same way")

    if rd.get("eligible_one_reader") and rd.get("eligible_two_readers"):
        rep["two_readers"] = table(
            segments, all_sys, lambda s: s["resolution"].get("readers") == 2,
            "3b. REFERENCES TWO GERMAN READERS HEARD",
            "the second reader rewrote most of these, so they are the most trustworthy text in "
            "the set AND the hardest audio in it — they were flagged for exactly that reason. "
            "Expect a higher WER here than in 3c and do not read that as a quality difference "
            "between the two halves of the reference.")
        rep["one_reader"] = table(
            segments, all_sys, lambda s: s["resolution"].get("readers") == 1,
            "3c. REFERENCES ONLY ONE READER HEARD",
            "nobody raised a question about these, which is not the same as nobody finding one. "
            "This is where any residual reference error lives.")

    if unseeded:
        rep["unseeded_ranking"] = table(segments, unseeded, lambda s: True,
                                        "4. THE SYSTEMS THAT DID NOT SEED — the defensible ranking",
                                        "no system here had any hand in writing the reference")

    # Pairwise tests over segments
    print("\n5. PAIRWISE — paired bootstrap over segments, Holm-corrected")
    shared = [s for s in segments
              if s["scoring"]["eligible"] and all(m in s["model_hypotheses"] for m in all_sys)]
    if len(shared) >= 2 and len(all_sys) >= 2:
        refs = [normalize(s["reference_text"], CFG) for s in shared]
        ref_words = [len(r.split()) for r in refs]
        errors: dict[str, list[int]] = {}
        for sysname in all_sys:
            counts = []
            for s, r in zip(shared, refs):
                m = jiwer.process_words([r], [normalize(s["model_hypotheses"][sysname], CFG)])
                counts.append(int(m.substitutions + m.insertions + m.deletions))
            errors[sysname] = counts
        seg_rows = paired_bootstrap_holm(errors, ref_words, seed=20260919)
        seg_sig = [r for r in seg_rows if r["significant_0.05"]]
        print(f"  n = {len(shared)} segments, {sum(ref_words)} reference words, "
              f"10,000 paired replicates")
        print(f"\n  {'pair':<44}{'dWER':>8}{'95% CI':>20}{'p raw':>9}{'p Holm':>9}{'sig':>6}")
        for r in seg_rows:
            lo, hi = r["ci95"]
            ci = f"[{lo:+.3f}, {hi:+.3f}]"
            flag = "YES" if r["significant_0.05"] else "no"
            print(f"  {r['a'] + ' vs ' + r['b']:<44}{r['delta_wer']:>8.3f}{ci:>20}"
                  f"{r['p_raw']:>9.4f}{r['p_holm']:>9.4f}{flag:>6}")
        print(f"\n  {len(seg_sig)} of {len(seg_rows)} pairs separate at p_holm < 0.05.")
        rep["pairwise_segment"] = seg_rows
        rep["pairwise_segment_meta"] = {"n_segments": len(shared),
                                        "ref_words": sum(ref_words),
                                        "n_boot": 10000, "seed": 20260919,
                                        "unit": "segment", "test": "paired bootstrap, micro WER"}

        if len(unseeded) >= 2:
            print("\n5a. THE SAME TEST, UNSEEDED SYSTEMS ONLY — the defensible comparison")
            print("  Holm corrected over this family alone. No system here helped write the "
                  "reference.")
            un_rows = paired_bootstrap_holm({m: errors[m] for m in unseeded if m in errors},
                                            ref_words, seed=20260919)
            print(f"\n  {'pair':<44}{'dWER':>8}{'95% CI':>20}{'p raw':>9}{'p Holm':>9}{'sig':>6}")
            for r in un_rows:
                lo, hi = r["ci95"]
                ci = f"[{lo:+.3f}, {hi:+.3f}]"
                flag = "YES" if r["significant_0.05"] else "no"
                print(f"  {r['a'] + ' vs ' + r['b']:<44}{r['delta_wer']:>8.3f}{ci:>20}"
                      f"{r['p_raw']:>9.4f}{r['p_holm']:>9.4f}{flag:>6}")
            rep["pairwise_segment_unseeded"] = un_rows

    # Per-meeting Wilcoxon tests, for contrast
    print("\n5b. THE SAME PAIRS over per-meeting WER (paired Wilcoxon) — reported for contrast")
    by_model: dict[str, list[float]] = {}
    meetings = sorted({s["meeting_id"] for s in segments})
    for sysname in all_sys:
        vals = []
        for mtg in meetings:
            recs = [s for s in collect(segments, sysname, lambda s: True)
                    if s["meeting_id"] == mtg]
            pairs = [(s["reference_text"], s["model_hypotheses"][sysname]) for s in recs]
            vals.append(micro_wer(pairs)["wer"] if pairs else None)
        by_model[sysname] = vals
    if len(by_model) >= 2:
        rows = paired_wilcoxon_holm(by_model)
        sig = [r for r in rows if r["significant_0.05"]]
        print(f"\n  {'pair':<44}{'p raw':>9}{'p Holm':>9}{'sig':>6}")
        for r in rows:
            print(f"  {r['a'] + ' vs ' + r['b']:<44}{r['p_raw']:>9.4f}{r['p_holm']:>9.4f}"
                  f"{('YES' if r['significant_0.05'] else 'no'):>6}")
        print(f"\n  {len(sig)} of {len(rows)} pairs separate at p_holm < 0.05.")
        power = min_detectable_p(len([v for v in next(iter(by_model.values())) if v is not None]),
                                 len(rows))
        rep["pairwise_power"] = power
        if not power["can_ever_reject"]:
            print(f"\n  ** THIS TEST CANNOT REJECT. Wilcoxon on {power['n_pairs']} meetings has a")
            print(f"     floor of p = {power['min_possible_p']:.4f}; Holm over {power['n_comparisons']}"
                  f" pairs needs p < {power['holm_strictest_threshold']:.4f}.")
            print("     A null here is arithmetic, not evidence. Read section 5 instead, and do")
            print("     not report this as 'the systems are not distinguishable'.")
        elif not sig:
            print("  That is a null result and it is reported as one: on this corpus these")
            print("  systems are not distinguishable.")
        if sig:
            both_seeded = [r for r in sig if r["a"] in SEEDING and r["b"] in SEEDING]
            if both_seeded:
                print("  ** A separation between two systems that BOTH seeded the reference is")
                print("     not evidence about the systems. Whichever contributed more of the")
                print("     seed text wins by construction — compare their 'free 0s' columns.")
                print("     This cannot be resolved until the unseeded systems have run.")
        rep["pairwise"] = rows

    # Noise floor: reference_reliability.json, else the gold file's noise_floor.
    rr_path = REPO / "goldset" / "reference_reliability.json"
    if rr_path.exists():
        rr = json.loads(rr_path.read_text(encoding="utf-8"))
        ms, bd = rr["measured"], rr["bound_for_eligible_set"]
        print("\n6. THE FLOOR")
        print(f"\n  reader-vs-reader disagreement: {ms['disagreement_rate']}  "
              f"({ms['clips_two_readers']} clips, {ms['words']} words)")
        print(f"  bound for the whole eligible set: {bd['lower_if_one_reader_clips_perfect']} .. "
              f"{bd['upper_if_as_bad_as_flagged']}")
        print("  Any gap in the tables above smaller than this is not distinguishable from")
        print("  two people hearing the same German meeting audio differently.")
        print(f"  NEVER quote as: {rr['never_quote_as']}")
        rep["noise_floor"] = {"source": "reference_reliability.json", **ms, "bound": bd}
    nf = {} if rr_path.exists() else meta.get("noise_floor", {})
    if nf.get("disagreement_rate_pooled"):
        print(f"\n6. THE FLOOR")
        print(f"\n  WARNING: {rr_path.name} is missing; this is the older floor stored in "
              "goldset_v10.json.")
        print(f"\n  human-human disagreement: {nf['disagreement_rate_pooled']}  "
              f"({nf.get('n_pairs')} pairs, {nf.get('words_compared')} words)")
        print(f"  Any gap in the tables above smaller than this is not distinguishable from")
        print(f"  two competent people hearing the same German meeting audio differently.")
        print(f"  {nf.get('caveat', '')}")
        rep["noise_floor"] = nf

    if a.json:
        # json rejects numpy scalars such as numpy.bool_ from scipy; .item() converts them.
        def plain(o):
            if hasattr(o, "item"):
                return o.item()
            raise TypeError(f"{type(o).__name__} is not serialisable")
        pathlib.Path(a.json).write_text(
            json.dumps(rep, ensure_ascii=False, indent=1, default=plain), encoding="utf-8")
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
