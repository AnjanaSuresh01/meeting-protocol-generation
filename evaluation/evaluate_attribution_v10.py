"""Compares speaker attribution of LISA protocols written from the flat and from the diarized transcript.

The two output trees differ in one variable, the transcript that generate_protocols.py gave to
LISA:

    outputs_v10_backup_2026-09-09_prediarization/   flat transcript.txt, the LLM guesses turns
    outputs_v10/                                    diarized transcript_speakers.txt
                                                    (`[HH:MM:SS] SPEAKER_NN: text`), the LLM
                                                    maps SPEAKER_NN to people

Both use the same six ASR systems, the same two LISA models (flash, pro), the same roster and the
same prompt skeleton, so each (ASR system, meeting, LISA model) cell is a pair. The reference is
the team's own minutes (person -> topic -> bullets), for the meetings with machine-readable
minutes (HUMAN); the scoring functions are imported unchanged from
evaluate_speaker_attribution.py, with topic-match threshold TH = 0.15. This is not DER (the
minutes have no timestamps); it measures whether a topic is credited to the right person.

For each arm the script reports attribution accuracy on the located topics, the chance level,
and the number of rostered attendees with no section at all. The paired comparison counts cells
where the diarized arm is better, worse or tied and applies a sign test and a Wilcoxon
signed-rank test. A per-ASR-system table follows for the diarized arm.

    python evaluate_attribution_v10.py    -> goldset/attribution_v10_ab.json
"""

from __future__ import annotations

import json
from pathlib import Path

from evaluate_speaker_attribution import (
    HUMAN, ROSTER, read_human, read_generated, score, oracle_remap,
)

REPO = Path(__file__).resolve().parent
TREES = {
    "pre-fix (flat transcript)": REPO / "outputs_v10_backup_2026-09-09_prediarization",
    "post-fix (diarized)": REPO / "outputs_v10",
}
MODELS = ["canary_diar", "cohere", "granite41",
          "whisper_large_v3", "whisper_turbo", "whisper_turbo_de"]
LISAS = ["flash", "pro"]
TH = 0.15          # topic-match threshold passed to score()


def collect(tree: Path) -> list[dict]:
    rows = []
    for mid in HUMAN:                       # meetings with machine-readable minutes
        human = read_human(mid)
        for model in MODELS:
            for lisa in LISAS:
                p = tree / model / mid / f"protocol_{lisa}.md"
                if not p.exists() or p.stat().st_size == 0:
                    continue
                gen, mapping = read_generated(p)
                loc, cor, n, chance = score(human, gen, TH)
                roster = ROSTER[mid]
                named = {g[0] for g in gen}
                rows.append({
                    "meeting": mid, "model": model, "lisa": lisa,
                    "human_topics": n, "located": loc, "correct": cor, "chance": chance,
                    "people_named": len(named),
                    "roster_n": len(roster),
                    # rostered attendees whose first name appears in no section heading
                    "attendees_missing": sum(
                        1 for r in roster
                        if not any(r.split()[0].lower() in g.lower() for g in named)),
                })
    return rows


def agg(rows: list[dict]) -> dict:
    loc = sum(r["located"] for r in rows)
    cor = sum(r["correct"] for r in rows)
    n = sum(r["human_topics"] for r in rows)
    ch = sum(r["chance"] * r["located"] for r in rows) / loc if loc else 0.0
    return {
        "protocols": len(rows), "human_topics": n, "located": loc, "correct": cor,
        "accuracy": cor / loc if loc else 0.0,
        "chance": ch,
        "coverage": loc / n if n else 0.0,
        "attendees_missing": sum(r["attendees_missing"] for r in rows),
    }


def main() -> None:
    print("=" * 90)
    print("SPEAKER ATTRIBUTION — flat transcript vs diarized transcript, paired A/B")
    print("=" * 90)
    print("Reference = the team's own minutes (who each topic is credited to).")
    print("NOT DER: the minutes carry no timestamps.\n")

    data = {}
    for label, tree in TREES.items():
        if not tree.exists():
            print(f"  !! {label}: tree not found ({tree.name}) — skipped")
            continue
        data[label] = collect(tree)

    print(f"{'arm':<28}{'protocols':>10}{'topics':>9}{'found':>8}"
          f"{'right person':>14}{'chance':>9}{'attendees missing':>19}")
    for label, rows in data.items():
        a = agg(rows)
        print(f"{label:<28}{a['protocols']:>10}{a['human_topics']:>9}{a['located']:>8}"
              f"{a['accuracy']:>13.1%}{a['chance']:>9.1%}{a['attendees_missing']:>19}")

    if len(data) == 2:
        (l1, r1), (l2, r2) = list(data.items())
        a1, a2 = agg(r1), agg(r2)
        d = a2["accuracy"] - a1["accuracy"]
        print(f"\n  delta (diarized - flat): {d:+.1%} attribution accuracy, "
              f"{a2['attendees_missing'] - a1['attendees_missing']:+d} attendees missing")

        # paired per (model, meeting, lisa) cell, since both arms share every cell
        key = lambda r: (r["model"], r["meeting"], r["lisa"])
        m1 = {key(r): r for r in r1}
        m2 = {key(r): r for r in r2}
        both = sorted(set(m1) & set(m2))
        wins = losses = ties = 0
        for k in both:
            x, y = m1[k], m2[k]
            ax = x["correct"] / x["located"] if x["located"] else 0.0
            ay = y["correct"] / y["located"] if y["located"] else 0.0
            if ay > ax:
                wins += 1
            elif ay < ax:
                losses += 1
            else:
                ties += 1
        print(f"  paired cells: {len(both)}  |  diarized better {wins}, "
              f"worse {losses}, tied {ties}")

        # The pooled delta is a ratio and can move on a handful of cells; the paired sign and
        # Wilcoxon tests decide whether the difference is distinguishable from noise (p < 0.05).
        try:
            from scipy.stats import binomtest, wilcoxon
            acc = lambda r: (r["correct"] / r["located"]) if r["located"] else 0.0
            xs = [acc(m1[k]) for k in both]
            ys = [acc(m2[k]) for k in both]
            p_sign = binomtest(wins, wins + losses, 0.5).pvalue if wins + losses else 1.0
            p_wil = wilcoxon(ys, xs).pvalue
            print(f"  sign test p = {p_sign:.3f}   Wilcoxon p = {p_wil:.3f}")
            verdict = ("NOT distinguishable from noise" if min(p_sign, p_wil) >= 0.05
                       else "a real difference")
            print(f"  -> attribution accuracy: {verdict}. "
                  f"The attendee-drop fix is categorical and stands on its own.")
        except ImportError:
            print("  (scipy not available — paired significance test skipped)")

    print("\nPER ASR SYSTEM (post-fix arm), attribution accuracy at threshold 0.15\n")
    post = data.get("post-fix (diarized)", [])
    print(f"  {'system':<20}{'flash':>10}{'pro':>10}{'attendees missing':>20}")
    for model in MODELS:
        cells = []
        for lisa in LISAS:
            rs = [r for r in post if r["model"] == model and r["lisa"] == lisa]
            a = agg(rs)
            cells.append(f"{a['accuracy']:.1%}" if rs else "-")
        miss = sum(r["attendees_missing"] for r in post if r["model"] == model)
        print(f"  {model:<20}{cells[0]:>10}{cells[1]:>10}{miss:>20}")

    out = REPO / "goldset" / "attribution_v10_ab.json"
    out.write_text(json.dumps(
        {label: {"aggregate": agg(rows), "rows": rows} for label, rows in data.items()},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwrote -> {out}")


if __name__ == "__main__":
    main()
