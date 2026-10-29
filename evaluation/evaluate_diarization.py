"""Compares the speaker diarization of the eight meetings with the attendance, without a reference annotation.

This is part 1 of evaluation layer 3. For each meeting it reads the diarization hypothesis
pipeline_outputs_extracted/outputs/canary_diar/<meeting>/diarization_hyp.rttm and reports:
  * whether the number of speakers matches the expected number of active speakers, taken from the
    attendance in data.csv (EXPECTED);
  * each speaker's share of the speaking time.
A label with less than MIN_SPEECH_S (15 s) of speech in total is treated as an over-segmentation
artifact, so both the raw label count and the count of speakers with substantial speech are reported.
DER and cpWER need a human who-spoke-when reference; der_cpwer_when_ready() computes DER once one exists.

    python evaluate_diarization.py
        -> pipeline_outputs_extracted/evaluation/layer3_diarization.md and layer3_diarization.json
"""

import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent
OUT = REPO / "pipeline_outputs_extracted" / "outputs" / "canary_diar"
EVAL = REPO / "pipeline_outputs_extracted" / "evaluation"

# expected number of active speakers per meeting, from the attendance in data.csv
EXPECTED = {"march09": 4, "april13": 4, "april27": 4, "may11": 5,
            "may18": 5, "june01": 4, "june15": 4, "june22": 4}
MIN_SPEECH_S = 15.0          # minimum total speech in seconds for a label to count as a real speaker


def parse_rttm(path):
    turns = []
    for line in path.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) >= 8 and p[0] == "SPEAKER":
            turns.append((float(p[3]), float(p[4]), p[7]))   # start, dur, speaker
    return turns


def meeting_stats(turns):
    by_spk = defaultdict(float)
    for _, dur, spk in turns:
        by_spk[spk] += dur
    total = sum(by_spk.values()) or 1.0
    real = {s: t for s, t in by_spk.items() if t >= MIN_SPEECH_S}
    shares = sorted(((s, t, 100 * t / total) for s, t in by_spk.items()),
                    key=lambda x: -x[1])
    return {"raw_speakers": len(by_spk), "real_speakers": len(real),
            "n_turns": len(turns), "speech_min": round(total / 60, 1),
            "avg_turn_s": round(total / max(len(turns), 1), 1), "shares": shares}


def main():
    rows = []
    detail = {}
    for m, exp in EXPECTED.items():
        p = OUT / m / "diarization_hyp.rttm"
        if not p.exists():
            continue
        st = meeting_stats(parse_rttm(p))
        st["expected"] = exp
        st["match"] = "✓" if st["real_speakers"] == exp else f"Δ{st['real_speakers'] - exp:+d}"
        rows.append((m, st))
        detail[m] = st

    n_ok = sum(1 for _, s in rows if s["real_speakers"] == s["expected"])
    print(f"SPEAKER-DETECTION ACCURACY vs attendance: {n_ok}/{len(rows)} meetings exact\n")
    print(f"{'meeting':9} {'expect':>6} {'raw':>4} {'real':>5} {'match':>6} {'turns':>6} {'min':>5} {'avg_turn':>9}")
    for m, s in rows:
        print(f"{m:9} {s['expected']:>6} {s['raw_speakers']:>4} {s['real_speakers']:>5} "
              f"{s['match']:>6} {s['n_turns']:>6} {s['speech_min']:>5} {s['avg_turn_s']:>8}s")

    print("\nspeaking-time shares (who held the floor):")
    for m, s in rows:
        top = "  ".join(f"{spk.replace('SPEAKER_','S')}:{pct:.0f}%" for spk, t, pct in s["shares"] if t >= MIN_SPEECH_S)
        tiny = sum(1 for spk, t, pct in s["shares"] if t < MIN_SPEECH_S)
        print(f"  {m:9} {top}" + (f"   (+{tiny} sliver labels <{MIN_SPEECH_S:.0f}s = over-seg.)" if tiny else ""))

    EVAL.mkdir(exist_ok=True)
    md = ["# Layer 3 (part 1) — Diarization vs attendance (reference-free)", "",
          f"Speaker-detection accuracy: **{n_ok}/{len(rows)} meetings exact** "
          f"(real speakers = those with ≥{MIN_SPEECH_S:.0f}s total speech; sliver labels are over-segmentation).", "",
          "| Meeting | Expected | Detected (real) | Match | Turns | Speech (min) |",
          "|---|--:|--:|:--:|--:|--:|"]
    for m, s in rows:
        md.append(f"| {m} | {s['expected']} | {s['real_speakers']} | {s['match']} | {s['n_turns']} | {s['speech_min']} |")
    md += ["", "DER / cpWER (against a human reference) are wired and pending references — see "
           "`evaluate_diarization.der_cpwer_when_ready`."]
    (EVAL / "layer3_diarization.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (EVAL / "layer3_diarization.json").write_text(json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote -> {EVAL/'layer3_diarization.md'}")


def der_cpwer_when_ready(ref_rttm, hyp_rttm, ref_stm=None, hyp_stm=None):
    """Return the DER of hyp_rttm against the reference ref_rttm (both RTTM files).

    DER is computed by thesis_pipeline.diarization_eval.der. ref_stm and hyp_stm are meant for
    cpWER (for example with MeetEval) and are not used yet."""
    from thesis_pipeline.diarization_eval import parse_rttm as _p, der
    return der(_p(ref_rttm), _p(hyp_rttm))   # pyannote.metrics can replace this DER implementation


if __name__ == "__main__":
    main()
