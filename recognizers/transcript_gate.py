"""Check whether each recognizer's meeting transcripts are fit to be compared.

For every model and meeting, the script reads transcript.txt (or transcript_speakers.txt without
the speaker labels) and the duration of the meeting recording; all eight recordings must be in
this folder. It labels each transcript missing, degenerate (under 100 words, under 60 or over 200
words per minute of audio, loop score above 0.20 or type-token ratio below 0.12), damaged (loop
score above 0.03), suspect (known boilerplate phrase) or clean.

    python transcript_gate.py                 # report on pipeline_outputs_extracted/outputs/
    python transcript_gate.py --apply         # also write transcript_gate.json
    python transcript_gate.py --v10           # report on outputs_v10/
    python transcript_gate.py --v10 --apply   # also write transcript_gate_v10.json
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import soundfile as sf

REPO = Path(__file__).resolve().parent
# --v10: transcripts that assemble_v9_transcripts.py builds from the clip decodes.
# Default: transcripts of the whole-meeting decodes.
USE_V10 = "--v10" in sys.argv
if USE_V10:
    OUT = REPO / "outputs_v10"
    MODELS = ["canary_diar", "cohere", "granite41",
              "whisper_large_v3", "whisper_turbo", "whisper_turbo_de"]
else:
    OUT = REPO / "pipeline_outputs_extracted" / "outputs"
    MODELS = ["whisper", "whisper_turbo", "whisper_turbo_de", "granite41", "cohere", "canary_diar"]
MEETINGS = ["march09", "april13", "april27", "may11", "may18", "june01", "june15", "june22"]
AUDIO = {"march09": "Team-Meeting_2026-03-09.mp3", "april13": "Team-Meeting_2026-04-13.mp3",
         "april27": "Team-Meeting_2026-04-27.flac", "may11": "Team-Meeting_2026-05-11.flac",
         "may18": "Team-Meeting_2026-05-18.flac", "june01": "Team-Meeting_2026-06-01.flac",
         "june15": "Team-Meeting_2026-06-15.flac", "june22": "Team-Meeting_2026-06-22.flac"}

# Allowed words per minute of audio; wide on purpose, to catch only impossible rates.
RATE_LO, RATE_HI = 60, 200
# Loop-score bounds: above LOOP_DAMAGED damaged, above LOOP_DEGENERATE degenerate.
LOOP_DAMAGED, LOOP_DEGENERATE, TTR_MIN, MIN_WORDS = 0.03, 0.20, 0.12, 100

BOILERPLATE = [
    "thank you very much for your question", "thank you for watching", "for watching",
    "subtitles by", "untertitelung des zdf", "vielen dank fürs zuschauen",
    "untertitel im auftrag des zdf", "amara.org", "copyright wdr",
]


def durations() -> dict[str, float]:
    return {m: sf.info(str(REPO / f)).duration / 60 for m, f in AUDIO.items()}


def loop_fraction(words: list[str], n: int = 6, k: int = 3) -> tuple[float, str]:
    """Return the loop score and the most repeated n-gram with its count.

    The score is n times the number of positions whose n-gram occurs k or more times,
    divided by the word count and capped at 1.0.
    """
    if len(words) < n * 2:
        return 0.0, ""
    grams = [" ".join(words[i:i + n]) for i in range(len(words) - n)]
    c = Counter(grams)
    hot = {g for g, v in c.items() if v >= k}
    if not hot:
        return 0.0, ""
    covered = sum(n for g in grams if g in hot)
    worst = max(hot, key=lambda g: c[g])
    return min(1.0, covered / len(words)), f"{worst[:46]} x{c[worst]}"


def judge(text: str, minutes: float) -> dict:
    words = text.split()
    n = len(words)
    low = [w.lower().strip(".,!?;:\"'()") for w in words]
    r = {"n_words": n, "wpm": round(n / minutes, 1) if minutes else 0.0}
    reasons, hard = [], False

    if n < MIN_WORDS:
        return {**r, "verdict": "degenerate", "reasons": [f"only {n} words"], "loop": 1.0,
                "ttr": 0.0, "phrase": ""}

    lf, phrase = loop_fraction(low)
    ttr = len(set(low)) / n
    r.update({"loop": round(lf, 3), "ttr": round(ttr, 3), "phrase": phrase})

    if not (RATE_LO <= r["wpm"] <= RATE_HI):
        reasons.append(f"{r['wpm']:.0f} words/min of audio (human German is 110-160)")
        hard = True
    if lf > LOOP_DEGENERATE:
        reasons.append(f"{lf:.0%} of tokens inside a repeated phrase: '{phrase}'")
        hard = True
    damaged = LOOP_DAMAGED < lf <= LOOP_DEGENERATE
    if damaged:
        reasons.append(f"{lf:.0%} of tokens inside a repeated phrase: '{phrase}'")
    if ttr < TTR_MIN:
        reasons.append(f"type-token ratio {ttr:.2f} — vocabulary collapsed")
        hard = True
    blob = " ".join(low)
    hits = [b for b in BOILERPLATE if b in blob]
    if hits:
        reasons.append(f"hallucinated boilerplate: {hits}")

    r["verdict"] = ("degenerate" if hard else "damaged" if damaged
                    else "suspect" if hits else "clean")
    r["reasons"] = reasons
    return r


def main() -> None:
    apply = "--apply" in sys.argv
    mins = durations()

    print("TRANSCRIPT GATE — what may enter the protocol comparison")
    print("=" * 78)
    print("Run BEFORE any ranking. A ranking that includes degenerate inputs is not a")
    print("ranking of ASR systems.\n")

    report: dict[str, dict[str, dict]] = {}
    for model in MODELS:
        report[model] = {}
        for mid in MEETINGS:
            d = OUT / model / mid
            p = d / "transcript.txt"
            if not p.exists():
                p2 = d / "transcript_speakers.txt"
                if p2.exists():
                    # Fallback for folders with only the speaker-attributed file (canary_diar).
                    text = re.sub(r"^\[\d\d:\d\d:\d\d\]\s*SPEAKER_\d+:\s*", " ",
                                  p2.read_text(encoding="utf-8", errors="replace"), flags=re.M)
                else:
                    report[model][mid] = {"verdict": "missing", "reasons": ["no transcript"],
                                          "n_words": 0, "wpm": 0.0, "loop": 0.0, "ttr": 0.0,
                                          "phrase": ""}
                    continue
            else:
                text = p.read_text(encoding="utf-8", errors="replace")
            report[model][mid] = judge(text, mins[mid])

    mark = {"clean": ".", "suspect": "?", "damaged": "!", "degenerate": "X", "missing": "-"}
    print(f"   {'model':<18}" + "".join(f"{m[:6]:>8}" for m in MEETINGS) + f"{'usable':>9}")
    for model in MODELS:
        row = "".join(f"{mark[report[model][m]['verdict']]:>8}" for m in MEETINGS)
        ok = sum(report[model][m]["verdict"] in ("clean", "suspect") for m in MEETINGS)
        print(f"   {model:<18}{row}{ok:>6} / 8")
    print(f"\n   . clean    ? suspect (usable, phrase noted)    X degenerate    - missing\n")

    print("   WHAT IS WRONG, IN DETAIL\n")
    for model in MODELS:
        for mid in MEETINGS:
            r = report[model][mid]
            if r["verdict"] in ("clean",):
                continue
            print(f"   {model:<18}{mid:<10}{r['n_words']:>6}w {r['wpm']:>6.0f}wpm "
                  f"{r['verdict']:<11}{'; '.join(r['reasons'])[:70]}")

    n_ok = sum(1 for m in MODELS for x in MEETINGS
               if report[m][x]["verdict"] in ("clean", "suspect"))
    n_dam = sum(1 for m in MODELS for x in MEETINGS if report[m][x]["verdict"] == "damaged")
    print(f"\n   {n_ok} of {len(MODELS) * len(MEETINGS)} model-meeting cells are admissible "
          f"({n_ok / (len(MODELS) * len(MEETINGS)):.0%}).")

    print("\n   CONSEQUENCE FOR evaluation/layer1_combined.md\n")
    for model in MODELS:
        bad = [m for m in MEETINGS if report[model][m]["verdict"] in ("degenerate", "missing")]
        if bad:
            print(f"   {model:<18}ranked on {8 - len(bad)} valid meetings, not 8 — "
                  f"excluded: {', '.join(bad)}")
    print("\n   Those rankings average over meetings where the input was a loop. They are not")
    print("   comparable across systems until recomputed on admissible cells only.")

    if apply:
        # One file per tree, so a --v10 run does not overwrite the default result.
        p = REPO / ("transcript_gate_v10.json" if USE_V10 else "transcript_gate.json")
        p.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nWROTE {p.name}")
    else:
        print("\n(report only — re-run with --apply)")


if __name__ == "__main__":
    main()
