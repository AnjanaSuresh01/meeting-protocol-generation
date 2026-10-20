"""Restore missing speaker labels in the Canary word lists from the diarization turns.

For each of the eight meetings, the script reads transcript_canary_ts.json and
diarization_hyp.rttm from pipeline_outputs_extracted/outputs/canary_diar/<meeting>/ next to the
script and relabels all words of the meeting if any word has no speaker. Without --apply it only
prints the result; with --apply it also rewrites transcript_canary_ts.json and
transcript_speakers.txt and keeps a .bak copy of each file that already exists.

    python repair_word_speakers.py           # report only
    python repair_word_speakers.py --apply   # also rewrite the files
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
OUT = REPO / "pipeline_outputs_extracted" / "outputs" / "canary_diar"
MEETINGS = ["march09", "april13", "april27", "may11", "may18", "june01", "june15", "june22"]


def read_rttm(mid: str) -> list[tuple[float, float, str]]:
    turns = []
    for line in (OUT / mid / "diarization_hyp.rttm").read_text(encoding="utf-8").splitlines():
        f = line.split()
        if len(f) > 7 and f[0] == "SPEAKER":
            turns.append((float(f[3]), float(f[3]) + float(f[4]), f[7]))
    return sorted(turns)


def assign(words: list[dict], turns: list[tuple[float, float, str]]) -> int:
    """Relabel the words in place with the rule of assign_speakers in
    pipeline/thesis_pipeline/canary_diarize.py and return how many labels changed.
    """
    n = 0
    for w in words:
        m = (w["start"] + w["end"]) / 2
        covering = [t for t in turns if t[0] <= m <= t[1]]
        if covering:
            spk = max(covering, key=lambda t: min(t[1], w["end"]) - max(t[0], w["start"]))[2]
        else:
            spk = min(turns, key=lambda t: 0.0 if t[0] <= m <= t[1]
                      else min(abs(t[0] - m), abs(t[1] - m)))[2]
        if w.get("speaker", "") != spk:
            n += 1
        w["speaker"] = spk
    return n


def hms(t: float) -> str:
    return f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{int(t % 60):02d}"


def transcript(words: list[dict]) -> str:
    lines, buf, spk, t0 = [], [], None, None
    for w in words:
        if w["speaker"] != spk and buf:
            lines.append(f"[{hms(t0)}] {spk}: {' '.join(buf)}")
            buf = []
        if w["speaker"] != spk:
            spk, t0 = w["speaker"], w["start"]
        buf.append(w["word"])
    if buf:
        lines.append(f"[{hms(t0)}] {spk}: {' '.join(buf)}")
    return "\n".join(lines)


def main() -> None:
    apply = "--apply" in sys.argv
    print("WORD -> SPEAKER LABELS — checking all eight meetings")
    print("=" * 78)
    print(f"   {'meeting':<10}{'words':>7}{'unlabeled':>12}{'turns':>7}{'speakers':>10}  status")

    broken = []
    for mid in MEETINGS:
        p = OUT / mid / "transcript_canary_ts.json"
        words = json.loads(p.read_text(encoding="utf-8"))
        turns = read_rttm(mid)
        empty = sum(1 for w in words if not w.get("speaker", "").strip())
        spk = len({t[2] for t in turns})
        state = "OK" if empty == 0 else f"BROKEN — {empty / len(words):.0%} unlabeled"
        print(f"   {mid:<10}{len(words):>7}{empty:>12}{len(turns):>7}{spk:>10}  {state}")
        if empty:
            broken.append((mid, p, words, turns))

    if not broken:
        print("\n   Nothing to repair.")
        return

    print(f"\n   {len(broken)} meeting(s) to repair. The diarization is intact in every case —")
    print("   only the word->speaker join is missing, and both inputs are on disk.\n")

    for mid, p, words, turns in broken:
        filled = assign(words, turns)
        dist = {}
        for w in words:
            dist[w["speaker"]] = dist.get(w["speaker"], 0) + 1
        share = ", ".join(f"{k}={v / len(words):.0%}"
                          for k, v in sorted(dist.items(), key=lambda x: -x[1]))
        print(f"   {mid}: {filled} words labeled -> {share}")
        if apply:
            shutil.copy2(p, p.with_suffix(".json.bak"))
            p.write_text(json.dumps(words, ensure_ascii=False), encoding="utf-8")
            t = OUT / mid / "transcript_speakers.txt"
            if t.exists():
                shutil.copy2(t, t.with_suffix(".txt.bak"))
            t.write_text(transcript(words), encoding="utf-8")
            print(f"      wrote {p.name} and {t.name} (.bak kept)")

    if apply:
        print("\n   Only the word level is repaired. Segment files keep their speaker labels;")
        print("   re-segmenting to update them would change the segment IDs that every")
        print("   recognizer output refers to.")
    else:
        print("\n   (report only — re-run with --apply)")


if __name__ == "__main__":
    main()
