"""Counts the first-name words that lie inside diarization turns shorter than 1.2 s.

These are the name words that a rule dropping every diarization turn shorter than 1.2 s would delete.
For each of the eight meetings it reads the turns of
pipeline_outputs_extracted/outputs/canary_diar/<meeting>/diarization_hyp.rttm and the word times of
transcript_canary_ts.json in the same folder, and records every word from the hard-coded first-name
list (NAMES) whose midpoint lies inside a short turn, with its time, the turn length and the turn's speaker.

    python short_turn_names_2026-10-26.py    -> goldset/short_turn_names_2026-10-26.json
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAMES = {"thomas", "johannes", "vincent", "osman", "jens"}
MEETINGS = ["march09", "april13", "april27", "may11", "may18", "june01", "june15", "june22"]

out = {"rule": "diarization turn shorter than 1.2 s", "names": sorted(NAMES), "short_turns": 0, "hits": []}
for m in MEETINGS:
    base = ROOT / "pipeline_outputs_extracted/outputs/canary_diar" / m
    turns = []
    for line in (base / "diarization_hyp.rttm").read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) > 8 and p[0] == "SPEAKER":
            s = float(p[3])
            turns.append((s, s + float(p[4]), p[7]))
    d = json.loads((base / "transcript_canary_ts.json").read_text(encoding="utf-8"))
    words = d["words"] if isinstance(d, dict) else d
    short = [t for t in turns if t[1] - t[0] < 1.2]
    out["short_turns"] += len(short)
    for w in words:
        if re.sub(r"[^\wäöüé]", "", w["word"].lower()) in NAMES:
            mid = (w["start"] + w["end"]) / 2
            for t in short:
                if t[0] <= mid <= t[1]:
                    out["hits"].append({"meeting": m, "word": w["word"], "time_s": round(mid, 2),
                                        "turn_s": round(t[1] - t[0], 2), "turn_speaker": t[2]})
                    break
(ROOT / "goldset" / "short_turn_names_2026-10-26.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                                                   encoding="utf-8")
print(out["short_turns"], "short turns;", len(out["hits"]), "name words inside them")
