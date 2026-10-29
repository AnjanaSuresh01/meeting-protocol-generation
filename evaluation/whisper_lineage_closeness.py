"""Measures how close each recognizer's output is to the scored output of Whisper large-v3-turbo and Canary-1B-v2.

Uses the normalizer and micro WER of score_v10.py on the eligible gold-set clips, with the output of
the anchor system as the reference. Also counts the clips whose output equals the gold reference after
normalization (bias.identical_to).

    python whisper_lineage_closeness.py     -> goldset/whisper_lineage_closeness_2026-10-29.json
"""
import json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
import score_v10 as S

doc = json.loads((ROOT / "goldset" / "goldset_v10.json").read_text(encoding="utf-8"))
segs = [s for s in doc["segments"] if s["scoring"]["eligible"]]
systems = ["whisper_turbo", "whisper_large_v3", "whisper_turbo_de", "cohere", "canary_diar", "granite41"]
print("clips", len(segs))
res = {"clips": len(segs), "scorer": "score_v10.micro_wer, NormConfig(casefold=True)", "distance": {}, "identical_to_reference": {}}

for anchor in ("whisper_turbo", "canary_diar"):
    print("\nDistance to the output of", anchor, "(WER with that output as reference)")
    for m in systems:
        if m == anchor:
            continue
        pairs = [(s["model_hypotheses"][anchor], s["model_hypotheses"][m]) for s in segs
                 if anchor in s["model_hypotheses"] and m in s["model_hypotheses"]]
        r = S.micro_wer(pairs)
        same = sum(1 for a, b in pairs if S.normalize(a, S.CFG) == S.normalize(b, S.CFG))
        print("  %-18s n=%3d  WER %.3f  identical clips %3d" % (m, r["n"], r["wer"], same))
        res["distance"].setdefault(anchor, {})[m] = {"wer": round(r["wer"], 4), "identical_clips": same}

print("\nClips identical to the reference after normalization")
for m in systems:
    k = sum(1 for s in segs if m in s["bias"]["identical_to"])
    print("  %-18s %3d" % (m, k))
    res["identical_to_reference"][m] = k
(ROOT / "goldset" / "whisper_lineage_closeness_2026-10-29.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print("wrote goldset/whisper_lineage_closeness_2026-10-29.json")
