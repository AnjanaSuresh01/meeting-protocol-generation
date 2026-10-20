"""Paired gain of the word vote of the final three recognizers over Canary-1B-v2 on gold-set subsets.

Uses the vote (rover_trio.vote_text), the error count (rover_trio.errors) and the paired
bootstrap (rover_trio.paired_p: 10,000 resamples of clips, seed 0) of rover_trio.py on three
subsets of the eligible clips: all of them, the clips the first reader edited, and the clips
read by two readers. For each subset it prints the micro WER difference with its 95% interval
and p-value.

    python vote_subsets_ci.py      -> goldset/vote_subsets_2026-10-10.json
"""
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import rover_trio as R  # noqa: E402

TRIO = [R.BASE, "whisper_turbo_de", "cohere"]
SUBSETS = {"all": lambda s: True,
           "edited": lambda s: bool(s.get("bias", {}).get("rater_edited")),
           "two_readers": lambda s: s.get("resolution", {}).get("readers") == 2}
OUT = REPO / "goldset" / "vote_subsets_2026-10-10.json"


def main():
    g = json.loads(R.GOLD.read_text(encoding="utf-8"))
    segs = [s for s in (g["segments"] if isinstance(g, dict) else g) if s["scoring"]["eligible"]]
    res = {"trio": TRIO, "subsets": {}}
    for name, keep in SUBSETS.items():
        S = [s for s in segs if keep(s)]
        ea, eb, w = [], [], []
        for s in S:
            e1, n1 = R.errors(s["reference_text"], R.vote_text(s["model_hypotheses"], TRIO))
            e2, _ = R.errors(s["reference_text"], s["model_hypotheses"].get(R.BASE, ""))
            ea.append(e1)
            eb.append(e2)
            w.append(n1)
        rng = random.Random(0)
        idx = range(len(S))
        diffs = []
        for _ in range(10000):
            b = [rng.choice(idx) for _ in idx]
            diffs.append((sum(ea[i] for i in b) - sum(eb[i] for i in b)) / sum(w[i] for i in b))
        diffs.sort()
        res["subsets"][name] = {
            "n": len(S), "wer_canary": round(sum(eb) / sum(w), 4), "wer_vote": round(sum(ea) / sum(w), 4),
            "difference": round((sum(ea) - sum(eb)) / sum(w), 4),
            "ci95": [round(diffs[249], 4), round(diffs[9749], 4)], "paired_p": R.paired_p(ea, eb, w)}
        r = res["subsets"][name]
        print(f"{name:12s} n={r['n']:3d}  {r['wer_canary']:.3f} -> {r['wer_vote']:.3f}  diff {r['difference']:+.4f} "
              f"[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]  p = {r['paired_p']:.4f}")
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
