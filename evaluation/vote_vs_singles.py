"""Tests the kept word vote against every single recognizer on the eligible gold-set clips.

rover_trio.py tests each vote only against Canary-1B-v2 alone, because the system needs the word times
of Canary-1B-v2. This script compares the kept vote (Canary-1B-v2, Whisper large-v3-turbo-German, Cohere
Transcribe) with each of the six single recognizers. It uses the functions and settings of rover_trio.py:
the normalizer and micro WER of score_v10.py, a paired bootstrap over clips (10,000 resamples, seed 0)
and Holm correction over the six recognizers.

    python vote_vs_singles.py       -> goldset/vote_vs_singles_2026-10-25.json
"""
from __future__ import annotations

import json
from pathlib import Path

from rover_trio import BASE, SINGLES, errors, holm, paired_p, vote_text
from score_v10 import GOLD

REPO = Path(__file__).resolve().parent
OUT = REPO / "goldset" / "vote_vs_singles_2026-10-25.json"
TRIO = [BASE, "whisper_turbo_de", "cohere"]


def main() -> int:
    g = json.loads(GOLD.read_text(encoding="utf-8"))
    segs = [s for s in (g["segments"] if isinstance(g, dict) else g) if s["scoring"]["eligible"]]
    refs = [s["reference_text"] for s in segs]
    vote = [vote_text(s["model_hypotheses"], TRIO) for s in segs]
    e_w = [errors(r, h) for r, h in zip(refs, vote)]
    e_vote = [e for e, _ in e_w]
    words = [w for _, w in e_w]
    total = sum(words)

    res = {"n_clips": len(segs), "ref_words": total, "vote": "+".join(TRIO), "vote_wer": sum(e_vote) / total,
           "test": "paired bootstrap over clips, 10,000 resamples, seed 0, two-sided; Holm over the six singles",
           "singles": {}}
    raw = {}
    for m in SINGLES:
        e_m = [errors(r, s["model_hypotheses"].get(m, ""))[0] for r, s in zip(refs, segs)]
        raw[m] = paired_p(e_vote, e_m, words)
        res["singles"][m] = {"wer": sum(e_m) / total, "vote_minus_single": (sum(e_vote) - sum(e_m)) / total,
                             "p_raw": raw[m],
                             "clips_vote_better": sum(a < b for a, b in zip(e_vote, e_m)),
                             "clips_vote_worse": sum(a > b for a, b in zip(e_vote, e_m))}
    for m, p in holm(raw).items():
        res["singles"][m]["p_holm"] = p
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"vote {res['vote']}: WER {res['vote_wer']:.4f} on {len(segs)} clips")
    for m, row in res["singles"].items():
        print(f"{m:18s} WER {row['wer']:.4f}  diff {row['vote_minus_single']:+.4f}  p_raw {row['p_raw']:.4f}  "
              f"p_holm {row['p_holm']:.4f}  better/worse {row['clips_vote_better']}/{row['clips_vote_worse']}")
    print("->", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
