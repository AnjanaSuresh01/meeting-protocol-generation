"""Gold-set WER with numbers normalized on both sides.

score_v10.py normalizes case and punctuation but not numbers, so "acht" against a reference
"8" counts as an error. This script applies the same transform to reference and hypothesis:
"%" becomes "prozent" before score_v10's normalizer, and German number words become digits
after it (german_numbers.value; the article forms ein/eine/einen/einem/einer stay). It then
recomputes the six micro WERs and score_v10's paired segment bootstrap (same seed) and prints
both versions side by side.

    python number_norm_wer.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

ARTICLES = {"ein", "eine", "einen", "einem", "einer", "eines"}


def numnorm(text: str) -> str:
    import german_numbers as GN
    from score_v10 import CFG
    from thesis_pipeline.text_norm import normalize
    t = normalize(text.replace("%", " prozent "), CFG)
    out = []
    for tok in t.split():
        v = None if tok in ARTICLES else GN.value(tok)
        out.append(str(v) if v is not None else tok)
    return " ".join(out)


def main() -> int:
    import jiwer
    from score_v10 import CFG, GOLD
    from thesis_pipeline.metrics import paired_bootstrap_holm
    from thesis_pipeline.text_norm import normalize
    segs = [s for s in json.loads(GOLD.read_text(encoding="utf-8"))["segments"]
            if s["scoring"]["eligible"]]
    systems = ["canary_diar", "whisper_large_v3", "whisper_turbo", "whisper_turbo_de",
               "cohere", "granite41"]
    segs = [s for s in segs if all(m in s["model_hypotheses"] for m in systems)]
    views = {"as published": lambda x: normalize(x, CFG), "numbers normalised": numnorm}
    res = {}
    for name, f in views.items():
        refs = [f(s["reference_text"]) for s in segs]
        words = [len(r.split()) for r in refs]
        errs = {}
        for m in systems:
            e = []
            for s, r in zip(segs, refs):
                o = jiwer.process_words([r], [f(s["model_hypotheses"][m])])
                e.append(o.substitutions + o.insertions + o.deletions)
            errs[m] = e
        wer = {m: sum(errs[m]) / sum(words) for m in systems}
        rows = paired_bootstrap_holm(errs, words, seed=20260919)
        res[name] = (wer, rows, sum(words))
    print(f"GOLD-SET WER, numbers as written vs numbers normalised (n = {len(segs)} clips)")
    print(f"  {'system':18s}{'as published':>14s}{'numbers normalised':>20s}{'change':>9s}")
    a, b = res["as published"][0], res["numbers normalised"][0]
    for m in systems:
        print(f"  {m:18s}{a[m]:14.4f}{b[m]:20.4f}{b[m] - a[m]:+9.4f}")
    for name in views:
        rows = res[name][1]
        sig = sum(r["significant_0.05"] for r in rows)
        cl = next(r for r in rows if {r["a"], r["b"]} == {"canary_diar", "whisper_large_v3"})
        print(f"  {name:20s}: {sig} of {len(rows)} pairs separate; canary vs large-v3 "
              f"p_holm {cl['p_holm']:.4f}; reference words {res[name][2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
