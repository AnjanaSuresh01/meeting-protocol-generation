"""Scores a word-level majority vote of Canary-1B-v2 and two other recognizers on the eligible gold-set clips.

At each aligned word position, Canary-1B-v2's choice (a word or no word) stands unless the other two
recognizers agree on a different one; an empty or English hypothesis does not vote. For each single
recognizer and each vote it reports the micro WER on all clips, on the reader-edited clips and on the
two-reader clips, and a 95% bootstrap interval on all clips; each vote also gets a Holm-corrected paired
bootstrap p-value against Canary-1B-v2 alone. Canary-1B-v2 is the base because its word
times are needed to assign speakers. The references were seeded from Canary-1B-v2 and Whisper
large-v3-turbo, so the edited and two-reader subsets are where the seed counts least.

    python rover_trio.py    -> goldset/rover_trio_2026-10-09.json
"""
from __future__ import annotations

import itertools
import json
import random
from pathlib import Path

import jiwer

from consensus import NULL, align, key, panel_of
from score_v10 import CFG, GOLD, micro_ci, micro_wer
from thesis_pipeline.text_norm import normalize

REPO = Path(__file__).resolve().parent
OUT = REPO / "goldset" / "rover_trio_2026-10-09.json"
BASE = "canary_diar"
OTHERS = ["whisper_large_v3", "whisper_turbo", "whisper_turbo_de", "cohere", "granite41"]
SINGLES = [BASE] + OTHERS


def vote_text(hyps: dict[str, str], trio: list[str]) -> str:
    sub = {m: hyps.get(m, "") for m in trio}
    panel, _ = panel_of(sub)
    if BASE not in panel:
        return hyps.get(BASE, "")
    order = [BASE] + [m for m in trio if m != BASE and m in panel]
    slots = align({m: sub[m] for m in order}, order)
    words = []
    for s in slots:
        ballots = {m: (key(s.votes[m]) if m in s.votes else NULL) for m in order}
        tally = {}
        for m, k in ballots.items():
            tally[k] = tally.get(k, 0) + 1
        maj = [k for k, v in tally.items() if v >= 2]
        win = maj[0] if maj else ballots[BASE]
        if win == NULL:
            continue
        surface = next(s.votes[m] for m in order if m in s.votes and key(s.votes[m]) == win)
        words.append(surface)
    return " ".join(words)


def errors(ref: str, hyp: str) -> tuple[int, int]:
    r, h = normalize(ref, CFG), normalize(hyp, CFG)
    m = jiwer.process_words([r], [h])
    return m.substitutions + m.deletions + m.insertions, len(r.split())


def paired_p(e_a: list[int], e_b: list[int], words: list[int], n_boot: int = 10000, seed: int = 0) -> float:
    """Two-sided paired bootstrap p-value of micro WER(a) - micro WER(b), resampling clips."""
    k = len(words)
    rng = random.Random(seed)
    le = ge = 0
    for _ in range(n_boot):
        idx = [rng.randrange(k) for _ in range(k)]
        w = sum(words[i] for i in idx)
        d = (sum(e_a[i] for i in idx) - sum(e_b[i] for i in idx)) / w
        le += d <= 0
        ge += d >= 0
    return min(1.0, 2 * min(le, ge) / n_boot)


def holm(ps: dict[str, float]) -> dict[str, float]:
    order = sorted(ps, key=ps.get)
    out, run = {}, 0.0
    for i, name in enumerate(order):
        run = max(run, min(1.0, ps[name] * (len(order) - i)))
        out[name] = run
    return out


def main() -> int:
    g = json.loads(GOLD.read_text(encoding="utf-8"))
    segs = [s for s in (g["segments"] if isinstance(g, dict) else g) if s["scoring"]["eligible"]]
    subsets = {"all": lambda s: True,
               "edited": lambda s: bool(s.get("bias", {}).get("rater_edited")),
               "two_readers": lambda s: s.get("resolution", {}).get("readers") == 2}
    systems = {m: [s["model_hypotheses"].get(m, "") for s in segs] for m in SINGLES}
    trios = [[BASE, a, b] for a, b in itertools.combinations(OTHERS, 2)]
    for t in trios:
        systems["vote:" + "+".join(t[1:])] = [vote_text(s["model_hypotheses"], t) for s in segs]
    refs = [s["reference_text"] for s in segs]

    res = {"n_clips": len(segs), "rule": "2 of 3 overrules Canary-1B-v2; otherwise Canary-1B-v2 stands", "systems": {}}
    errs = {}
    for name, hyps in systems.items():
        e_w = [errors(r, h) for r, h in zip(refs, hyps)]
        errs[name] = [e for e, _ in e_w]
        words = [w for _, w in e_w]
        row = {}
        for sub, keep in subsets.items():
            pairs = [(r, h) for r, h, s in zip(refs, hyps, segs) if keep(s)]
            row[sub] = micro_wer(pairs)
        row["ci95"] = micro_ci(list(zip(refs, hyps)))
        res["systems"][name] = row
    raw = {n: paired_p(errs[n], errs[BASE], words) for n in systems if n.startswith("vote:")}
    adj = holm(raw)
    for n in raw:
        res["systems"][n]["vs_canary"] = {"p_raw": raw[n], "p_holm": adj[n]}
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"{'system':42s} {'WER all':>8s} {'95% CI':>17s} {'edited':>7s} {'2 readers':>9s} {'p_holm vs Canary':>17s}")
    for name, row in res["systems"].items():
        ci = row["ci95"]
        p = row.get("vs_canary", {}).get("p_holm")
        print(f"{name:42s} {row['all']['wer']:8.3f} [{ci[0]:.3f}, {ci[1]:.3f}] {row['edited']['wer']:7.3f} "
              f"{row['two_readers']['wer']:9.3f} {'' if p is None else f'{p:17.4f}'}")
    print("->", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
