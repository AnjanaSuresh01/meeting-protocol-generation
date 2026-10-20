"""Repeat the Holm-corrected paired WER tests of score_v10.py with four bootstrap resampling
schemes: independent clips (iid), meeting-speaker blocks, whole meetings, and meetings then
clips within each meeting (hierarchical). Clips of the same meeting and speaker are not independent, so the iid scheme
can give intervals that are too narrow and p-values that are too small. For all system pairs, the iid scheme must give the
same results as score_v10.py and as the stored goldset/score_v10_2026-10-01.json, or the script
stops. It reads goldset/goldset_v10.json and writes the test results and the per-system 95%
intervals to goldset/wer_block_bootstrap_2026-10-01.json.

    python wer_block_bootstrap.py
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
OUT = REPO / "goldset" / "wer_block_bootstrap_2026-10-01.json"
N_BOOT = 10000
SEED = 20260919  # same seed as score_v10.py; the iid check needs it
UNSEEDED: list[str] = []  # set by load(), same rule as score_v10.py


def load():
    import jiwer
    import score_v10 as SV
    from thesis_pipeline.text_norm import normalize
    doc = json.loads(SV.GOLD.read_text(encoding="utf-8"))
    segs = doc["segments"]
    # Same system order as score_v10.py, so each pair has the same a - b orientation.
    systems = [m for m in doc["meta"]["panel"]["systems"]
               if any(m in s["model_hypotheses"] for s in segs)]
    global UNSEEDED
    UNSEEDED = [m for m in systems if m not in SV.SEEDING and m not in SV.SAME_FAMILY]
    shared = [s for s in segs
              if s["scoring"]["eligible"] and all(m in s["model_hypotheses"] for m in systems)]
    refs = [normalize(s["reference_text"], SV.CFG) for s in shared]
    words = [len(r.split()) for r in refs]
    errors = {}
    for m in systems:
        e = []
        for s, r in zip(shared, refs):
            w = jiwer.process_words([r], [normalize(s["model_hypotheses"][m], SV.CFG)])
            e.append(int(w.substitutions + w.insertions + w.deletions))
        errors[m] = e
    return shared, systems, words, errors


def iid_replicates(k: int) -> list[list[int]]:
    rng = random.Random(SEED)  # same draws as metrics.paired_bootstrap_holm
    return [[rng.randrange(k) for _ in range(k)] for _ in range(N_BOOT)]


def block_replicates(blocks: list[list[int]], seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    out = []
    for _ in range(N_BOOT):
        idx: list[int] = []
        for _ in range(len(blocks)):
            idx.extend(blocks[rng.randrange(len(blocks))])
        out.append(idx)
    return out


def hierarchical_replicates(blocks: list[list[int]], seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    out = []
    for _ in range(N_BOOT):
        idx: list[int] = []
        for _ in range(len(blocks)):
            b = blocks[rng.randrange(len(blocks))]
            idx.extend(b[rng.randrange(len(b))] for _ in range(len(b)))
        out.append(idx)
    return out


def paired_holm(errors: dict[str, list[int]], words: list[int], reps: list[list[int]],
                models: list[str]) -> list[dict]:
    """Paired bootstrap test with Holm correction, as in metrics.paired_bootstrap_holm,
    on replicate index lists given by the caller.
    """
    def micro(e, idx):
        w = sum(words[i] for i in idx)
        return sum(e[i] for i in idx) / w if w else 0.0

    base = list(range(len(words)))
    raw = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            ea, eb = errors[models[i]], errors[models[j]]
            obs = micro(ea, base) - micro(eb, base)
            diffs = sorted(micro(ea, r) - micro(eb, r) for r in reps)
            n_cross = sum(1 for d in diffs if (d >= 0) != (obs >= 0))
            p = min(1.0, 2.0 * (n_cross + 1) / (len(reps) + 1))
            raw.append((models[i], models[j], obs, diffs[int(0.025 * len(reps))],
                        diffs[int(0.975 * len(reps))], p))
    raw.sort(key=lambda r: r[5])
    out, running = [], 0.0
    for rank, (a, b, obs, lo, hi, p) in enumerate(raw):
        running = max(running, min(1.0, p * (len(raw) - rank)))
        out.append({"a": a, "b": b, "delta_wer": round(obs, 4), "ci95": [round(lo, 4), round(hi, 4)],
                    "p_raw": round(p, 4), "p_holm": round(running, 4),
                    "significant_0.05": running < 0.05})
    return out


def system_ci(errors: dict[str, list[int]], words: list[int], reps: list[list[int]]) -> dict:
    out = {}
    for m, e in errors.items():
        v = sorted(sum(e[i] for i in r) / sum(words[i] for i in r) for r in reps)
        out[m] = [round(v[int(0.025 * len(v))], 4), round(v[int(0.975 * len(v))], 4)]
    return out


def main() -> int:
    from thesis_pipeline.metrics import paired_bootstrap_holm
    shared, systems, words, errors = load()
    k = len(shared)
    by_meeting, by_speaker = defaultdict(list), defaultdict(list)
    for i, s in enumerate(shared):
        by_meeting[s["meeting_id"]].append(i)
        by_speaker[(s["meeting_id"], s["speaker"])].append(i)
    meetings = [by_meeting[m] for m in sorted(by_meeting)]
    speakers = [by_speaker[b] for b in sorted(by_speaker)]

    schemes = {"iid": iid_replicates(k),
               "speaker": block_replicates(speakers, SEED + 1),
               "meeting": block_replicates(meetings, SEED + 2),
               "hierarchical": hierarchical_replicates(meetings, SEED + 3)}

    # Reproduction checks against score_v10.py
    frozen = paired_bootstrap_holm(errors, words, seed=SEED)
    mine = paired_holm(errors, words, schemes["iid"], systems)
    key = lambda r: (r["a"], r["b"])
    assert sorted(frozen, key=key) == sorted(mine, key=key), "iid scheme does not reproduce score_v10"
    stored = json.loads((REPO / "goldset/score_v10_2026-10-01.json").read_text(encoding="utf-8"))
    assert sorted(stored["pairwise_segment"], key=key) == sorted(frozen, key=key), \
        "stored score_v10 result differs from a fresh run"

    res = {"built": "2026-10-01", "n_segments": k, "ref_words": sum(words), "n_boot": N_BOOT,
           "blocks": {"meeting": len(meetings), "speaker": len(speakers)},
           "method": {"iid": "Bisani & Ney 2004 (= score_v10)",
                      "speaker": "Liu & Peng, Interspeech 2020, arXiv 1912.09508",
                      "meeting": "cluster bootstrap; Anglin, arXiv 2606.26422: under-covers below 20 clusters",
                      "hierarchical": "two-stage bootstrap; Anglin, arXiv 2606.26422"},
           "all_pairs": {}, "unseeded_pairs": {}, "system_ci95": {}}
    for name, reps in schemes.items():
        res["all_pairs"][name] = paired_holm(errors, words, reps, systems)
        res["unseeded_pairs"][name] = paired_holm(errors, words, reps, UNSEEDED)
        res["system_ci95"][name] = system_ci(errors, words, reps)

    print(f"WER PAIRS UNDER FOUR RESAMPLING SCHEMES — {k} clips, {sum(words)} reference words, "
          f"{len(meetings)} meetings, {len(speakers)} meeting-speaker blocks, {N_BOOT} replicates")
    print("gate: the iid scheme reproduces score_v10 and its stored result exactly — passed")
    print()
    order = sorted(res["all_pairs"]["iid"], key=lambda r: abs(r["delta_wer"]), reverse=True)
    names = list(schemes)
    print(f"  {'pair':40s}{'dWER':>8s}" + "".join(f"{n:>14s}" for n in names))
    for r in order:
        cells = []
        for n in names:
            q = next(x for x in res["all_pairs"][n] if x["a"] == r["a"] and x["b"] == r["b"])
            cells.append(f"{q['p_holm']:>9.4f} {'YES' if q['significant_0.05'] else 'no ':>3s} ")
        print(f"  {r['a'] + ' vs ' + r['b']:40s}{r['delta_wer']:>8.3f}" + "".join(f"{c:>14s}" for c in cells))
    print()
    print("  pairs separating at p_holm < 0.05:  " +
          ",  ".join(f"{n} {sum(x['significant_0.05'] for x in res['all_pairs'][n])}/15" for n in names))
    print()
    print("  UNSEEDED SYSTEMS ONLY (Holm over these 3 pairs):")
    for r in res["unseeded_pairs"]["iid"]:
        cells = []
        for n in names:
            q = next(x for x in res["unseeded_pairs"][n] if x["a"] == r["a"] and x["b"] == r["b"])
            cells.append(f"{q['p_holm']:>9.4f} {'YES' if q['significant_0.05'] else 'no ':>3s} ")
        print(f"  {r['a'] + ' vs ' + r['b']:40s}{r['delta_wer']:>8.3f}" + "".join(f"{c:>14s}" for c in cells))
    print()
    print("  95 % interval of each system's micro WER:")
    for m in systems:
        print(f"  {m:20s}" + "".join(f"  {n}: [{res['system_ci95'][n][m][0]:.4f}, {res['system_ci95'][n][m][1]:.4f}]"
                                       for n in names))
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
