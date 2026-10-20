"""Measure how much the two readers of the gold set disagree on the clips they both read.

The script compares the snapshot taken before the second reader's corrections were merged
with goldset/goldset_v10.json. It prints the word-level disagreement rate and a lower and
upper bound for all eligible clips, and writes goldset/reference_reliability.json.

Usage:
    python reference_reliability.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent
BEFORE = REPO / "goldset" / "goldset_v10.BEFORE_peinl_merge_20260907.json"
AFTER = REPO / "goldset" / "goldset_v10.json"
OUT = REPO / "goldset" / "reference_reliability.json"

# Same normalization as goldset/rater_noise_floor.json, so the two rates are comparable. Case is
# kept because German capitalization carries meaning.
from thesis_pipeline.text_norm import NormConfig, normalize      # noqa: E402

CFG = NormConfig(casefold=False, strip_punct=True, remove_filled_pauses=True, collapse_ws=True)


def main() -> int:
    import jiwer

    bef = {s["segment_id"]: s for s in json.loads(BEFORE.read_text(encoding="utf-8"))["segments"]}
    aft = {s["segment_id"]: s for s in json.loads(AFTER.read_text(encoding="utf-8"))["segments"]}
    elig = {k: v for k, v in aft.items() if v.get("scoring", {}).get("eligible")}
    two = [k for k, v in elig.items() if v["resolution"]["readers"] == 2 and k in bef]
    one = [k for k, v in elig.items() if v["resolution"]["readers"] == 1]

    edits = words = changed = 0
    per_clip = []
    for sid in two:
        a = normalize(bef[sid]["reference_text"], CFG)
        b = normalize(aft[sid]["reference_text"], CFG)
        m = jiwer.process_words([a], [b])
        e = m.substitutions + m.deletions + m.insertions
        n = max(len(a.split()), len(b.split()))
        edits += e
        words += n
        changed += e > 0
        per_clip.append({"segment_id": sid, "edits": e, "words": n,
                         "anjana": bef[sid]["reference_text"], "peinl": aft[sid]["reference_text"]})
    words_one = sum(len(normalize(elig[s]["reference_text"], CFG).split()) for s in one)
    rate = edits / words
    lo = edits / (words + words_one)

    print("REFERENCE RELIABILITY — the two readers compared on the clips both read")
    print("=" * 92)
    print(f"  eligible clips                    {len(elig)}")
    print(f"  read by two readers   (measured)  {len(two):3d}   {words} words, {edits} edits")
    print(f"  of those, second reader changed   {changed:3d}")
    print(f"  read by one reader  (unmeasured)  {len(one):3d}   {words_one} words")
    print()
    print(f"  DISAGREEMENT on the clips both read      {rate:.3f}")
    print(f"  bound for the whole eligible set         {lo:.3f} .. {rate:.3f}")
    print()
    print("  The two-reader clips are the ones the FIRST reader flagged as doubtful, so the")
    print(f"  {rate:.3f} is an UPPER bound for that stratum, not an average. Both readers are the")
    print("  two people who built the set; this is not an independent inter-rater study.")
    print()
    print("  system WER on the same 147 clips, for scale:")
    for n_, w in [("whisper_turbo", 0.067), ("whisper_large_v3", 0.087), ("whisper_turbo_de", 0.088),
                  ("cohere", 0.098), ("canary_diar", 0.099), ("granite41", 0.294)]:
        mark = "  <- inside the reference band" if w <= rate else ""
        print(f"    {n_:18s} {w:.3f}{mark}")
    print()
    print("  Five of six systems sit inside or just above the band; granite41 is far outside it.")
    print("  So the granite41 gap survives reference noise and the differences among the top five")
    print("  are of the same order as two humans disagreeing on the same audio.")

    OUT.write_text(json.dumps({
        "built": "2026-09-28",
        "what_this_is": "Reader-vs-reader disagreement on the gold-set clips both readers read, "
                        "from the pre-merge snapshot against the final file.",
        "never_quote_as": "the reference error rate of the gold set. The 48 clips were SELECTED "
                          "by the first reader as doubtful, so the rate is an upper bound for "
                          "that stratum; the other 99 eligible clips have one reader and are "
                          "unmeasured. Both readers built the set, so this is not an independent "
                          "inter-rater study.",
        "normalisation": CFG.as_dict(),
        "metric": "(S+D+I) / max(len(a), len(b)) over normalised tokens, pooled",
        "measured": {"clips_two_readers": len(two), "words": words, "edits": edits,
                     "clips_changed_by_second_reader": changed,
                     "disagreement_rate": round(rate, 4)},
        "unmeasured": {"clips_one_reader": len(one), "words": words_one},
        "bound_for_eligible_set": {"lower_if_one_reader_clips_perfect": round(lo, 4),
                                   "upper_if_as_bad_as_flagged": round(rate, 4)},
        "what_would_settle_it": "A second German reader on a random sample of the 99 "
                                "one-reader clips. Needs a German speaker, not code.",
        "per_clip": sorted(per_clip, key=lambda r: -r["edits"]),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  wrote -> {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
