"""Decides which segments of goldset/segments_v9_all.json the ASR evaluation may score.

A segment is excluded when its boundary was forced (`boundary_why == "forced"`) or its
clip was rated unusable; segments with `long_form_risk` are kept as a separate group.
Run as a script, it prints the groups and writes the exclusion list to
goldset/asr_eval_exclude_v9.json.

Usage:
    python asr_eval_scope.py
"""

from __future__ import annotations

import json
import pathlib
import sys

G = pathlib.Path(__file__).resolve().parent / "goldset"
SEGMENTS = G / "segments_v9_all.json"
EXCLUDE_FILE = G / "asr_eval_exclude_v9.json"


def load(path: pathlib.Path = SEGMENTS) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def excluded(segs: list[dict] | None = None) -> dict[str, str]:
    """Map each excluded segment_id to its reason; clips rated unusable are always listed."""
    import rater_notes as PN
    segs = load() if segs is None else segs
    out = {s["segment_id"]: "no-agreement: boundary forced, no witness endorses the cut"
           for s in segs if s.get("boundary_why") == "forced"}
    for sid, why in PN.UNUSABLE.items():
        out[sid] = f"rated unusable: {why}"
    return out


def partition(segs: list[dict] | None = None) -> dict[str, list[dict]]:
    """Split segments into "scored", "scored_long_form" and "dropped" (excluded) lists."""
    segs = load() if segs is None else segs
    bad = excluded(segs)
    keep = [s for s in segs if s["segment_id"] not in bad]
    return {"scored": [s for s in keep if not s.get("long_form_risk")],
            "scored_long_form": [s for s in keep if s.get("long_form_risk")],
            "dropped": [s for s in segs if s["segment_id"] in bad]}


def main() -> int:
    segs = load()
    p = partition(segs)
    bad = excluded(segs)
    w = sum(s["n_words"] for s in segs)

    print(f"ASR EVALUATION SCOPE — {SEGMENTS.name}\n")
    print(f"  {'group':<22}{'segments':>10}{'words':>9}{'% words':>9}{'hours':>8}")
    for k in ("scored", "scored_long_form", "dropped"):
        g = p[k]
        print(f"  {k:<22}{len(g):>10}{sum(s['n_words'] for s in g):>9}"
              f"{100 * sum(s['n_words'] for s in g) / w:>8.1f}%"
              f"{sum(s['duration_s'] for s in g) / 3600:>8.2f}")
    print(f"  {'TOTAL':<22}{len(segs):>10}{w:>9}{100.0:>8.1f}%"
          f"{sum(s['duration_s'] for s in segs) / 3600:>8.2f}")

    here = {s["segment_id"] for s in segs}
    reasons: dict[str, list[int]] = {}
    for sid, why in bad.items():
        k = why.split(":")[0]
        r = reasons.setdefault(k, [0, 0])
        r[0] += 1
        r[1] += sid in here
    print("\n  why dropped:                 rule   in this file")
    for k, (n, n_here) in sorted(reasons.items(), key=lambda x: -x[1][0]):
        print(f"    {k:<20} {n:>7} {n_here:>13}")
    print("    (a rule can name a segment from a retired set, such as april27_v5p_0244, and it")
    print("     stays on the list so no later build can quietly score it again)")

    EXCLUDE_FILE.write_text(json.dumps(
        {"instruction": ("Peinl, 3 Sep 2026: \"exclude them for the ASR evaluation. "
                         "It will just contaminate it.\""),
         "source": SEGMENTS.name,
         "n_excluded": len(bad),
         "excluded": bad}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwrote {EXCLUDE_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
