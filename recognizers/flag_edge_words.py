"""Flag segments whose first or last word lies mostly in silence.

Canary's word timestamps fall on 80 ms encoder frames, so an edge word can point at silence.
Such segments are flagged so the rater checks that word by ear; they are not dropped.

For the first and last word of each segment, the script stores the word, the share of its
timestamped span that is above the meeting's speech gate, and a flag that is true when this
share is below VOICED. Each file named in --files is read from goldset/ next to this script
and rewritten in place, and the flag counts are printed.

    python flag_edge_words.py --files FILE [FILE ...]   # files in goldset/, rewritten in place
    python flag_edge_words.py --files segments_v9_rerun.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import boundary_lib as B
import resegment_v7 as V7

G = pathlib.Path(__file__).resolve().parent / "goldset"
VOICED = 0.20       # a word must be above the speech gate for at least this share of its span


def voiced_fraction(env, gate: float, t0: float, t1: float) -> float:
    """Return the share of envelope frames between t0 and t1 (in seconds) that are above gate."""
    i0 = int(t0 / B.HOP)
    i1 = max(i0 + 1, int(t1 / B.HOP))
    span = env[i0:i1]
    return float((span > gate).mean()) if len(span) else 0.0


def annotate(segs: list[dict]) -> list[dict]:
    for m in sorted({s["meeting_id"] for s in segs}):
        ws, env = B.words(m), B.envelope(m)
        gate = V7.HOT * B.levels(env)[0]
        for s in segs:
            if s["meeting_id"] != m:
                continue
            first, last = ws[s["word_i"]], ws[s["word_j"]]
            fv = voiced_fraction(env, gate, first["start"], first["end"])
            lv = voiced_fraction(env, gate, last["start"], last["end"])
            s["first_word"] = first["word"]
            s["last_word"] = last["word"]
            s["first_word_voiced"] = round(fv, 3)
            s["last_word_voiced"] = round(lv, 3)
            s["first_word_in_silence"] = fv < VOICED
            s["last_word_in_silence"] = lv < VOICED
    return segs


def report(segs: list[dict], label: str) -> None:
    n = len(segs)
    f = sum(s["first_word_in_silence"] for s in segs)
    l = sum(s["last_word_in_silence"] for s in segs)
    both = sum(s["first_word_in_silence"] or s["last_word_in_silence"] for s in segs)
    print(f"\n{label}  ({n} segments)")
    print(f"  first word points at silence : {f:>5} ({f / n:>4.0%})")
    print(f"  last  word points at silence : {l:>5} ({l / n:>4.0%})")
    print(f"  either edge                  : {both:>5} ({both / n:>4.0%})"
          f"   <- flagged for the rater's ear")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True,
                    help="segment files in goldset/ to annotate in place, "
                         "e.g. goldset_v9_draw.json segments_v9_all.json")
    a = ap.parse_args()
    for name in a.files:
        p = G / name
        segs = json.loads(p.read_text(encoding="utf-8"))
        annotate(segs)
        p.write_text(json.dumps(segs, ensure_ascii=False, indent=1), encoding="utf-8")
        report(segs, name)
        print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
