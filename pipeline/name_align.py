"""Split a second ASR transcript into the speaker turns of the main transcript, using a word
alignment with difflib instead of timestamps.

Run as a script, it prints for 8 meetings the median word overlap per turn, paired by index
and after alignment. It reads outputs_v11_final/canary_diar/<meeting>/transcript_speakers.txt
and outputs_v10/cohere/<meeting>/transcript_speakers.txt from the working directory.

    python name_align.py
"""
from __future__ import annotations

import difflib
import re
from bisect import bisect_right

_TOKEN = re.compile(r"\S+")
_NORM = re.compile(r"[^\wäöüß]+")


def _norm(tok: str) -> str:
    return _NORM.sub("", tok.lower())


def align_to_segments(main_segs: list[str], other_segs: list[str]) -> list[str]:
    """Return one text per main segment, taken from the other transcript at the same place."""
    main_tok, starts = [], []
    for seg in main_segs:
        starts.append(len(main_tok))
        main_tok += _TOKEN.findall(seg)
    other_tok = [t for seg in other_segs for t in _TOKEN.findall(seg)]
    if not main_tok or not other_tok:
        return ["" for _ in main_segs]
    a = [_norm(t) for t in main_tok]
    b = [_norm(t) for t in other_tok]
    blocks = difflib.SequenceMatcher(None, a, b).get_matching_blocks()
    # Anchors: main token index -> other token index for each matched token.
    # Both lists increase, which bisect_right in map_index relies on.
    ai, bj = [], []
    for blk in blocks:
        for k in range(blk.size):
            ai.append(blk.a + k)
            bj.append(blk.b + k)

    def map_index(i: int) -> int:
        """Return the other-transcript position for main token position i: interpolated between
        anchors, extrapolated beyond them, scaled by length if there are none.
        """
        if not ai:
            return round(i * len(b) / len(a))
        p = bisect_right(ai, i) - 1
        if p < 0:
            return max(0, bj[0] - (ai[0] - i))
        if ai[p] == i or p == len(ai) - 1:
            return min(len(b), bj[p] + (i - ai[p]))
        x0, x1, y0, y1 = ai[p], ai[p + 1], bj[p], bj[p + 1]
        return y0 + round((i - x0) * (y1 - y0) / (x1 - x0))

    cuts = [map_index(s) for s in starts] + [len(b)]
    cuts = [max(0, min(len(b), c)) for c in cuts]
    for k in range(1, len(cuts)):
        cuts[k] = max(cuts[k], cuts[k - 1])
    return [" ".join(other_tok[cuts[k]:cuts[k + 1]]) for k in range(len(main_segs))]


def _overlap(x: str, y: str) -> float:
    a = set(re.findall(r"[a-zäöüß]{4,}", x.lower()))
    b = set(re.findall(r"[a-zäöüß]{4,}", y.lower()))
    return len(a & b) / len(a | b) if a | b else 0.0


def main() -> int:
    import statistics
    import time
    from pathlib import Path
    from protokoll import _segments
    meets = ["march09", "april13", "april27", "may11", "may18", "june01", "june15", "june22"]
    print(f"{'meeting':9s} {'turns':>6s} {'by index (old)':>16s} {'aligned (new)':>15s} {'secs':>6s}")
    for mid in meets:
        main_t = Path(f"outputs_v11_final/canary_diar/{mid}/transcript_speakers.txt").read_text(
            encoding="utf-8")
        other_t = Path(f"outputs_v10/cohere/{mid}/transcript_speakers.txt").read_text(
            encoding="utf-8")
        ms, os_ = _segments(main_t), _segments(other_t)
        n = min(len(ms), len(os_))
        old = statistics.median(_overlap(ms[i], os_[i]) for i in range(n))
        t0 = time.time()
        al = align_to_segments(ms, os_)
        dt = time.time() - t0
        new = statistics.median(_overlap(ms[i], al[i]) for i in range(len(ms)))
        print(f"{mid:9s} {len(ms):6d} {old:16.2f} {new:15.2f} {dt:6.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
