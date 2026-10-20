"""Merge the per-sentence segments of resegment_v7 into same-speaker segments whose end is chosen
by three boundary votes: silence, pyannote turns and Canary word gaps (boundary_signals.score_gap).
Each segment ends where at least NEED votes agree, preferably within TARGET seconds, otherwise
within HARD_MAX seconds, and is marked "forced" when no such boundary exists. anchor_end then
moves the speech end out to the first silence after the last word, where there is one.

    python resegment_v8.py [--meetings march09,april13,...] [--out FILE]

--meetings defaults to all meetings in boundary_lib.MEETINGS. The segments are written as JSON
to --out (default goldset/segments_v8_packed.json, relative to the working directory), and a
summary is printed.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys

import boundary_lib as B
import boundary_signals as BS
import resegment_v7 as V7

TARGET = 15.0  # seconds; boundaries within this span are preferred (upper end of the usual 5-15 s ASR segment length)
HARD_MAX = 25.0  # seconds, longest span a merge may produce
PACK_GAP = 2.0  # seconds, longest gap a merge may cross
WANT = 3  # votes (of 3) for a fully agreed boundary
NEED = 2  # minimum votes (of 3) for a boundary that is not forced
ANCHOR_MIN = 0.20  # seconds of quiet in the gap that anchor_end requires


def anchor_end(seg: dict, v: dict, limit: float) -> dict:
    """Move the segment's speech end later, to v["cut_at"], the start of the first quiet run after
    its last word, but never past `limit`; no change if cut_at is None or v["sil"] < ANCHOR_MIN.

    Up to V7.SIL_END s of trailing silence is kept, but not beyond cut_at + v["sil"] or `limit`,
    and the rest becomes padding.
    """
    if v.get("cut_at") is None or v["sil"] < ANCHOR_MIN:
        return seg
    offset = seg["end_s"] - seg["silence_after_s"]  # speech end without the trailing silence
    new = max(offset, min(v["cut_at"], limit))
    avail = max(0.0, min(v["cut_at"] + v["sil"], limit) - new)
    keep = min(V7.SIL_END, avail)
    seg = dict(seg)
    seg["end_s"] = round(new + keep, 3)
    seg["silence_after_s"] = round(keep, 3)
    seg["pad_end_s"] = round(V7.SIL_END - keep, 3)
    seg["duration_s"] = round(seg["end_s"] - seg["start_s"]
                              + seg["pad_start_s"] + seg["pad_end_s"], 2)
    seg["end_anchored_s"] = round(new - offset, 3)  # seconds the speech end was moved later
    return seg


def pack(segs: list[dict], ws, env, gate, tl) -> list[dict]:
    """Merge consecutive same-speaker sentences, ending each segment at the longest candidate
    with WANT votes, else NEED votes, first within TARGET s and then within HARD_MAX s.

    If none qualifies, the segment ends at the best candidate by votes, then silence (within
    TARGET s where possible), and is marked "forced".
    """
    out, i = [], 0
    while i < len(segs):
        j = i
        while j + 1 < len(segs):
            nxt, cur = segs[j + 1], segs[j]
            if (nxt["speaker"] != segs[i]["speaker"]
                    or nxt["meeting_id"] != segs[i]["meeting_id"]
                    or nxt["start_s"] - cur["end_s"] > PACK_GAP
                    or nxt["end_s"] - segs[i]["start_s"] > HARD_MAX):
                break
            j += 1

        def dur(k):
            return segs[k]["end_s"] - segs[i]["start_s"]

        cands = []
        for k in range(i, j + 1):
            v = BS.score_gap(ws, segs[k]["word_j"], env, gate, tl)
            cands.append((k, v, dur(k)))

        in_band = [c for c in cands if c[2] <= TARGET]
        over = [c for c in cands if c[2] > TARGET]

        pick, why = None, ""
        for pool, bar, tag in ((in_band, WANT, "agreed-in-band"),
                               (in_band, NEED, "majority-in-band"),
                               (over, WANT, "agreed-over-length"),
                               (over, NEED, "majority-over-length")):
            hits = [c for c in pool if c[1]["votes"] >= bar]
            if hits:
                pick, why = max(hits, key=lambda c: c[2]), tag
                break
        if pick is None:
            pick = max(in_band or cands,
                       key=lambda c: (c[1]["votes"], c[1]["sil"]))
            why = "forced"

        k, v, _ = pick
        seg = V7.merge(segs[i:k + 1]) if k > i else dict(segs[i])
        wj = segs[k]["word_j"]
        limit = (ws[wj + 1]["start"] - V7.GUARD) if wj + 1 < len(ws) else float("inf")
        seg = anchor_end(seg, v, limit)
        seg["boundary_votes"] = v["votes"]
        seg["boundary_why"] = why
        seg["boundary_sil_s"] = v["sil"]
        seg["boundary_turn_s"] = v["turn"]
        seg["boundary_wgap_s"] = v["wgap"]
        out.append(seg)
        i = k + 1
    return out


def describe(segs: list[dict], label: str) -> None:
    n = len(segs)
    d = sorted(s["duration_s"] for s in segs)
    print(f"\n{label}  ({n} segments)")
    print(f"  duration  median {st.median(d):.1f} s   range {d[0]:.1f}-{d[-1]:.1f} s")
    for lo, hi in ((0, 5), (5, 15), (15, 25), (25, 1e9)):
        k = sum(1 for x in d if lo <= x < hi)
        tag = "  <- the 5-15 s target band" if (lo, hi) == (5, 15) else ""
        print(f"  {lo:>3}-{hi if hi < 1e9 else '':<4} s {k:>5} ({k / n:>4.0%}) "
              f"{'#' * int(38 * k / n)}{tag}")
    print(f"  total audio           : {sum(s['duration_s'] for s in segs) / 3600:.2f} h")
    print(f"  speaker-pure          : {sum(s['speaker_pure'] for s in segs)}/{n}")
    for v in (3, 2, 1, 0):
        k = sum(s["boundary_votes"] == v for s in segs)
        print(f"  ends on {v} of 3 votes : {k:>5} ({k / n:>4.0%})")
    print("  how the boundary was reached:")
    for why in ("agreed-in-band", "majority-in-band", "agreed-over-length",
                "majority-over-length", "forced"):
        k = sum(s["boundary_why"] == why for s in segs)
        if k:
            print(f"    {why:<22} {k:>5} ({k / n:>4.0%})")
    moved = [s["end_anchored_s"] for s in segs if s.get("end_anchored_s", 0) > 0.005]
    print(f"  cut moved out to real silence: {len(moved)}/{n} ({len(moved) / n:.0%})"
          f"   median {st.median(moved) * 1000 if moved else 0:.0f} ms")
    clamp = [s["tail_clipped_s"] for s in segs if s["tail_clipped_s"] > V7.CLIP_TOL]
    print(f"  ends mid-word (v7 clamp): {len(clamp)}/{n} = {len(clamp) / n:.0%}"
          f"   median loss {st.median(clamp) * 1000 if clamp else 0:.0f} ms")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meetings", default=",".join(B.MEETINGS))
    ap.add_argument("--out", default="goldset/segments_v8_packed.json")
    a = ap.parse_args()

    nlp = B.german_nlp()
    allsegs: list[dict] = []
    for m in a.meetings.split(","):
        ws, env = B.words(m), B.envelope(m)
        gate = V7.HOT * B.levels(env)[0]
        tl = BS.turns(m)
        spans = B.sentence_spans(ws, nlp)
        sents = [V7.as_segment(ws, i, j, m, n, env, gate) for n, (i, j) in enumerate(spans)]
        packed = pack(sents, ws, env, gate, tl)
        allsegs += packed
        print(f"  {m:<9} {len(ws):>6} words -> {len(sents):>5} sentences -> "
              f"{len(packed):>4} segments")
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(allsegs, f, ensure_ascii=False, indent=1)
    describe(allsegs, "v8 — three witnesses vote, length gives way")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
