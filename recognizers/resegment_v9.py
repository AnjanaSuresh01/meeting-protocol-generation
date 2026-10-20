"""Split each meeting's Canary word list into sentences and merge them into ASR segments.

A segment ends where at least NEED of the three boundary votes from boundary_signals.score_gap
agree, searching up to --reach seconds; a segment without such a boundary is still cut and
marked "forced" (asr_eval_ok False). With --sweep the script prints segment statistics for
reach values from 15 to 90 s and writes no file.

Usage:
    python resegment_v9.py [--meetings M1,M2,...] [--reach SECONDS] [--out FILE]
    python resegment_v9.py --sweep [--meetings M1,M2,...]

Defaults: all meetings in boundary_lib.MEETINGS, --reach 35 and --out
goldset/segments_v9_all.new.json, relative to the current directory.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys

import boundary_lib as B
import boundary_signals as BS
import resegment_v7 as V7
import resegment_v8 as V8

TARGET = 15.0  # preferred maximum segment length in seconds, the upper end of the usual 5-15 s ASR segment length
REACH = 35.0  # seconds searched for a boundary; read off the --sweep curve: fewer forced cuts than at 25 s, while a longer reach puts much more speech into segments over 30 s
WINDOW = 30.0  # seconds in one Whisper decode window
PACK_GAP = 2.0  # longest pause in seconds between merged sentences
WANT = 3  # votes (of 3) for an agreed boundary
NEED = 2  # minimum votes for an accepted boundary

TAG = "v9"


def pack(segs: list[dict], ws, env, gate, tl, reach: float = REACH) -> list[dict]:
    """Merge consecutive same-speaker sentences into segments, cutting each at the longest
    candidate within `reach` seconds that has WANT, or failing that NEED, boundary votes;
    candidates of at most TARGET seconds are tried first. If none has NEED votes, the cut is
    still made and the segment is marked "forced".
    """
    out, i = [], 0
    while i < len(segs):
        j = i
        while j + 1 < len(segs):
            nxt, cur = segs[j + 1], segs[j]
            if (nxt["speaker"] != segs[i]["speaker"]
                    or nxt["meeting_id"] != segs[i]["meeting_id"]
                    or nxt["start_s"] - cur["end_s"] > PACK_GAP
                    or nxt["end_s"] - segs[i]["start_s"] > reach):
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
            pick = max(in_band or cands, key=lambda c: (c[1]["votes"], c[1]["sil"]))
            why = "forced"

        k, v, _ = pick
        seg = V7.merge(segs[i:k + 1]) if k > i else dict(segs[i])
        wj = segs[k]["word_j"]
        limit = (ws[wj + 1]["start"] - V7.GUARD) if wj + 1 < len(ws) else float("inf")
        seg = V8.anchor_end(seg, v, limit)
        seg["segment_id"] = seg["segment_id"].replace("_v7_", f"_{TAG}_")
        seg["boundary_votes"] = v["votes"]
        seg["boundary_why"] = why
        seg["boundary_sil_s"] = v["sil"]
        seg["boundary_turn_s"] = v["turn"]
        seg["boundary_wgap_s"] = v["wgap"]
        seg["asr_eval_ok"] = why != "forced"
        seg["over_length_s"] = round(max(0.0, seg["duration_s"] - TARGET), 2)
        seg["long_form_risk"] = seg["duration_s"] > WINDOW
        out.append(seg)
        i = k + 1
    return out


def sentences(meeting: str, nlp):
    """Return (sentences, words, envelope, gate, turns) of one meeting, the inputs of pack()."""
    ws, env = B.words(meeting), B.envelope(meeting)
    gate = V7.HOT * B.levels(env)[0]
    tl = BS.turns(meeting)
    spans = B.sentence_spans(ws, nlp)
    sents = [V7.as_segment(ws, i, j, meeting, n, env, gate)
             for n, (i, j) in enumerate(spans)]
    return sents, ws, env, gate, tl


def describe(segs: list[dict], label: str) -> None:
    n = len(segs)
    d = sorted(s["duration_s"] for s in segs)
    print(f"\n{label}  ({n} segments)")
    print(f"  duration  median {st.median(d):.1f} s   range {d[0]:.1f}-{d[-1]:.1f} s")
    for lo, hi in ((0, 5), (5, 15), (15, 25), (25, 45), (45, 1e9)):
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
    lng = [s for s in segs if s["long_form_risk"]]
    print(f"  over one decode window : {len(lng)} segments, "
          f"{100 * sum(s['n_words'] for s in lng) / sum(s['n_words'] for s in segs):.1f}% of words")
    bad = [s for s in segs if not s["asr_eval_ok"]]
    print(f"  excluded from ASR eval : {len(bad)} segments, "
          f"{sum(s['n_words'] for s in bad)} words, "
          f"{sum(s['duration_s'] for s in bad) / 3600:.2f} h")


def speed_bias(segs: list[dict]) -> tuple[float, float, float]:
    """Return the median words per second of segments with at least WANT votes (a) and of the
    rest (b), and 100 * (b / a - 1); all three are 0.0 if either group is empty.
    """
    def wps(s):
        return s["n_words"] / s["duration_s"] if s["duration_s"] else 0.0
    good = [s for s in segs if s["boundary_votes"] >= WANT]
    rest = [s for s in segs if s["boundary_votes"] < WANT]
    if not good or not rest:
        return 0.0, 0.0, 0.0
    a, b = st.median([wps(s) for s in good]), st.median([wps(s) for s in rest])
    return a, b, 100 * (b / a - 1)


def sweep(meetings: list[str], reaches: list[float]) -> None:
    """Print, for each reach, the pooled segment count, forced count and share, median and
    95th-percentile duration, and speed_bias() percentage over the given meetings.
    """
    nlp = B.german_nlp()
    print(f"reach sweep over {', '.join(meetings)}\n")
    # Meetings in the outer loop: boundary_lib caches the audio of only one meeting at a time.
    by_reach: dict[float, list[dict]] = {r: [] for r in reaches}
    for m in meetings:
        sents, ws, env, gate, tl = sentences(m, nlp)
        for r in reaches:
            by_reach[r] += pack(sents, ws, env, gate, tl, reach=r)
    print("  %6s %8s %8s %9s %9s %9s %9s"
          % ("reach", "segs", "forced", "forced%", "median s", "p95 s", "bias%"))
    for r in reaches:
        allsegs = by_reach[r]
        d = sorted(s["duration_s"] for s in allsegs)
        f = sum(s["boundary_why"] == "forced" for s in allsegs)
        _, _, bias = speed_bias(allsegs)
        print("  %6.0f %8d %8d %8.1f%% %9.1f %9.1f %8.1f%%"
              % (r, len(allsegs), f, 100 * f / len(allsegs), st.median(d),
                 d[int(0.95 * len(d))], bias))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meetings", default=",".join(B.MEETINGS))
    ap.add_argument("--out", default="goldset/segments_v9_all.new.json",
                    help="output file; the frozen goldset/segments_v9_all.json is only "
                         "replaced when it is named here")
    ap.add_argument("--reach", type=float, default=REACH)
    ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args()
    ms = a.meetings.split(",")

    if a.sweep:
        sweep(ms, [15.0, 25.0, 35.0, 45.0, 60.0, 90.0])
        return 0

    nlp = B.german_nlp()
    allsegs: list[dict] = []
    for m in ms:
        sents, ws, env, gate, tl = sentences(m, nlp)
        packed = pack(sents, ws, env, gate, tl, reach=a.reach)
        allsegs += packed
        print(f"  {m:<9} {len(ws):>6} words -> {len(sents):>5} sentences -> "
              f"{len(packed):>4} segments   forced "
              f"{sum(s['boundary_why'] == 'forced' for s in packed):>3}")
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(allsegs, f, ensure_ascii=False, indent=1)
    describe(allsegs, f"v9 — reach {a.reach:.0f} s, no break forced where the audio has none")
    g, b, bias = speed_bias(allsegs)
    print(f"  speech rate  agreed {g:.2f} w/s   not agreed {b:.2f} w/s   bias {bias:+.1f}%")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
