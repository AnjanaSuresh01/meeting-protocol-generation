"""Split each meeting's Canary word list into one segment per sentence, with boundaries at
the speech edges in the audio and a fixed leading and trailing silence budget per clip.
With --pack, consecutive same-speaker sentences are then merged while the span stays within
PACK_TARGET seconds; --retreat first moves clipped final words into the next sentence.

    python resegment_v7.py [--meetings a,b] [--pack] [--retreat]

Output goes to goldset/ in this folder: segments_v7.json and, with --pack,
segments_v7_packed.json (segments_v7r.json and segments_v7r_packed.json with --retreat).
"""

from __future__ import annotations

import json
import statistics as st
import sys

import numpy as np

import boundary_lib as B

G = B.G
HOT = 0.25  # speech gate, as a fraction of the meeting's p75 frame RMS
HANG = 0.12  # seconds of sub-gate audio that end a word; a shorter gap is a stop closure
LOOK_OUT = 0.80  # seconds searched past a timestamp for the speech edge
LOOK_BACK = 0.08  # seconds searched before the timestamp as well
GUARD = 0.05  # seconds kept clear of the neighboring word
MIN_TAIL = 0.02  # minimum seconds from the last word's end timestamp to the end limit

SIL_END = 0.50  # seconds of trailing silence in every clip
# In as_segment, real_after is capped by room_after, so a longer SIL_END never reaches
# the next word; the rest of the budget is digital padding.
SIL_START = 0.20  # seconds of leading silence in every clip

PACK_TARGET = 15.0  # seconds, longest span a merge may produce; the upper end of the usual 5-15 s ASR segment length
PACK_GAP = 2.0  # seconds, longest gap a merge may cross
CLEAN_PAUSE = 0.25  # seconds of pause before the next speech that make an end clean
CLIP_TOL = 0.02  # seconds of last-word speech the neighbor clamp may remove


# Stage 1: sentence segments

def speech_offset(env, t: float, gate: float, limit: float) -> float:
    """Find where the speech near `t` ends, searching forward up to LOOK_OUT seconds.

    Return (offset, capped, clipped): the end found from `t` onward, clamped to `limit`; whether
    speech was still going when the search ran out; and the seconds of speech the clamp removed.
    """
    # Search the full LOOK_OUT window and clamp only the result, so the clipped amount is known.
    k0 = max(0, int((t - LOOK_BACK) / B.HOP))
    kmax = min(len(env) - 1, int((t + LOOK_OUT) / B.HOP))
    last = k0
    hang = int(HANG / B.HOP)
    k = k0
    while k <= kmax:
        if env[k] > gate:
            last = k
        elif k - last >= hang:
            break
        k += 1
    capped = bool(k > kmax and (last + hang) * B.HOP >= t + LOOK_OUT)
    raw = max(t, (last + 1) * B.HOP)
    return min(limit, raw), capped, max(0.0, raw - limit)


def speech_onset(env, t: float, gate: float, limit: float) -> float:
    """Mirror of speech_offset: find where the speech near `t` starts, searching backward.

    Return (onset, clipped): the start found from `t` backward, clamped to `limit`, and the
    seconds of speech the clamp removed.
    """
    k0 = min(len(env) - 1, int((t + LOOK_BACK) / B.HOP))
    kmin = max(0, int((t - LOOK_OUT) / B.HOP))
    first = k0
    hang = int(HANG / B.HOP)
    k = k0
    while k >= kmin:
        if env[k] > gate:
            first = k
        elif first - k >= hang:
            break
        k -= 1
    raw = min(t, first * B.HOP)
    return max(limit, raw), max(0.0, limit - raw)


def quiet_run(env, t: float, gate: float, forward: bool, cap: float) -> float:
    """Return the seconds of continuous sub-gate audio from `t`, forward or backward, up to `cap`."""
    k, step = int(t / B.HOP), int(cap / B.HOP)
    rng = range(k, k + step) if forward else range(k - 1, k - step - 1, -1)
    n = 0
    for x in rng:
        if x < 0 or x >= len(env) or env[x] > gate:
            break
        n += 1
    return n * B.HOP


def as_segment(ws, i, j, meeting, n, env, gate) -> dict:
    """Build the segment record for the sentence ws[i..j] of one meeting.

    Boundaries come from the speech edges in `env`; the silence budget is taken from the
    recording where there is room, and the rest is recorded as digital padding (pad_start_s,
    pad_end_s).
    """
    w0, w1 = ws[i], ws[j]
    prev_end = ws[i - 1]["end"] if i > 0 else 0.0
    next_start = ws[j + 1]["start"] if j + 1 < len(ws) else len(env) * B.HOP

    # Limit the clip at the neighbors' acoustic speech edges, not at their timestamps.
    # Each neighbor search is bounded by this sentence's own word timestamp.
    prev_off = (speech_offset(env, prev_end, gate, limit=w0["start"])[0] if i > 0 else 0.0)
    next_on = (speech_onset(env, next_start, gate, limit=w1["end"])[0]
               if j + 1 < len(ws) else len(env) * B.HOP)

    # Contiguous: the next speech starts within 20 ms of this sentence's last word end,
    # so no acoustic boundary exists.
    contiguous = next_on <= w1["end"] + 0.02
    # Without a pause, the end limit is the next word's start timestamp less GUARD, and the trailing
    # silence is all digital. Canary's word start timestamps have no median error against the
    # waveform, while its end timestamps run early.
    right = max(w1["end"] + MIN_TAIL,
                (next_start if contiguous else next_on) - GUARD)

    onset, head_clipped = speech_onset(env, w0["start"], gate, limit=prev_off + GUARD)
    offset, capped, tail_clipped = speech_offset(env, w1["end"], gate, limit=right)
    turn_ends = j + 1 >= len(ws) or ws[j + 1]["speaker"] != ws[j]["speaker"]

    room_before = max(0.0, onset - (prev_off + GUARD))
    room_after = max(0.0, (next_on - GUARD) - offset)
    real_before = min(SIL_START, room_before, quiet_run(env, onset, gate, False, SIL_START))
    real_after = min(SIL_END, room_after, quiet_run(env, offset, gate, True, SIL_END))

    a = max(0.0, onset - real_before)
    b = offset + real_after
    txt = " ".join(w["word"] for w in ws[i:j + 1])
    return {
        "segment_id": f"{meeting}_v7_{n:04d}",
        "meeting_id": meeting,
        # sorted() makes the majority-speaker tie-break independent of the string hash seed,
        # so repeated runs give identical output.
        "speaker": max(sorted({s["speaker"] for s in ws[i:j + 1]}),
                       key=[s["speaker"] for s in ws[i:j + 1]].count),
        "speaker_pure": len({s["speaker"] for s in ws[i:j + 1]}) == 1,
        "start_s": round(a, 3),
        "end_s": round(b, 3),
        # digital silence (s) the clip cutter must add
        "pad_start_s": round(SIL_START - real_before, 3),
        "pad_end_s": round(SIL_END - real_after, 3),
        "silence_before_s": round(real_before, 3),
        "silence_after_s": round(real_after, 3),
        # how far (s) the acoustic edges lie beyond the word timestamps
        "tail_overhang_s": round(offset - w1["end"], 3),
        "head_overhang_s": round(w0["start"] - onset, 3),
        "room_after_s": round(room_after, 3),
        "next_speech_s": round(next_on, 3),
        "speech_capped": bool(capped),
        "contiguous_end": bool(contiguous),
        # speech (s) the neighbor clamp removed from the last and first word
        "tail_clipped_s": round(tail_clipped, 3),
        "head_clipped_s": round(head_clipped, 3),
        # clean_end: acoustic evidence that the speaker finished (a pause or a turn change)
        "clean_end": bool(not capped and not contiguous
                          and (room_after >= CLEAN_PAUSE or turn_ends)),
        # safe_end: the neighbor clamp removed at most CLIP_TOL of the last word
        # (an energy-based screen only)
        "safe_end": bool(tail_clipped <= CLIP_TOL),
        "duration_s": round(b - a + (SIL_START - real_before) + (SIL_END - real_after), 2),
        "n_words": j - i + 1,
        "n_sentences": 1,
        "word_i": i, "word_j": j,
        "text": txt,
    }


# Stage 2: merge-only packing

def merge(group: list[dict]) -> dict:
    """Join a run of consecutive sentence segments into one segment record."""
    a, b = group[0], group[-1]
    return {**a,
            "segment_id": a["segment_id"].replace("_v6_", "_v6p_"),
            "end_s": b["end_s"],
            "pad_end_s": b["pad_end_s"],
            "silence_after_s": b["silence_after_s"],
            "tail_overhang_s": b["tail_overhang_s"],
            "room_after_s": b["room_after_s"],
            "speech_capped": b["speech_capped"],
            "clean_end": b["clean_end"],
            "safe_end": b["safe_end"],
            "tail_clipped_s": b["tail_clipped_s"],
            "head_clipped_s": a["head_clipped_s"],
            "word_j": b["word_j"],
            "speaker_pure": all(x["speaker_pure"] for x in group)
                            and len({x["speaker"] for x in group}) == 1,
            "duration_s": round(b["end_s"] - a["start_s"]
                                + a["pad_start_s"] + b["pad_end_s"], 2),
            "n_words": sum(x["n_words"] for x in group),
            "n_sentences": len(group),
            "text": " ".join(x["text"] for x in group)}


def pack(segs: list[dict]) -> list[dict]:
    """Merge consecutive same-speaker sentences of one meeting, never splitting a sentence.

    A merge stops at a speaker or meeting change, at a gap over PACK_GAP, or when the span would
    exceed PACK_TARGET seconds. Each segment ends at the last sentence end that is both safe and
    clean, else the last safe one, else the last clean one, else the furthest one.
    """
    out: list[dict] = []
    i = 0
    while i < len(segs):
        j = i
        while j + 1 < len(segs):
            nxt = segs[j + 1]
            if (nxt["speaker"] != segs[i]["speaker"]
                    or nxt["meeting_id"] != segs[i]["meeting_id"]
                    or nxt["start_s"] - segs[j]["end_s"] > PACK_GAP
                    or nxt["end_s"] - segs[i]["start_s"] > PACK_TARGET):
                break
            j += 1

        def within(limit: float) -> int:
            k = i
            while k + 1 <= j and segs[k + 1]["end_s"] - segs[i]["start_s"] <= limit:
                k += 1
            return k

        in_band = within(PACK_TARGET)
        safe = [k for k in range(i, in_band + 1) if segs[k]["safe_end"]]
        clean = [k for k in range(i, in_band + 1) if segs[k]["clean_end"]]
        both = [k for k in safe if segs[k]["clean_end"]]
        m = both[-1] if both else (safe[-1] if safe else (clean[-1] if clean else in_band))
        out.append(merge(segs[i:m + 1]))
        i = m + 1
    return out


# Report and command line

def describe(segs: list[dict], label: str) -> None:
    d = sorted(s["duration_s"] for s in segs)
    n = len(segs)
    print(f"\n{label}  ({n} segments)")
    print(f"  duration  median {st.median(d):.1f} s   range {d[0]:.1f}-{d[-1]:.1f} s")
    for lo, hi in ((0, 2), (2, 5), (5, 15), (15, 30), (30, 1e9)):
        k = sum(1 for x in d if lo <= x < hi)
        tag = "  <- the 5-15 s target band" if (lo, hi) == (5, 15) else ""
        print(f"  {lo:>3}-{hi if hi < 1e9 else '':<4} s {k:>5} ({k / n:>4.0%}) "
              f"{'#' * int(40 * k / n)}{tag}")
    print(f"  speaker-pure          : {sum(s['speaker_pure'] for s in segs)}/{n}")
    print(f"  end at real evidence  : {sum(s['clean_end'] for s in segs) / n:.0%}"
          f"   (pause >= {CLEAN_PAUSE} s, or the turn changes)")
    print(f"  next speaker touches  : {sum(s['contiguous_end'] for s in segs) / n:.0%}"
          f"   (no silence exists between the two — digital silence supplies the end)")
    print(f"  SAFE end (v7)         : {sum(s['safe_end'] for s in segs) / n:.0%}"
          f"   (energy screen only — edge_audit.py overrules it)")
    print(f"  clamped mid-word      : {sum(not s['safe_end'] for s in segs) / n:.0%}"
          f"   median loss "
          f"{st.median([s['tail_clipped_s'] for s in segs if not s['safe_end']] or [0]) * 1000:.0f} ms")
    print(f"  speech never stopped  : {sum(s['speech_capped'] for s in segs) / n:.0%}"
          f"   (search hit {LOOK_OUT} s still hearing speech)")
    sil = [s["end_s"] - s["start_s"] for s in segs]
    print(f"  trailing silence in clip (median): "
          f"{st.median([s['silence_after_s'] + s['pad_end_s'] for s in segs]) * 1000:.0f} ms"
          f"   leading: "
          f"{st.median([s['silence_before_s'] + s['pad_start_s'] for s in segs]) * 1000:.0f} ms")
    print(f"  needed digital silence: end {sum(s['pad_end_s'] > 0.01 for s in segs) / n:.0%}"
          f"   start {sum(s['pad_start_s'] > 0.01 for s in segs) / n:.0%}")
    print(f"  end timestamp was early by (median): "
          f"{st.median([s['tail_overhang_s'] for s in segs]) * 1000:.0f} ms")
    print(f"  total audio           : {sum(s['duration_s'] for s in segs) / 3600:.2f} h")


def retreat_unsafe_ends(spans, ws, env, gate):
    """Move final words that the neighbor clamp would cut by more than CLIP_TOL into the next
    sentence span; used only with --retreat. Off by default: forced alignment on re-cut clips showed no fewer
    truncated final words, because safe_end is only an energy screen.
    """
    out, carry = [], None
    for (i, j) in spans:
        i = carry if carry is not None else i
        carry = None
        if j < i:
            continue
        while j > i:
            w1, nxt = ws[j], (ws[j + 1] if j + 1 < len(ws) else None)
            if nxt is None:
                break
            next_on, _ = speech_onset(env, nxt["start"], gate, limit=w1["end"])
            cont = next_on <= w1["end"] + 0.02
            right = max(w1["end"] + MIN_TAIL,
                        (nxt["start"] if cont else next_on) - GUARD)
            if speech_offset(env, w1["end"], gate, limit=right)[2] <= CLIP_TOL:
                break
            j -= 1
            carry = j + 1
        out.append((i, j))
    return out


def main() -> None:
    nlp = B.german_nlp()
    allsegs: list[dict] = []
    want = B.MEETINGS
    for k, arg in enumerate(sys.argv):
        if arg == "--meetings" and k + 1 < len(sys.argv):
            want = sys.argv[k + 1].split(",")
    for m in want:
        ws = B.words(m)
        env = B.envelope(m)
        gate = HOT * B.levels(env)[0]
        spans = B.sentence_spans(ws, nlp)
        if "--retreat" in sys.argv:
            spans = retreat_unsafe_ends(spans, ws, env, gate)
        segs = [as_segment(ws, i, j, m, n, env, gate) for n, (i, j) in enumerate(spans)]
        allsegs += segs
        over = np.median([s["tail_overhang_s"] for s in segs])
        print(f"  {m:<9} {len(ws):>6} words -> {len(segs):>5} sentences   "
              f"end extended by median {over * 1000:>4.0f} ms")

    seg_out = G / ("segments_v7r.json" if "--retreat" in sys.argv else "segments_v7.json")
    seg_out.write_text(
        json.dumps(allsegs, ensure_ascii=False, indent=1), encoding="utf-8")
    describe(allsegs, "STAGE 1 — every sentence, boundaries from the waveform")

    if "--pack" in sys.argv:
        packed = pack(allsegs)
        packed_out = G / ("segments_v7r_packed.json" if "--retreat" in sys.argv
                          else "segments_v7_packed.json")
        packed_out.write_text(
            json.dumps(packed, ensure_ascii=False, indent=1), encoding="utf-8")
        describe(packed, f"STAGE 2 — merged towards {PACK_TARGET:.0f} s (merge only, no splits)")
        print(f"\nwrote {packed_out}")
    print(f"\nwrote {seg_out}")


if __name__ == "__main__":
    main()
