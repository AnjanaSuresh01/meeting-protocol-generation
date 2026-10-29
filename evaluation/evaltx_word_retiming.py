"""Counts how the evaluated transcript re-timed its words: matched, nearest matched neighbor, or segment fallback.

Replays the alignment of speaker_turns_v11.v10_words without changing anything. For every segment of
goldset/segments_v9_all.json, the words of the segment-wise decode of `source`
(goldset/clip_asr_v10/<source>__v10.json) are aligned with difflib.SequenceMatcher to the long-form
Canary words whose midpoint lies inside the segment with a 0.3 s margin
(pipeline_outputs_extracted/outputs/canary_diar/<meeting>/transcript_canary_ts.json). A word is matched
when it aligns to a long-form word; the other words take the times of the nearest matched word, and in a
segment where nothing matched, every word falls back to times spread evenly over the segment.

    python evaltx_word_retiming.py [source]    -> goldset/evaltx_word_retiming_2026-10-29.json

`source` defaults to canary_diar; the output file name is the same for every source.
"""
import json, os, sys
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
import speaker_turns_v11 as T

source = sys.argv[1] if len(sys.argv) > 1 else "canary_diar"
segs_all = json.loads(T.SEGS.read_text(encoding="utf-8"))
hyp = json.loads((T.HYPS / f"{source}__v10.json").read_text(encoding="utf-8"))
tot = matched = neighbour = fallback = segs_fallback = 0
for mid in sorted({s["meeting_id"] for s in segs_all}):
    lf = json.loads((T.SRC / mid / "transcript_canary_ts.json").read_text(encoding="utf-8"))
    for sg in sorted((s for s in segs_all if s["meeting_id"] == mid), key=lambda x: x["start_s"]):
        h = hyp.get(sg["segment_id"])
        text = (h.get("text", "") if isinstance(h, dict) else h or "").strip()
        if not text:
            continue
        hw = text.split()
        ref = [w for w in lf if sg["start_s"] - 0.3 <= (w["start"] + w["end"]) / 2 <= sg["end_s"] + 0.3]
        sm = SequenceMatcher(None, [T._norm(w) for w in hw], [T._norm(w["word"]) for w in ref], autojunk=False)
        anchor = [None] * len(hw)
        for a, b, n in sm.get_matching_blocks():
            for k in range(n):
                anchor[a + k] = ref[b + k]
        m = sum(1 for x in anchor if x is not None)
        tot += len(hw)
        matched += m
        if m == 0:
            fallback += len(hw)
            segs_fallback += 1
        else:
            neighbour += len(hw) - m
print(source, "words", tot, "matched", matched, "%.1f%%" % (100 * matched / tot),
      "nearest neighbour", neighbour, "%.1f%%" % (100 * neighbour / tot),
      "segment fallback", fallback, "in", segs_fallback, "segments")
(ROOT / "goldset" / "evaltx_word_retiming_2026-10-29.json").write_text(json.dumps({
    "source": source, "rule": "speaker_turns_v11.v10_words", "words": tot, "matched": matched,
    "nearest_matched_neighbour": neighbour, "segment_fallback_words": fallback,
    "segment_fallback_segments": segs_fallback}, indent=1), encoding="utf-8")
print("wrote goldset/evaltx_word_retiming_2026-10-29.json")
