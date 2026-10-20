"""Builds speaker turns from words with diarized speaker labels, after a per-sentence speaker vote.

A speaker change starts a new turn, except that a run shorter than 1.2 s with at most two words
joins the previous run; once a turn is 25 s long, it closes at the next pause of at least 0.3 s,
and at the latest at 60 s. Run as a script, it gives the words of one system's segment-wise decode
the time and speaker of the matching long-form Canary words and writes one transcript per meeting.

    python speaker_turns_v11.py [system]

    system   name in goldset/clip_asr_v10/<system>__v10.json (default: canary_diar)
    reads    goldset/segments_v9_all.json, goldset/clip_asr_v10/<system>__v10.json,
             pipeline_outputs_extracted/outputs/canary_diar/<meeting>/transcript_canary_ts.json
    writes   outputs_v11_turns/<system>/<meeting>/transcript_speakers.txt
    (all paths relative to the folder of this file)
"""

from __future__ import annotations

import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

REPO = Path(__file__).resolve().parent
SRC = REPO / "pipeline_outputs_extracted" / "outputs" / "canary_diar"
OUT = REPO / "outputs_v11_turns"
MEETINGS = ["march09", "april13", "april27", "may11", "may18", "june01", "june15", "june22"]
MIN_RUN, TARGET, MAX_TURN, PAUSE = 1.2, 25.0, 60.0, 0.3
BACKCHANNEL_WORDS = 2  # a run under MIN_RUN s with at most this many words is a backchannel


def hhmmss(t: float) -> str:
    t = int(t)
    return f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}"


def runs(words):
    """Groups consecutive words of one speaker into [speaker, start, end, words] lists.
    A word without a speaker joins the previous run.
    """
    out = []
    for w in words:
        spk = w.get("speaker") or (out[-1][0] if out else "SPEAKER_00")
        if out and out[-1][0] == spk:
            out[-1][2] = w["end"]
            out[-1][3].append(w)
        else:
            out.append([spk, w["start"], w["end"], [w]])
    return out


def absorb_short(rs):
    """Folds runs shorter than MIN_RUN seconds with at most BACKCHANNEL_WORDS words into the
    preceding run and merges neighboring runs of the same speaker.
    """
    out = []
    for r in rs:
        if out and (r[2] - r[1]) < MIN_RUN and len(r[3]) <= BACKCHANNEL_WORDS:
            out[-1][2] = r[2]
            out[-1][3].extend(r[3])
        elif out and out[-1][0] == r[0]:
            out[-1][2] = r[2]
            out[-1][3].extend(r[3])
        else:
            out.append(r)
    return out


def split_long(r):
    """Splits one run into (speaker, start, end, text) turns: once a turn is TARGET seconds long,
    it closes at the next pause of at least PAUSE seconds, and at the latest at MAX_TURN seconds.
    """
    spk, ws = r[0], r[3]
    out, cur = [], [ws[0]]
    for prev, w in zip(ws, ws[1:]):
        dur = w["start"] - cur[0]["start"]
        if (dur >= TARGET and w["start"] - prev["end"] >= PAUSE) or dur >= MAX_TURN:
            out.append(cur)
            cur = []
        cur.append(w)
    out.append(cur)
    return [(spk, c[0]["start"], c[-1]["end"], " ".join(x["word"] for x in c)) for c in out]


HYPS = REPO / "goldset" / "clip_asr_v10"
SEGS = REPO / "goldset" / "segments_v9_all.json"


def _norm(w: str) -> str:
    return re.sub(r"[^a-zäöüß0-9]", "", w.lower())


def v10_words(mid: str, source: str) -> list[dict]:
    """Returns one meeting's words from the segment-wise decode of `source`, each with the start,
    end and speaker of the long-form Canary word it aligns to within its segment. Unmatched words
    copy the nearest matched word, or, if nothing in the segment matched, are spread evenly over
    it with the segment's speaker; the result passes through sentence_vote.
    """
    segs = [s for s in json.loads(SEGS.read_text(encoding="utf-8")) if s["meeting_id"] == mid]
    hyp = json.loads((HYPS / f"{source}__v10.json").read_text(encoding="utf-8"))
    lf = json.loads((SRC / mid / "transcript_canary_ts.json").read_text(encoding="utf-8"))
    out = []
    for sg in sorted(segs, key=lambda x: x["start_s"]):
        h = hyp.get(sg["segment_id"])
        text = (h.get("text", "") if isinstance(h, dict) else h or "").strip()
        if not text:
            continue
        hw = text.split()
        ref = [w for w in lf if sg["start_s"] - 0.3 <= (w["start"] + w["end"]) / 2 <= sg["end_s"] + 0.3]
        sm = SequenceMatcher(None, [_norm(w) for w in hw], [_norm(w["word"]) for w in ref],
                             autojunk=False)
        anchor = [None] * len(hw)
        for a, b, n in sm.get_matching_blocks():
            for k in range(n):
                anchor[a + k] = ref[b + k]
        for i, w in enumerate(hw):
            near = anchor[i] or next((anchor[j] for d in range(1, len(hw))
                                      for j in (i - d, i + d)
                                      if 0 <= j < len(hw) and anchor[j]), None)
            if near is None:
                t0 = sg["start_s"] + (sg["end_s"] - sg["start_s"]) * i / len(hw)
                near = {"start": t0, "end": t0, "speaker": sg["speaker"]}
            out.append({"word": w, "start": near["start"], "end": near["end"],
                        "speaker": near.get("speaker") or sg["speaker"]})
    return sentence_vote(out)


EDGE_WORDS = 5  # longest run of another speaker, in words, that sentence_vote relabels


def sentence_vote(words: list[dict]) -> list[dict]:
    """Gives all words of a sentence (up to a word that ends in '.', '!' or '?') the speaker of most
    of its words, in place. A sentence stays unchanged when that speaker has no strict majority,
    when a run of another speaker is longer than EDGE_WORDS words, or when the speaker with the
    most runs in the input would lose words. That speaker is protected because, without this rule,
    most wrong relabelings took words away from that speaker.
    """
    counts: dict[str, int] = {}
    for a, b in zip([None] + words, words):
        if a is None or a["speaker"] != b["speaker"]:
            counts[b["speaker"]] = counts.get(b["speaker"], 0) + 1
    chair = max(counts, key=counts.get) if counts else None
    sent, cur = [], []
    for w in words:
        cur.append(w)
        if re.search(r"[.!?]$", w["word"]):
            sent.append(cur)
            cur = []
    if cur:
        sent.append(cur)
    for s in sent:
        labs = [w["speaker"] for w in s]
        if len(set(labs)) < 2:
            continue
        keep = max(set(labs), key=labs.count)
        if labs.count(keep) * 2 <= len(labs):  # no strict majority
            continue
        # A run of another speaker longer than EDGE_WORDS words is treated as real speech.
        runs_, i = [], 0
        while i < len(labs):
            j = i
            while j + 1 < len(labs) and labs[j + 1] == labs[i]:
                j += 1
            runs_.append((labs[i], j - i + 1))
            i = j + 1
        if any(lab != keep and n > EDGE_WORDS for lab, n in runs_):
            continue
        if chair in labs and keep != chair:
            continue
        for w in s:
            w["speaker"] = keep
    return words


def build(mid: str, source: str = "canary_diar") -> list[tuple[str, float, float, str]]:
    turns = []
    for r in absorb_short(runs(v10_words(mid, source))):
        turns.extend(split_long(r))
    return turns


def main() -> int:
    print(f"{'meeting':9s}{'words':>7s}{'turns':>7s}{'mean s':>8s}{'max s':>7s}{'changes kept':>14s}")
    source = sys.argv[1] if len(sys.argv) > 1 else "canary_diar"
    for mid in MEETINGS:
        t = build(mid, source)
        d = OUT / source / mid
        d.mkdir(parents=True, exist_ok=True)
        (d / "transcript_speakers.txt").write_text(
            "\n".join(f"[{hhmmss(s)}] {spk}: {txt}" for spk, s, e, txt in t) + "\n",
            encoding="utf-8")
        durs = [e - s for _, s, e, _ in t]
        changes = sum(1 for a, b in zip(t, t[1:]) if a[0] != b[0])
        nw = sum(len(x[3].split()) for x in t)
        print(f"{mid:9s}{nw:7d}{len(t):7d}{sum(durs) / len(durs):8.1f}{max(durs):7.1f}{changes:14d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
