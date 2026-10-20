"""Finds clips whose decoded text in goldset/clip_asr_v10/<system>__v10.json (relative to
this folder) is empty or stuck in a loop. A clip with 10 or more words is a token loop when
one word makes up at least 30% of its words, and a phrase loop when one 6-word sequence
occurs at least 3 times and 6 times that count is at least 25% of its words. Prints counts
and every looping clip (empty clips are only counted); --apply also writes
goldset/segment_loop_gate.json, which assemble_v9_transcripts.py reads to leave the looping
clips out.

    python segment_loop_gate.py            # report only
    python segment_loop_gate.py --apply    # also write goldset/segment_loop_gate.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent
CLIPS = REPO / "goldset" / "clip_asr_v10"
SYSTEMS = ["canary_diar", "cohere", "granite41",
           "whisper_large_v3", "whisper_turbo", "whisper_turbo_de"]

TOKEN_LOOP_FRAC = 0.30  # one word at this share or more: token loop
PHRASE_N = 6
PHRASE_MIN_REPEATS = 3
PHRASE_LOOP_FRAC = 0.25
MIN_WORDS = 10  # non-empty clips with fewer words count as clean


def classify(text: str) -> tuple[str, str]:
    """Return (label, reason) for one clip; label is clean, empty, token_loop or phrase_loop."""
    w = text.split()
    if not w:
        return "empty", "no text"
    if len(w) < MIN_WORDS:
        return "clean", ""
    top, n = Counter(w).most_common(1)[0]
    if n / len(w) >= TOKEN_LOOP_FRAC:
        return "token_loop", f"{top!r} x{n} of {len(w)}w ({n/len(w):.0%})"
    grams = [" ".join(w[i:i + PHRASE_N]) for i in range(len(w) - PHRASE_N + 1)]
    if grams:
        g, gn = Counter(grams).most_common(1)[0]
        if gn >= PHRASE_MIN_REPEATS and (gn * PHRASE_N) / len(w) >= PHRASE_LOOP_FRAC:
            return "phrase_loop", f"6-gram x{gn} ({gn*PHRASE_N/len(w):.0%}): {g[:48]!r}"
    return "clean", ""


def main() -> int:
    apply = "--apply" in sys.argv
    report: dict[str, dict] = {}

    print("SEGMENT LOOP GATE — degenerate clips in the v10 segment-wise decodes")
    print("=" * 94)
    print("Unit is the CLIP. One bad window no longer condemns a whole meeting.\n")
    print(f"{'system':18s}{'clips':>7}{'clean':>7}{'empty':>7}{'tokenLp':>9}{'phraseLp':>10}"
          f"{'degen%':>8}")

    worst = []
    for sysname in SYSTEMS:
        p = CLIPS / f"{sysname}__v10.json"
        if not p.exists():
            print(f"{sysname:18s}  (no clip decode file)")
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        counts = Counter()
        per_meeting: dict[str, Counter] = {}
        bad: dict[str, dict] = {}
        for sid, text in d.items():
            v, why = classify(text if isinstance(text, str) else (text or {}).get("text", ""))
            counts[v] += 1
            mid = sid.split("_")[0]
            per_meeting.setdefault(mid, Counter())[v] += 1
            if v in ("token_loop", "phrase_loop"):
                bad[sid] = {"verdict": v, "why": why}
                worst.append((sysname, sid, v, why))
        n = sum(counts.values())
        degen = counts["token_loop"] + counts["phrase_loop"]
        print(f"{sysname:18s}{n:>7}{counts['clean']:>7}{counts['empty']:>7}"
              f"{counts['token_loop']:>9}{counts['phrase_loop']:>10}{degen/n:>7.1%}")
        report[sysname] = {
            "n_clips": n, "counts": dict(counts),
            "degenerate_ids": bad,
            "per_meeting": {m: dict(c) for m, c in per_meeting.items()},
        }

    print(f"\nDEGENERATE CLIPS IN FULL ({len(worst)}):\n")
    for sysname, sid, v, why in worst:
        print(f"   {sysname:18s}{sid:22s}{v:12s}{why}")

    print("\nMEETINGS THE OLD MEETING-LEVEL GATE CONDEMNED, RE-JUDGED PER CLIP:\n")
    for sysname, info in report.items():
        for mid, c in info["per_meeting"].items():
            tot = sum(c.values())
            deg = c.get("token_loop", 0) + c.get("phrase_loop", 0)
            if deg:
                print(f"   {sysname:18s}{mid:10s}{deg} of {tot} clips degenerate "
                      f"({deg/tot:.1%}) -> exclude {deg}, keep {tot-deg}")

    if apply:
        out = REPO / "goldset" / "segment_loop_gate.json"
        out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nWROTE {out.relative_to(REPO)}")
    else:
        print("\n(report only — re-run with --apply)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
