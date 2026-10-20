"""Join the segment texts into one speaker transcript per meeting and source.

The text comes from goldset/segments_v9_all.json itself (source "segments") or from a
recognizer file goldset/clip_asr_v10/<source>__v10.json, without the clips that
goldset/segment_loop_gate.json marks as degenerate (--keep-degenerate keeps them). Consecutive segments of one speaker
become one turn, and each meeting gets <out>/<source>/<meeting>/transcript_speakers.txt
("[HH:MM:SS] SPEAKER_NN: text" per turn) and transcript.txt (the same words, no labels).

Usage:
    python assemble_v9_transcripts.py                        # source "segments"
    python assemble_v9_transcripts.py --source canary_diar   # one recognizer
    python assemble_v9_transcripts.py --source all           # "segments" and every recognizer
    python assemble_v9_transcripts.py --source all --out DIR # default: outputs_v10 in this folder
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent
G = REPO / "goldset"
SEGMENTS = G / "segments_v9_all.json"
HYPS = G / "clip_asr_v10"
OUT = REPO / "outputs_v10"

# Word counts of the long-form Canary decode per meeting,
# used for the "long-form" and "delta" columns.
JULY_CANARY = {"march09": 8343, "april13": 6122, "april27": 6581, "may11": 9782,
               "may18": 8725, "june01": 8228, "june15": 8666, "june22": 11205}


def hhmmss(t: float) -> str:
    t = int(t)
    return f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}"


def degenerate_ids(source: str) -> dict[str, str]:
    """Return {segment_id: verdict} for the clips of this source that
    goldset/segment_loop_gate.json marks as degenerate, or {} if that file is missing.
    """
    p = REPO / "goldset" / "segment_loop_gate.json"
    if not p.exists():
        return {}
    gate = json.loads(p.read_text(encoding="utf-8"))
    return {sid: info["verdict"]
            for sid, info in gate.get(source, {}).get("degenerate_ids", {}).items()}


def text_for(segs: list[dict], source: str) -> tuple[dict[str, str], str]:
    """Return {segment_id: text} for one source and a one-line note on where the text
    came from. Degenerate clips get an empty text unless --keep-degenerate is given.
    """
    if source == "segments":
        return ({s["segment_id"]: s.get("text", "") for s in segs},
                "segments_v9_all.json — the long-form decode re-sliced at the v9 boundaries")
    f = HYPS / f"{source}__v10.json"
    if not f.exists():
        sys.exit(f"no hypotheses for {source!r}: {f} does not exist.\n"
                 f"Run the cluster notebook first, or use --source segments for the baseline.")
    h = json.loads(f.read_text(encoding="utf-8"))
    text = {k: (v.get("text", "") if isinstance(v, dict) else v) for k, v in h.items()}
    note = f"{f.name} — each segment decoded as its own clip"
    if "--keep-degenerate" not in sys.argv:
        bad = degenerate_ids(source)
        for sid in bad:
            text[sid] = ""
        if bad:
            note += (f"; {len(bad)} degenerate clip(s) EXCLUDED "
                     f"({', '.join(sorted(bad))})")
    return text, note


def turns(segs: list[dict], text: dict[str, str]) -> list[tuple[float, str, str]]:
    """Merge consecutive same-speaker segments into (start_s, speaker, text) turns."""
    out: list[tuple[float, str, list[str]]] = []
    for s in sorted(segs, key=lambda x: x["start_s"]):
        t = (text.get(s["segment_id"]) or "").strip()
        if not t:
            continue
        spk = s.get("speaker") or "SPEAKER_00"
        if out and out[-1][1] == spk:
            out[-1][2].append(t)
        else:
            out.append((s["start_s"], spk, [t]))
    return [(a, b, " ".join(c)) for a, b, c in out]


def write_meeting(dst: pathlib.Path, tn: list[tuple[float, str, str]]) -> tuple[int, int]:
    dst.mkdir(parents=True, exist_ok=True)
    speakers = "\n".join(f"[{hhmmss(t)}] {spk}: {txt}" for t, spk, txt in tn)
    plain = " ".join(txt for _, _, txt in tn)
    (dst / "transcript_speakers.txt").write_text(speakers + "\n", encoding="utf-8")
    (dst / "transcript.txt").write_text(plain + "\n", encoding="utf-8")
    return len(tn), len(plain.split())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="segments",
                    help="'segments' (baseline), a system name, or 'all'")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--keep-degenerate", action="store_true",
                    help="keep the text of clips flagged as degenerate (default: emptied)")
    a = ap.parse_args()

    segs = json.loads(SEGMENTS.read_text(encoding="utf-8"))
    by_meeting: dict[str, list[dict]] = collections.defaultdict(list)
    for s in segs:
        by_meeting[s["meeting_id"]].append(s)

    if a.source == "all":
        sources = ["segments"] + sorted(f.name.split("__")[0] for f in HYPS.glob("*__v10.json")) \
            if HYPS.exists() else ["segments"]
    else:
        sources = [a.source]

    root = pathlib.Path(a.out)
    for source in sources:
        text, whence = text_for(segs, source)
        print(f"\n=== {source} ===\n{whence}\n")
        print(f"  {'meeting':<10}{'turns':>7}{'segs':>7}{'words':>8}"
              f"{'long-form':>11}{'delta':>8}")
        tot_w = tot_s = 0
        for m in sorted(by_meeting):
            tn = turns(by_meeting[m], text)
            n_turns, n_words = write_meeting(root / source / m, tn)
            n_segs = sum(1 for s in by_meeting[m] if (text.get(s["segment_id"]) or "").strip())
            lf = JULY_CANARY.get(m, 0)
            d = f"{100 * (n_words / lf - 1):+.1f}%" if lf else "—"
            print(f"  {m:<10}{n_turns:>7}{n_segs:>7}{n_words:>8}{lf:>11}{d:>8}")
            tot_w += n_words
            tot_s += n_segs
        print(f"  {'TOTAL':<10}{'':>7}{tot_s:>7}{tot_w:>8}{sum(JULY_CANARY.values()):>11}")

        missing = sum(1 for s in segs if not (text.get(s["segment_id"]) or "").strip())
        if missing:
            print(f"\n  {missing} of {len(segs)} segments have no text under this source "
                  f"and were skipped")
        print(f"  wrote {root / source}/<meeting>/transcript{{,_speakers}}.txt")

    return 0


if __name__ == "__main__":
    sys.exit(main())
