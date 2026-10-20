"""Three signals that show where meeting audio can be cut between two words: a quiet run
in the waveform, the pyannote speaker turns, and the gap between the recognizer's word
timestamps. score_gap() turns them into zero to three votes for each gap. With --calibrate,
the votes at the end of each gold-set clip are compared with the support score of the
clip's last word.

    python boundary_signals.py --calibrate [--meetings march09,april13,april27]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

import boundary_lib as B

HOT = 0.25  # speech gate as a fraction of the meeting's p75 frame RMS
SIL_VOTE = 0.20  # seconds; a quiet run or turn gap this long is a vote
TURN_TOL = 0.25  # seconds; distance to a turn start or end that counts as an edge
WORD_VOTE = 0.16  # seconds; two 80 ms encoder frames, the smallest word gap that is a vote
LOOK_OUT = 0.80  # seconds after the word end that are searched for cut_at


def turns(meeting: str) -> list[tuple[float, float]]:
    """Return the sorted (start, end) seconds of every turn in the meeting's RTTM file."""
    p = (B.OUT / "canary_diar" / meeting / "diarization_hyp.rttm")
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        f = line.split()
        if f and f[0] == "SPEAKER":
            out.append((float(f[3]), float(f[3]) + float(f[4])))
    return sorted(out)


def quiet_run(env: np.ndarray, t0: float, t1: float, gate: float) -> tuple[float, float]:
    """Return (length, start) in seconds of the longest run of frames at or below `gate`
    inside [t0, t1].
    """
    i0, i1 = max(0, int(t0 / B.HOP)), min(len(env), int(t1 / B.HOP) + 1)
    if i1 <= i0:
        return 0.0, t0
    quiet = env[i0:i1] <= gate
    best = run = beststart = 0
    for k, q in enumerate(quiet):
        run = run + 1 if q else 0
        if run > best:
            best, beststart = run, k - run + 1
    return best * B.HOP, (i0 + beststart) * B.HOP


def turn_gap(tl: list[tuple[float, float]], t0: float, t1: float) -> float:
    """Return the length of [t0, t1] minus its overlap with each turn in `tl` (sorted by
    start), not below zero. Turns that overlap each other are not merged first.
    """
    if t1 <= t0:
        return 0.0
    covered = 0.0
    for a, b in tl:
        if b <= t0:
            continue
        if a >= t1:
            break
        covered += min(b, t1) - max(a, t0)
    return max(0.0, (t1 - t0) - covered)


def near_turn_edge(tl: list[tuple[float, float]], t: float, tol: float = TURN_TOL) -> bool:
    """True if a turn in `tl` (sorted by start) starts or ends within `tol` seconds of t."""
    for a, b in tl:
        if a - tol > t:
            break
        if abs(a - t) <= tol or abs(b - t) <= tol:
            return True
    return False


def first_quiet(env: np.ndarray, t0: float, gate: float, need: float, horizon: float
                ) -> float | None:
    """Return the start in seconds of the first run of at least `need` seconds at or below
    `gate` between t0 and t0 + horizon, or None.
    """
    i0 = max(0, int(t0 / B.HOP))
    i1 = min(len(env), int((t0 + horizon) / B.HOP) + 1)
    nneed = max(1, int(need / B.HOP))
    run = 0
    for k in range(i0, i1):
        if env[k] <= gate:
            run += 1
            if run >= nneed:
                return (k - run + 1) * B.HOP
        else:
            run = 0
    return None


def score_gap(ws, k, env, gate, tl) -> dict:
    """Return how many of the three signals (silence, diarization, word gap) vote for cutting
    after word k, the measured values, and `cut_at`, the start of the first quiet run after
    the word (or None). The last word of `ws` always gets three votes.
    """
    if k + 1 >= len(ws):
        return {"votes": 3, "sil": 9.9, "sil_start": ws[k]["end"], "cut_at": None,
                "turn": 9.9, "wgap": 9.9, "edge": True}
    t0, t1 = ws[k]["end"], ws[k + 1]["start"]
    lo, hi = min(t0, t1) - 0.10, max(t0, t1) + 0.10
    sil, sil_start = quiet_run(env, lo, hi, gate)
    tg = turn_gap(tl, lo, hi)
    edge = near_turn_edge(tl, t0)
    wgap = max(0.0, t1 - t0)
    votes = int(sil >= SIL_VOTE) + int(tg >= SIL_VOTE or edge) + int(wgap >= WORD_VOTE)
    cut = first_quiet(env, t0, gate, SIL_VOTE, LOOK_OUT)
    return {"votes": votes, "sil": round(sil, 3), "sil_start": round(sil_start, 3),
            "cut_at": (round(cut, 3) if cut is not None else None),
            "turn": round(tg, 3), "wgap": round(wgap, 3), "edge": bool(edge)}


def calibrate(meetings: list[str]) -> None:
    """Compare the end-boundary votes of the gold-set clips with the support score of each
    clip's last word, print the tables and write goldset/boundary_signals.json. The gold-set
    files are read and written relative to the working directory.
    """
    import statistics as st
    align = json.loads(pathlib.Path("goldset/clip_align_v7.json").read_text(encoding="utf-8"))
    recut = {s["segment_id"]: s for s in
             json.loads(pathlib.Path("goldset/goldset_v7_recut.json").read_text("utf-8"))}
    import rater_notes as PN
    label = {r["segment_id"]: r["boundary_label"] for r in PN.resolve()}

    rows = []
    for m in meetings:
        ws, env, tl = B.words(m), B.envelope(m), turns(m)
        gate = HOT * B.levels(env)[0]
        for sid, seg in recut.items():
            if seg["meeting_id"] != m or sid not in align:
                continue
            a = align[sid]
            if "error" in a or not a.get("last"):
                continue
            s = score_gap(ws, seg["word_j"], env, gate, tl)
            s.update(sid=sid, meeting=m, tail_support=a["last"]["support"],
                     trail_gap=round(a["n_frames"] * a["sec_per_frame"] - a["last"]["end_s"], 3),
                     label=label.get(sid, "-"))
            rows.append(s)

    print(f"{len(rows)} gold-set segments in {', '.join(meetings)}\n")
    print("  %-6s %5s %9s %9s %9s %9s" % ("votes", "n", "med sup", "sup<-1.5", "med sil",
                                          "med gap"))
    for v in (0, 1, 2, 3):
        g = [r for r in rows if r["votes"] == v]
        if not g:
            continue
        print("  %-6d %5d %9.2f %8.0f%% %9.3f %9.3f"
              % (v, len(g), st.median([r["tail_support"] for r in g]),
                 100 * sum(r["tail_support"] < -1.5 for r in g) / len(g),
                 st.median([r["sil"] for r in g]), st.median([r["trail_gap"] for r in g])))

    print("\n  each signal on its own, as a predictor of a destroyed last word "
          "(tail_support < -1.5):")
    print("  %-34s %6s %6s %7s" % ("rule", "flags", "of bad", "precision"))
    bad = [r for r in rows if r["tail_support"] < -1.5]
    for name, f in [("silence >= %.2f s" % SIL_VOTE, lambda r: r["sil"] < SIL_VOTE),
                    ("diarization says gap/edge", lambda r: not (r["turn"] >= SIL_VOTE
                                                                 or r["edge"])),
                    ("word gap >= %.2f s" % WORD_VOTE, lambda r: r["wgap"] < WORD_VOTE),
                    ("fewer than 2 of 3 votes", lambda r: r["votes"] < 2),
                    ("fewer than 3 of 3 votes", lambda r: r["votes"] < 3)]:
        hit = [r for r in rows if f(r)]
        tp = [r for r in hit if r["tail_support"] < -1.5]
        print("  %-34s %6d %6d %6.0f%%"
              % ("NOT " + name, len(hit), len(tp),
                 100 * len(tp) / len(hit) if hit else 0))
    print("  (%d of %d segments have a destroyed last word by that threshold)"
          % (len(bad), len(rows)))

    named = [r for r in rows if r["label"] not in ("-", "CLEAN")]
    if named:
        print("\n  the segments the rater labeled, in these meetings:")
        print("  %-20s %-15s %6s %7s %7s %7s" % ("segment", "rater label", "votes", "sil",
                                                 "turn", "wgap"))
        for r in sorted(named, key=lambda r: r["votes"]):
            print("  %-20s %-15s %6d %7.3f %7.3f %7.3f"
                  % (r["sid"], r["label"], r["votes"], r["sil"], r["turn"], r["wgap"]))
    pathlib.Path("goldset/boundary_signals.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nwrote goldset/boundary_signals.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--meetings", default="march09,april13,april27")
    a = ap.parse_args()
    if a.calibrate:
        calibrate(a.meetings.split(","))
    else:
        ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
