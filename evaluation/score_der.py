"""Estimates the diarization error rate of three systems from a sample of hand-labeled frames.

DER is a per-frame average, so a uniform random sample of frames estimates it without bias, with
a binomial (Wilson) confidence interval. The three components are scored as the standard defines
them, on the 2 s frames drawn by build_der_app.py:

    MISS          a human hears speech, the system says nobody
    FALSE ALARM   the system says somebody, the human hears silence
    CONFUSION     both hear speech, but the system names the wrong person

    DER = (miss + false alarm + confusion) / frames with reference speech

SYSTEMS (all scored on the same frames, with the same rule)
  pyannote raw   the diarization RTTM of the Canary run, cluster -> name by speaker_identity.json
  v10 shipped    the speaker of each ASR segment the protocols were written from (frames.json)
  v11 final      word-level speakers: v10 text on long-form word times, sentence rule,
                 verified short-turn speaker moves, recovered gaps
A frame is credited when the heard person is AMONG the speakers the system puts in it; a system
that puts two speakers into every frame would be flattered, so the mean number of names per
speech frame is printed next to each DER. v10 and v11 are also compared frame by frame with an
exact McNemar test.

LIMITS
* No collar. Standard DER usually forgives 250 ms around each boundary; a frame sample cannot
  apply a collar, so a frame straddling a real speaker change counts as an error here and would
  often be forgiven by pyannote's own scorer. This estimate is therefore pessimistic.
* Overlap is reported separately rather than folded in, because the pipeline assigns one speaker
  per segment and cannot win an overlapped frame by construction.
* "cannot tell" frames are excluded and counted, not guessed.

Writes goldset/der_result.json, which lists every error frame.

    python score_der.py dist/der_annotation/der_answers.json
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent
FRAMES = REPO / "dist" / "der_annotation" / "frames.json"
FRAME = 2.0
IDENT = json.loads((REPO / "goldset" / "speaker_identity.json").read_text(encoding="utf-8"))["mapping"]
RTTM = REPO / "pipeline_outputs_extracted" / "outputs" / "canary_diar"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson interval — correct for small n and for rates near 0, unlike the normal approximation."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _name(mid: str, lab: str) -> str:
    return IDENT.get(mid, {}).get(lab, lab)


def pyannote_hyp(frames: list[dict]) -> dict[str, list[str]]:
    out, cache = {}, {}
    for f in frames:
        mid, t = f["meeting"], f["t"]
        if mid not in cache:
            cache[mid] = [(float(p[3]), float(p[3]) + float(p[4]), p[7]) for p in
                          (l.split() for l in open(RTTM / mid / "diarization_hyp.rttm"))]
        out[f["id"]] = sorted({_name(mid, k) for s, e, k in cache[mid] if e > t and s < t + FRAME})
    return out


def v11_hyp(frames: list[dict]) -> dict[str, list[str]]:
    """Rebuild the v11 final speaker of every word, with times, and read off each frame."""
    from speaker_turns_v11 import v10_words
    moves = json.loads((REPO / "goldset" / "verify_short_turns_canary_diar.json")
                       .read_text(encoding="utf-8"))
    gaps = json.loads((REPO / "goldset" / "gap_recovered.json").read_text(encoding="utf-8"))
    out, words = {}, {}
    for f in frames:
        mid, t = f["meeting"], f["t"]
        if mid not in words:
            ws = v10_words(mid, "canary_diar")          # includes the sentence rule
            sent, cur = [], []
            for w in ws:
                cur.append(w)
                if re.search(r"[.?!]$", w["word"]):
                    sent.append(cur)
                    cur = []
            sent.append(cur)
            for mv in (m for m in moves if m["meeting"] == mid):
                for s in sent:                          # the verified sentence starts at mv["t"]
                    if s and abs(round(s[0]["start"], 1) - mv["t"]) < 0.051 and \
                            s[0]["speaker"] == mv["from"]:
                        for w in s:
                            w["speaker"] = mv["to"]
                        break
            words[mid] = ws
        hyp = {_name(mid, w["speaker"]) for w in words[mid]
               if w["end"] > t and w["start"] < t + FRAME}
        # recovered speech: the gap's diarized speakers present in this frame
        for g in (g for g in gaps if g["meeting"] == mid):
            if g["end_s"] > t and g["start_s"] < t + FRAME:
                hyp |= {_name(mid, k) for k, sec in g["speakers"].items() if sec >= 1}
        out[f["id"]] = sorted(hyp)
    return out


def score(ans: dict, frames: dict, hyps: dict[str, list[str]]) -> dict:
    n = miss = fa = conf = correct = overlap = unclear = names_in_speech = 0
    per_meeting: dict[str, Counter] = {}
    errors = []
    for fid, a in ans.items():
        f = frames.get(fid)
        if f is None:
            continue
        pm = per_meeting.setdefault(f["meeting"], Counter())
        hyp = hyps.get(fid, [])
        if a == "__unclear__":
            unclear += 1
            continue
        if a == "__multi__":
            overlap += 1
            continue
        ref_speech, hyp_speech = a != "__none__", bool(hyp)
        if not ref_speech and not hyp_speech:
            continue                        # agreed silence is not in the DER denominator
        if ref_speech:
            n += 1
            pm["ref_speech"] += 1
            names_in_speech += len(hyp)
        if ref_speech and not hyp_speech:
            miss += 1
            pm["miss"] += 1
            errors.append((fid, "miss", a, hyp))
        elif hyp_speech and not ref_speech:
            fa += 1
            pm["fa"] += 1
            errors.append((fid, "false alarm", a, hyp))
        elif a in hyp:
            correct += 1
        else:
            conf += 1
            pm["conf"] += 1
            errors.append((fid, "confusion", a, hyp))
    lo, hi = wilson(miss + fa + conf, n)
    return {"ref_speech_frames": n, "miss": miss, "false_alarm": fa, "confusion": conf,
            "correct": correct, "overlap_excluded": overlap, "unclear_excluded": unclear,
            "der": (miss + fa + conf) / n if n else None, "ci95": [lo, hi],
            "names_per_speech_frame": names_in_speech / n if n else None,
            "per_meeting": {k: dict(v) for k, v in per_meeting.items()}, "errors": errors}


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__.strip().splitlines()[-1])
        return 1
    ans = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    flist = json.loads(FRAMES.read_text(encoding="utf-8"))
    frames = {f["id"]: f for f in flist}
    systems = {
        "pyannote raw": pyannote_hyp(flist),
        "v10 shipped": {f["id"]: f["hyp"] for f in flist},
        "v11 final": v11_hyp(flist),
    }
    res = {k: score(ans, frames, h) for k, h in systems.items()}
    if not res["v10 shipped"]["ref_speech_frames"]:
        print("no scoreable frames yet — answer some clips and export again")
        return 1

    r0 = res["v10 shipped"]
    print("DIARIZATION ERROR RATE — estimated from a uniform random frame sample")
    print("=" * 92)
    print(f"answered frames: {len(ans)}   excluded 'cannot tell': {r0['unclear_excluded']}   "
          f"excluded 'two or more at once': {r0['overlap_excluded']}")
    print(f"\n{'system':14}{'ref frames':>11}{'miss':>6}{'FA':>5}{'conf':>6}{'DER':>8}"
          f"{'95% CI':>18}{'names/frame':>13}")
    print("-" * 92)
    for k, r in res.items():
        print(f"{k:14}{r['ref_speech_frames']:>11}{r['miss']:>6}{r['false_alarm']:>5}"
              f"{r['confusion']:>6}{r['der']:>8.1%}   [{r['ci95'][0]:.1%}, {r['ci95'][1]:.1%}]"
              f"{r['names_per_speech_frame']:>11.2f}")

    # paired comparison v10 -> v11 on the same frames (exact McNemar on 'frame right')
    from scipy.stats import binomtest
    bad10 = {e[0] for e in res["v10 shipped"]["errors"]}
    bad11 = {e[0] for e in res["v11 final"]["errors"]}
    b, c = len(bad10 - bad11), len(bad11 - bad10)
    p = binomtest(b, b + c).pvalue if b + c else 1.0
    print(f"\npaired v10 -> v11: fixed {b} frames, broke {c}, exact McNemar p = {p:.3f}")

    print("\n  No collar is applied, so a frame straddling a real speaker change counts as an")
    print("  error here that pyannote's own scorer would usually forgive. Treat this as a")
    print("  PESSIMISTIC estimate. Overlapped frames are excluded, because the pipeline")
    print("  assigns one speaker per segment and cannot win them by construction.")

    out = REPO / "goldset" / "der_result.json"
    out.write_text(json.dumps(
        {"frames_answered": len(ans), "answers_file": str(Path(sys.argv[1])),
         "systems": res, "paired_v10_v11": {"fixed": b, "broke": c, "p": p},
         "collar": None, "note": "uniform random frame sample, no collar, pessimistic"},
        indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote -> {out.relative_to(REPO)}   (every error frame is listed there for reading)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
