"""Drift-correct Canary word times and cut the gold-set candidate clips.

drift_map() maps the Canary times of one meeting to corrected times, using words matched
with the whole-meeting Whisper transcript. Run as a script, the file cuts the clips listed
in goldset/goldset_candidate.json from their recordings, each shifted by up to 1.5 s to the
position with the most speech, and writes:
    goldset/clips/<segment_id>.wav   16 kHz mono clips
    goldset/clips_manifest.json      one entry per clip, with a QC flag
    goldset/clips_drift_report.json  the drift map summary per meeting

Usage:
    python build_clip_dataset.py
"""

from __future__ import annotations

import difflib
import json
import re
import statistics
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from thesis_pipeline.goldset_local import parse_timestamped, tokenize

REPO = Path(__file__).resolve().parent
OUT = REPO / "pipeline_outputs_extracted" / "outputs"
GD = REPO / "goldset"
CLIPS = GD / "clips"
SR = 16000
PAD = 0.25  # seconds added before and after each clip
MIN_SPEECH = 0.35  # minimum speech fraction for qc_ok


def _norm(w):
    return re.sub(r"[^0-9a-zA-ZäöüÄÖÜß]", "", w).lower()


def drift_map(meeting):
    """Return (f, info): f maps a Canary time in seconds to the drift-corrected time.

    f is the identity when the Whisper transcript is missing or too few words match.
    """
    cw = json.loads((OUT / "canary_diar" / meeting / "transcript_canary_ts.json").read_text(encoding="utf-8"))
    wsegs = parse_timestamped(OUT / "whisper" / meeting / "transcript_no_prompt_timestamped.txt")
    if not cw or not wsegs:
        return (lambda t: t), {"anchors": 0, "note": "no data - identity map"}

    wwords = []
    for s in wsegs:
        toks = [t for _, t in tokenize(s.text)]
        for k, t in enumerate(toks):
            wwords.append((t, s.t0 + (k + 0.5) / max(len(toks), 1) * (s.t1 - s.t0)))
    cwords = [(_norm(w["word"]), (float(w["start"]) + float(w["end"])) / 2) for w in cw]

    sm = difflib.SequenceMatcher(a=[w for w, _ in cwords], b=[w for w, _ in wwords], autojunk=False)
    anchors = []
    for a, b, size in sm.get_matching_blocks():
        for k in range(size):
            ct, wt = cwords[a + k][1], wwords[b + k][1]
            if abs(ct - wt) < 15:  # max anchor offset, seconds
                anchors.append((ct, wt - ct))  # (Canary time, correction in seconds)
    if len(anchors) < 40:
        return (lambda t: t), {"anchors": len(anchors), "note": "too few anchors - identity map"}

    anchors.sort()
    # Median correction per time bin (4 to 40 bins of about 120 s, at least 5 anchors each).
    # np.interp is linear between bin centers and constant beyond the outer ones.
    span = anchors[-1][0] - anchors[0][0]
    nbins = max(4, min(40, int(span // 120)))
    edges = np.linspace(anchors[0][0], anchors[-1][0], nbins + 1)
    xs, ys = [], []
    for i in range(nbins):
        chunk = [d for t, d in anchors if edges[i] <= t <= edges[i + 1]]
        if len(chunk) >= 5:
            xs.append((edges[i] + edges[i + 1]) / 2)
            ys.append(statistics.median(chunk))
    if len(xs) < 2:
        return (lambda t: t), {"anchors": len(anchors), "note": "flat - identity map"}
    xs, ys = np.array(xs), np.array(ys)

    def f(t):
        return float(t + np.interp(t, xs, ys))

    return f, {"anchors": len(anchors), "bins": len(xs),
               "correction_start_s": round(float(ys[0]), 2),
               "correction_end_s": round(float(ys[-1]), 2),
               "max_abs_correction_s": round(float(np.abs(ys).max()), 2)}


def read_region(path, t0, t1):
    """Read seconds t0 to t1 of an audio file as 16 kHz mono float32."""
    info = sf.info(str(path))
    s0 = max(0, int(t0 * info.samplerate))
    n = int((t1 - t0) * info.samplerate)
    n = min(n, info.frames - s0)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    y, _ = sf.read(str(path), start=s0, frames=n, dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if info.samplerate != SR:
        y = resample_poly(y, SR, info.samplerate).astype(np.float32)
    return y


def speech_frames(y, win=0.02):
    n = int(SR * win)
    m = len(y) // n
    if m < 1:
        return np.zeros(0, dtype=bool)
    e = np.sqrt((y[:m * n].reshape(m, n) ** 2).mean(1)) + 1e-9
    return e > max(np.percentile(e, 30) * 3, e.max() * 0.02)


def speech_ratio(y):
    m = speech_frames(y)
    return float(m.mean()) if len(m) else 0.0


def main():
    gold = json.loads((GD / "goldset_candidate.json").read_text(encoding="utf-8"))
    CLIPS.mkdir(exist_ok=True)
    by_meeting = {}
    for c in gold:
        by_meeting.setdefault(c["meeting_id"], []).append(c)

    manifest, drift_report = [], {}
    for meeting, clips in sorted(by_meeting.items()):
        src = REPO / clips[0]["audio_file"]
        if not src.exists():
            print(f"{meeting}: SOURCE MISSING {src.name}")
            continue
        f, info = drift_map(meeting)
        drift_report[meeting] = info
        print(f"{meeting:9} drift map: {info}", flush=True)

        for c in clips:
            t0, t1 = f(c["start_s"]), f(c["end_s"])
            # Try shifts of -1.5 to +1.5 s in 0.25 s steps and keep the first one
            # with the highest speech ratio.
            lo, hi = max(0.0, t0 - 1.5 - PAD), t1 + 1.5 + PAD
            region = read_region(src, lo, hi)
            best_d, best_sc = 0.0, -1.0
            for d in np.arange(-1.5, 1.51, 0.25):
                a = int((t0 - PAD + d - lo) * SR)
                b = int((t1 + PAD + d - lo) * SR)
                if a < 0 or b > len(region) or b <= a:
                    continue
                sc = speech_ratio(region[a:b])
                if sc > best_sc:
                    best_sc, best_d = sc, float(d)
            a = int((t0 - PAD + best_d - lo) * SR)
            b = int((t1 + PAD + best_d - lo) * SR)
            clip = region[max(0, a):max(0, b)]
            if len(clip) < SR:  # under 1 s: use the whole search region
                clip = region
            peak = float(np.abs(clip).max()) if len(clip) else 0.0
            if peak > 0:
                clip = (clip / peak * 0.89).astype(np.float32)  # peak-normalize to 0.89
            path = CLIPS / f"{c['segment_id']}.wav"
            sf.write(str(path), clip, SR, subtype="PCM_16")

            sr_ = speech_ratio(clip)
            manifest.append({
                "segment_id": c["segment_id"], "meeting_id": meeting,
                "file": f"clips/{c['segment_id']}.wav",
                "orig_start_s": c["start_s"], "orig_end_s": c["end_s"],
                "corrected_start_s": round(t0 - PAD + best_d, 2),
                "drift_correction_s": round(t0 - c["start_s"], 2),
                "micro_align_s": best_d,
                "duration_s": round(len(clip) / SR, 2),
                "speech_ratio": round(sr_, 3), "peak": round(peak, 4),
                "qc_ok": bool(sr_ >= MIN_SPEECH and peak > 0.01 and len(clip) > 3 * SR),
            })

    (GD / "clips_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (GD / "clips_drift_report.json").write_text(json.dumps(drift_report, ensure_ascii=False, indent=2), encoding="utf-8")

    ok = sum(1 for m in manifest if m["qc_ok"])
    tot_mb = sum((CLIPS / f"{m['segment_id']}.wav").stat().st_size for m in manifest) / 1e6
    print(f"\nwrote {len(manifest)} clips ({tot_mb:.0f} MB) -> {CLIPS}")
    print(f"QC passed: {ok}/{len(manifest)}   (speech>={MIN_SPEECH:.0%}, audible, >3s)")
    bad = [m for m in manifest if not m["qc_ok"]]
    if bad:
        print("flagged clips:")
        for m in bad[:15]:
            print(f"   {m['segment_id']:16} speech={m['speech_ratio']:.2f} peak={m['peak']:.3f} dur={m['duration_s']}")
    corr = [abs(m["drift_correction_s"]) for m in manifest]
    print(f"drift correction applied: median {statistics.median(corr):.2f}s  max {max(corr):.2f}s")


if __name__ == "__main__":
    main()
