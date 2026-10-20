"""Speaker verification of the short sentences at the edges of speaker turns, with WavLM x-vectors.

An edge sentence of MIN_S to MAX_S seconds moves to the closest speaker centroid if that speaker
is the meeting lead (the label with the most turns) or the current label has no centroid, and the
similarity beats the current label by at least MARGIN. As a script, it reads the recordings named
in AUDIO from this folder and writes outputs_v11_verified/<system>/<meeting>/transcript_speakers.txt
and goldset/verify_short_turns_<system>.json for the eight meetings.

    python verify_short_turns.py [system]    (default system: canary_diar)
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from speaker_turns_v11 import (MEETINGS, absorb_short, hhmmss, runs, split_long, v10_words)

REPO = Path(__file__).resolve().parent
OUT = REPO / "outputs_v11_verified"
AUDIO = {
    "march09": "Team-Meeting_2026-03-09.mp3", "april13": "Team-Meeting_2026-04-13.mp3",
    "april27": "Team-Meeting_2026-04-27.flac", "may11": "Team-Meeting_2026-05-11.flac",
    "may18": "Team-Meeting_2026-05-18.flac", "june01": "Team-Meeting_2026-06-01.flac",
    "june15": "Team-Meeting_2026-06-15.flac", "june22": "Team-Meeting_2026-06-22.flac",
}
MARGIN = 0.05  # cosine-similarity margin over the current label
MIN_S, MAX_S, LONG_S = 0.8, 6.0, 8.0


def load_audio(mid: str) -> np.ndarray:
    y, sr = sf.read(str(REPO / AUDIO[mid]), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if sr != 16000:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(16000, sr)
        y = resample_poly(y, 16000 // g, sr // g).astype(np.float32)
    return y


class Embedder:
    def __init__(self):
        from transformers import AutoFeatureExtractor, WavLMForXVector
        name = "microsoft/wavlm-base-plus-sv"
        self.fe = AutoFeatureExtractor.from_pretrained(name)
        self.m = WavLMForXVector.from_pretrained(name).eval()

    @torch.no_grad()
    def __call__(self, clip: np.ndarray) -> np.ndarray:
        x = self.fe(clip, sampling_rate=16000, return_tensors="pt")
        e = self.m(**x).embeddings[0].numpy()
        return e / (np.linalg.norm(e) + 1e-9)


def sentences(words):
    """Split a list of word dicts into sentences at words ending in '.', '?' or '!'."""
    out, cur = [], []
    for w in words:
        cur.append(w)
        if re.search(r"[.?!]$", w["word"]):
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def verify(mid: str, source: str, emb: Embedder, log: list) -> list[dict]:
    words = v10_words(mid, source)
    y = load_audio(mid)
    clip = lambda s, e: y[int(max(0, s) * 16000):int(e * 16000)]
    rs = absorb_short(runs(words))
    # Speaker centroids from long runs
    vecs = defaultdict(list)
    for spk, s, e, ws in rs:
        if e - s >= LONG_S and len(vecs[spk]) < 25:
            mid_t = (s + e) / 2
            vecs[spk].append(emb(clip(mid_t - 4, mid_t + 4)))
    cent = {k: np.mean(v, axis=0) / np.linalg.norm(np.mean(v, axis=0)) for k, v in vecs.items() if v}
    turns_per = defaultdict(int)
    for r in rs:
        turns_per[r[0]] += 1
    chair = max(turns_per, key=turns_per.get)
    # Check each run's edge sentences
    for spk, s, e, ws in rs:
        sents = sentences(ws)
        edge = sents[:2] + (sents[-2:] if len(sents) > 2 else [])
        for sw in edge:
            a, b = sw[0]["start"], sw[-1]["end"]
            if not (MIN_S <= b - a <= MAX_S):
                continue
            v = emb(clip(a - 0.1, b + 0.1))
            sims = {k: float(v @ c) for k, c in cent.items()}
            best = max(sims, key=sims.get)
            sliver = spk not in cent  # no run of LONG_S or more, so no centroid
            own = sims.get(spk, -1.0)
            # Move only toward the meeting lead or out of a label without a centroid: short
            # sentences of the meeting lead can carry the label of the long turn next to them.
            if best != spk and (sliver or best == chair) and sims[best] - own >= MARGIN:
                for w in sw:
                    w["speaker"] = best
                log.append({"meeting": mid, "t": round(a, 1), "from": spk, "to": best,
                            "margin": round(sims[best] - own, 3),
                            "text": " ".join(w["word"] for w in sw)})
    return words


def main() -> int:
    source = sys.argv[1] if len(sys.argv) > 1 else "canary_diar"
    emb = Embedder()
    log: list = []
    for mid in MEETINGS:
        words = verify(mid, source, emb, log)
        turns = []
        for r in absorb_short(runs(words)):
            turns.extend(split_long(r))
        d = OUT / source / mid
        d.mkdir(parents=True, exist_ok=True)
        (d / "transcript_speakers.txt").write_text(
            "\n".join(f"[{hhmmss(s)}] {spk}: {txt}" for spk, s, e, txt in turns) + "\n",
            encoding="utf-8")
        print(f"{mid}: {sum(1 for x in log if x['meeting'] == mid)} sentences moved", flush=True)
    (REPO / "goldset" / f"verify_short_turns_{source}.json").write_text(
        json.dumps(log, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"total moved: {len(log)}  -> goldset/verify_short_turns_{source}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
