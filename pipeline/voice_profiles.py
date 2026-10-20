"""Voice-profile store that names speakers from earlier meetings.

enroll() labels the voices of a recording with names from its own corrected minutes and adds
their WavLM x-vectors to a JSON store. match() names the voices of a later recording by cosine
similarity to the stored profiles.

    python voice_profiles.py enroll AUDIO TRANSCRIPT CORRECTED_PROTOCOL --store voices.json
"""
from __future__ import annotations

import json
import re
import sys
from math import gcd
from pathlib import Path

import numpy as np

SR = 16000
MODEL = "microsoft/wavlm-base-plus-sv"
MIN_TURN, MAX_TURN, N_TURNS = 2.0, 8.0, 12  # seconds, seconds, turns per speaker
# Cosine similarity between a voice and a stored profile. In the meeting recordings the same
# person scored at least 0.975 and different people up to 0.985.
MATCH_MIN = 0.975  # match() names a voice only at or above this
SURELY_OTHER = 0.985  # enroll() refuses a new name at or above this
_TURN = re.compile(r"\[(\d\d):(\d\d):(\d\d)\]\s+(SPEAKER_\d+):")
_model = None


def _load_model():
    global _model
    if _model is None:
        from transformers import AutoFeatureExtractor, WavLMForXVector
        _model = (AutoFeatureExtractor.from_pretrained(MODEL), WavLMForXVector.from_pretrained(MODEL).eval())
    return _model


def load_audio(path) -> np.ndarray:
    import soundfile as sf
    from scipy.signal import resample_poly
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if sr != SR:
        g = gcd(SR, sr)
        y = resample_poly(y, SR // g, sr // g).astype(np.float32)
    return y


def _trim_silence(seg: np.ndarray, sr: int = SR, floor: float = 0.10,
                  min_s: float = 1.0) -> np.ndarray:
    """Cut leading and trailing 20 ms frames with RMS below `floor` times the peak frame RMS.

    Return the window unchanged if it has fewer than three frames, is silent, or would become
    shorter than `min_s` seconds.
    """
    frame = int(0.02 * sr)
    k = len(seg) // frame
    if k < 3:
        return seg
    rms = np.sqrt((seg[: k * frame].reshape(k, frame).astype(np.float64) ** 2).mean(axis=1))
    peak = float(rms.max())
    if peak <= 0:
        return seg
    loud = np.flatnonzero(rms >= floor * peak)
    if loud.size == 0:
        return seg
    a, b = int(loud[0]) * frame, min(len(seg), (int(loud[-1]) + 1) * frame)
    if b - a < int(min_s * sr):
        return seg
    return seg[a:b]


def embed(audio, transcript: str) -> dict[str, np.ndarray]:
    """Return one L2-normalized WavLM x-vector per SPEAKER label, the mean over its longest turns.

    A turn lasts until the next turn starts, at most MAX_TURN seconds.
    """
    import torch
    y = load_audio(audio) if not isinstance(audio, np.ndarray) else audio
    rows = [(int(h) * 3600 + int(m) * 60 + int(s), lab) for h, m, s, lab in _TURN.findall(transcript)]
    turns: dict[str, list[tuple[float, float]]] = {}
    for i, (t, lab) in enumerate(rows):
        d = (rows[i + 1][0] - t) if i + 1 < len(rows) else 10
        turns.setdefault(lab, []).append((d, t))
    fe, model = _load_model()
    out = {}
    for lab, tl in turns.items():
        use = sorted([x for x in tl if x[0] >= MIN_TURN], reverse=True)[:N_TURNS] or sorted(tl, reverse=True)[:N_TURNS]
        vecs = []
        for d, s in use:
            a, b = int(s * SR), int(min(s + min(d, MAX_TURN), len(y) / SR) * SR)
            if b - a > SR // 2:
                w = _trim_silence(y[a:b])
                with torch.no_grad():
                    e = model(**fe(w, sampling_rate=SR, return_tensors="pt")).embeddings[0].numpy()
                vecs.append(e / (np.linalg.norm(e) + 1e-9))
        if vecs:
            v = np.mean(vecs, axis=0)
            out[lab] = v / (np.linalg.norm(v) + 1e-9)
    return out


def load_store(path) -> dict:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def match(embs: dict[str, np.ndarray], store: dict, fixed: dict[str, str] | None = None) -> dict[str, str]:
    """Map labels to stored names one-to-one, keeping pairs with cosine similarity >= MATCH_MIN.

    Labels in `fixed` keep their names, and those names are not assigned again.
    """
    from scipy.optimize import linear_sum_assignment
    out = dict(fixed or {})
    names = [n for n in store if n not in out.values()]
    labs = [l for l in embs if l not in out]
    if not names or not labs:
        return out
    P = np.array([store[n]["emb"] for n in names], dtype=np.float32)
    # +1e-9: an all-zero profile (for example in a hand-edited store) would otherwise become NaN,
    # which linear_sum_assignment rejects.
    P /= (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
    S = np.stack([embs[l] for l in labs]) @ P.T
    for r, c in zip(*linear_sum_assignment(-S)):
        if S[r, c] >= MATCH_MIN:
            out[labs[r]] = names[c]
    return out


def enroll(audio, transcript: str, corrected_protocol, store_path, source: str = "") -> dict[str, str]:
    """Name this recording's voices from its own corrected minutes and add them to the store.

    If `source` is already in the store, or the minutes are not .md or .docx, nothing is added and
    {} is returned.
    """
    import handover_naming as HN
    src = source or str(audio)
    already = load_store(store_path)
    if any(src in r.get("sources", []) for r in already.values()):
        print(f"   {src} is already in the voice store, not enrolled again")
        return {}
    if Path(str(corrected_protocol)).suffix.lower() not in (".md", ".docx"):
        print(f"   {Path(str(corrected_protocol)).name} is not .md or .docx, no voices enrolled")
        return {}
    from attribution_baselines import read_any
    # A bold, unindented "Backlog" heading is read as a person; keep it out of the roster.
    roster = list(dict.fromkeys(r[:1].upper() + r[1:] for r, _ in read_any(str(corrected_protocol))
                                if r and r != "backlog"))
    names = HN.name_speakers([transcript], roster) if roster else {}
    # The minutes belong to the same meeting, so topic naming runs without a margin.
    keep, HN.CONTENT_MARGIN = HN.CONTENT_MARGIN, 0.0
    try:
        names = HN.name_by_content(transcript, corrected_protocol, names)
    finally:
        HN.CONTENT_MARGIN = keep
    embs = embed(audio, transcript)
    store = load_store(store_path)
    # Consistency guard: a voice is not stored under a name when it clearly
    # belongs to another person already in the store.
    known = {n: np.array(r["emb"], dtype=np.float32) for n, r in store.items()}
    for lab, name in list(names.items()):
        if lab not in embs or not known:
            continue
        sims = {n: float(embs[lab] @ (v / (np.linalg.norm(v) + 1e-9))) for n, v in known.items()}
        best = max(sims, key=sims.get)
        # A known name must be the voice's best match; a new name is refused only if
        # another profile reaches SURELY_OTHER.
        wrong = (best != name and name in known) or (best != name and sims[best] >= SURELY_OTHER)
        if wrong:
            print(f"   not enrolled: {lab} labelled {name}, but the voice is {best} ({sims[best]:.2f})")
            del names[lab]
    for lab, name in names.items():
        if lab not in embs:
            continue
        rec = store.get(name, {"emb": np.zeros_like(embs[lab]).tolist(), "n": 0, "sources": []})
        n = rec["n"]
        v = (np.array(rec["emb"]) * n + embs[lab]) / (n + 1)
        store[name] = {"emb": (v / (np.linalg.norm(v) + 1e-9)).tolist(), "n": n + 1,
                       "sources": rec["sources"] + [src]}
    Path(store_path).write_text(json.dumps(store, ensure_ascii=False), encoding="utf-8")
    return names


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["enroll"]:
        store = a[a.index("--store") + 1]
        got = enroll(a[1], Path(a[2]).read_text(encoding="utf-8", errors="replace"), Path(a[3]), store, a[1])
        print("enrolled:", ", ".join(f"{k}={v}" for k, v in sorted(got.items())))
