"""Shared audio and word-list helpers: meeting waveform, frame RMS envelope, drift-corrected
Canary word list, spaCy sentence spans and clip cutting with added silence.

Recordings and pipeline outputs are read from paths relative to this folder.
"""

from __future__ import annotations

import json
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

REPO = Path(__file__).resolve().parent
G = REPO / "goldset"
OUT = REPO / "pipeline_outputs_extracted" / "outputs"
SR, HOP, WIN = 16000, 0.010, 0.030

MEETINGS = ["march09", "april13", "april27", "may11", "may18", "june01", "june15", "june22"]
AUDIO = {"march09": "Team-Meeting_2026-03-09.mp3", "april13": "Team-Meeting_2026-04-13.mp3",
         "april27": "Team-Meeting_2026-04-27.flac", "may11": "Team-Meeting_2026-05-11.flac",
         "may18": "Team-Meeting_2026-05-18.flac", "june01": "Team-Meeting_2026-06-01.flac",
         "june15": "Team-Meeting_2026-06-15.flac", "june22": "Team-Meeting_2026-06-22.flac"}

_wav: dict[str, np.ndarray] = {}
_env: dict[str, np.ndarray] = {}


def waveform(meeting: str) -> np.ndarray:
    """Return the whole meeting as 16 kHz mono float32; the cache holds one meeting at a time."""
    if meeting not in _wav:
        _wav.clear()
        _env.clear()
        # The same recording may be stored as FLAC or MP3. If the mapped file is missing,
        # the first file with the same stem and another extension is used.
        f = REPO / AUDIO[meeting]
        if not f.exists():
            alt = sorted(REPO.glob(f.stem + ".*"))
            if not alt:
                raise FileNotFoundError(f"no audio for {meeting}: looked for {f.name}")
            f = alt[0]
        y, sr = sf.read(str(f), dtype="float32", always_2d=True)
        y = y.mean(axis=1)
        if sr != SR:
            g = gcd(SR, sr)
            y = resample_poly(y, SR // g, sr // g).astype(np.float32)
        _wav[meeting] = y
    return _wav[meeting]


def envelope(meeting: str) -> np.ndarray:
    """Return the frame RMS envelope of the meeting (30 ms frames every 10 ms), cached."""
    if meeting not in _env:
        y = waveform(meeting)
        h, w = int(HOP * SR), int(WIN * SR)
        n = max(0, (len(y) - w) // h)
        fr = np.lib.stride_tricks.as_strided(
            y, shape=(n, w), strides=(y.strides[0] * h, y.strides[0]))
        _env[meeting] = np.sqrt((fr.astype(np.float64) ** 2).mean(axis=1) + 1e-12)
    return _env[meeting]


def levels(env: np.ndarray) -> tuple[float, float]:
    """Return (speech level, noise floor) as the 75th and 10th percentiles of an envelope."""
    return float(np.percentile(env, 75)), float(np.percentile(env, 10))


def words(meeting: str) -> list[dict]:
    """Return the meeting's Canary words with timestamps corrected by drift_map."""
    from build_clip_dataset import drift_map
    ws = json.loads((OUT / "canary_diar" / meeting / "transcript_canary_ts.json")
                    .read_text(encoding="utf-8"))
    f, _ = drift_map(meeting)
    for w in ws:
        w["start"], w["end"] = f(float(w["start"])), f(float(w["end"]))
    return ws


def sentence_spans(ws: list[dict], nlp) -> list[tuple[int, int]]:
    """Return inclusive (first word, last word) index pairs, one per spaCy sentence."""
    text, starts, pos = [], [], 0
    for w in ws:
        starts.append(pos)
        text.append(w["word"])
        pos += len(w["word"]) + 1
    doc = nlp(" ".join(text))
    st = np.array(starts)
    out = []
    for s in doc.sents:
        i = int(np.searchsorted(st, s.start_char, "right")) - 1
        j = int(np.searchsorted(st, s.end_char - 1, "right")) - 1
        if 0 <= i <= j < len(ws):
            out.append((i, j))
    return out


def german_nlp():
    import spacy
    nlp = spacy.load("de_core_news_sm",
                     exclude=["ner", "lemmatizer", "tagger", "attribute_ruler"])
    if "parser" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    nlp.max_length = 3_000_000
    return nlp


FADE = 0.010  # seconds; fade at each join with added silence


def cut(meeting: str, seg: dict, y: np.ndarray | None = None) -> np.ndarray:
    """Return the segment's audio from `start_s` to `end_s` with `pad_start_s` and `pad_end_s`
    seconds of digital silence added. Where silence is added, the audio fades over FADE seconds
    so the join does not click. The pads (from resegment_v7.py) are the part of the clip's silence
    budget that the recording cannot supply because a neighboring word is too close.
    """
    if y is None:
        y = waveform(meeting)
    a, b = int(seg["start_s"] * SR), int(seg["end_s"] * SR)
    clip = np.array(y[max(0, a):min(len(y), b)], dtype=np.float32)
    pad0, pad1 = int(seg.get("pad_start_s", 0) * SR), int(seg.get("pad_end_s", 0) * SR)
    f = int(FADE * SR)
    if pad1 > 0 and len(clip) > f:
        clip[-f:] *= np.linspace(1.0, 0.0, f, dtype=np.float32)
    if pad0 > 0 and len(clip) > f:
        clip[:f] *= np.linspace(0.0, 1.0, f, dtype=np.float32)
    return np.concatenate([np.zeros(pad0, np.float32), clip, np.zeros(pad1, np.float32)])
