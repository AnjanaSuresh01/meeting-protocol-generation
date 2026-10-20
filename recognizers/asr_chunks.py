"""Split audio that is too long for one recognizer pass at its quietest points.

This is the version NB_whisper_long.ipynb used: each piece is at most MAX_S seconds long and
is cut at the quietest 20 ms frame in the last LOOK_S seconds before the limit.
pipeline/asr_chunks.py is a later version that averages the energy over 100 ms before it
chooses the cut.

    python asr_chunks.py      # self-test
"""
from __future__ import annotations

import numpy as np

MAX_S = 28.0      # below Whisper's 30 s window with margin; also below Canary's 40 s limit
LOOK_S = 10.0     # search the last 10 s before the limit for the quietest point
FRAME_S = 0.02


def split(y: np.ndarray, sr: int = 16000, max_s: float = MAX_S, look_s: float = LOOK_S) -> list[np.ndarray]:
    """Pieces of y, each at most max_s long, cut at the quietest frame; concatenation == y."""
    n_max = int(max_s * sr)
    if len(y) <= n_max:
        return [y]
    frame = int(FRAME_S * sr)
    pieces, start = [], 0
    while len(y) - start > n_max:
        lo = start + int((max_s - look_s) * sr)
        hi = start + n_max
        seg = y[lo:hi]
        k = len(seg) // frame
        if k == 0:
            cut = hi
        else:
            rms = np.sqrt((seg[: k * frame].reshape(k, frame).astype(np.float64) ** 2).mean(axis=1))
            cut = lo + int(np.argmin(rms)) * frame + frame // 2
        pieces.append(y[start:cut])
        start = cut
    pieces.append(y[start:])
    return pieces


def join(texts: list[str]) -> str:
    return " ".join(t.strip() for t in texts if t and t.strip())


if __name__ == "__main__":                                # self-test
    sr = 16000
    rng = np.random.default_rng(0)
    y = rng.normal(0, 0.1, sr * 70).astype(np.float32)
    y[int(25.0 * sr):int(25.4 * sr)] = 0.0                # a pause at 25 s
    y[int(52.0 * sr):int(52.3 * sr)] = 0.0                # and one at 52 s
    p = split(y, sr)
    assert np.array_equal(np.concatenate(p), y), "pieces must reassemble the input exactly"
    assert all(len(x) <= MAX_S * sr for x in p), "every piece within the limit"
    cuts = np.cumsum([len(x) for x in p])[:-1] / sr
    assert abs(cuts[0] - 25.2) < 0.3 and abs(cuts[1] - 52.15) < 0.3, cuts
    assert split(y[: sr * 20], sr)[0] is not None and len(split(y[: sr * 20], sr)) == 1
    print(f"self-test OK: 70 s -> {len(p)} pieces, cuts at {', '.join(f'{c:.2f}s' for c in cuts)} (pauses at 25.0-25.4, 52.0-52.3)")
