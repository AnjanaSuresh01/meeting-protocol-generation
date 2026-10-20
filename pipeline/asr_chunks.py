"""Split audio that is too long for one recognizer pass at its quietest points.

Whisper's feature extractor pads or cuts every input to exactly 30 s, and the Whisper models
returned no text for longer clips. Canary in the NeMo environment fails on clips longer than
40 s. split() returns pieces of at most MAX_S seconds, each cut in a pause, so every piece is
an ordinary short decode. join() puts the texts of the pieces back together.

    python asr_chunks.py      # self-test
"""
from __future__ import annotations

import numpy as np

MAX_S = 28.0      # below Whisper's 30 s window with margin; also below Canary's 40 s limit
LOOK_S = 10.0     # search the last 10 s before the limit for the quietest point
FRAME_S = 0.02
SMOOTH_S = 0.10  # a real pause lasts, a stop consonant does not (see split())


def split(y: np.ndarray, sr: int = 16000, max_s: float = MAX_S, look_s: float = LOOK_S) -> list[np.ndarray]:
    """Pieces of y, each at most max_s long, cut in a real pause; their concatenation is y.

    The cut is the center of the quietest SMOOTH_S window in the last look_s seconds before the
    limit. A single quiet 20 ms frame is often the closure of a stop consonant inside a word, so
    the energy is averaged over SMOOTH_S first. The window start is clamped to the current
    piece, so look_s > max_s cannot read audio from the end of the array.
    """
    n_max = int(max_s * sr)
    if len(y) <= n_max:
        return [y]
    frame = int(FRAME_S * sr)
    win = max(1, int(round(SMOOTH_S / FRAME_S)))
    pieces, start = [], 0
    while len(y) - start > n_max:
        lo = max(start + frame, start + int((max_s - look_s) * sr))     # never before `start`
        hi = start + n_max
        seg = y[lo:hi]
        k = len(seg) // frame
        if k == 0:
            cut = hi
        else:
            rms = np.sqrt((seg[: k * frame].reshape(k, frame).astype(np.float64) ** 2).mean(axis=1))
            if k >= win:
                # mean energy of every SMOOTH_S window; the center of the quietest window is the cut
                c = np.cumsum(np.concatenate(([0.0], rms)))
                env = (c[win:] - c[:-win]) / win
                cut = lo + (int(np.argmin(env)) + win // 2) * frame + frame // 2
            else:
                cut = lo + int(np.argmin(rms)) * frame + frame // 2
        cut = min(max(cut, lo), hi)                                     # stay inside the window
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
