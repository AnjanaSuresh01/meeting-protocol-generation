"""Quality checks for ASR transcripts: compression ratio, type-token ratio, repeated n-grams,
a sliding-window loop scan, subtitle boilerplate and words per minute.
transcript_health() combines them into the verdict "clean", "suspect" or "degenerate".
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass, asdict, field

# Subtitle phrases that Whisper outputs on silence or non-speech.
# Keep the list narrow: phrases that also occur in real meetings cause false flags.
GERMAN_BOILERPLATE = [
    "amara.org", "untertitel der amara", "untertitelung des zdf", "untertitelung aufgrund",
    "vielen dank fürs zuschauen", "abonniert den kanal", "abonniert unseren kanal",
    "copyright wdr", "wdr 20", "swr 20",
]

# Thresholds for transcript_health(). CR = gzip compression ratio,
# TTR = type-token ratio, _DE = values for German text. Clean German transcripts already
# reach a CR of about 2.7, above Whisper's English default threshold of 2.4.
CR_SUSPECT_DE = 3.1  # above this: suspect
CR_DEGENERATE_DE = 3.5  # above this: degenerate
TTR_DEGENERATE = 0.12  # below this: degenerate
WPM_FLOOR = 60.0  # words/min of audio; below: suspect, below half: degenerate
LOOP_WINDOW = 60  # tokens per window
# A window counts as looped when its share of distinct bigrams is below LOOP_DISTINCT2;
# windows of clean German speech reach about 0.85 to 0.95.
# LOOP_FRACTION_SUSPECT and LOOP_FRACTION_DEGENERATE: minimum share of looped windows.
LOOP_DISTINCT2 = 0.55
LOOP_FRACTION_SUSPECT = 0.05
LOOP_FRACTION_DEGENERATE = 0.20


def gzip_compression_ratio(text: str) -> float:
    """Return the UTF-8 byte length of text divided by its gzip-compressed length.
    Repeated text gives a higher value.
    """
    b = text.encode("utf-8")
    if not b:
        return 0.0
    return len(b) / max(1, len(gzip.compress(b)))


def type_token_ratio(toks: list[str]) -> float:
    return (len(set(toks)) / len(toks)) if toks else 1.0


def distinct_n(toks: list[str], n: int = 2) -> float:
    if len(toks) < n:
        return 1.0
    grams = list(zip(*[toks[i:] for i in range(n)]))
    return len(set(grams)) / max(1, len(grams))


def max_ngram_repeat(toks: list[str], n: int = 4) -> tuple[int, str]:
    """Return (count, phrase) for the longest run of one n-gram repeated back to back.
    A loop is found only when its cycle length divides n.
    """
    best_count, best_phrase = 1, ""
    if len(toks) < n * 2:
        return 1, ""
    i = 0
    while i < len(toks) - n:
        gram = toks[i:i + n]
        count = 1
        j = i + n
        while toks[j:j + n] == gram:
            count += 1
            j += n
        if count > best_count:
            best_count, best_phrase = count, " ".join(gram)
        i = j if count > 1 else i + 1
    return best_count, best_phrase


def windowed_loop_scan(toks: list[str], window: int = LOOP_WINDOW,
                       step: int = 20) -> tuple[float, int | None]:
    """Return the share of windows whose distinct-bigram ratio is below LOOP_DISTINCT2
    and the token index where the first such window starts (None if there is none).
    """
    if len(toks) < window:
        return 0.0, None
    starts = range(0, len(toks) - window, step)
    onset, n_bad, n_tot = None, 0, 0
    for start in starts:
        n_tot += 1
        if distinct_n(toks[start:start + window], 2) < LOOP_DISTINCT2:
            n_bad += 1
            if onset is None:
                onset = start
    return (n_bad / n_tot if n_tot else 0.0), onset


@dataclass
class HealthReport:
    n_words: int
    wpm: float | None
    compression_ratio: float
    type_token_ratio: float
    loop_fraction: float  # share of looped windows
    max_ngram_repeat: int
    repeated_phrase: str
    boilerplate_hits: list[str]
    degeneracy_onset: int | None
    verdict: str  # "clean" | "suspect" | "degenerate"
    reasons: list[str] = field(default_factory=list)

    def as_dict(self):
        return asdict(self)


def transcript_health(text: str, audio_duration_s: float | None = None) -> HealthReport:
    """Return a HealthReport with the verdict "clean", "suspect" or "degenerate" and its reasons.
    audio_duration_s (seconds) enables the words-per-minute check.
    """
    toks = text.split()
    n = len(toks)
    cr = gzip_compression_ratio(text)
    ttr = type_token_ratio(toks)
    rep_count, rep_phrase = max_ngram_repeat(toks, 4)
    low = text.casefold()
    hits = [b for b in GERMAN_BOILERPLATE if b in low]
    wpm = (n / (audio_duration_s / 60.0)) if audio_duration_s else None
    loop_frac, onset = windowed_loop_scan(toks)

    reasons: list[str] = []
    if loop_frac >= LOOP_FRACTION_SUSPECT:
        reasons.append(f"loop_fraction {loop_frac:.2f} (onset~word {onset})")
    if rep_count >= 4:
        reasons.append(f"phrase x{rep_count}: '{rep_phrase}'")
    if cr > CR_SUSPECT_DE:
        reasons.append(f"gzipCR {cr:.2f} > {CR_SUSPECT_DE} (German baseline ~2.7)")
    if n >= 200 and ttr < TTR_DEGENERATE:
        reasons.append(f"TTR {ttr:.2f} < {TTR_DEGENERATE} (massive loop)")
    if hits:
        reasons.append(f"boilerplate: {hits}")
    if wpm is not None and wpm < WPM_FLOOR:
        reasons.append(f"words/min {wpm:.0f} < {WPM_FLOOR} (near-silence)")
    if n < 50:
        reasons.append(f"only {n} words (truncated/empty)")

    degenerate = (
        loop_frac >= LOOP_FRACTION_DEGENERATE
        or rep_count >= 6
        or ttr < TTR_DEGENERATE
        or cr > CR_DEGENERATE_DE
        or n < 50
        or (wpm is not None and wpm < WPM_FLOOR / 2)
    )
    verdict = "degenerate" if degenerate else ("suspect" if reasons else "clean")

    return HealthReport(
        n_words=n, wpm=wpm, compression_ratio=round(cr, 3),
        type_token_ratio=round(ttr, 3), loop_fraction=round(loop_frac, 3),
        max_ngram_repeat=rep_count, repeated_phrase=rep_phrase,
        boilerplate_hits=hits, degeneracy_onset=onset,
        verdict=verdict, reasons=reasons,
    )


def flag_segment(avg_logprob: float | None, no_speech_prob: float | None,
                 seg_text: str,
                 logprob_threshold: float = -1.0,
                 no_speech_threshold: float = 0.6) -> bool:
    """Return True if a Whisper segment should be decoded again: avg_logprob below
    logprob_threshold, no_speech_prob above no_speech_threshold, or a transcript_health()
    verdict other than "clean".
    """
    if avg_logprob is not None and avg_logprob < logprob_threshold:
        return True
    if no_speech_prob is not None and no_speech_prob > no_speech_threshold:
        return True
    return transcript_health(seg_text).verdict != "clean"
