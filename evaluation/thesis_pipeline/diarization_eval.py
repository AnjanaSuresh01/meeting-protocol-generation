"""Diarization metrics for the speaker side of the pipeline (who spoke when).

The protocol skeleton (## Person -> ### Topic -> bullets) depends on the diarization, so its
errors are measured directly. The metrics need only a speaker-only reference (who spoke when,
labels such as S1..S5 by voice), which can be annotated without understanding the German words.

Metrics:
  * DER (md-eval style: (missed + false-alarm + confusion) / total reference speaker time) on a
    frame grid (10 ms by default), with the optimal hyp->ref speaker mapping (Hungarian
    algorithm); overlapping speech is counted per speaker.
  * overlap-region DER (DER restricted to frames with more than one reference speaker).
  * speaker-count error (hypothesis speakers minus reference speakers).
  * WDER, transcript-free: for each word time and predicted speaker, the reference speaker active
    at that time is looked up; the result is the fraction of words with the wrong speaker. No
    transcript text is needed.
  * share of misattributed protocol bullets: the same check at bullet level (needs each bullet's
    source-word times and the hypothesis speaker of the ## Person it was filed under).

Input is an RTTM file (parse_rttm) or a list of (start, duration, speaker). pyannote.metrics is
the reference implementation; this one depends only on numpy and scipy. Running the module prints
DER and WDER for a small synthetic example:

    python thesis_pipeline/diarization_eval.py    -> stdout only
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy.optimize import linear_sum_assignment

Segment = tuple[float, float, str]   # (start_s, duration_s, speaker_label)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def parse_rttm(path: str) -> list[Segment]:
    """Read a standard RTTM file. Lines: SPEAKER <file> 1 <start> <dur> <NA> <NA> <spk> ..."""
    out: list[Segment] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 8 and p[0] == "SPEAKER":
                out.append((float(p[3]), float(p[4]), p[7]))
    return out


# ---------------------------------------------------------------------------
# frame grid
# ---------------------------------------------------------------------------

def _grid(ref: list[Segment], hyp: list[Segment], res: float):
    end = max([s + d for s, d, _ in ref + hyp], default=0.0)
    n = int(np.ceil(end / res)) + 1
    ref_spk = sorted({s for *_, s in ref})
    hyp_spk = sorted({s for *_, s in hyp})
    ri = {s: i for i, s in enumerate(ref_spk)}
    hi = {s: i for i, s in enumerate(hyp_spk)}
    R = np.zeros((len(ref_spk), n), dtype=bool)
    H = np.zeros((len(hyp_spk), n), dtype=bool)
    for st, du, s in ref:
        R[ri[s], int(st / res):int((st + du) / res)] = True
    for st, du, s in hyp:
        H[hi[s], int(st / res):int((st + du) / res)] = True
    return R, H, ref_spk, hyp_spk


def optimal_mapping(ref: list[Segment], hyp: list[Segment], res: float = 0.01):
    """Hungarian mapping hyp-speaker -> ref-speaker maximizing co-occurrence frames."""
    R, H, ref_spk, hyp_spk = _grid(ref, hyp, res)
    if not hyp_spk or not ref_spk:
        return {}
    cost = -(H.astype(int) @ R.astype(int).T)          # [hyp x ref] shared frames (neg for max)
    rows, cols = linear_sum_assignment(cost)
    return {hyp_spk[r]: ref_spk[c] for r, c in zip(rows, cols)}


# ---------------------------------------------------------------------------
# DER
# ---------------------------------------------------------------------------

@dataclass
class DerReport:
    der: float
    miss: float
    false_alarm: float
    confusion: float
    overlap_der: float | None
    n_ref_speakers: int
    n_hyp_speakers: int
    speaker_count_error: int
    mapping: dict

    def as_dict(self):
        return asdict(self)


def der(ref: list[Segment], hyp: list[Segment], res: float = 0.01) -> DerReport:
    R, H, ref_spk, hyp_spk = _grid(ref, hyp, res)
    mapping = optimal_mapping(ref, hyp, res)
    n = R.shape[1] if R.size else 0
    # map hyp rows onto ref-speaker index space
    Hmap = np.zeros_like(R)
    ri = {s: i for i, s in enumerate(ref_spk)}
    for j, hs in enumerate(hyp_spk):
        tgt = mapping.get(hs)
        if tgt in ri:
            Hmap[ri[tgt]] |= H[j]
    n_ref = R.sum(0)                                    # ref speakers active per frame
    n_hyp_all = H.sum(0)                                # hyp speakers active per frame (unmapped count)
    n_corr = (R & Hmap).sum(0)                          # correctly matched speakers per frame
    miss = np.maximum(0, n_ref - np.minimum(n_ref, n_hyp_all))
    fa = np.maximum(0, n_hyp_all - n_ref)
    conf = np.minimum(n_ref, n_hyp_all) - n_corr
    total = n_ref.sum()
    miss_s, fa_s, conf_s = miss.sum() * res, fa.sum() * res, conf.sum() * res
    total_s = total * res
    der_val = (miss_s + fa_s + conf_s) / total_s if total_s else 0.0

    ov = n_ref > 1                                      # overlap frames
    ov_total = n_ref[ov].sum() * res
    ov_der = (((miss + fa + conf)[ov]).sum() * res / ov_total) if ov_total else None

    return DerReport(
        der=round(der_val, 4), miss=round(miss_s / total_s, 4) if total_s else 0.0,
        false_alarm=round(fa_s / total_s, 4) if total_s else 0.0,
        confusion=round(conf_s / total_s, 4) if total_s else 0.0,
        overlap_der=round(ov_der, 4) if ov_der is not None else None,
        n_ref_speakers=len(ref_spk), n_hyp_speakers=len(hyp_spk),
        speaker_count_error=len(hyp_spk) - len(ref_spk), mapping=mapping,
    )


# ---------------------------------------------------------------------------
# WDER (transcript-free) and bullet attribution
# ---------------------------------------------------------------------------

def _ref_speaker_at(ref: list[Segment], t: float) -> str | None:
    """Reference speaker active at time t: the first segment that covers t, or None."""
    best, best_ov = None, 0.0
    for st, du, s in ref:
        if st <= t < st + du:
            return s                                     # first covering segment, overlap is not resolved
    return best


def wder(words: list[tuple[float, str]], ref: list[Segment],
         mapping: dict) -> dict:
    """words = [(time_s, predicted_hyp_speaker), ...] (time = word midpoint). Transcript-free.
    Counts words whose mapped predicted speaker differs from the reference speaker active at that
    time; words that fall outside all reference speech are not scored."""
    wrong = counted = 0
    for t, hyp_spk in words:
        ref_spk = _ref_speaker_at(ref, t)
        if ref_spk is None:
            continue                                     # word in a non-speech ref gap, skip
        counted += 1
        if mapping.get(hyp_spk) != ref_spk:
            wrong += 1
    return {"wder": round(wrong / counted, 4) if counted else None,
            "words_scored": counted, "words_wrong": wrong}


def bullet_attribution_error(bullets: list[dict], ref: list[Segment], mapping: dict) -> dict:
    """Reader-facing metric: % of protocol bullets filed under the wrong ## Person.
    Each bullet = {"person_speaker": <hyp speaker id of the ## heading>, "word_times": [t,...]}.
    A bullet is misattributed if the majority reference speaker over its word times differs from
    the reference speaker its heading maps to. Needs word-level provenance, e.g. from forced
    alignment."""
    wrong = counted = 0
    for b in bullets:
        times = b.get("word_times", [])
        if not times:
            continue
        ref_votes = [_ref_speaker_at(ref, t) for t in times]
        ref_votes = [r for r in ref_votes if r]
        if not ref_votes:
            continue
        majority = max(set(ref_votes), key=ref_votes.count)
        heading_ref = mapping.get(b.get("person_speaker"))
        counted += 1
        if heading_ref != majority:
            wrong += 1
    return {"misattributed_bullet_pct": round(100 * wrong / counted, 2) if counted else None,
            "bullets_scored": counted, "bullets_wrong": wrong}


if __name__ == "__main__":
    # synthetic self-test: 3 reference speakers; the hypothesis gives S2's first turn to A (= S1),
    # which is a confusion
    ref = [(0, 10, "S1"), (10, 10, "S2"), (20, 10, "S3"), (30, 5, "S1"), (35, 5, "S2")]
    hyp = [(0, 10, "A"), (10, 10, "A"), (20, 10, "B"), (30, 5, "A"), (35, 5, "C")]  # A=S1 plus S2's turn
    r = der(ref, hyp)
    print("DER:", r.as_dict())
    words = [(5, "A"), (15, "A"), (25, "B"), (32, "A"), (37, "C")]  # word@15 is really S2 -> wrong
    print("WDER:", wder(words, ref, r.mapping))
