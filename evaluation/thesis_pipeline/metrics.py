"""Evaluation metrics: micro-averaged WER and CER, fuzzy named-entity recall and precision,
cpWER, gold-set scoring, bootstrap intervals, and paired tests with Holm correction.

Run from the evaluation folder to print a small demo:

    python -m thesis_pipeline.metrics
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, asdict

import jiwer
from rapidfuzz import fuzz

from .text_norm import normalize, to_nfc, NormConfig

# Normalizer for entity matching: casefold, strip punctuation, keep umlauts.
_ENT_CFG = NormConfig(casefold=True, strip_punct=True, remove_filled_pauses=False)

# ---------------------------------------------------------------------------
# WER and CER
# ---------------------------------------------------------------------------


@dataclass
class WerReport:
    n_items: int
    ref_words: int
    wer: float          # micro: total (S+D+I) / total ref words
    cer: float          # micro character error rate
    sub: int
    ins: int
    dele: int
    per_item_wer: list[float]

    def as_dict(self):
        return asdict(self)


def wer_cer(refs: list[str], hyps: list[str], cfg: NormConfig | None = None) -> WerReport:
    """Micro-averaged WER and CER over aligned (ref, hyp) pairs, with the same normalizer on both
    sides; WER is not clipped at 1.
    """
    cfg = cfg or NormConfig()
    refs_n = [normalize(r, cfg) for r in refs]
    hyps_n = [normalize(h, cfg) for h in hyps]

    measures = jiwer.process_words(refs_n, hyps_n)
    ref_words = sum(len(r.split()) for r in refs_n)
    cer = jiwer.cer(refs_n, hyps_n)
    per_item = [jiwer.wer([r], [h]) for r, h in zip(refs_n, hyps_n)]

    return WerReport(
        n_items=len(refs), ref_words=ref_words,
        wer=round(measures.wer, 4), cer=round(float(cer), 4),
        sub=measures.substitutions, ins=measures.insertions, dele=measures.deletions,
        per_item_wer=[round(x, 4) for x in per_item],
    )


# ---------------------------------------------------------------------------
# Named-entity recall and precision
# ---------------------------------------------------------------------------

def build_attendance_roster(reference_text: str, known_entities: list[str],
                            cutoff: int = 88) -> list[str]:
    """Return the entries of known_entities that occur in the reference text (fuzzy word match,
    see entity_in_text).
    """
    present = []
    ref_nfc = to_nfc(reference_text)
    for ent in known_entities:
        if entity_in_text(ent, ref_nfc, cutoff=cutoff):
            present.append(ent)
    return present


def entity_in_text(entity: str, text: str, cutoff: int = 88) -> bool:
    """Return True if the normalized entity matches one word, or a span of as many words, in the
    text with a RapidFuzz ratio of at least cutoff.
    """
    ent = normalize(entity, _ENT_CFG)
    if not ent:
        return False
    words = normalize(text, _ENT_CFG).split()
    span = len(ent.split())
    if span == 1:
        return any(fuzz.ratio(ent, w) >= cutoff for w in words)
    for i in range(len(words) - span + 1):
        cand = " ".join(words[i:i + span])
        if fuzz.ratio(ent, cand) >= cutoff:
            return True
    return False


@dataclass
class NeReport:
    n_entities_present: int
    found: int
    ne_fnr: float  # 1 - recall
    recall: float
    precision: float  # roster hits / (roster hits + extra-candidate hits)
    per_entity: dict

    def as_dict(self):
        return asdict(self)


def ne_scores(hypothesis: str, roster: list[str],
              extra_candidates: list[str] | None = None, cutoff: int = 88) -> NeReport:
    """Score a transcript against a meeting roster: recall, NE-FNR (1 - recall) and precision.
    Each extra_candidates name found in the text counts as a false positive.
    """
    per = {}
    found = 0
    for ent in roster:
        hit = entity_in_text(ent, hypothesis, cutoff)
        per[ent] = hit
        found += int(hit)
    recall = found / len(roster) if roster else 0.0

    false_pos = 0
    for ent in (extra_candidates or []):
        if entity_in_text(ent, hypothesis, cutoff):
            false_pos += 1
    precision = found / (found + false_pos) if (found + false_pos) else 1.0

    return NeReport(
        n_entities_present=len(roster), found=found,
        ne_fnr=round(1 - recall, 4), recall=round(recall, 4),
        precision=round(precision, 4), per_entity=per,
    )


# ---------------------------------------------------------------------------
# Gold-set scoring
# ---------------------------------------------------------------------------

def score_goldset(gold_records: list[dict], model: str, cfg: NormConfig | None = None) -> dict:
    """Score one model on the gold-set records that have a filled corrected_text: WER/CER overall
    and per stratum (easy/hard), plus a population-weighted WER. Records need model_hypotheses,
    stratum and population_weight.
    """
    cfg = cfg or NormConfig(casefold=True)
    filled = [r for r in gold_records if str(r.get("corrected_text", "")).strip()]
    scored = [r for r in filled if model in (r.get("model_hypotheses") or {})]
    if not scored:
        return {"model": model, "n_gold_filled": len(filled), "n_scored": 0,
                "coverage": 0.0, "note": "no segments carry this model's hypothesis "
                "(needs the cluster re-segmentation step)"}

    def _wer(recs):
        if not recs:
            return None
        refs = [r["corrected_text"] for r in recs]
        hyps = [r["model_hypotheses"][model] for r in recs]
        return wer_cer(refs, hyps, cfg).as_dict()

    easy = [r for r in scored if r.get("stratum") == "easy"]
    hard = [r for r in scored if r.get("stratum") == "hard"]
    overall, e, h = _wer(scored), _wer(easy), _wer(hard)

    # population_weight of an easy record is the population share of easy segments;
    # the hard share is 1 minus that.
    easy_share = next((r["population_weight"] for r in scored if r.get("stratum") == "easy"), None)
    corpus_wer = None
    if easy_share is not None and e and h:
        corpus_wer = round(easy_share * e["wer"] + (1 - easy_share) * h["wer"], 4)

    return {
        "model": model,
        "n_gold_filled": len(filled),
        "n_scored": len(scored),
        "coverage": round(len(scored) / len(filled), 4),
        "overall": overall,
        "easy": e,
        "hard": h,
        "population_easy_share": easy_share,
        "corpus_weighted_wer": corpus_wer,
    }


# ---------------------------------------------------------------------------
# cpWER (concatenated minimum-permutation WER)
# ---------------------------------------------------------------------------

def cpwer(ref_by_speaker: dict[str, str], hyp_by_speaker: dict[str, str],
          cfg: NormConfig | None = None) -> dict:
    """Concatenated minimum-permutation WER over {speaker: text} dicts, using the speaker mapping
    with the fewest word errors. The search tries all k! mappings.
    """
    import itertools
    cfg = cfg or NormConfig(casefold=True)
    ref_spk = list(ref_by_speaker)
    hyp_spk = list(hyp_by_speaker)
    k = max(len(ref_spk), len(hyp_spk))
    ref_texts = [normalize(ref_by_speaker.get(s, ""), cfg) for s in ref_spk] + [""] * (k - len(ref_spk))
    ref_slots = ref_spk + [f"_pad{i}" for i in range(k - len(ref_spk))]
    total_ref_words = sum(len(t.split()) for t in ref_texts) or 1

    best = None
    for perm in itertools.permutations(range(k)):
        hyp_texts = []
        for slot in range(k):
            h = hyp_spk[perm[slot]] if perm[slot] < len(hyp_spk) else None
            hyp_texts.append(normalize(hyp_by_speaker.get(h, ""), cfg) if h else "")
        errs = 0
        for r, h in zip(ref_texts, hyp_texts):
            m = jiwer.process_words([r] if r else [" "], [h] if h else [" "])
            errs += m.substitutions + m.deletions + m.insertions
        if best is None or errs < best[0]:
            best = (errs, {ref_slots[s]: (hyp_spk[perm[s]] if perm[s] < len(hyp_spk) else None)
                           for s in range(k)})
    return {"cpwer": round(best[0] / total_ref_words, 4),
            "total_errors": best[0], "ref_words": total_ref_words, "mapping": best[1]}


# ---------------------------------------------------------------------------
# Small-sample statistics
# ---------------------------------------------------------------------------

def bootstrap_ci(values: list[float], n_boot: int = 10000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float, float]:
    """Percentile bootstrap interval of the mean; returns (mean, lo, hi)."""
    import random
    if not values:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    m = statistics.fmean(values)
    boots = []
    k = len(values)
    for _ in range(n_boot):
        sample = [values[rng.randrange(k)] for _ in range(k)]
        boots.append(statistics.fmean(sample))
    boots.sort()
    lo = boots[int((alpha / 2) * n_boot)]
    hi = boots[int((1 - alpha / 2) * n_boot)]
    return (round(m, 4), round(lo, 4), round(hi, 4))


def min_detectable_p(n_pairs: int, n_comparisons: int, alpha: float = 0.05) -> dict:
    """Check whether a Wilcoxon signed-rank test on n_pairs pairs can reach Holm's strictest
    threshold alpha / n_comparisons; its smallest two-sided p is 2 / 2**n_pairs.
    """
    floor = 2.0 / (2 ** n_pairs) if n_pairs > 0 else 1.0
    threshold = alpha / n_comparisons if n_comparisons else alpha
    return {"n_pairs": n_pairs, "n_comparisons": n_comparisons,
            "min_possible_p": round(floor, 6),
            "holm_strictest_threshold": round(threshold, 6),
            "can_ever_reject": bool(floor < threshold)}


def paired_bootstrap_holm(errors_by_model: dict[str, list[int]], ref_words: list[int],
                          n_boot: int = 10000, seed: int = 0, alpha: float = 0.05) -> list[dict]:
    """Paired bootstrap over segments of the micro-WER difference for every pair of systems, with
    Holm-adjusted p-values. errors_by_model maps each system to per-segment edit counts aligned
    with ref_words.
    """
    import random
    models = list(errors_by_model)
    k = len(ref_words)
    if k < 2 or len(models) < 2:
        return []

    def micro(errs: list[int], idx: list[int]) -> float:
        w = sum(ref_words[i] for i in idx)
        return sum(errs[i] for i in idx) / w if w else 0.0

    base = list(range(k))
    rng = random.Random(seed)
    # one shared set of index draws, so every pair sees the same resamples
    replicates = [[rng.randrange(k) for _ in range(k)] for _ in range(n_boot)]

    raw = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            ea, eb = errors_by_model[models[i]], errors_by_model[models[j]]
            observed = micro(ea, base) - micro(eb, base)
            diffs = [micro(ea, r) - micro(eb, r) for r in replicates]
            diffs.sort()
            # two-sided p: share of replicates on the other side of zero
            n_cross = sum(1 for d in diffs if (d >= 0) != (observed >= 0))
            p = min(1.0, 2.0 * (n_cross + 1) / (n_boot + 1))
            lo = diffs[int((alpha / 2) * n_boot)]
            hi = diffs[int((1 - alpha / 2) * n_boot)]
            raw.append((models[i], models[j], observed, lo, hi, p))

    raw.sort(key=lambda r: r[5])
    m = len(raw)
    out = []
    running = 0.0
    for rank, (mi, mj, obs, lo, hi, p) in enumerate(raw):
        # Holm step-down: adjusted p-values never decrease
        running = max(running, min(1.0, p * (m - rank)))
        p_holm = running
        out.append({"a": mi, "b": mj,
                    "delta_wer": round(obs, 4),
                    "ci95": [round(lo, 4), round(hi, 4)],
                    "p_raw": round(p, 4), "p_holm": round(p_holm, 4),
                    "significant_0.05": p_holm < alpha})
    return out


def paired_wilcoxon_holm(scores_by_model: dict[str, list[float]]) -> list[dict]:
    """Paired Wilcoxon signed-rank test on per-meeting scores for every pair of models, with
    Holm-adjusted p-values. Entries with a None score are skipped; rows are sorted by raw p.
    """
    from scipy.stats import wilcoxon
    models = list(scores_by_model)
    raw = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a, b = scores_by_model[models[i]], scores_by_model[models[j]]
            pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
            if len(pairs) < 2 or all(x == y for x, y in pairs):
                raw.append((models[i], models[j], float("nan"), 1.0))
                continue
            xa = [x for x, _ in pairs]
            xb = [y for _, y in pairs]
            try:
                stat, p = wilcoxon(xa, xb)
            except ValueError:
                stat, p = float("nan"), 1.0
            raw.append((models[i], models[j], float(stat) if p == p else float("nan"), p))
    # Holm correction
    raw.sort(key=lambda r: r[3])
    m = len(raw)
    out = []
    running = 0.0
    for rank, (mi, mj, stat, p) in enumerate(raw):
        running = max(running, min(1.0, p * (m - rank)))
        p_holm = running
        out.append({"a": mi, "b": mj, "p_raw": round(p, 4),
                    "p_holm": round(p_holm, 4), "significant_0.05": p_holm < 0.05})
    return out


if __name__ == "__main__":
    ref = "Matthias berichtete über die Personalratssitzung mit Zoé Brandt und Malte."
    hyp = "Mathias berichtete über die Personal Rat Sitzung mit Zoe Brandt."
    print("WER/CER:", wer_cer([ref], [hyp], NormConfig(casefold=True)).as_dict())
    roster = build_attendance_roster(ref, ["Matthias", "Zoé Brandt", "Malte", "Lena"])
    print("roster (present this meeting):", roster)
    print("NE:", ne_scores(hyp, roster, extra_candidates=["Lena", "Magpie"]).as_dict())
    print("bootstrap CI:", bootstrap_ci([0.6, 0.7, 0.65, 0.72, 0.58]))
