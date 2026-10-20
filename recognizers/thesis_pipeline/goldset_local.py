"""Selects up to 150 Whisper windows of 5 to 15 s from the transcripts in
pipeline_outputs_extracted/outputs/ for human correction and writes them to goldset/.

A window's word error rate against the Cohere and Granite transcripts labels it easy or hard,
and the sample keeps the easy:hard ratio of all scored windows. parse_timestamped() reads a
timestamped Whisper transcript, and tokenize() splits text into words.

Usage, from recognizers/:
    python -m thesis_pipeline.goldset_local
"""

from __future__ import annotations

import bisect
import difflib
import hashlib
import json
import random
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

import jiwer

from .text_norm import normalize, NormConfig
from .asr_quality import transcript_health

# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #

REPO = Path(__file__).resolve().parent.parent
OUTPUTS = REPO / "pipeline_outputs_extracted" / "outputs"
GOLDDIR = REPO / "goldset"

MEETINGS = ["march09", "april13", "april27", "may11", "may18", "june01", "june15", "june22"]

AUDIO_FILE = {
    "march09": "Team-Meeting_2026-03-09.mp3", "april13": "Team-Meeting_2026-04-13.mp3",
    "april27": "Team-Meeting_2026-04-27.flac", "may11": "Team-Meeting_2026-05-11.flac",
    "may18": "Team-Meeting_2026-05-18.flac", "june01": "Team-Meeting_2026-06-01.flac",
    "june15": "Team-Meeting_2026-06-15.flac", "june22": "Team-Meeting_2026-06-22.flac",
}
MEETING_DATE = {m: f"2026-{m[:-2].replace('march','03').replace('april','04').replace('may','05').replace('june','06')}-{m[-2:]}"
                for m in MEETINGS}

# Timestamped Whisper files for the window grid, in order of preference.
GRID_VARIANTS = ["transcript_no_prompt_timestamped.txt", "transcript_with_prompt_timestamped.txt"]
# Partners as (model folder under OUTPUTS, transcript file name).
PARTNERS = [("cohere", "transcript.txt"), ("granite41", "transcript.txt")]

MIN_WIN, TARGET_WIN, MAX_WIN = 5.0, 10.0, 15.0
MIN_TOKENS = 8  # minimum words per window
N_TOTAL = 150
SEED = 20260717
# A window is easy when its mean partner WER is at most AGREE_WER, otherwise hard.
AGREE_WER = 0.10
SEED_MODEL = "cohere"  # partner whose text seeds candidate_text

_NORM = NormConfig(casefold=True, strip_punct=True)
_LINE = re.compile(r"^\[(\d{2}):(\d{2}):(\d{2})\s*\S+\s*(\d{2}):(\d{2}):(\d{2})\]\s*(.*)$")
_WORD = re.compile(r"[0-9A-Za-zÄÖÜäöüß][0-9A-Za-zÄÖÜäöüß'’-]*")


# --------------------------------------------------------------------------- #
# parsing + tokenizing
# --------------------------------------------------------------------------- #

def _read_utf8(path: Path) -> str:
    """Read a file as strict UTF-8 and name the file in the error if decoding fails."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        raise UnicodeDecodeError(e.encoding, e.object, e.start, e.end,
                                 f"{e.reason} — non-UTF-8 source transcript: {path}") from None


@dataclass
class Seg:
    t0: float
    t1: float
    text: str


def parse_timestamped(path: Path) -> list[Seg]:
    segs: list[Seg] = []
    if not path.exists() or path.stat().st_size == 0:
        return segs
    for line in _read_utf8(path).splitlines():
        m = _LINE.match(line.strip())
        if not m:
            continue
        h0, m0, s0, h1, m1, s1, txt = m.groups()
        t0 = int(h0) * 3600 + int(m0) * 60 + int(s0)
        t1 = int(h1) * 3600 + int(m1) * 60 + int(s1)
        if txt.strip():
            segs.append(Seg(float(t0), float(max(t1, t0)), txt.strip()))
    return segs


def tokenize(text: str) -> list[tuple[str, str]]:
    """Split text into (original, lowercase) word pairs."""
    text = text.replace("...", " ")
    return [(w, w.lower()) for w in _WORD.findall(text)]


# --------------------------------------------------------------------------- #
# loop-guarded loading
# --------------------------------------------------------------------------- #

def pick_clean_grid(meeting: str) -> tuple[list[Seg], str, str]:
    """Return (segments, variant, health label) of the first Whisper variant rated clean by
    transcript_health, else of the first rated suspect, else ([], 'none', 'degenerate').
    """
    fallback = ([], "none", "degenerate")
    suspect = None
    for variant in GRID_VARIANTS:
        segs = parse_timestamped(OUTPUTS / "whisper" / meeting / variant)
        if not segs:
            continue
        v = transcript_health(" ".join(s.text for s in segs)).verdict
        if v == "clean":
            return segs, variant, "clean"
        if v == "suspect" and suspect is None:
            suspect = (segs, variant, "suspect")
    return suspect or fallback


def load_partner(model: str, fname: str, meeting: str) -> list[tuple[str, str]] | None:
    """Return the tokens of a partner transcript, or None if it is missing, empty, degenerate or
    shorter than 50 words.
    """
    path = OUTPUTS / model / meeting / fname
    if not path.exists() or path.stat().st_size == 0:
        return None
    text = _read_utf8(path)
    if transcript_health(text).verdict == "degenerate":
        return None
    toks = tokenize(text)
    return toks if len(toks) >= 50 else None


# --------------------------------------------------------------------------- #
# windowing
# --------------------------------------------------------------------------- #

def build_windows(grid: list[Seg]) -> list[tuple[float, float, int, int]]:
    """Merge consecutive grid segments into windows of about TARGET_WIN s, as (t0, t1, first index,
    end index exclusive). Windows can fall outside [MIN_WIN, MAX_WIN]; the caller drops them.
    """
    windows: list[tuple[float, float, int, int]] = []
    n = len(grid)
    i = 0
    while i < n:
        i0 = i
        w0, w1 = grid[i].t0, grid[i].t1
        while (w1 - w0) < TARGET_WIN and i + 1 < n:
            if grid[i + 1].t1 - w0 > MAX_WIN:
                break
            i += 1
            w1 = grid[i].t1
        dur = w1 - w0
        if dur >= MIN_WIN or not windows:
            windows.append((round(w0, 2), round(w1, 2), i0, i + 1))
        else:
            pw0, _, pi0, _ = windows[-1]
            windows[-1] = (pw0, round(w1, 2), pi0, i + 1)
        i += 1
    return windows


# --------------------------------------------------------------------------- #
# alignment + per-window WER
# --------------------------------------------------------------------------- #

def align_map(src_low: list[str], tgt_low: list[str]):
    """Return a function mapping a src word index to a tgt word index, and the share of src words
    matched. Both come from difflib matching blocks; an index between blocks keeps the offset of
    the block before it.
    """
    sm = difflib.SequenceMatcher(a=src_low, b=tgt_low, autojunk=False)
    blocks = sm.get_matching_blocks()
    a_idx = [b.a for b in blocks]
    b_idx = [b.b for b in blocks]
    matched = sum(b.size for b in blocks)

    def mp(i: int) -> int:
        j = bisect.bisect_right(a_idx, i) - 1
        if j < 0:
            return 0
        return min(len(tgt_low), b_idx[j] + (i - a_idx[j]))

    return mp, (matched / len(src_low) if src_low else 0.0)


def window_wer(w_low: list[str], p_low: list[str]) -> float:
    if len(w_low) < 4 or len(p_low) < 4:
        return -1.0
    try:
        return min(1.0, jiwer.wer([" ".join(p_low)], [" ".join(w_low)]))  # ref=partner, hyp=whisper
    except Exception:
        return -1.0


# --------------------------------------------------------------------------- #
# candidate build
# --------------------------------------------------------------------------- #

@dataclass
class Candidate:
    segment_id: str
    meeting_id: str
    date: str
    audio_file: str
    start_s: float
    end_s: float
    duration_s: float
    difficulty: float
    difficulty_source: str
    partners_used: list = field(default_factory=list)
    partner_wer: dict = field(default_factory=dict)
    stratum: str = ""
    candidate_text: str = ""
    model_hypotheses: dict = field(default_factory=dict)
    population_weight: float = 0.0
    grid_variant: str = ""
    # filled by the human rater:
    corrected_text: str = ""
    rater: str = ""
    confidence: str = ""          # "ok" | "edge_case"
    needs_supervisor: bool = False


def _degenerate_window(low: list[str]) -> bool:
    """Return True if the window has fewer than MIN_TOKENS words, under 45% distinct words, or one
    word more than four times in a row.
    """
    if len(low) < MIN_TOKENS:
        return True
    if len(set(low)) / len(low) < 0.45:
        return True
    run = best = 1
    for a, b in zip(low, low[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best > 4


def build_candidates_for_meeting(meeting: str) -> tuple[list[Candidate], dict]:
    grid, variant, grid_verdict = pick_clean_grid(meeting)
    info = {"grid_variant": variant, "grid_verdict": grid_verdict, "partners": [], "skipped": False}
    if not grid or grid_verdict == "degenerate":
        info["skipped"] = True
        return [], info

    seg_tokens = [tokenize(s.text) for s in grid]
    prefix = [0]
    for st in seg_tokens:
        prefix.append(prefix[-1] + len(st))
    grid_orig = [o for st in seg_tokens for (o, _) in st]
    grid_low = [l for st in seg_tokens for (_, l) in st]

    maps = {}
    for model, fname in PARTNERS:
        toks = load_partner(model, fname, meeting)
        if toks is None:
            continue
        p_low = [l for (_, l) in toks]
        p_orig = [o for (o, _) in toks]
        mp, cover = align_map(grid_low, p_low)
        if cover >= 0.35:  # minimum share of grid words matched
            maps[model] = (mp, p_low, p_orig)
            info["partners"].append({"model": model, "coverage": round(cover, 3)})

    windows = build_windows(grid)
    cands: list[Candidate] = []
    for k, (t0, t1, i0, i1) in enumerate(windows):
        if not (MIN_WIN <= (t1 - t0) <= MAX_WIN):
            continue
        wi0, wi1 = prefix[i0], prefix[i1]
        w_low = grid_low[wi0:wi1]
        if _degenerate_window(w_low):
            continue
        w_text = " ".join(s.text for s in grid[i0:i1]).strip()

        partner_wer, hyps = {}, {}
        for model, (mp, p_low, p_orig) in maps.items():
            cs, ce = mp(wi0), mp(wi1)
            if ce - cs >= 4:
                wer = window_wer(w_low, p_low[cs:ce])
                if wer >= 0:
                    partner_wer[model] = round(wer, 4)
                    hyps[model] = " ".join(p_orig[cs:ce])

        if partner_wer:
            difficulty = round(sum(partner_wer.values()) / len(partner_wer), 4)
            source = "cross_model_disagreement"
        else:
            difficulty, source = -1.0, "unscored"

        seed = hyps.get(SEED_MODEL) or w_text
        hypotheses = {"whisper_large_v3": w_text}
        hypotheses.update(hyps)

        cands.append(Candidate(
            segment_id=f"{meeting}_{k:04d}", meeting_id=meeting, date=MEETING_DATE[meeting],
            audio_file=AUDIO_FILE[meeting], start_s=t0, end_s=t1, duration_s=round(t1 - t0, 2),
            difficulty=difficulty, difficulty_source=source,
            partners_used=list(partner_wer.keys()), partner_wer=partner_wer,
            candidate_text=seed, model_hypotheses=hypotheses, grid_variant=variant,
        ))
    return cands, info


# --------------------------------------------------------------------------- #
# sampling
# --------------------------------------------------------------------------- #

def sample_goldset(cands: list[Candidate], n_total: int = N_TOTAL,
                   agree_wer: float = AGREE_WER, seed: int = SEED):
    """Set each candidate's stratum (easy if difficulty <= agree_wer, else hard; unscored without a
    score) and draw up to n_total scored candidates with the population easy:hard ratio, spread
    over meetings by window count. Return (selection, stats).
    """
    rng = random.Random(seed)
    scored = [c for c in cands if c.difficulty_source == "cross_model_disagreement"]

    for c in cands:
        c.stratum = ("easy" if c.difficulty <= agree_wer else "hard") \
            if c.difficulty_source == "cross_model_disagreement" else "unscored"

    strata = {s: [c for c in scored if c.stratum == s] for s in ("easy", "hard")}
    total = len(scored) or 1
    easy_share = len(strata["easy"]) / total
    for c in scored:
        c.population_weight = round(easy_share if c.stratum == "easy" else (1 - easy_share), 4)

    def spread_by_meeting(items: list[Candidate], n: int) -> list[Candidate]:
        by: dict[str, list[Candidate]] = {}
        for c in items:
            by.setdefault(c.meeting_id, []).append(c)
        if not items or n <= 0:
            return []
        alloc = {m: len(v) / len(items) * n for m, v in by.items()}
        base = {m: int(a) for m, a in alloc.items()}
        rem = n - sum(base.values())
        for m in sorted(alloc, key=lambda k: alloc[k] - int(alloc[k]), reverse=True):
            if rem <= 0:
                break
            base[m] += 1
            rem -= 1
        out = []
        for m, v in by.items():
            out.extend(rng.sample(v, min(base.get(m, 0), len(v))))
        return out

    n_easy = min(len(strata["easy"]), round(n_total * easy_share))
    n_hard = min(len(strata["hard"]), n_total - n_easy)
    sel = spread_by_meeting(strata["easy"], n_easy) + spread_by_meeting(strata["hard"], n_hard)
    sel.sort(key=lambda c: (c.meeting_id, c.start_s))

    stats = {
        "corpus_scored_windows": len(scored),
        "agree_wer_threshold": agree_wer,
        "population_easy": len(strata["easy"]), "population_hard": len(strata["hard"]),
        "population_easy_share": round(easy_share, 4),
        "sampled_total": len(sel),
        "sampled_easy": sum(1 for c in sel if c.stratum == "easy"),
        "sampled_hard": sum(1 for c in sel if c.stratum == "hard"),
        "sampled_easy_share": round(sum(1 for c in sel if c.stratum == "easy") / max(1, len(sel)), 4),
        "sampled_by_meeting": {m: sum(1 for c in sel if c.meeting_id == m) for m in MEETINGS},
        "seed": seed,
    }
    return sel, stats


# --------------------------------------------------------------------------- #
# outputs
# --------------------------------------------------------------------------- #

def _sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()[:12] if path.exists() else "missing"


def write_outputs(sel: list[Candidate], stats: dict, meeting_info: dict):
    GOLDDIR.mkdir(exist_ok=True)
    records = [asdict(c) for c in sel]
    (GOLDDIR / "goldset_candidate.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    # Same records as a script, so a page opened from file:// can load them with <script src>.
    (GOLDDIR / "goldset_data.js").write_text(
        "window.GOLDSET = " + json.dumps(records, ensure_ascii=False) + ";", encoding="utf-8")

    inputs = {}
    for m in MEETINGS:
        for v in GRID_VARIANTS:
            p = OUTPUTS / "whisper" / m / v
            if p.exists():
                inputs[f"whisper/{m}/{v}"] = _sha1(p)
        for model, fname in PARTNERS:
            inputs[f"{model}/{m}"] = _sha1(OUTPUTS / model / m / fname)

    manifest = {
        "spec": "GROUND_TRUTH_REQUIREMENTS.md",
        "method": "loop-guarded difflib word-alignment; Whisper-lv3 grid vs Cohere + Granite-4.1",
        "window_bounds_s": [MIN_WIN, MAX_WIN], "n_total": N_TOTAL, "stats": stats,
        "meeting_info": meeting_info,
        "disclosure": (
            "LOCAL build. candidate_text is COHERE Transcribe's text for each span (Peinl: the gold "
            "'can be based on Cohere Transcribe transcriptions') — a seed to EDIT, not accept; raters "
            "must listen first. Difficulty = word-level WER of Whisper-large-v3 vs Cohere over the same "
            "span, aligned by text since the saved transcripts lack word timestamps (Granite-4.1 "
            "summarises, align coverage 0.25-0.33, so it is excluded locally). easy = WER <= "
            + f"{AGREE_WER} (models agree), hard = disagree; the 150 keep the population easy:hard ratio. "
            "Every transcript was loop-guarded with asr_quality.transcript_health; looping variants and "
            "whole meetings with no clean Whisper grid are SKIPPED locally and deferred to "
            "build_goldset.segment_audio_and_transcribe [CLUSTER], which recomputes disagreement over ALL "
            "7 models on identical clips. Gold LABELS come only from the human rater."
        ),
        "input_hashes": inputs,
    }
    (GOLDDIR / "goldset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    tasks = [{
        "data": {"audio": c.audio_file, "segment_id": c.segment_id, "meeting": c.meeting_id,
                 "stratum": c.stratum, "start_s": c.start_s, "end_s": c.end_s,
                 "asr_prefill": c.candidate_text},
        "predictions": [{"model_version": "whisper_large_v3_seed",
                         "result": [{"from_name": "transcription", "to_name": "audio",
                                     "type": "textarea", "value": {"text": [c.candidate_text]}}]}],
    } for c in sel]
    (GOLDDIR / "label_studio_tasks.json").write_text(
        json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    all_cands: list[Candidate] = []
    meeting_info = {}
    print("meeting    grid_variant                          verdict     partners            windows")
    for meeting in MEETINGS:
        mc, info = build_candidates_for_meeting(meeting)
        meeting_info[meeting] = info
        all_cands.extend(mc)
        pnames = ",".join(p["model"] for p in info["partners"]) or "-"
        print(f"  {meeting:8} {info['grid_variant']:38} {info['grid_verdict']:10}  {pnames:18} {len(mc):5}"
              + ("   [SKIPPED->cluster]" if info["skipped"] else ""))
    if not all_cands:
        print("No candidates built — check OUTPUTS path."); return
    sel, stats = sample_goldset(all_cands)
    if not sel:
        print("No scored windows: the partner transcripts are missing or not aligned. Nothing written.")
        return
    write_outputs(sel, stats, meeting_info)
    print("\n=== gold-set selection ===")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    durs = [c.duration_s for c in sel]
    print(f"\nsegment duration s: min={min(durs)} max={max(durs)} mean={round(sum(durs)/len(durs),1)} (target 5-15)")
    print(f"wrote -> {GOLDDIR/'goldset_candidate.json'}  ({len(sel)} segments)")


if __name__ == "__main__":
    main()
