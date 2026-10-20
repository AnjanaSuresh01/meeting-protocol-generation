"""Repair the speaker transcript of one meeting with functions and thresholds from
speaker_turns_v11.py, verify_short_turns.py and gap_recover.py: sentence rule, short-turn voice
check, turn shaping, and re-transcription of long word gaps that contain diarized speech.

Inputs are the word list with start, end and diarized speaker per word
(transcript_canary_ts.json), the RTTM speaker turns (diarization_hyp.rttm) and the recording.
Given out.txt, the script writes the repaired transcript there and the report to
out.repair.json; it always prints a summary.

    python transcript_repair.py <words.json> <diarization.rttm> <audio> [out.txt]
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from speaker_turns_v11 import absorb_short, hhmmss, runs, sentence_vote, split_long  # noqa: E402

from gap_recover import GAP_S, MIN_SPEECH_S, PAD, MODEL as GAP_MODEL  # noqa: E402
SR = 16000


# Inputs
def load_words(path: Path) -> list[dict]:
    words = json.loads(Path(path).read_text(encoding="utf-8"))
    words = [{"word": w["word"], "start": float(w["start"]), "end": float(w["end"]),
              "speaker": w.get("speaker") or ""} for w in words if str(w.get("word", "")).strip()]
    return sorted(words, key=lambda w: (w["start"], w["end"]))


def load_rttm(path: Path) -> list[tuple[float, float, str]]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) >= 8 and p[0] == "SPEAKER":
            s = float(p[3])
            out.append((s, s + float(p[4]), p[7]))
    return sorted(out)


class Audio:
    """Read short mono 16 kHz float32 clips of a recording without loading the whole file."""

    def __init__(self, path: Path):
        import soundfile as sf
        self.path = str(path)
        self.sr = sf.info(self.path).samplerate

    def clip(self, s: float, e: float):
        import numpy as np
        import soundfile as sf
        s = max(0.0, s)
        a, _ = sf.read(self.path, start=int(s * self.sr), stop=max(int(s * self.sr) + 1, int(e * self.sr)),
                       dtype="float32", always_2d=True)
        a = a.mean(axis=1)
        if self.sr != SR:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(SR, self.sr)
            a = resample_poly(a, SR // g, self.sr // g).astype(np.float32)
        return a


# Short-turn voice check
def verify_short(words: list[dict], y, emb, log: list) -> list[dict]:
    """For the first two and last two sentences of each speaker run, relabel each sentence of MIN_S
    to MAX_S seconds by voice similarity, and append each move to log. A sentence moves only to the
    speaker with the most runs, or away from a speaker with no run of at least LONG_S seconds,
    and only if the similarity gain is at least MARGIN.
    """
    import numpy as np
    from verify_short_turns import LONG_S, MARGIN, MAX_S, MIN_S, sentences
    clip = y.clip
    rs = absorb_short(runs(words))
    vecs = defaultdict(list)
    for spk, s, e, ws in rs:
        if e - s >= LONG_S and len(vecs[spk]) < 25:
            mid_t = (s + e) / 2
            vecs[spk].append(emb(clip(mid_t - 4, mid_t + 4)))
    cent = {k: np.mean(v, axis=0) / np.linalg.norm(np.mean(v, axis=0)) for k, v in vecs.items() if v}
    turns_per = defaultdict(int)
    for r in rs:
        turns_per[r[0]] += 1
    if not turns_per or not cent:
        return words
    chair = max(turns_per, key=turns_per.get)
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
            sliver = spk not in cent
            own = sims.get(spk, -1.0)
            if best != spk and (sliver or best == chair) and sims[best] - own >= MARGIN:
                for w in sw:
                    w["speaker"] = best
                log.append({"t": round(a, 1), "from": spk, "to": best, "margin": round(sims[best] - own, 3),
                            "text": " ".join(w["word"] for w in sw)})
    return words


# Gap recovery
def find_gaps(words: list[dict], rttm: list[tuple[float, float, str]]) -> list[dict]:
    """Return the pauses of at least GAP_S seconds between two consecutive words in which
    the diarization marks at least MIN_SPEECH_S seconds of speech.
    """
    out = []
    for a, b in zip(words, words[1:]):
        s, e = a["end"], b["start"]
        if e - s < GAP_S:
            continue
        spk = defaultdict(float)
        for rs, re_, k in rttm:
            o = min(e, re_) - max(s, rs)
            if o > 0:
                spk[k] += o
        if sum(spk.values()) >= MIN_SPEECH_S:
            out.append({"start_s": round(s, 3), "end_s": round(e, 3),
                        "diarized_speech_s": round(sum(spk.values()), 1),
                        "speakers": {k: round(v, 1) for k, v in spk.items()}})
    return out


def gap_turns(g: dict, rttm) -> list[tuple[float, str, str]]:
    """Split the recovered text of a gap into sentences and share them among the speakers with
    at least 1 s of speech in the gap, in the order in which they first speak and in proportion
    to their speech. Returns (time, speaker, text) tuples.
    """
    order = []
    for s, e, k in rttm:
        if e > g["start_s"] and s < g["end_s"] and k not in order and g["speakers"].get(k, 0) >= 1:
            order.append(k)
    sents = [x for x in re.split(r"(?<=[.?!])\s+", g["recovered_text"]) if x]
    total = sum(g["speakers"][k] for k in order) or 1
    out, i = [], 0
    for n, k in enumerate(order):
        share = g["speakers"][k] / total
        take = len(sents) - i if n == len(order) - 1 else max(1, round(share * len(sents)))
        chunk = sents[i:i + take]
        i += take
        if chunk:
            t = g["start_s"] + (g["end_s"] - g["start_s"]) * (i - len(chunk)) / max(1, len(sents))
            out.append((t, k, " ".join(chunk)))
    return out


CACHE = REPO / "cache_gap_recovery.json"


def recover(gaps: list[dict], y) -> list[dict]:
    """Transcribe each gap, padded by PAD seconds on both sides, with GAP_MODEL on the CPU,
    and set its recovered_text and model. Texts are cached in CACHE by recording file name,
    span and model.
    """
    if not gaps:
        return gaps
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    key = lambda g: f"{Path(y.path).name}|{g['start_s']:.3f}|{g['end_s']:.3f}|{GAP_MODEL}"
    todo = [g for g in gaps if key(g) not in cache]
    if todo:
        import torch
        from transformers import pipeline
        asr = pipeline("automatic-speech-recognition", model=GAP_MODEL, torch_dtype=torch.bfloat16, device="cpu")
        for g in todo:
            a = y.clip(g["start_s"] - PAD, g["end_s"] + PAD)
            r = asr({"raw": a, "sampling_rate": SR}, chunk_length_s=30,
                    generate_kwargs={"language": "german", "task": "transcribe"})
            cache[key(g)] = r["text"].strip()
            CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
        del asr
    for g in gaps:
        g["recovered_text"] = cache[key(g)]
        g["model"] = GAP_MODEL
    return gaps


# Full repair
def repair(words_path: Path, rttm_path: Path, audio_path: Path, *, verify: bool = True,
           gaps: bool = True) -> tuple[str, dict]:
    """Repair the transcript from the word list, the RTTM and the recording, in this order:
    sentence rule, short-turn voice check (verify), turn shaping, gap recovery (gaps). Returns
    the text, one [HH:MM:SS] SPEAKER_NN: text line per turn, and a report dict.
    """
    words = load_words(words_path)
    rttm = load_rttm(rttm_path)
    before = [w["speaker"] for w in words]
    words = sentence_vote(words)
    report = {"words": len(words),
              "sentence_rule_words": sum(1 for b, w in zip(before, words) if b != w["speaker"])}
    y = None
    if verify or gaps:
        y = Audio(audio_path)
    moved: list = []
    if verify:
        from verify_short_turns import Embedder
        emb = Embedder()
        words = verify_short(words, y, emb, moved)
        del emb  # free WavLM before Whisper is loaded for gap recovery
        import gc
        gc.collect()
    report["short_turn_moves"] = moved
    turns = []
    for r in absorb_short(runs(words)):
        turns.extend(split_long(r))
    lines = [(s, spk, txt) for spk, s, e, txt in turns]
    found = find_gaps(words, rttm) if gaps else []
    if found:
        recover(found, y)
        for g in found:
            lines.extend(gap_turns(g, rttm))
    report["turns"] = len(turns)
    report["gaps"] = found
    report["gap_seconds"] = round(sum(g["end_s"] - g["start_s"] for g in found), 1)
    report["gap_words"] = sum(len(g.get("recovered_text", "").split()) for g in found)
    lines.sort(key=lambda x: x[0])
    text = "\n".join(f"[{hhmmss(t)}] {spk}: {txt}" for t, spk, txt in lines) + "\n"
    return text, report


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) < 4:
        print("usage: python transcript_repair.py <words.json> <diarization.rttm> <audio> [out.txt]")
        return 1
    words, rttm, audio = (Path(x) for x in sys.argv[1:4])
    text, rep = repair(words, rttm, audio)
    if len(sys.argv) > 4:
        Path(sys.argv[4]).write_text(text, encoding="utf-8")
        Path(sys.argv[4]).with_suffix(".repair.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1),
                                                                 encoding="utf-8")
    print(f"{rep['words']} words, {rep['turns']} turns, sentence rule {rep['sentence_rule_words']} words, "
          f"{len(rep['short_turn_moves'])} short sentences moved, {len(rep['gaps'])} gaps "
          f"({rep['gap_seconds']} s, {rep['gap_words']} words recovered)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
