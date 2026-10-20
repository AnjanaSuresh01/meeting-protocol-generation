"""Transcription with speaker labels: Canary-1B-v2 word timestamps, pyannote speaker turns, and
each word assigned to the speaker whose turn covers its midpoint.

process_meeting() writes transcript_canary_ts.json, transcript_speakers.txt,
diarization_hyp.rttm and goldset_segments_canary.json for one recording. NeMo, pyannote,
librosa, soundfile and torch are imported inside the functions that use them, so the module
imports with the standard library alone.

On JupyterHub, notebooks 1a and 1b (pipeline/cluster/) call these functions in two
environments. main() needs a config.json (keys base_output and meetings) that is not part of
the repository, NeMo and pyannote in one environment, and HF_TOKEN. Without config.json it
points to the notebooks:
    python -m thesis_pipeline.canary_diarize
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

# Data types

@dataclass
class Word:
    word: str
    start: float
    end: float
    speaker: str = ""

    @property
    def mid(self) -> float:
        return (self.start + self.end) / 2.0


@dataclass
class Turn:
    start: float
    end: float
    speaker: str


@dataclass
class Segment:
    segment_id: str
    meeting_id: str
    speaker: str
    start_s: float
    end_s: float
    duration_s: float
    text: str
    n_words: int


# Canary transcription

def ensure_16k_mono(path: str, tmp_dir: str = "/tmp") -> str:
    """Convert the audio at `path` to a 16 kHz mono WAV, `<tmp_dir>/<stem>_16k_mono.wav`, and
    return the new path.
    """
    import librosa, soundfile as sf
    from pathlib import Path as _P
    audio, _ = librosa.load(path, sr=16000, mono=True)
    out = f"{tmp_dir}/{_P(path).stem}_16k_mono.wav"
    sf.write(out, audio, 16000)
    return out


def transcribe_words(model, wav_path: str, lang: str = "de") -> list[Word]:
    """Transcribe a WAV file with Canary in language `lang`, with punctuation and capitalization,
    and return its words with start and end times in seconds.
    """
    # pnc must be the string "yes" or "no"; a bool raises AssertionError in the prompt formatter.
    # NeMo chunks audio longer than 40 s by itself, so one call covers a whole recording.
    out = model.transcribe([wav_path], source_lang=lang, target_lang=lang,
                           timestamps=True, pnc="yes")
    words: list[Word] = []
    for w in out[0].timestamp["word"]:
        txt = (w.get("word") or w.get("segment") or "").strip()
        if txt:
            words.append(Word(txt, float(w["start"]), float(w["end"])))
    return words


def load_canary(model_name: str = "nvidia/canary-1b-v2"):
    """Load the NeMo model (default Canary-1B-v2) in eval mode, on the GPU if there is one."""
    import torch
    from nemo.collections.asr.models import ASRModel
    model = ASRModel.from_pretrained(model_name=model_name, map_location=torch.device("cpu"))
    return model.cuda().eval() if torch.cuda.is_available() else model.eval()


# pyannote diarization

def diarize(wav_path: str, hf_token: str | None = None,
            model_name: str = "pyannote/speaker-diarization-community-1") -> list[Turn]:
    """Run a pyannote diarization pipeline (default community-1) on a WAV file and return the
    speaker turns. Works with the pyannote 4.x and 3.x APIs.
    """
    import sys
    if sys.modules.get("torchaudio", "x") is None:
        del sys.modules["torchaudio"]  # undo the NeMo-load workaround (a None entry that blocks torchaudio); pyannote needs it
    # pyannote.audio 4.x sends usage metrics by default; switch them off.
    import os
    os.environ["PYANNOTE_METRICS_ENABLED"] = "0"
    from pyannote.audio import Pipeline
    try:
        from pyannote.audio.telemetry import set_telemetry_metrics
        set_telemetry_metrics(False)
    except ImportError:
        pass
    try:
        pipe = Pipeline.from_pretrained(model_name, token=hf_token)           # pyannote >= 4.0
    except TypeError:
        pipe = Pipeline.from_pretrained(model_name, use_auth_token=hf_token)  # pyannote 3.x
    try:
        import torch
        if torch.cuda.is_available():
            pipe.to(torch.device("cuda"))
    except Exception:
        pass
    output = pipe(wav_path)
    annotation = getattr(output, "speaker_diarization", output)   # 4.x wraps; 3.x is the Annotation
    return [Turn(float(seg.start), float(seg.end), str(spk))
            for seg, _, spk in annotation.itertracks(yield_label=True)]


# Word-to-speaker assignment

def assign_speakers(words: list[Word], turns: list[Turn]) -> list[Word]:
    """Give each word the speaker of the turn that covers its midpoint (largest overlap wins), or
    of the nearest turn if it falls in a gap. Changes `words` in place and returns it.
    """
    if not turns:
        for w in words:
            w.speaker = "SPK?"
        return words
    turns = sorted(turns, key=lambda t: t.start)
    for w in words:
        m = w.mid
        covering = [t for t in turns if t.start <= m <= t.end]
        if covering:
            w.speaker = max(covering, key=lambda t: min(t.end, w.end) - max(t.start, w.start)).speaker
        else:
            w.speaker = min(turns, key=lambda t: 0.0 if t.start <= m <= t.end
                            else min(abs(t.start - m), abs(t.end - m))).speaker
    return words


# Outputs

def speaker_attributed_transcript(words: list[Word]) -> str:
    """Join runs of same-speaker words into lines `[HH:MM:SS] SPEAKER: text`."""
    lines, buf, spk, t0 = [], [], None, None
    for w in words:
        if w.speaker != spk and buf:
            lines.append(f"[{_hms(t0)}] {spk}: {' '.join(buf)}")
            buf = []
        if w.speaker != spk:
            spk, t0 = w.speaker, w.start
        buf.append(w.word)
    if buf:
        lines.append(f"[{_hms(t0)}] {spk}: {' '.join(buf)}")
    return "\n".join(lines)


def write_rttm(turns: list[Turn], uri: str, path: Path) -> None:
    """Write the turns to `path` as RTTM lines sorted by start time, with `uri` as the file ID."""
    lines = [f"SPEAKER {uri} 1 {t.start:.3f} {t.end - t.start:.3f} <NA> <NA> {t.speaker} <NA> <NA>"
             for t in sorted(turns, key=lambda t: t.start)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_speaker_pure_segments(words: list[Word], meeting_id: str,
                                min_s: float = 5.0, target_s: float = 10.0,
                                max_s: float = 15.0, floor_s: float = 3.0) -> list[Segment]:
    """Cut the words into single-speaker clips that start and end on word boundaries.
    A clip ends at a speaker change, when it reaches `target_s` seconds, or before it would pass
    `max_s`; clips shorter than `floor_s` are dropped.
    """
    segs: list[Segment] = []
    i, n, k = 0, len(words), 0
    while i < n:
        spk = words[i].speaker
        j, seg_start = i, words[i].start
        while j + 1 < n and words[j + 1].speaker == spk \
                and (words[j + 1].end - seg_start) <= max_s:
            if (words[j].end - seg_start) >= target_s:
                break
            j += 1
        chunk = words[i:j + 1]
        dur = chunk[-1].end - chunk[0].start
        if dur >= floor_s:
            segs.append(Segment(
                segment_id=f"{meeting_id}_{k:04d}", meeting_id=meeting_id, speaker=spk,
                start_s=round(chunk[0].start, 2), end_s=round(chunk[-1].end, 2),
                duration_s=round(dur, 2),
                text=" ".join(w.word for w in chunk), n_words=len(chunk)))
            k += 1
        i = j + 1
    return segs


def _hms(s: float) -> str:
    s = int(s or 0)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


# Run all steps

def process_meeting(model, meeting_id: str, wav_path: str, out_dir: Path,
                    hf_token: str | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = ensure_16k_mono(wav_path)
    words = transcribe_words(model, wav_path)
    turns = diarize(wav_path, hf_token=hf_token)
    assign_speakers(words, turns)

    (out_dir / "transcript_canary_ts.json").write_text(
        json.dumps([asdict(w) for w in words], ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "transcript_speakers.txt").write_text(
        speaker_attributed_transcript(words), encoding="utf-8")
    write_rttm(turns, meeting_id, out_dir / "diarization_hyp.rttm")
    segs = build_speaker_pure_segments(words, meeting_id)
    (out_dir / "goldset_segments_canary.json").write_text(
        json.dumps([asdict(s) for s in segs], ensure_ascii=False, indent=2), encoding="utf-8")

    return {"meeting": meeting_id, "words": len(words), "turns": len(turns),
            "speakers": len({t.speaker for t in turns}), "segments": len(segs)}


def main():
    import os
    repo = Path(__file__).resolve().parent.parent
    if not (repo / "config.json").exists():
        raise SystemExit(f"{repo / 'config.json'} not found. On JupyterHub, run the notebooks "
                         "pipeline/cluster/step1a_canary_transcribe.ipynb and "
                         "pipeline/cluster/step1b_pyannote_speakers.ipynb instead.")
    cfg = json.loads((repo / "config.json").read_text(encoding="utf-8"))
    hf_token = os.environ.get("HF_TOKEN")
    model = load_canary()
    base = Path(cfg["base_output"]) / "canary_diar"
    for m in cfg["meetings"]:
        info = process_meeting(model, m["id"], m["audio"], base / m["id"], hf_token=hf_token)
        print(f"{info['meeting']:10s} words={info['words']:5d} turns={info['turns']:4d} "
              f"speakers={info['speakers']} segments={info['segments']}")


if __name__ == "__main__":
    main()
