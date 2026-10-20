"""Transcribe with Whisper the long stretches between transcript segments in which the diarization
finds speech.

A stretch lies between two consecutive segments of goldset/segments_v9_all.json, lasts at least
GAP_S seconds and overlaps the speaker turns of
pipeline_outputs_extracted/outputs/canary_diar/<meeting>/diarization_hyp.rttm for at least
MIN_SPEECH_S seconds. MODEL transcribes each stretch on the CPU from the recording named in AUDIO,
with PAD seconds of context on each side, and the results go to goldset/gap_recovered.json.

    python gap_recover.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parent
AUDIO = {
    "march09": "Team-Meeting_2026-03-09.mp3", "april13": "Team-Meeting_2026-04-13.mp3",
    "april27": "Team-Meeting_2026-04-27.flac", "may11": "Team-Meeting_2026-05-11.flac",
    "may18": "Team-Meeting_2026-05-18.flac", "june01": "Team-Meeting_2026-06-01.flac",
    "june15": "Team-Meeting_2026-06-15.flac", "june22": "Team-Meeting_2026-06-22.flac",
}
GAP_S = 30.0
MIN_SPEECH_S = 2.0
PAD = 0.5
MODEL = "primeline/whisper-large-v3-turbo-german"


def gaps():
    segs = json.loads((REPO / "goldset" / "segments_v9_all.json").read_text(encoding="utf-8"))
    by = defaultdict(list)
    for r in segs:
        by[r["meeting_id"]].append(r)
    out = []
    for mid, R in by.items():
        R.sort(key=lambda r: r["start_s"])
        rttm = [(float(p[3]), float(p[3]) + float(p[4]), p[7]) for p in
                (l.split() for l in open(REPO / "pipeline_outputs_extracted" / "outputs" /
                                         "canary_diar" / mid / "diarization_hyp.rttm"))]
        for a, b in zip(R, R[1:]):
            s, e = a["end_s"], b["start_s"]
            if e - s < GAP_S:
                continue
            spk = defaultdict(float)
            for rs, re_, k in rttm:
                o = min(e, re_) - max(s, rs)
                if o > 0:
                    spk[k] += o
            if sum(spk.values()) >= MIN_SPEECH_S:
                out.append({"meeting": mid, "start_s": s, "end_s": e,
                            "diarized_speech_s": round(sum(spk.values()), 1),
                            "speakers": {k: round(v, 1) for k, v in spk.items()},
                            "next_segment_speaker": b["speaker"],
                            "next_segment_text": b["text"][:120]})
    return out


def load(mid, s, e):
    path = REPO / AUDIO[mid]
    info = sf.info(str(path))
    sr = info.samplerate
    a, _ = sf.read(str(path), start=int(max(0, s - PAD) * sr), stop=int((e + PAD) * sr),
                   dtype="float32", always_2d=True)
    a = a.mean(axis=1)
    if sr != 16000:
        import torchaudio.functional as F
        a = F.resample(torch.from_numpy(a), sr, 16000).numpy()
    return a


def main() -> int:
    from transformers import pipeline
    asr = pipeline("automatic-speech-recognition", model=MODEL, torch_dtype=torch.bfloat16,
                   device="cpu")
    todo = gaps()
    print(f"{len(todo)} gaps with diarized speech, {sum(g['diarized_speech_s'] for g in todo):.0f} s")
    for g in todo:
        audio = load(g["meeting"], g["start_s"], g["end_s"])
        r = asr({"raw": audio, "sampling_rate": 16000}, chunk_length_s=30,
                generate_kwargs={"language": "german", "task": "transcribe"})
        g["recovered_text"] = r["text"].strip()
        g["model"] = MODEL
        print(f"\n{g['meeting']} {g['start_s'] / 60:.2f}-{g['end_s'] / 60:.2f}m "
              f"{g['speakers']}\n  RECOVERED: {g['recovered_text']}\n  next seg : "
              f"{g['next_segment_text']}")
    (REPO / "goldset" / "gap_recovered.json").write_text(
        json.dumps(todo, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
