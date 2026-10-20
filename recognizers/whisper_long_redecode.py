"""Fill the 219 clips that the three Whisper models of NB_v10_asr.ipynb left empty.

--merge reads the texts NB_whisper_long.ipynb wrote, from
goldset/whisper_long_redecode/<system>.json (step 3 of README.md moves them there), and writes them into goldset/clip_asr_v10/<system>__v10.json and keeps a
backup of each file. It fills only the entries of goldset/whisper_empty_segments.json and stops
if one of them is not empty. Without --merge, the script decodes the clips itself on the CPU,
with the same pipeline call as the notebook: clips longer than 28 s are split at their quietest
points (asr_chunks.split). It saves after every clip, so it can be restarted.

    python whisper_long_redecode.py --merge
    python whisper_long_redecode.py whisper_turbo [--limit 2] [--fp32]
"""
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import asr_chunks

IDS = {"whisper_large_v3": "openai/whisper-large-v3",
       "whisper_turbo": "openai/whisper-large-v3-turbo",
       "whisper_turbo_de": "primeline/whisper-large-v3-turbo-german"}
OUT = REPO / "goldset" / "whisper_long_redecode"
EMPTY = json.loads((REPO / "goldset" / "whisper_empty_segments.json").read_text(encoding="utf-8"))
CLIPS = REPO / "goldset" / "clips_v10"


def decode(system: str, limit: int | None = None) -> None:
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    torch.set_num_threads(max(1, (torch.get_num_threads() or 8)))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{system}.json"
    done = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    todo = [k for k in EMPTY if k not in done][:limit]
    if not todo:
        print(f"{system}: all {len(done)} done"); return
    dt = torch.float32 if "--fp32" in sys.argv else torch.bfloat16  # bf16 by default to halve memory; --fp32 can be faster on a CPU
    m = AutoModelForSpeechSeq2Seq.from_pretrained(IDS[system], dtype=dt, low_cpu_mem_usage=True)
    pr = AutoProcessor.from_pretrained(IDS[system])
    pipe = pipeline("automatic-speech-recognition", model=m, tokenizer=pr.tokenizer,
                    feature_extractor=pr.feature_extractor, device="cpu", dtype=dt)
    t0 = time.time(); audio_s = 0.0
    for i, k in enumerate(todo, 1):
        a, sr = sf.read(str(CLIPS / f"{k}.mp3"), dtype="float32")
        if a.ndim > 1:
            a = a.mean(axis=1)
        assert sr == 16000, (k, sr)
        pieces = asr_chunks.split(a, sr)
        texts = [pipe({"raw": p, "sampling_rate": sr}, generate_kwargs={"language": "german"})["text"]
                 for p in pieces]
        done[k] = {"text": asr_chunks.join(texts), "pieces": len(pieces), "seconds": round(len(a) / sr, 1)}
        audio_s += len(a) / sr
        path.write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding="utf-8")
        el = time.time() - t0
        print(f"  {system} {i}/{len(todo)} {k} {len(pieces)} piece(s)  "
              f"{el:.0f}s elapsed, {audio_s/el:.2f}x realtime", flush=True)


def merge() -> None:
    for system in IDS:
        red = json.loads((OUT / f"{system}.json").read_text(encoding="utf-8"))
        missing = [k for k in EMPTY if not red.get(k, {}).get("text", "").strip()]
        assert not missing, f"{system}: {len(missing)} still empty, e.g. {missing[:3]} — finish decoding first"
        tgt = REPO / "goldset" / "clip_asr_v10" / f"{system}__v10.json"
        bak = tgt.with_name(f"{system}__v10.BEFORE_long_redecode_2026-09-25.json")
        if not bak.exists():
            shutil.copy2(tgt, bak)
        d = json.loads(tgt.read_text(encoding="utf-8"))
        for k in EMPTY:
            assert not (d.get(k) or "").strip(), f"{system} {k} is not empty any more — refusing to overwrite"
            d[k] = red[k]["text"]
        tgt.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{system}: {len(EMPTY)} segments filled -> {tgt.name}  (backup {bak.name})")


if __name__ == "__main__":
    if "--merge" in sys.argv:
        merge()
    else:
        lim = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
        decode(sys.argv[1], lim)
