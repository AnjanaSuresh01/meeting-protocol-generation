"""Builds a browser page for annotating who speaks in randomly drawn 2-second frames, as a reference for the diarization error rate (DER).

Frames are drawn at random across the meetings, in proportion to meeting length and uniformly
within each meeting (default 240 frames, seed 20260918). Because DER is a per-frame average, a
uniform frame sample gives an unbiased estimate of each DER component (missed speech, false
alarm, speaker confusion) with a confidence interval. Silent frames stay in the sample, so
missed speech and false alarm can be estimated as well as confusion. The annotator answers one
question per frame (one of the speakers, nobody / silence, two or more at once, or cannot tell),
which needs no knowledge of German. Each frame is shown with a reference voice panel: a 6-second
sample per named speaker of that meeting, cut around the middle of the speaker's longest
segment, so the task is matching a voice rather than recognizing a person. Speaker names come
from goldset/speaker_identity.json and the diarized segments from goldset/segments_v9_all.json;
the system's speakers for each frame are stored with the frame but not shown on the page.

    python build_der_app.py                    # 240 frames, all meetings
    python build_der_app.py --n 120            # fewer frames
    python build_der_app.py --meeting june22   # one meeting (--seed sets the random seed)

Writes dist/der_annotation/: index.html, frames.json and clips/ (a 2-second WAV per frame, a
second WAV with 3 s of context on each side, and ref_<meeting>_<name>.wav voice samples). The
page exports der_answers.json, which score_der.py scores.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

G = REPO / "goldset"
SEGS = G / "segments_v9_all.json"
IDENT = G / "speaker_identity.json"
OUT = REPO / "dist" / "der_annotation"
FRAME = 2.0                      # seconds judged per question
SR = 16000

AUDIO = {
    "march09": "Team-Meeting_2026-03-09.mp3",
    "april13": "Team-Meeting_2026-04-13.mp3",
    "april27": "Team-Meeting_2026-04-27.flac",
    "may11":   "Team-Meeting_2026-05-11.flac",
    "may18":   "Team-Meeting_2026-05-18.flac",
    "june01":  "Team-Meeting_2026-06-01.flac",
    "june15":  "Team-Meeting_2026-06-15.flac",
    "june22":  "Team-Meeting_2026-06-22.flac",
}


def load_audio(path: Path):
    import numpy as np
    import soundfile as sf
    from math import gcd
    from scipy.signal import resample_poly
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if sr != SR:
        g = gcd(SR, sr)
        y = resample_poly(y, SR // g, sr // g).astype("float32")
    return y


def main() -> int:
    import numpy as np
    import soundfile as sf

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--meeting")
    ap.add_argument("--seed", type=int, default=20260918)
    a = ap.parse_args()

    segs = json.loads(SEGS.read_text(encoding="utf-8"))
    ident = json.loads(IDENT.read_text(encoding="utf-8"))["mapping"]

    by_meeting: dict[str, list] = defaultdict(list)
    for s in segs:
        by_meeting[s["meeting_id"]].append(s)
    meetings = [a.meeting] if a.meeting else sorted(by_meeting)
    meetings = [m for m in meetings if m in by_meeting and (REPO / AUDIO.get(m, "")).exists()]
    if not meetings:
        print("no meetings with both segments and audio")
        return 1

    spans = {m: max(s["end_s"] for s in by_meeting[m]) for m in meetings}
    total = sum(spans.values())
    rng = random.Random(a.seed)

    # draw frames proportional to meeting length, uniformly within each meeting
    draws: list[tuple[str, float]] = []
    for m in meetings:
        k = max(1, round(a.n * spans[m] / total))
        for _ in range(k):
            draws.append((m, rng.uniform(0, max(0.1, spans[m] - FRAME))))
    rng.shuffle(draws)
    draws = draws[:a.n]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "clips").mkdir(exist_ok=True)

    # ---- reference voice panel -------------------------------------------------------
    # One labeled sample of REF_S seconds per named speaker, centered on the middle of that
    # speaker's longest segment (never starting before it), so the annotator can match a voice
    # instead of recognizing it.
    REF_S = 6.0
    refs: dict[str, list] = defaultdict(list)
    ref_spans: dict[str, list] = defaultdict(list)
    by_spk: dict[tuple, list] = defaultdict(list)
    for s_ in segs:
        by_spk[(s_["meeting_id"], s_["speaker"])].append(s_)
    for (mid_, spk), v in by_spk.items():
        if mid_ not in meetings:
            continue
        who = ident.get(mid_, {}).get(spk)
        if not who:
            continue
        best = max(v, key=lambda x: x["end_s"] - x["start_s"])
        mid_t = (best["start_s"] + best["end_s"]) / 2 - REF_S / 2
        refs[mid_].append({"name": who, "t": max(best["start_s"], mid_t)})
        ref_spans[mid_].append((max(best["start_s"], mid_t), max(best["start_s"], mid_t) + REF_S))

    items = []
    cache: dict[str, "np.ndarray"] = {}
    for i, (mid, t0) in enumerate(draws):
        if mid not in cache:
            print(f"  decoding {AUDIO[mid]} ...", flush=True)
            cache[mid] = load_audio(REPO / AUDIO[mid])
        y = cache[mid]
        i0, i1 = int(t0 * SR), int((t0 + FRAME) * SR)
        clip = y[i0:i1]
        if len(clip) < int(FRAME * SR):
            clip = np.pad(clip, (0, int(FRAME * SR) - len(clip)))
        # a little context on demand, written as a second file
        c0, c1 = max(0, int((t0 - 3) * SR)), min(len(y), int((t0 + FRAME + 3) * SR))
        name = f"{i:04d}_{mid}"
        sf.write(OUT / "clips" / f"{name}.wav", clip, SR, subtype="PCM_16")
        sf.write(OUT / "clips" / f"{name}_ctx.wav", y[c0:c1], SR, subtype="PCM_16")

        # what the system claims for this frame (overlapping segments -> several speakers)
        hyp = sorted({s["speaker"] for s in by_meeting[mid]
                      if s["start_s"] < t0 + FRAME and s["end_s"] > t0})
        names = ident.get(mid, {})
        items.append({"id": name, "meeting": mid, "t": round(t0, 2),
                      "hyp": [names.get(h, h) for h in hyp],
                      "roster": sorted(set(names.values()))})

    # write the reference voice samples (one per speaker per meeting)
    for mid_, lst in refs.items():
        if mid_ not in cache:
            print(f"  decoding {AUDIO[mid_]} for reference voices ...", flush=True)
            cache[mid_] = load_audio(REPO / AUDIO[mid_])
        y = cache[mid_]
        for r in lst:
            i0, i1 = int(r["t"] * SR), int((r["t"] + REF_S) * SR)
            seg = y[i0:i1]
            if len(seg) < int(REF_S * SR):
                seg = np.pad(seg, (0, int(REF_S * SR) - len(seg)))
            safe = r["name"].replace(" ", "_")
            sf.write(OUT / "clips" / f"ref_{mid_}_{safe}.wav", seg, SR, subtype="PCM_16")
    REFS = {m: sorted(x["name"] for x in v) for m, v in refs.items()}

    (OUT / "frames.json").write_text(json.dumps(items, ensure_ascii=False, indent=1),
                                     encoding="utf-8")

    html = (_PAGE.replace("__ITEMS__", json.dumps(items, ensure_ascii=False))
                 .replace("__REFS__", json.dumps(REFS, ensure_ascii=False)))
    (OUT / "index.html").write_text(html, encoding="utf-8")

    mb = sum(f.stat().st_size for f in (OUT / "clips").glob("*.wav")) / 1e6
    print(f"\nwrote -> {OUT.relative_to(REPO)}  ({len(items)} frames, {mb:.0f} MB of audio)")
    print(f"  {len(items)} x {FRAME:.0f}s = {len(items)*FRAME/60:.0f} minutes of listening")
    print("  open dist\\der_annotation\\index.html, answer, then Export")
    print("  then:  python score_der.py dist\\der_annotation\\der_answers.json")
    return 0


_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Who is speaking?</title>
<style>
 :root{--bg:#fbfbfa;--fg:#1c1b19;--mut:#6b6862;--line:#e3e0da;--acc:#2f6f4f}
 @media (prefers-color-scheme:dark){:root{--bg:#191817;--fg:#e9e7e3;--mut:#9a958c;--line:#33312e;--acc:#7fba97}}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
 .wrap{max-width:640px;margin:0 auto;padding:20px 16px 60px}
 h1{font-size:18px;margin:0 0 4px}
 .sub{color:var(--mut);font-size:13px;margin-bottom:18px}
 .card{border:1px solid var(--line);border-radius:12px;padding:18px;margin-bottom:14px}
 .meta{font-size:12px;color:var(--mut);margin-bottom:10px}
 audio{width:100%;margin-bottom:8px}
 button{font:inherit;padding:10px 12px;border:1px solid var(--line);border-radius:9px;
        background:var(--bg);color:var(--fg);cursor:pointer;margin:0 6px 6px 0}
 button:hover{border-color:var(--acc)}
 button.sel{background:var(--acc);color:#fff;border-color:var(--acc)}
 .panel{margin-top:14px;padding:10px;border:1px dashed var(--line);border-radius:9px}
 .refrow{display:flex;align-items:center;gap:10px;margin:4px 0}
 .refrow span{min-width:130px;font-size:14px}
 .refrow audio{height:32px;flex:1}
 .bar{height:6px;background:var(--line);border-radius:3px;overflow:hidden;margin:14px 0}
 .bar i{display:block;height:100%;background:var(--acc)}
 .row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
 .ghost{color:var(--mut);font-size:13px}
 .done{text-align:center;padding:40px 0}
</style></head><body><div class="wrap">
<h1>Who is speaking?</h1>
<div class="sub">Two seconds per clip. You do not need to understand the words &mdash; only who is
talking. Answers save in this browser as you go.</div>
<div class="bar"><i id="bar" style="width:0%"></i></div>
<div id="app"></div>
<div class="row"><button id="exp">Export answers</button>
<span class="ghost" id="cnt"></span></div>
</div>
<script>
const ITEMS=__ITEMS__;
const REFS=__REFS__;
const KEY="der_answers_v1";
let ans=JSON.parse(localStorage.getItem(KEY)||"{}");
let i=ITEMS.findIndex(x=>!(x.id in ans)); if(i<0)i=ITEMS.length;
const $=id=>document.getElementById(id);
function save(){try{localStorage.setItem(KEY,JSON.stringify(ans))}catch(e){}}
function draw(){
  $("bar").style.width=(100*Object.keys(ans).length/ITEMS.length)+"%";
  $("cnt").textContent=Object.keys(ans).length+" / "+ITEMS.length+" answered";
  if(i>=ITEMS.length){$("app").innerHTML='<div class="done"><b>All done.</b><br>'+
    'Press <b>Export answers</b> and run the scorer.</div>';return}
  const it=ITEMS[i];
  const opts=it.roster.map(n=>`<button data-v="${n}">${n}</button>`).join("")+
    `<button data-v="__none__">nobody / silence</button>`+
    `<button data-v="__multi__">two or more at once</button>`+
    `<button data-v="__unclear__">cannot tell</button>`;
  $("app").innerHTML=`<div class="card">
    <div class="meta">${it.meeting} &middot; ${Math.floor(it.t/60)}:${String(Math.floor(it.t%60)).padStart(2,"0")}
      &middot; clip ${i+1} of ${ITEMS.length}</div>
    <audio controls autoplay src="clips/${it.id}.wav"></audio>
    <details><summary class="ghost">play with 3s of context</summary>
      <audio controls src="clips/${it.id}_ctx.wav"></audio></details>
    <div class="panel"><div class="ghost" style="margin-bottom:6px">
      <b>Reference voices</b> — play these to compare. They are from a different part of the
      same meeting. You do not need to know these people; just match the voice.</div>
      ${(REFS[it.meeting]||[]).map(n=>`<div class="refrow"><span>${n}</span>
        <audio controls preload=none src="clips/ref_${it.meeting}_${n.replace(/ /g,"_")}.wav"></audio>
      </div>`).join("")}</div>
    <div style="margin-top:12px">${opts}</div>
    <div class="row" style="margin-top:10px">
      <button id="back">&larr; back</button><span class="ghost">the system's guess is hidden on purpose</span>
    </div></div>`;
  $("app").querySelectorAll("button[data-v]").forEach(b=>b.onclick=()=>{
    ans[it.id]=b.dataset.v; save(); i++; draw();
  });
  $("back").onclick=()=>{ if(i>0){i--;draw();} };
}
$("exp").onclick=()=>{
  const blob=new Blob([JSON.stringify(ans,null,1)],{type:"application/json"});
  const a=document.createElement("a");
  a.href=URL.createObjectURL(blob); a.download="der_answers.json"; a.click();
};
draw();
</script></body></html>"""


if __name__ == "__main__":
    raise SystemExit(main())
