"""Cut the segments out of the meeting recordings as MP3 clips.

boundary_lib.cut adds the silence the recording does not supply at a clip's edges and fades
into it, so a clip does not end in a fragment of the next speaker. Without that silence,
recognizers tend to guess a final word. The clips are MP3 because
Safari and the browsers on iOS cannot play Ogg.

    python cut_clips_v9.py                  # goldset/goldset_v9_draw.json -> goldset/clips_v9/clips/
    python cut_clips_v9.py --all            # every segment of goldset/segments_v9_all.json
    python cut_clips_v9.py --segments FILE --out DIR
"""

from __future__ import annotations

import argparse
import io
import json
import pathlib
import sys

import soundfile as sf

import boundary_lib as B

G = pathlib.Path("goldset")


def write_mp3(path: pathlib.Path, y, sr: int = B.SR) -> int:
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="MP3", subtype="MPEG_LAYER_III")
    raw = buf.getvalue()
    path.write_bytes(raw)
    return len(raw)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments", default=str(G / "goldset_v9_draw.json"))
    ap.add_argument("--all", action="store_true",
                    help="cut every v9 segment instead of the drawn 150")
    ap.add_argument("--out", default=str(G / "clips_v9"))
    a = ap.parse_args()

    src = str(G / "segments_v9_all.json") if a.all else a.segments
    segs = json.loads(pathlib.Path(src).read_text(encoding="utf-8"))
    out = pathlib.Path(a.out)
    (out / "clips").mkdir(parents=True, exist_ok=True)

    by_meeting: dict[str, list[dict]] = {}
    for s in segs:
        by_meeting.setdefault(s["meeting_id"], []).append(s)

    total = 0
    for m in sorted(by_meeting):
        y = B.waveform(m)            # one meeting in memory at a time, as the cache expects
        n = 0
        for s in by_meeting[m]:
            clip = B.cut(m, s, y)
            total += write_mp3(out / "clips" / f"{s['segment_id']}.mp3", clip)
            n += 1
        print(f"  {m:<9} {n:>4} clips")
    print(f"\n{len(segs)} clips -> {out / 'clips'}   ({total / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
