"""Does whisper_turbo stay first on clips where the seeded references give no free zeros?

The gold-set references were seeded from recognizer output, so a system can score a clip as
error-free because the reader kept its text. The script prints the micro WER of the six
systems on all eligible clips, on the clips where no system matches the reference exactly, on
the two-reader clips and on the reader-edited clips, then the exact matches per system.

    python check_turbo_seeding.py
"""
import json
import statistics
import sys
from pathlib import Path

R = Path(__file__).resolve().parent
sys.path.insert(0, str(R))
import score_v10 as S

d = json.loads((R / "goldset/goldset_v10.json").read_text(encoding="utf-8"))
segs = [s for s in d["segments"] if s["scoring"]["eligible"]]
SYS = ["whisper_turbo", "whisper_large_v3", "whisper_turbo_de", "cohere", "canary_diar", "granite41"]
print("eligible clips:", len(segs))
dur = [s["duration_s"] for s in segs]
print(f"clip duration: median {statistics.median(dur):.1f} s, mean {statistics.mean(dur):.1f} s, "
      f"min {min(dur):.1f}, max {max(dur):.1f}")


def norm(x):
    return S.normalize(x, S.CFG)


def show(title, sub):
    print(f"\n{title}  (n={len(sub)})")
    for name in SYS:
        pairs = [(s["reference_text"], s["model_hypotheses"][name]) for s in sub]
        w = S.micro_wer(pairs)
        print(f"  {name:18s} {w}")


show("ALL eligible", segs)

# clips where no system's output equals the reference after normalization
none_equal = [s for s in segs
              if all(norm(s["model_hypotheses"][n]) != norm(s["reference_text"]) for n in SYS)]
show("NO system equals the reference (no free zeros for anyone)", none_equal)

two = [s for s in segs if s["resolution"].get("readers") == 2]
show("TWO readers", two)

edited = [s for s in segs if s["bias"].get("rater_edited")]
show("rater_edited = True", edited)

# free zeros per system
print("\nfree zeros (ref == hyp after normalization):")
for name in SYS:
    print(f"  {name:18s} {sum(norm(s['model_hypotheses'][name]) == norm(s['reference_text']) for s in segs)}")
