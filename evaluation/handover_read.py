"""Prints the evidence for each reference topic whose owner changed since last week, for reading by hand.

handover_test.py decides two things automatically: that a topic of this week is the same topic as
one in last week's protocol (bag-of-words Jaccard >= 0.10), and where that topic sits in a generated
protocol (block matcher with a keyword fallback). Either step can be wrong, and with few cases a few
such errors decide the result, so this script prints every case for reading instead of scoring it.

A case is a reference bullet with a person (attribution_baselines.read_human, Backlog excluded, at
least three words in its bag of words) whose best Jaccard match among last week's non-Backlog items
reaches 0.10 and belongs to a different person. For each case it prints the meeting, the reference
owner, up to four keywords from the bullet's first line, the bullet, the matched item of last week
with its score and owner, and, for each arm, up to six lines of
outputs_deliverable/<arm>/<meeting>/protocol_pro.md that contain a keyword, each with its '## '
person section.

    python handover_read.py v6 v10    # arms to show; default v10

Prints to stdout only.
"""
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from attribution_baselines import MIDS, PREV, bag, jac, read_any, read_human   # noqa: E402
from evaluate_speaker_attribution import person_key                            # noqa: E402
import handover_test as HT                                                     # noqa: E402

STOP = set("diese woche noch nicht auch soll wird wurde haben macht gemacht will sich".split())


def keywords(text):
    ws = re.findall(r"[A-Za-zÄÖÜäöüß0-9][\w.-]{3,}", text.splitlines()[0])
    return [w for w in ws if w.lower() not in STOP][:4]


def where(path: Path, kws):
    """Returns every line of the protocol that contains one of the keywords, with its person section."""
    out, who = [], None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## ") and not line.startswith("###"):
            who = person_key(line[3:])
        elif any(k.lower() in line.lower() for k in kws):
            out.append((who, line.strip()[:95]))
    return out


def main():
    arms = sys.argv[1:] or ["v10"]
    n = 0
    for mid in MIDS:
        human = [h for h in read_human(mid) if h[0]]
        prev = [(p, t) for p, t in read_any(PREV[mid]) if p and not HT.is_backlog(t)]
        for i, (hp, ht) in enumerate(human):
            if HT.is_backlog(ht) or len(bag(ht)) < 3:
                continue
            best, bs = None, 0.0
            for pp, pt in prev:
                s = jac(bag(ht), bag(pt))
                if s > bs:
                    bs, best = s, (pp, pt)
            if bs < 0.10 or best[0] == hp:
                continue
            n += 1
            kws = keywords(ht)
            print("=" * 100)
            print(f"[{n}] {mid}   truth: {hp}   keywords: {kws}")
            print(f"  REF THIS WEEK : {' / '.join(l.strip() for l in ht.splitlines()[:3])[:160]}")
            print(f"  MATCHED LAST WEEK (Jaccard {bs:.2f}) under {best[0]}: "
                  f"{' / '.join(l.strip() for l in best[1].splitlines()[:3])[:160]}")
            for a in arms:
                hits = where(REPO / f"outputs_deliverable/{a}/{mid}/protocol_pro.md", kws)
                print(f"  {a}:")
                for who, line in hits[:6]:
                    print(f"      [{who}] {line}")
                if not hits:
                    print("      (no line with these keywords)")


if __name__ == "__main__":
    main()
