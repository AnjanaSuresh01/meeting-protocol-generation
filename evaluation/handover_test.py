"""Attribution accuracy on reference topics whose owner changed since the previous week.

The no-audio baseline credits each topic to whoever owned it in last week's protocol. Pooled over
all topics it is right for free whenever the owner stays the same, so a pooled comparison has
little power. This script restricts the test to "changed-owner" topics: a reference topic whose
best match in last week's protocol (Jaccard overlap of word bags without stop words >= 0.10)
sits under a different person. On these topics the baseline is wrong by construction, so a
correct attribution can only come from the audio. Backlog topics are excluded, because the
system does not produce the Backlog.

For each arm (v6, v8, v9, v10) every changed-owner topic is located in
outputs_deliverable/<arm>/<meeting>/protocol_pro.md with attribution_blocklevel.block_hits. A
topic the block matcher cannot locate falls back to the person section that contains a
distinctive word of its first line; a topic located in neither way is a coverage miss and is not
counted. Each arm's total is tested with a one-sided binomial test against chance, the mean over
meetings of 1 / (number of people in the room).

Limitation: the Jaccard >= 0.10 match can be met by filler words alone (for example
"diese Woche"), so many changed-owner topics are new topics rather than handovers, and the
p-values are not reliable.

    python handover_test.py    -> stdout only
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

from attribution_baselines import (MIDS, PREV, bag, jac, read_any, read_human,
                                      read_generated, TH)
# block_hits, not topic_hits: topic_hits pools a person's whole section into one bag, so the
# largest section wins the argmax by coincidence. block_hits matches block against block and
# takes the margin between persons, so section size does not decide the match.
from attribution_blocklevel import block_hits
from pathlib import Path

REPO = Path(__file__).resolve().parent
ARMS = ("v6", "v8", "v9", "v10")
CUT = 0.10                       # same-topic threshold against last week, as in the ladder

# The Backlog is excluded: the system is not meant to produce it, so scoring Backlog topics would
# count their absence as attribution errors. evaluate_protocols.strip_backlog does the same for
# the post-edit measures.
_BACKLOG = re.compile(r"^\s*(\*\*)?backlog\b", re.I)


def is_backlog(topic: str) -> bool:
    first = topic.splitlines()[0] if topic.strip() else ""
    return bool(_BACKLOG.match(first))


_STOP = set("und der die das für mit von den dem des ist sind auf aus bei nach über ein eine "
            "einen einem einer noch nicht auch werden wurde wird kann soll hat haben zu im in "
            "am an als dann diese dieser".split())


def _locate_by_keyword(human_topic: str, generated: str) -> str | None:
    """Return the person of the '##' section where a distinctive word of the topic's first line
    first occurs, trying the longest words first, or None. Used only when the block matcher could
    not locate the topic, so that a failure of the matcher is not counted as an attribution error
    of the pipeline."""
    from evaluate_speaker_attribution import person_key
    first_line = human_topic.splitlines()[0] if human_topic.strip() else ""
    words = [w for w in re.findall(r"[A-Za-zÄÖÜäöüß][\w-]{4,}", first_line)
             if w.lower() not in _STOP]
    if not words:
        return None
    # longest first: a long, capitalized, project-like token discriminates best
    for w in sorted(set(words), key=lambda x: (-len(x), x)):
        who = None
        for line in generated.splitlines():
            if line.startswith("## ") and not line.startswith("###"):
                who = line[3:].strip()
            elif w.lower() in line.lower() and who:
                return person_key(who)
    return None


def main() -> int:
    print("=" * 94)
    print("THE HANDOVER TEST — topics that CHANGED OWNER since last week")
    print("=" * 94)
    print("  On these, 'copy last week's owner' is wrong by construction (0% by definition).")
    print("  Anything the pipeline gets right here came from the audio and nowhere else.\n")
    print("  !! RESULT WITHDRAWN. Most of these 'changed-owner' topics are NEW topics that")
    print("  !! matched last week only through filler words; read one by one, v10 put 7 of 13 under the")
    print("  !! right person, 0 under a wrong one, 6 missing.")
    print("  !! Do not quote the p-values below.")

    per_arm = {a: [0, 0] for a in ARMS}       # [correct, n]
    rows = []
    for mid in MIDS:
        human = [h for h in read_human(mid) if h[0]]
        prev = [h for h in read_any(PREV[mid]) if h[0] and not is_backlog(h[1])]
        prev_bags = [(p, bag(t)) for p, t in prev]
        if not prev_bags:
            continue
        # locate each of this week's topics in last week's document
        changed = {}                            # reference topic index -> (owner, last week's owner)
        for i, (hp, ht) in enumerate(human):
            if is_backlog(ht):
                continue                      # the system does not produce the Backlog
            hb = bag(ht)
            if len(hb) < 3:
                continue
            best_p, best_s = None, 0.0
            for pp, pb in prev_bags:
                s = jac(hb, pb)
                if s > best_s:
                    best_s, best_p = s, pp
            if best_s >= CUT and best_p != hp:
                changed[i] = (hp, best_p)       # same topic, DIFFERENT owner
        if not changed:
            rows.append((mid, 0, {a: (0, 0) for a in ARMS}))
            continue
        got = {}
        for arm in ARMS:
            p = REPO / "outputs_deliverable" / arm / mid / "protocol_pro.md"
            if not p.exists():
                got[arm] = (0, 0)
                continue
            raw = p.read_text(encoding="utf-8")
            hits = block_hits(human, read_generated(p)[0], TH)
            ok = n = 0
            for i, (true_owner, _last) in changed.items():
                who = hits.get(i)
                if who is None:
                    # The block matcher could not locate the topic, which is not the same as
                    # putting it under the wrong person. An arm that splits a topic across finer
                    # headings can leave each block with too few of the topic's tokens to clear
                    # TH, so fall back to the keyword search.
                    who = _locate_by_keyword(human[i][1], raw)
                if who is None:
                    continue                      # not in the protocol: a coverage miss
                n += 1
                ok += who == true_owner
            got[arm] = (ok, n)
            per_arm[arm][0] += ok
            per_arm[arm][1] += n
        rows.append((mid, len(changed), got))

    hdr = f"  {'meeting':<10}{'changed':>9}" + "".join(f"{a:>12}" for a in ARMS)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for mid, nch, got in rows:
        line = f"  {mid:<10}{nch:>9}"
        for a in ARMS:
            ok, n = got[a]
            line += f"{(f'{ok}/{n}'):>12}"
        print(line)
    print("  " + "-" * (len(hdr) - 2))
    line = f"  {'ALL':<10}{sum(r[1] for r in rows):>9}"
    for a in ARMS:
        ok, n = per_arm[a]
        line += f"{(f'{ok}/{n}'):>12}"
    print(line)
    line = f"  {'':<10}{'':>9}"
    for a in ARMS:
        ok, n = per_arm[a]
        line += f"{(f'{100*ok/n:.0f}%' if n else '-'):>12}"
    print(line + "   <- baseline is 0% here BY CONSTRUCTION\n")

    # Significance. The no-audio baseline scores 0% here by construction, but the null is chance:
    # a random guess among the people in the room is right 1/|roster| of the time, and the audio
    # has to beat that floor, not zero.
    from scipy.stats import binomtest
    from attribution_baselines import ROSTER
    # A meeting without a ROSTER entry (april13, whose minutes are a Wiki PDF export) falls back
    # to the number of distinct people its reference protocol names, at least 2.
    def room(m: str) -> int:
        if m in ROSTER:
            return len(ROSTER[m])
        return max(2, len({p for p, _ in read_human(m) if p}))
    floor = sum(1 / room(m) for m in MIDS) / len(MIDS)
    print(f"  Null used: random guess among the people in the room = {floor:.0%}")
    print("  (NOT 0%. The baseline scores 0 here by construction, but chance does not.)")
    print()
    for a in ARMS:
        ok, n = per_arm[a]
        if n:
            pv = binomtest(ok, n, floor, alternative="greater").pvalue
            print(f"  {a:<5} {ok}/{n} = {100*ok/n:3.0f}%  vs {floor:.0%} chance: p = {pv:.3f}"
                  + ("   SIGNIFICANT" if pv < 0.05 else "   not significant"))
    print()
    print("  POWER. Only 14 topics changed hands across all 8 meetings, and each arm located")
    print("  5-8 of them. With n that small nothing short of a near-perfect score can reach")
    print("  significance, so 'not significant' here means UNDERPOWERED, not disproved.")
    print()
    print("  !! RESULT WITHDRAWN. Most of these 'changed-owner' topics are NEW topics that")
    print("  !! matched last week only through filler words; read one by one, v10 put 7 of 13 under the")
    print("  !! right person, 0 under a wrong one, 6 missing.")
    print("  !! Do not quote the p-values above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
