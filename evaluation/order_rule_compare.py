"""Compares speaking-order rules for the person sections on the same named transcript.

Two rules order the people who spoke: by first utterance, with the speaker who has the most turns placed
last, and by the start of each person's longest continuous block of speech
(speaking_order_test.longest_block, with gaps of 15, 30 and 60 s). Both run on the Canary-1B-v2 transcript
(outputs_v11_final) named with the speaker names of the v10 run (outputs_deliverable/v10/<meeting>/run_pro.log);
a turn ends at the next transcript line, at most 60 s after its start. Each order is scored against the order of the
reference minutes (attribution_baselines.human_order) in two ways: over the people who spoke only, and with
the people who did not speak placed at their slot of the previous protocol (place_absent). For each rule
the script prints the number of meetings in exact order and the mean pairwise agreement
(attribution_baselines.kendall_like).

    python order_rule_compare.py    -> stdout only
"""
from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
MEET = ["march09", "april13", "april27", "may11", "may18", "june01", "june15", "june22"]
TURN_CAP = 60.0


def named_turns(mid: str) -> list[tuple[float, float, str]]:
    from evaluate_speaker_attribution import person_key
    log = (REPO / f"outputs_deliverable/v10/{mid}/run_pro.log").read_text(encoding="utf-8",
                                                                              errors="replace")
    m = re.findall(r"speaker names \(voice, handover, topic\): (.*)", log)
    names = dict(x.split("=", 1) for x in m[-1].strip().split(", ")) if m else {}
    rows = []
    for line in (REPO / f"outputs_v11_final/canary_diar/{mid}/transcript_speakers.txt").read_text(
            encoding="utf-8").splitlines():
        mm = re.match(r"\[(\d\d):(\d\d):(\d\d)\]\s+(SPEAKER_\d+):", line.strip())
        if mm:
            t = int(mm[1]) * 3600 + int(mm[2]) * 60 + int(mm[3])
            who = names.get(mm[4])
            rows.append((t, person_key(who) if who else None))
    out = []
    for i, (t, who) in enumerate(rows):
        end = rows[i + 1][0] if i + 1 < len(rows) else t + 10
        if who:
            out.append((float(t), float(min(end, t + TURN_CAP)), who))
    return out


def order_first_chair_last(turns):
    first, count = {}, Counter()
    for s, _, w in turns:
        first.setdefault(w, s)
        count[w] += 1
    chair = count.most_common(1)[0][0]
    return [w for w in sorted(first, key=first.get) if w != chair] + [chair]


def order_longest_block(turns, gap: float):
    from attribution_baselines import kendall_like  # noqa: F401  (same family of metric)
    from speaking_order_test import longest_block
    by = defaultdict(list)
    for s, e, w in turns:
        by[w].append((s, e))
    starts = {w: longest_block(tl, gap)[0] for w, tl in by.items()}
    return sorted(starts, key=starts.get)


def place_absent(order, prev_order):
    """Place each person of last week's order who did not speak this week directly after the
    last person before them in last week's order who did speak."""
    rank = {p: float(i) for i, p in enumerate(order)}
    spoke = dict(rank)
    after = {}
    for i, p in enumerate(prev_order):
        if p in rank:
            continue
        before = [q for q in prev_order[:i] if q in spoke]
        anchor = spoke[before[-1]] if before else -1.0
        after[anchor] = after.get(anchor, 0) + 1
        rank[p] = anchor + after[anchor] / (len(prev_order) + 1)
    return sorted(rank, key=rank.get)


def main() -> int:
    from attribution_baselines import human_order, kendall_like, read_any, PREV
    rules = {"first utterance + moderator last (shipped)": order_first_chair_last,
             "longest block, gap 15 s": lambda t: order_longest_block(t, 15.0),
             "longest block, gap 30 s": lambda t: order_longest_block(t, 30.0),
             "longest block, gap 60 s": lambda t: order_longest_block(t, 60.0)}
    res = {r: {"ex_spoke": 0, "pw_spoke": [], "ex_full": 0, "pw_full": []} for r in rules}
    rows = []
    for mid in MEET:
        his = human_order(mid)
        prev = list(dict.fromkeys(p for p, _ in read_any(PREV[mid]) if p))
        turns = named_turns(mid)
        line = [mid, " ".join(his)]
        for r, fn in rules.items():
            o = fn(turns)
            spoke_ref = [h for h in his if h in o]
            res[r]["ex_spoke"] += [x for x in o if x in his] == spoke_ref
            res[r]["pw_spoke"].append(kendall_like(his, o))
            full = [x for x in place_absent(o, prev) if x in his]
            res[r]["ex_full"] += full == [h for h in his if h in full]
            res[r]["pw_full"].append(kendall_like(his, full))
            line.append(" ".join(o))
        rows.append(line)
    print("ORDER RULES on the deliverable's own input, 8 meetings with reference minutes")
    print("=" * 100)
    for mid, his, *orders in rows:
        print(f"  {mid:8s} ref: {his}")
        for r, o in zip(rules, orders):
            print(f"           {r[:38]:38s} {o}")
    print()
    print(f"  {'rule':40s}{'exact (spoke)':>14s}{'pairwise (spoke)':>18s}{'exact (with absent)':>21s}{'pairwise (w/ absent)':>22s}")
    for r, v in res.items():
        n = len(MEET)
        print(f"  {r:40s}{v['ex_spoke']:>10d}/{n}{100 * sum(v['pw_spoke']) / n:>17.1f}%"
              f"{v['ex_full']:>17d}/{n}{100 * sum(v['pw_full']) / n:>21.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
