"""Topic attribution matched block-to-block, as a cross-check on the pooled-per-person matcher.

`evaluate_speaker_attribution.topic_hits` locates a reference topic by pooling every topic block
of a person into a single bag and taking the person with the highest token containment:

    pp[person] |= bag(every topic block of that person)
    score      = |reference_topic_tokens & pp[person]| / |reference_topic_tokens|

That score is biased by section size: the person with the most text has the largest bag and the
best chance of containing a topic's words by coincidence, and merging topic headings merges bags
and moves the bias without changing where the content sits. `read_generated` already returns one
(person, "heading + bullets") item per topic block, and the reference protocol is also a heading
with bullets, so this module compares topic with topic:

1. Score each block, and attribute the topic to the owner of the best-scoring block.
2. Take the abstention margin between persons: the best block of the winner against the best
   block of the next person. A margin between blocks would abstain whenever a topic is split
   over two blocks of the same person, which is a property of formatting, not of attribution.

Everything else is identical to `topic_hits`, so the matching is the only variable: the same
qualifying topics (>= 3 tokens after filtering, previous-week Jaccard >= CUT, Backlog excluded),
the same threshold 0.15, the same margin 0.05, and the same person-level document-frequency cap
on tokens shared by more than half of the people. Results are split into all, spoken and carried
topics (ladder_split.spoken).

Both matchers are printed side by side, and every disagreement is written out so that it can be
checked by reading the protocol. goldset/ladder_split.json is not overwritten.

    python attribution_blocklevel.py v6 v8    -> outputs_deliverable/attribution_blocklevel.json
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent
OUT = REPO / "outputs_deliverable" / "attribution_blocklevel.json"

from evaluate_speaker_attribution import bag, person_key, read_generated, read_human  # noqa: E402
from attribution_baselines import MIDS, PREV, TH, read_any, jac                    # noqa: E402
from ladder_split import CUT, owner_vocab, spoken                                     # noqa: E402


def block_hits(human, gen, th: float = TH, margin: float = 0.05) -> dict[int, str]:
    """Reference topic index -> the person owning the best-matching topic block.

    Parallel to topic_hits: the same th, the same margin and the same person-level df cap. The one
    difference is that each block is scored on its own instead of pooled per person."""
    blocks = [(p, bag(t)) for p, t in gen if p]
    if not blocks:
        return {}
    # df cap computed over people, as topic_hits does, so the vocabulary filter is the same and
    # only the matching changes. Pooling is right here because the cap is a stop-word rule.
    per_person: dict[str, set[str]] = defaultdict(set)
    for p, b in blocks:
        per_person[p] |= b
    df = Counter(tok for b in per_person.values() for tok in b)
    cap = max(1, len(per_person) // 2)

    out: dict[int, str] = {}
    for i, (_, ht) in enumerate(human):
        hb = {t for t in bag(ht) if df.get(t, 0) <= cap}
        if len(hb) < 3:
            continue
        best: dict[str, float] = {}
        for p, b in blocks:
            s = len(hb & b) / len(hb)
            if s > best.get(p, -1.0):
                best[p] = s
        r = sorted(((s, p) for p, s in best.items()), reverse=True)
        if r[0][0] >= th and r[0][0] - (r[1][0] if len(r) > 1 else 0.0) >= margin:
            out[i] = r[0][1]
    return out


def rows_for(arms: list[str]) -> list[dict]:
    from evaluate_speaker_attribution import topic_hits
    rows = []
    for mid in MIDS:
        human = [h for h in read_human(mid) if h[0]]
        prev = [(p, bag(t)) for p, t in read_any(PREV[mid]) if p]
        if not prev:
            continue
        vocab = owner_vocab(mid)
        pooled, block = {}, {}
        for a in arms:
            p = REPO / "outputs_deliverable" / a / mid / "protocol_pro.md"
            if not p.exists():
                continue
            gen = read_generated(p)[0]
            pooled[a] = topic_hits(human, gen, TH)
            block[a] = block_hits(human, gen, TH)
        for i, (hp, ht) in enumerate(human):
            hb = bag(ht)
            if len(hb) < 3:
                continue
            if max((jac(hb, pb) for _, pb in prev), default=0.0) < CUT:
                continue
            if re.match(r"\s*[*_]*backlog", ht, re.I):
                continue                      # the system does not produce the Backlog
            rows.append({"meeting": mid, "owner": hp, "text": ht[:120],
                         "spoken": spoken(ht, hp, vocab),
                         **{f"pooled:{a}": pooled.get(a, {}).get(i) for a in arms},
                         **{f"block:{a}": block.get(a, {}).get(i) for a in arms}})
    return rows


def main() -> int:
    arms = sys.argv[1:] or ["v6", "v8"]
    rows = rows_for(arms)
    if not rows:
        print("no rows — are outputs_deliverable/<arm>/<meeting>/protocol_pro.md present?")
        return 1

    def rate(rs, field):
        got = [r for r in rs if r[field]]
        ok = sum(person_key(r[field]) == person_key(r["owner"]) for r in got)
        return ok, len(got), len(rs)

    print("ATTRIBUTION — pooled-per-person matching vs block-to-block matching")
    print("=" * 88)
    print("  Same topics, same threshold, same margin. The matcher is the only variable.")
    for label, sel in (("ALL", lambda r: True), ("SPOKEN", lambda r: r["spoken"]),
                       ("CARRIED", lambda r: not r["spoken"])):
        rs = [r for r in rows if sel(r)]
        print(f"\n  {label}   ({len(rs)} reference topics)")
        for a in arms:
            po, pl, pt = rate(rs, f"pooled:{a}")
            bo, bl, bt = rate(rs, f"block:{a}")
            print(f"      {a:4s} pooled {po:3d}/{pt} = {100*po/pt:5.1f}%  (located {pl})"
                  f"   block {bo:3d}/{bt} = {100*bo/bt:5.1f}%  (located {bl})")

    dis = []
    for r in rows:
        for a in arms:
            p, b = r[f"pooled:{a}"], r[f"block:{a}"]
            if p != b:
                dis.append({"arm": a, "meeting": r["meeting"], "his_owner": r["owner"],
                            "pooled": p, "block": b, "text": r["text"],
                            "pooled_right": bool(p) and person_key(p) == person_key(r["owner"]),
                            "block_right": bool(b) and person_key(b) == person_key(r["owner"])})
    print(f"\n  the two matchers disagree on {len(dis)} (topic, arm) pairs")
    fixed = [d for d in dis if d["block_right"] and not d["pooled_right"]]
    broke = [d for d in dis if d["pooled_right"] and not d["block_right"]]
    print(f"      block right where pooled was wrong : {len(fixed)}")
    print(f"      block wrong where pooled was right : {len(broke)}")
    print("\n  EVERY disagreement, to be checked by reading the protocol (the only authority):")
    for d in dis[:40]:
        mark = "block fixes it" if d in fixed else ("block breaks it" if d in broke else "both wrong/abstain")
        print(f"    [{d['arm']} {d['meeting']}] ref={d['his_owner']:9s} "
              f"pooled={str(d['pooled']):9s} block={str(d['block']):9s}  {mark}")
        print(f"        {d['text'][:84]}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "built": "2026-09-29",
        "what_this_is": "The attribution ladder recomputed with block-to-block matching, beside "
                        "the pooled-per-person matcher the thesis currently quotes.",
        "why": "topic_hits pools a person's whole section into one bag, so the score is biased by "
               "section size. Whoever has the most text wins argmaxes by coincidence, and merging "
               "topic headings (which v8 did to René) merges bags and moves the bias.",
        "held_identical": ["qualifying topics", "threshold 0.15", "margin 0.05",
                           "person-level document-frequency cap", "Backlog excluded"],
        "changed": ["match each topic block separately, not the pooled person",
                    "abstention margin taken between PERSONS, not between blocks"],
        "authority": "reading the protocol. This file lists disagreements so each can be checked.",
        "arms": arms,
        "summary": {label: {a: {"pooled": rate([r for r in rows if sel(r)], f"pooled:{a}"),
                                "block": rate([r for r in rows if sel(r)], f"block:{a}")}
                            for a in arms}
                    for label, sel in (("ALL", lambda r: True), ("SPOKEN", lambda r: r["spoken"]),
                                       ("CARRIED", lambda r: not r["spoken"]))},
        "disagreements": dis,
        "rows": rows,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  wrote -> {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
