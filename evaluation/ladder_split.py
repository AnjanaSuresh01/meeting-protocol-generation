"""Scores topic attribution of each protocol arm against the no-audio baseline "last week's owner",
split by whether the topic was spoken in the meeting.

The baseline credits each topic of the reference minutes to whoever owned the most similar topic
in the previous week's minutes. The reference minutes are written from a copy of the previous
protocol, so some of their topics were never said in the meeting; for those, last week's owner is
right by construction and no audio-based system can produce them. Each reference topic with at
least three words whose best match in last week's minutes has a Jaccard similarity of at least CUT
is therefore labeled
    SPOKEN     at least 40 % of its distinctive words (at least 3) occur in the owner's own speech
               (voice cluster -> person from goldset/speaker_identity.json), pooled over the
               ASR systems
    UNSPOKEN   the rest, i.e. text carried over from earlier minutes
and the ladder is reported on each part separately, for every protocol arm, on the same topics.
An arm credits a topic to the person most of its protocols (ASR systems x LISA models) file it
under; the arms are the raw protocol_{lisa}.md, the _fixed output and the shipped _fixed_carry
output, plus, when outputs_v11_final holds v11 protocols, the v11 Canary arm and the shipped arm
restricted to Canary. For each arm the script reports the share of located topics credited to
the right person, and an exact McNemar test against last week's owner on the topics the arm
located. When goldset/ladder_hand_labels.json lines up with the rows, its hand labels replace the
script's SPOKEN/UNSPOKEN labels.

    python ladder_split.py    -> goldset/ladder_split.json
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import re

from absent_reachability import distinctive
from evaluate_speaker_attribution import read_human, read_generated, bag, topic_hits, person_key
from attribution_baselines import MIDS, MODELS, LISAS, PREV, TH, V10, read_any, jac

REPO = Path(__file__).resolve().parent
ARMS = {"raw": "", "fixed": "_fixed", "shipped (fixed_carry)": "_fixed_carry"}
# arm -> (tree, models) for arms that do not use the default tree and model list. The v11 outputs
# exist for Canary only, so they are scored beside the shipped arm restricted to Canary; comparing
# them with the pool of all systems would change two variables at once. Both arms are added only
# when `generate_protocols.py --v11 --fixed --carry` has written its files.
V11 = Path(__file__).resolve().parent / "outputs_v11_final"
ARM_SRC: dict[str, tuple] = {}
if any(V11.glob("canary_diar/*/protocol_*_fixed_carry.md")):
    ARMS["shipped, canary only"] = "_fixed_carry"
    ARMS["v11 (canary)"] = "_fixed_carry"
    ARM_SRC["shipped, canary only"] = (None, ["canary_diar"])
    ARM_SRC["v11 (canary)"] = (V11, ["canary_diar"])
CUT = 0.10                       # Jaccard threshold for 'same topic as last week'


def owner_vocab(mid: str) -> dict[str, set[str]]:
    """person_key -> every token that person's own voice produced in this meeting, pooled over
    the ASR systems in MODELS (they share one diarization, so the cluster labels agree).

    Only the owner's own speech counts: a topic that someone else mentions in passing is not
    spoken by its owner. Cluster -> person comes from goldset/speaker_identity.json (voice
    x-vectors)."""
    from term_source import tokens, _clean
    ident = json.loads((REPO / "goldset" / "speaker_identity.json").read_text(encoding="utf-8"))
    names = {lab: person_key(n) for lab, n in ident["mapping"].get(mid, {}).items()}
    out: dict[str, set[str]] = defaultdict(set)
    for model in MODELS:
        p = V10 / model / mid / "transcript_speakers.txt"
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.match(r"\[[\d:]+\]\s+(SPEAKER_\d+):\s*(.*)", line.strip())
            if m and m.group(1) in names:
                out[names[m.group(1)]] |= {_clean(t).lower() for t in tokens(m.group(2))}
    return out


def spoken(topic_text: str, owner: str, vocab: dict[str, set[str]]) -> bool:
    terms = {t for t in distinctive(topic_text) if t}
    if len(terms) < 3 or owner not in vocab:
        return False
    return sum(t.lower() in vocab[owner] for t in terms) / len(terms) >= 0.4


def main() -> int:
    rows = []                    # one row per qualifying human topic
    for mid in MIDS:
        human = [h for h in read_human(mid) if h[0]]
        prev = [(p, bag(t)) for p, t in read_any(PREV[mid]) if p]
        vocab = owner_vocab(mid)
        votes = {arm: defaultdict(list) for arm in ARMS}
        for arm, suf in ARMS.items():
            root, models = ARM_SRC.get(arm, (None, MODELS))
            for model in models:
                for lisa in LISAS:
                    p = (root or V10) / model / mid / f"protocol_{lisa}{suf}.md"
                    if p.exists() and p.stat().st_size > 80:
                        for i, who in topic_hits(human, read_generated(p)[0], TH).items():
                            votes[arm][i].append(who)
        for i, (hp, ht) in enumerate(human):
            hb = bag(ht)
            if len(hb) < 3 or not prev:
                continue
            best_p, best_s = max(((pp, jac(hb, pb)) for pp, pb in prev), key=lambda x: x[1])
            if best_s < CUT:
                continue
            rows.append({
                "meeting": mid, "owner": hp, "spoken": spoken(ht, hp, vocab), "text": ht[:120],
                "last_week_ok": best_p == hp,
                **{f"arm:{a}": (Counter(v[i]).most_common(1)[0][0] if v.get(i) else None)
                   for a, v in votes.items()},
            })

    # The 40 % rule is conservative: it can label a topic as carried that its owner did say.
    # Hand labels replace the script labels when they line up with these rows one to one; the
    # script label is kept as script_spoken.
    hand = REPO / "goldset" / "ladder_hand_labels.json"
    if hand.exists():
        lab = json.loads(hand.read_text(encoding="utf-8"))["labels"]
        if len(lab) == len(rows) and all(a["meeting"] == r["meeting"] and a["owner"] == r["owner"]
                                         for a, r in zip(lab, rows)):
            for a, r in zip(lab, rows):
                r["script_spoken"], r["spoken"] = r["spoken"], a["hand_spoken"]
            print("(spoken/carried from the hand labels: goldset/ladder_hand_labels.json)\n")
        else:
            print("(hand labels do not line up with these rows — script labels used)\n")

    def rate(rs, key):
        loc = [r for r in rs if r[key] is not None]
        ok = sum(r[key] == r["owner"] for r in loc)
        return ok, len(loc), len(rs)

    def mcnemar(rs, key):
        """Exact McNemar test (binomial), arm vs. last week's owner, on the topics the arm located."""
        from scipy.stats import binomtest
        loc = [r for r in rs if r[key] is not None]
        b = sum((r[key] == r["owner"]) and not r["last_week_ok"] for r in loc)
        c = sum((r[key] != r["owner"]) and r["last_week_ok"] for r in loc)
        lw = sum(r["last_week_ok"] for r in loc)
        p = binomtest(b, b + c).pvalue if b + c else 1.0
        return lw, len(loc), b, c, p

    print("BASELINE LADDER, SPLIT BY WHETHER THE TOPIC WAS SPOKEN  (Jaccard >= %.2f)" % CUT)
    print("=" * 96)
    out = {}
    for label, sel in (("ALL", lambda r: True), ("SPOKEN", lambda r: r["spoken"]),
                       ("UNSPOKEN (carried)", lambda r: not r["spoken"])):
        rs = [r for r in rows if sel(r)]
        lw = sum(r["last_week_ok"] for r in rs)
        print(f"\n  {label:<20} {len(rs):4d} topics   last week (no audio): "
              f"{lw}/{len(rs)} = {lw / len(rs):.1%}" if rs else f"\n  {label}: none")
        out[label] = {"n": len(rs), "last_week": lw / len(rs) if rs else None}
        for arm in ARMS:
            ok, n, tot = rate(rs, f"arm:{arm}")
            lw_l, _, b, c, p = mcnemar(rs, f"arm:{arm}")
            print(f"      {arm:<24} located {n:3d}/{tot:<3d}  correct {ok:3d} = "
                  f"{ok / n if n else 0:6.1%} of located   {ok / tot if tot else 0:6.1%} counting "
                  f"unlocated as wrong   | same topics, last week {lw_l / n if n else 0:6.1%}; "
                  f"McNemar arm-only {b} / last-week-only {c}, p = {p:.3f}")
            out[label][arm] = {"ok": ok, "located": n, "total": tot}
    print("\n  per meeting — share of the reference topics that were SPOKEN")
    for mid in MIDS:
        rs = [r for r in rows if r["meeting"] == mid]
        s = sum(r["spoken"] for r in rs)
        print(f"      {mid:<9} {s:3d}/{len(rs):<3d} = {s / len(rs) if rs else 0:.0%}")
    (REPO / "goldset" / "ladder_split.json").write_text(
        json.dumps({"cut": CUT, "summary": out, "topics": rows}, indent=1, ensure_ascii=False),
        encoding="utf-8")
    print("\n  -> goldset/ladder_split.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
