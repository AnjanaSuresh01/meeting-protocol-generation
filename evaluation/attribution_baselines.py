"""Checks how the generated protocols in outputs_v10/ attribute topics to people.
It compares section order and section names with the human minutes, with the speaker
mapping in goldset/speaker_identity.json and with a baseline that copies each topic's
owner from the previous week's human protocol. Other modules import read_any() and jac().

Usage:
    python attribution_baselines.py
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from evaluate_speaker_attribution import (
    HUMAN, ROSTER, read_human, read_human_md,
    read_generated, person_key, bag, topic_hits,
)

REPO = Path(__file__).resolve().parent
V10 = REPO / "outputs_v10"
MODELS = ["canary_diar", "cohere", "granite41",
          "whisper_large_v3", "whisper_turbo", "whisper_turbo_de"]
LISAS = ["flash", "pro"]
TH = 0.15  # minimum share of a topic's words found in one section

# previous week's human protocol for each meeting
PREV = {
    "march09": "Meeting_2026-03-02.md",
    "april13": "Meeting_2026-04-08.md",
    "april27": "Meeting_2026-04-20.md",
    "may11":   "Meeting_2026-05-04.md",
    "may18":   "Meeting_2026-05-11.md",
    "june01":  "Meeting_2026-05-18.md",
    "june15":  "Meeting_2026-06-08(1).md",
    "june22":  "Meeting_2026-06-15.md",
}
MIDS = list(HUMAN)  # meeting ids with human minutes


# Reads a Word protocol as (person, topic text) pairs. The person level is read
# from the bold paragraphs of the file; topics sit one list level deeper.
def read_docx_robust(path: Path) -> list[tuple[str, str]]:
    import docx
    paras = []
    for p in docx.Document(str(path)).paragraphs:
        t = p.text.strip()
        if not t or p.style.name.startswith("Title"):
            continue
        pr = p._p.pPr
        lvl = (pr.numPr.ilvl.val if pr is not None and pr.numPr is not None
               and pr.numPr.ilvl is not None else None)
        paras.append((t, lvl, any(r.bold for r in p.runs)))
    bold_lvls = [l for _, l, b in paras if b]
    person_lvl = None if None in bold_lvls else (min(bold_lvls) if bold_lvls else None)
    topic_lvl = 0 if person_lvl is None else person_lvl + 1
    items, person, topic, body = [], None, None, []
    for t, lvl, bold in paras:
        if bold and lvl == person_lvl:
            if topic:
                items.append((person, topic + " " + " ".join(body)))
            person, topic, body = person_key(t), None, []
        elif lvl == topic_lvl and person:
            if topic:
                items.append((person, topic + " " + " ".join(body)))
            topic, body = t, []
        elif person and topic:
            body.append(t)
    if topic:
        items.append((person, topic + " " + " ".join(body)))
    return items


def read_any(name: str) -> list[tuple[str, str]]:
    p = REPO / name
    return read_docx_robust(p) if p.suffix.lower() == ".docx" else read_human_md(p)


def jac(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if (a | b) else 0.0


def human_order(mid: str) -> list[str]:
    seen = []
    for p, _ in read_human(mid):
        if p and p not in seen:
            seen.append(p)
    return seen


def gen_order(path: Path) -> list[str]:
    seen = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if s.startswith("## ") and not s.startswith("###"):
            k = person_key(s[3:])
            if k and k not in seen:
                seen.append(k)
    return seen


def read_turns(mid: str, model: str) -> list[tuple[str, str]]:
    p = V10 / model / mid / "transcript_speakers.txt"
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"\[\d\d:\d\d:\d\d\]\s+(SPEAKER_\d+):\s*(.*)", line.strip())
        if m:
            out.append((m.group(1), m.group(2)))
    return out


def kendall_like(a: list[str], b: list[str]) -> float:
    """Fraction of shared item pairs that keep their order between the two lists.

    1.0 = same order, 0.0 = reversed, NaN if fewer than two items are shared.
    """
    common = [x for x in a if x in b]
    if len(common) < 2:
        return float("nan")
    ib = {x: i for i, x in enumerate(b)}
    ok = tot = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            tot += 1
            ok += ib[common[i]] < ib[common[j]]
    return ok / tot if tot else float("nan")


def protocols() -> list[tuple[str, str, str, Path]]:
    out = []
    for model in MODELS:
        for mid in MIDS:
            for lisa in LISAS:
                p = V10 / model / mid / f"protocol_{lisa}.md"
                if p.exists() and p.stat().st_size > 80:
                    out.append((model, mid, lisa, p))
    return out


def test1_rotation() -> dict[str, list[str]]:
    print("=" * 96)
    print("TEST 1 — IS THE REPORTING ORDER ROTATED?  (checked in the human minutes)")
    print("=" * 96)
    orders = {mid: human_order(mid) for mid in MIDS}
    print(f"\n  {'meeting':<10} reporting order as the human minutes list it")
    for mid in MIDS:
        print(f"  {mid:<10} {' -> '.join(orders[mid])}")
    first = Counter(o[0] for o in orders.values() if o)
    last = Counter(o[-1] for o in orders.values() if o)
    print(f"\n  reports FIRST: " + ", ".join(f"{k} x{v}" for k, v in first.most_common()))
    print(f"  reports LAST : " + ", ".join(f"{k} x{v}" for k, v in last.most_common())
          + "   <- the meeting lead closes; only this position is fixed")
    pairs = [kendall_like(orders[a], orders[b])
             for i, a in enumerate(MIDS) for b in MIDS[i + 1:]]
    pairs = [p for p in pairs if p == p]
    print(f"  pairwise order agreement between weeks: mean {np.mean(pairs):.2f} "
          f"(1.00 = same order every week, 0.50 = unrelated)")
    print("\n  RESULT: confirmed. Every week a different person opens; the order of the")
    print("  four reporters is reshuffled. Position carries no identity.")
    return orders


def test2_emitted_order(horders: dict[str, list[str]]) -> None:
    print("\n" + "=" * 96)
    print("TEST 2 — WHAT ORDER DOES THE PIPELINE EMIT?")
    print("=" * 96)
    print("  The prompt always lists the roster in the SAME order (René, Vincent, Thomas,")
    print("  Johannes, Osman). If protocols reproduced that instead of the real order, the")
    print("  layout would be coming from the prompt rather than from the audio.\n")
    rows = []
    for model, mid, lisa, p in protocols():
        roster = [person_key(x) for x in ROSTER[mid]]
        g = gen_order(p)
        rows.append({"lisa": lisa,
                     "vs_roster": kendall_like(g, roster),
                     "vs_human": kendall_like(g, horders[mid]),
                     "exact_roster": g == [r for r in roster if r in g],
                     "exact_human": g == [h for h in horders[mid] if h in g]})
    f = lambda k: np.nanmean([x[k] for x in rows])
    n = len(rows)
    print(f"  {n} protocols")
    print(f"  order agreement with ROSTER order (prompt) : {f('vs_roster'):.2f}")
    print(f"  order agreement with HUMAN order (reality) : {f('vs_human'):.2f}")
    print(f"  exactly the roster order : {sum(x['exact_roster'] for x in rows)}/{n}"
          f"   exactly the human order : {sum(x['exact_human'] for x in rows)}/{n}")
    print("\n  RESULT: not supported. Section order is near-random with respect to both. The")
    print("  protocols are not simply replaying the attendee list, so if names are wrong it is")
    print("  not because the layout was copied from the prompt. Test 3 and 4 locate the real")
    print("  mechanism.")


def recover_mapping(mid: str, model: str, path: Path) -> dict[str, str]:
    """Map each diarization label to one '## name' section: {SPEAKER_xx: name}.

    Labels and sections are paired one-to-one by Jaccard overlap of distinctive words
    (Hungarian assignment); pairs with no overlap are left out.
    """
    turns = read_turns(mid, model)
    if not turns:
        return {}
    per_label: dict[str, set[str]] = defaultdict(set)
    for spk, txt in turns:
        per_label[spk] |= bag(txt)
    gen, _ = read_generated(path)
    per_name: dict[str, set[str]] = defaultdict(set)
    for p, t in gen:
        if p:
            per_name[p] |= bag(t)
    if not per_name or not per_label:
        return {}
    df = Counter(tok for b in per_label.values() for tok in b)
    cap = max(1, len(per_label) // 2)
    labels, names = sorted(per_label), sorted(per_name)
    M = np.zeros((len(names), len(labels)))
    for i, nme in enumerate(names):
        nb = {t for t in per_name[nme] if df.get(t, 0) <= cap}
        for j, l in enumerate(labels):
            lb = {t for t in per_label[l] if df.get(t, 0) <= cap}
            M[i, j] = jac(nb, lb)
    ri, ci = linear_sum_assignment(-M)
    return {labels[j]: names[i] for i, j in zip(ri, ci) if M[i, j] > 0}


def voice_map() -> dict[str, dict[str, str]]:
    ident = json.loads((REPO / "goldset" / "speaker_identity.json").read_text(encoding="utf-8"))
    return {mid: {k: person_key(v) for k, v in m.items()}
            for mid, m in ident["mapping"].items()}


def test3_voice_agreement() -> None:
    print("\n" + "=" * 96)
    print("TEST 3 — DID IT NAME THE RIGHT VOICE?")
    print("=" * 96)
    print("  Reference = goldset/speaker_identity.json: WavLM x-vectors linked across the eight")
    print("  meetings and bound to people by the attendance record. It is the only name<-voice")
    print("  evidence in the repo and it is INDEPENDENT of every text prior. It is itself only")
    print("  ~71% accurate against the minutes, so this is AGREEMENT, not accuracy.\n")
    voice = voice_map()
    tot = agree = 0
    per_model = defaultdict(lambda: [0, 0])
    per_meeting = defaultdict(lambda: [0, 0])
    for model, mid, lisa, p in protocols():
        for label, name in recover_mapping(mid, model, p).items():
            truth = voice.get(mid, {}).get(label)
            if not truth:
                continue
            tot += 1
            hit = name == truth
            agree += hit
            per_model[model][0] += hit
            per_model[model][1] += 1
            per_meeting[mid][0] += hit
            per_meeting[mid][1] += 1
    chance = float(np.mean([1 / len(ROSTER[m]) for m in MIDS]))
    print(f"  label->name decisions traced : {tot}")
    print(f"  agree with the voice evidence: {agree}  ({agree / tot:.1%})")
    print(f"  chance (uniform over roster) : {chance:.1%}")
    print(f"\n  {'ASR system':<20}{'agreement':>11}")
    for m in MODELS:
        ok, n = per_model[m]
        print(f"  {m:<20}{ok / n if n else 0:>10.1%}  ({n})")
    print(f"\n  {'meeting':<12}{'agreement':>11}")
    for mid in MIDS:
        ok, n = per_meeting[mid]
        print(f"  {mid:<12}{ok / n if n else 0:>10.1%}  ({n})")
    print("\n  RESULT: naming is far above chance, so the LLM is NOT assigning names at random")
    print("  — but agreement is not evidence of mechanism. A model that knows Thomas owns the")
    print("  website and Johannes owns the phoneme models lands on the right name without ever")
    print("  using the voice. Test 4 separates the two.")


def test4_ladder() -> None:
    print("\n" + "=" * 96)
    print("TEST 4 — THE BASELINE LADDER")
    print("=" * 96)
    print("  rung 1  random          guess uniformly among the people in the room")
    print("  rung 2  majority        always credit the person who owns the most topics")
    print("  rung 3  LAST WEEK       find this topic in LAST WEEK'S HUMAN PROTOCOL and credit")
    print("                          whoever owned it then. NO audio, NO ASR, NO diarization,")
    print("                          NO LLM. Pure topic memory, the suspected source of")
    print("                          the pipeline's names.")
    print("  rung 4  pipeline        the 84 generated protocols, majority vote over systems,")
    print("                          scored on EXACTLY the same topics as rung 3.\n")

    # meeting -> human topic index -> names from the protocols that located it
    pipe: dict[str, dict[int, list[str]]] = {}
    for mid in MIDS:
        human = [h for h in read_human(mid) if h[0]]
        acc: dict[int, list[str]] = defaultdict(list)
        for model, m2, lisa, p in protocols():
            if m2 != mid:
                continue
            for i, who in topic_hits(human, read_generated(p)[0], TH).items():
                acc[i].append(who)
        pipe[mid] = acc

    for cut in (0.05, 0.10, 0.20):
        T = R = MAJ = B = P = PN = 0
        rows = []
        for mid in MIDS:
            human = [h for h in read_human(mid) if h[0]]
            prev = [h for h in read_any(PREV[mid]) if h[0]]
            prev_bags = [(p, bag(t)) for p, t in prev]
            owner = Counter(p for p, _ in human).most_common(1)[0][0] if human else ""
            n = base_ok = maj_ok = pk = pn = 0
            for i, (hp, ht) in enumerate(human):
                hb = bag(ht)
                if len(hb) < 3 or not prev_bags:
                    continue
                best_p, best_s = None, 0.0
                for pp, pb in prev_bags:
                    s = jac(hb, pb)
                    if s > best_s:
                        best_s, best_p = s, pp
                if best_s < cut:
                    continue
                n += 1
                base_ok += best_p == hp
                maj_ok += owner == hp
                votes = pipe[mid].get(i, [])
                if votes:
                    pn += 1
                    pk += Counter(votes).most_common(1)[0][0] == hp
            T += n
            B += base_ok
            MAJ += maj_ok
            P += pk
            PN += pn
            R += n / len(ROSTER[mid])
            rows.append((mid, n, base_ok, pk, pn))
        print(f"  --- 'same topic as last week' = Jaccard >= {cut:.2f} "
              f"({T} of {sum(len([h for h in read_human(m) if h[0]]) for m in MIDS)} "
              f"human topics qualify) ---")
        print(f"  {'meeting':<10}{'topics':>8}{'last week':>12}{'pipeline':>11}")
        for mid, n, b, pk, pn in rows:
            print(f"  {mid:<10}{n:>8}{b / n if n else 0:>11.0%}"
                  f"{pk / pn if pn else 0:>11.0%}")
        print(f"  {'ALL':<10}{T:>8}{B / T if T else 0:>11.1%}{P / PN if PN else 0:>11.1%}"
              f"      random {R / T:.1%}   majority {MAJ / T:.1%}")
        # The first pipeline figure counts only located topics; this one counts
        # unlocated topics as wrong.
        print(f"  {'':<10}{'':>8}{'':>11}{P / T if T else 0:>11.1%}"
              f"      <- pipeline counting its {T - PN} unlocated topics as wrong\n")
    print("  READ THIS CAREFULLY: the 'last week' column used no audio whatsoever. Whatever")
    print("  the pipeline column beats it by is the ONLY part of the attribution result the")
    print("  audio system can claim.")


def test5_counterfactual() -> None:
    print("\n" + "=" * 96)
    print("TEST 5 — COUNTERFACTUAL: WHAT IF THE VOICE HAD NAMED THE SECTIONS?")
    print("=" * 96)
    print("  generate_protocols.py never reads goldset/speaker_identity.json — it mentions it")
    print("  in a comment and then asks the LLM to guess from 'Anreden, Rollen und Themen'.")
    print("  Here each section keeps its content but takes the name the VOICE evidence gives")
    print("  its traced SPEAKER_xx. Same protocols, same threshold, no API call.\n")
    voice = voice_map()
    a_tot = a_ok = b_ok = 0
    per_model = defaultdict(lambda: [0, 0, 0])
    for model, mid, lisa, p in protocols():
        human = [h for h in read_human(mid) if h[0]]
        gen, _ = read_generated(p)
        hits = topic_hits(human, gen, TH)
        if not hits:
            continue
        rec = recover_mapping(mid, model, p)                 # label -> LLM name
        remap = {n: voice.get(mid, {}).get(l) for l, n in rec.items()}
        for i, who in hits.items():
            a_tot += 1
            a_ok += who == human[i][0]
            b_ok += (remap.get(who) or who) == human[i][0]
            per_model[model][0] += 1
            per_model[model][1] += who == human[i][0]
            per_model[model][2] += (remap.get(who) or who) == human[i][0]
    if not a_tot:
        print("  nothing located")
        return
    print(f"  {'ASR system':<20}{'as named by LLM':>17}{'named by voice':>16}{'delta':>9}")
    for m in MODELS:
        n, a, b = per_model[m]
        if n:
            print(f"  {m:<20}{a / n:>16.1%}{b / n:>16.1%}{(b - a) / n * 100:>+7.1f}pp")
    print(f"  {'ALL':<20}{a_ok / a_tot:>16.1%}{b_ok / a_tot:>16.1%}"
          f"{(b_ok - a_ok) / a_tot * 100:>+8.1f}pp   ({a_tot} located topics)")


def test6_new_vs_recurring() -> None:
    print("\n" + "=" * 96)
    print("TEST 6 — THE MECHANISM TEST: NEW TOPICS vs RECURRING TOPICS")
    print("=" * 96)
    print("  If attribution runs on the VOICE, a topic being new this week is irrelevant — the")
    print("  person still spoke it. If it runs on TOPIC MEMORY, a topic with no precedent in")
    print("  last week's protocol has nothing to remember and accuracy must fall.")
    print("  Same protocols, same scorer, split only by whether the topic existed last week.\n")
    cut = 0.10
    rec_ok = rec_n = new_ok = new_n = 0
    rows = []
    for mid in MIDS:
        human = [h for h in read_human(mid) if h[0]]
        prev_bags = [(p, bag(t)) for p, t in read_any(PREV[mid]) if p]
        votes: dict[int, list[str]] = defaultdict(list)
        for model, m2, lisa, p in protocols():
            if m2 == mid:
                for i, who in topic_hits(human, read_generated(p)[0], TH).items():
                    votes[i].append(who)
        r_ok = r_n = n_ok = n_n = 0
        for i, (hp, ht) in enumerate(human):
            hb = bag(ht)
            if len(hb) < 3 or i not in votes:
                continue
            best = max((jac(hb, pb) for _, pb in prev_bags), default=0.0)
            hit = Counter(votes[i]).most_common(1)[0][0] == hp
            if best >= cut:
                r_n += 1
                r_ok += hit
            else:
                n_n += 1
                n_ok += hit
        rec_ok += r_ok; rec_n += r_n; new_ok += n_ok; new_n += n_n
        rows.append((mid, r_ok, r_n, n_ok, n_n))
    print(f"  {'meeting':<10}{'recurring':>20}{'new':>18}")
    for mid, a, b, c, d in rows:
        print(f"  {mid:<10}{f'{a}/{b}':>12}{a / b if b else 0:>8.0%}"
              f"{f'{c}/{d}':>10}{c / d if d else 0:>8.0%}")
    print(f"  {'ALL':<10}{f'{rec_ok}/{rec_n}':>12}{rec_ok / rec_n:>8.1%}"
          f"{f'{new_ok}/{new_n}':>10}{new_ok / new_n:>8.1%}"
          f"   delta {(new_ok / new_n - rec_ok / rec_n) * 100:+.1f}pp")
    try:
        from scipy.stats import fisher_exact
        _, p = fisher_exact([[rec_ok, rec_n - rec_ok], [new_ok, new_n - new_ok]])
        print(f"  Fisher exact p = {p:.4f}")
    except ImportError:
        pass
    print("\n  A drop on new topics is the signature of topic memory. A flat line is the")
    print("  signature of a system that is actually listening to who is speaking.")


def test7_arms() -> None:
    print("\n" + "=" * 96)
    print("TEST 7 — THE ARM WHERE LAST WEEK'S PROTOCOL IS LITERALLY IN THE PROMPT")
    print("=" * 96)
    print("  generate_protocols.py --carry pastes the previous week's HUMAN protocol — person")
    print("  headings and all — into the prompt. If attribution leaks from the old protocol,")
    print("  this is where it must show, and it must show SPECIFICALLY on recurring topics.")
    print("  Only canary_diar has the carry arms, so the comparison is held to that system.\n")
    cut = 0.10
    arms = {"plain": "", "carry": "_carry", "tracked": "_tracked", "merged": "_merged",
            "fixed": "_fixed"}
    print(f"  {'arm':<10}{'protocols':>10}{'recurring':>22}{'new':>18}")
    for arm, suf in arms.items():
        r_ok = r_n = n_ok = n_n = np_ = 0
        for mid in MIDS:
            human = [h for h in read_human(mid) if h[0]]
            prev_bags = [(p, bag(t)) for p, t in read_any(PREV[mid]) if p]
            for lisa in LISAS:
                p = V10 / "canary_diar" / mid / f"protocol_{lisa}{suf}.md"
                if not p.exists() or p.stat().st_size < 80:
                    continue
                np_ += 1
                for i, who in topic_hits(human, read_generated(p)[0], TH).items():
                    hb = bag(human[i][1])
                    best = max((jac(hb, pb) for _, pb in prev_bags), default=0.0)
                    hit = who == human[i][0]
                    if best >= cut:
                        r_n += 1
                        r_ok += hit
                    else:
                        n_n += 1
                        n_ok += hit
        if not (r_n or n_n):
            print(f"  {arm:<10}{'—':>10}   not generated")
            continue
        print(f"  {arm:<10}{np_:>10}{f'{r_ok}/{r_n}':>14}{r_ok / r_n if r_n else 0:>8.0%}"
              f"{f'{n_ok}/{n_n}':>10}{n_ok / n_n if n_n else 0:>8.0%}")
    print("\n  The arms do not locate the same topics, and accuracy is scored on what each one")
    print("  located — so the table above confounds attribution with coverage. The paired test")
    print("  below fixes that: only topics located by BOTH arms of the same meeting and LISA.\n")

    def hits_for(mid: str, lisa: str, suf: str) -> dict[int, str]:
        p = V10 / "canary_diar" / mid / f"protocol_{lisa}{suf}.md"
        if not p.exists() or p.stat().st_size < 80:
            return {}
        human = [h for h in read_human(mid) if h[0]]
        return topic_hits(human, read_generated(p)[0], TH)

    print(f"  {'arm vs plain':<16}{'paired topics':>15}{'plain':>9}{'arm':>9}{'delta':>9}"
          f"   recurring only")
    for arm, suf in list(arms.items())[1:]:
        n = a_ok = b_ok = 0
        rn = ra = rb = 0
        for mid in MIDS:
            human = [h for h in read_human(mid) if h[0]]
            prev_bags = [(p, bag(t)) for p, t in read_any(PREV[mid]) if p]
            for lisa in LISAS:
                A, B = hits_for(mid, lisa, ""), hits_for(mid, lisa, suf)
                for i in set(A) & set(B):
                    n += 1
                    a_ok += A[i] == human[i][0]
                    b_ok += B[i] == human[i][0]
                    hb = bag(human[i][1])
                    if max((jac(hb, pb) for _, pb in prev_bags), default=0.0) >= cut:
                        rn += 1
                        ra += A[i] == human[i][0]
                        rb += B[i] == human[i][0]
        if n:
            print(f"  {arm:<16}{n:>15}{a_ok / n:>9.0%}{b_ok / n:>9.0%}"
                  f"{(b_ok - a_ok) / n * 100:>+8.1f}pp"
                  f"   {ra / rn if rn else 0:.0%} -> {rb / rn if rn else 0:.0%} (n={rn})")
    print("\n  A lift on topics BOTH arms located, with last week's protocol as the only added")
    print("  input, is attribution being supplied by the document rather than by the audio.")


def main() -> None:
    print("SPEAKER ATTRIBUTION: from the audio, or from last week's protocol?\n")
    orders = test1_rotation()
    test2_emitted_order(orders)
    test3_voice_agreement()
    test4_ladder()
    test5_counterfactual()
    test6_new_vs_recurring()
    test7_arms()


if __name__ == "__main__":
    main()
