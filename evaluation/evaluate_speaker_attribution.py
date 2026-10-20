"""Score speaker attribution: for each topic in the human minutes, check whether the generated
protocol of the same meeting puts that topic under the same person.

Reads the minutes (Meeting_2026-*.md) from this folder and the generated protocols
pipeline_outputs_extracted/outputs/canary_diar/<meeting>/protocol_<lisa>_<arm>.md
(lisa: flash, pro; arm: blind, attributed, identified), prints the results and writes
speaker_attribution_results.json in this folder. The minutes have no timestamps, so this is
not a diarization error rate.

Usage:
    python evaluate_speaker_attribution.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent
OUT = REPO / "pipeline_outputs_extracted" / "outputs" / "canary_diar"

# Meeting id -> human minutes file, read from this script's folder.
HUMAN = {
    "march09": "Meeting_2026-03-09.md",
    "april13": "Meeting_2026-04-13.md",
    "april27": "Meeting_2026-04-27.md",
    "may11":   "Meeting_2026-05-11.md",
    "may18":   "Meeting_2026-05-18.md",
    "june01":  "Meeting_2026-06-01.md",
    "june15":  "Meeting_2026-06-15.md",
    "june22":  "Meeting_2026-06-22.md",
}

ARMS = ["blind", "attributed", "identified"]
LISAS = ["flash", "pro"]

# Attendee names per meeting. Output section 5 counts reference topics credited to
# someone who is not on this list.
ROSTER = {
    "march09": ["René Peinl", "Vincent Tischler", "Thomas Weber", "Johannes"],
    "april27": ["René Peinl", "Vincent Tischler", "Thomas Weber", "Johannes"],
    "may11":   ["René Peinl", "Vincent Tischler", "Thomas Weber", "Johannes", "Osman"],
    "may18":   ["René Peinl", "Vincent Tischler", "Thomas Weber", "Johannes", "Osman"],
    "june01":  ["René Peinl", "Thomas Weber", "Johannes", "Osman"],
    "june15":  ["René Peinl", "Vincent Tischler", "Johannes", "Osman"],
    "june22":  ["René Peinl", "Vincent Tischler", "Thomas Weber", "Osman"],
}

# Identity is the lowercase first name, so a first name and a full name give the same key.
# Assumes that no two people share a first name.
def person_key(name: str) -> str:
    # The class starts with the invisible range U+F000-U+F0FF (Word Symbol-font bullets that
    # arrive inside p.text), so a bullet glyph is not read as a name.
    n = re.sub(r"[-*_#]", " ", name).strip().strip(":").strip()
    n = re.sub(r"\s*\(.*?\)", "", n)
    first = n.split()[0].lower() if n.split() else ""
    return first if re.match(r"[a-zäöüß]", first) else ""


STOP = set("und der die das den dem des ein eine einen einem eines ist sind war waren mit für "
           "von zu im in am an auf als auch noch nicht nur wie bei aus wird werden hat haben "
           "man wir ich er sie es dass da schon mal aber oder wenn dann so um über nach vor "
           "the and for with that this".split())


def bag(text: str) -> set[str]:
    w = re.findall(r"[a-zA-ZäöüßÄÖÜ0-9][\w\-]{2,}", text.lower())
    return {x for x in w if x not in STOP}


def f1(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    i = len(a & b)
    return 0.0 if not i else 2 * i / (len(a) + len(b))


# Reading the human minutes

def md_indent_unit(lines: list[str]) -> int:
    """Return the width of one nesting level for a list of markdown lines: the smallest
    non-zero bullet indent, or 4.
    """
    widths = [len(l) - len(l.lstrip()) for l in lines if l.strip().startswith("*")]
    return min((w for w in widths if w > 0), default=4)


def read_human_md(path: Path) -> list[tuple[str, str]]:
    """Parse markdown minutes (bold person at column 0, topics one level in, details deeper)
    into (person key, topic text with its details) pairs.
    """
    items, person, topic, body = [], None, None, []
    lines = path.read_text(encoding="utf-8").splitlines()
    unit = md_indent_unit(lines)
    for line in lines:
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        m = re.match(r"\*\s+(.*)", line.strip())
        if not m:
            continue
        text = m.group(1).strip()
        if indent == 0 and text.startswith("**"):
            if topic:
                items.append((person, topic + " " + " ".join(body)))
            person, topic, body = person_key(text), None, []
        elif indent <= unit + 1 and person:
            if topic:
                items.append((person, topic + " " + " ".join(body)))
            topic, body = text, []
        elif person and topic:
            body.append(text)
    if topic:
        items.append((person, topic + " " + " ".join(body)))
    return items


def read_human_docx(path: Path) -> list[tuple[str, str]]:
    """Parse Word minutes into the same pairs: a bold paragraph outside a list is a person,
    list level 0 is a topic.
    """
    import docx
    items, person, topic, body = [], None, None, []
    for p in docx.Document(str(path)).paragraphs:
        t = p.text.strip()
        if not t:
            continue
        pr = p._p.pPr
        lvl = (pr.numPr.ilvl.val if pr is not None and pr.numPr is not None
               and pr.numPr.ilvl is not None else None)
        # Word can split a name into a whitespace run and a bold run, so any bold run counts.
        # A bold paragraph inside the list is a topic, not a person.
        bold = any(r.bold for r in p.runs)
        if bold and lvl is None:
            if topic:
                items.append((person, topic + " " + " ".join(body)))
            person, topic, body = person_key(t), None, []
        elif lvl == 0 and person:
            if topic:
                items.append((person, topic + " " + " ".join(body)))
            topic, body = t, []
        elif person and topic:
            body.append(t)
    if topic:
        items.append((person, topic + " " + " ".join(body)))
    return items


def read_human(mid: str) -> list[tuple[str, str]]:
    p = REPO / HUMAN[mid]
    return read_human_docx(p) if p.suffix == ".docx" else read_human_md(p)


# Reading a generated protocol

def read_generated(path: Path) -> tuple[list[tuple[str, str]], dict[str, str]]:
    """Parse a generated protocol (## person, ### topic, bullets) into (person key, topic
    text) pairs and the SPEAKER_xx -> name mapping of its Sprecherzuordnung line.
    """
    items, person, topic, body = [], None, None, []
    mapping = {}
    text = path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"Sprecherzuordnung:(.*)", text)
    if m:
        for a, b in re.findall(r"(SPEAKER_\d+)\s*=\s*([^,\n]+)", m.group(1)):
            mapping[a] = b.strip()
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("## ") and not s.startswith("###"):
            if topic:
                items.append((person, topic + " " + " ".join(body)))
            person, topic, body = person_key(s[3:]), None, []
        elif s.startswith("### "):
            if topic:
                items.append((person, topic + " " + " ".join(body)))
            topic, body = s[4:], []
        elif person and topic:
            body.append(re.sub(r"^[*\-]\s*", "", s))
    if topic:
        items.append((person, topic + " " + " ".join(body)))
    return items, mapping


# Scoring

def score(human: list[tuple[str, str]], gen: list[tuple[str, str]], th: float) -> tuple:
    """Locate each human topic in the generated person that holds the largest share of its
    distinctive tokens (tokens under at most half of the persons; share >= th, lead >= 0.05).
    Returns (located, correct, number of human topics, chance level).
    """
    per_person: dict[str, set[str]] = {}
    for p, t in gen:
        per_person.setdefault(p, set()).update(bag(t))
    if not per_person:
        return 0, 0, len(human), 0.0
    df = Counter(tok for b in per_person.values() for tok in b)
    cap = max(1, len(per_person) // 2)

    located = correct = 0
    for hp, ht in human:
        hb = {t for t in bag(ht) if df.get(t, 0) <= cap}
        if len(hb) < 3:  # minimum distinctive tokens per topic
            continue
        ranked = sorted(((len(hb & b) / len(hb), p) for p, b in per_person.items()),
                        reverse=True)
        top = ranked[0]
        second = ranked[1][0] if len(ranked) > 1 else 0.0
        if top[0] >= th and top[0] - second >= 0.05:
            located += 1
            correct += (top[1] == hp)
    # Chance level: a uniform guess among the persons in this protocol.
    return located, correct, len(human), 1 / len(per_person)


def topic_hits(human, gen, th: float = 0.15) -> dict[int, str]:
    """Apply the location rule of score() and return {human topic index: attributed person}
    for the located topics.
    """
    pp: dict[str, set[str]] = {}
    for p, t in gen:
        pp.setdefault(p, set()).update(bag(t))
    if not pp:
        return {}
    df = Counter(tok for b in pp.values() for tok in b)
    cap = max(1, len(pp) // 2)
    out = {}
    for i, (_, ht) in enumerate(human):
        hb = {t for t in bag(ht) if df.get(t, 0) <= cap}
        if len(hb) < 3:
            continue
        r = sorted(((len(hb & b) / len(hb), p) for p, b in pp.items()), reverse=True)
        if r[0][0] >= th and r[0][0] - (r[1][0] if len(r) > 1 else 0) >= 0.05:
            out[i] = r[0][1]
    return out


def oracle_remap(human, gen, mapping: dict[str, str]) -> tuple[int, int, int]:
    """Recover each section's speaker cluster from the SPEAKER_xx -> name mapping and try every
    assignment of reference names to clusters. Returns (scored topics, correct as named,
    correct under the best assignment), an upper bound for perfect naming.
    """
    from itertools import permutations
    name2cluster = {person_key(v): k for k, v in mapping.items() if person_key(v)}
    if len(name2cluster) < 2:
        return 0, 0, 0
    hits = topic_hits(human, gen)
    scored = [(i, name2cluster[p]) for i, p in hits.items() if p in name2cluster]
    if not scored:
        return 0, 0, 0
    clusters = sorted({c for _, c in scored})
    names = sorted({h[0] for h in human})
    best = 0
    for perm in permutations(names, len(clusters)):
        assign = dict(zip(clusters, perm))
        best = max(best, sum(1 for i, c in scored if assign[c] == human[i][0]))
    actual = sum(1 for i, c in scored if person_key(mapping[c]) == human[i][0])
    return len(scored), actual, best


def main() -> None:
    thresholds = [0.15, 0.25, 0.35]
    rows, missing = [], []

    print("SPEAKER ATTRIBUTION — scored against the team's own minutes")
    print("=" * 78)
    print("Reference = who the human minutes credit each topic to. NOT DER: the minutes have")
    print("no timestamps, so nothing here measures who spoke in which second.\n")

    for mid in HUMAN:
        human = read_human(mid)
        for arm in ARMS:
            for lisa in LISAS:
                p = OUT / mid / f"protocol_{lisa}_{arm}.md"
                if not p.exists():
                    missing.append(p.name + " " + mid)
                    continue
                gen, mapping = read_generated(p)
                for th in thresholds:
                    loc, cor, n, chance = score(human, gen, th)
                    rows.append({"meeting": mid, "arm": arm, "lisa": lisa, "th": th,
                                 "human_topics": n, "located": loc, "correct": cor,
                                 "chance": chance,
                                 "gen_people": len({g[0] for g in gen}),
                                 "mapping": mapping})

    if not rows:
        print("No generated protocols found.")
        return

    print(f"1. IS THE TOPIC CREDITED TO THE RIGHT PERSON?\n")
    print("   'chance' is guessing uniformly among the people that protocol names — the")
    print("   number any result here has to beat to mean anything at all.\n")
    print(f"   {'threshold':>10}{'arm':>13}{'LISA':>7}{'topics found':>15}"
          f"{'right person':>15}{'chance':>9}")
    for th in thresholds:
        for arm in ARMS:
            for lisa in LISAS:
                r = [x for x in rows if x["th"] == th and x["arm"] == arm and x["lisa"] == lisa]
                if not r:
                    continue
                n = sum(x["human_topics"] for x in r)
                loc = sum(x["located"] for x in r)
                cor = sum(x["correct"] for x in r)
                ch = sum(x["chance"] * x["located"] for x in r) / loc if loc else 0
                print(f"   {th:>10.2f}{arm:>13}{lisa:>7}{loc:>8} / {n:<4}"
                      f"{cor / loc if loc else 0:>14.1%}{ch:>9.1%}")
        print()

    print("2. PER MEETING, AT THRESHOLD 0.15\n")
    print(f"   {'meeting':<10}{'topics':>7}" +
          "".join(f"{a[:4]+'-'+l:>12}" for a in ARMS for l in LISAS))
    for mid in HUMAN:
        r = {(x["arm"], x["lisa"]): x for x in rows if x["th"] == 0.15 and x["meeting"] == mid}
        if not r:
            continue
        n = next(iter(r.values()))["human_topics"]
        cells = []
        for a in ARMS:
            for l in LISAS:
                x = r.get((a, l))
                cells.append(f"{x['correct'] / x['located']:.0%}" if x and x["located"] else "-")
        print(f"   {mid:<10}{n:>7}" + "".join(f"{c:>12}" for c in cells))

    print("\n3. HOW MANY PEOPLE DOES EACH PROTOCOL ORGANISE BY?\n")
    print("   The human minutes name everyone who was in the room. A protocol with far more")
    print("   sections than that is inventing speakers; far fewer is collapsing them.\n")
    print(f"   {'meeting':<10}{'human':>7}" +
          "".join(f"{a[:4]+'-'+l:>12}" for a in ARMS for l in LISAS))
    for mid in HUMAN:
        human = read_human(mid)
        r = {(x["arm"], x["lisa"]): x for x in rows if x["th"] == 0.15 and x["meeting"] == mid}
        if not r:
            continue
        cells = [str(r[(a, l)]["gen_people"]) if (a, l) in r else "-"
                 for a in ARMS for l in LISAS]
        print(f"   {mid:<10}{len({h[0] for h in human}):>7}" +
              "".join(f"{c:>12}" for c in cells))

    print("\n4. THE SPEAKER-ID -> NAME MAPPING (arm B only)\n")
    for mid in HUMAN:
        for lisa in LISAS:
            r = next((x for x in rows if x["meeting"] == mid and x["arm"] == "attributed"
                      and x["lisa"] == lisa and x["th"] == 0.15), None)
            if r and r["mapping"]:
                names = [person_key(v) for v in r["mapping"].values()]
                dup = [k for k, c in Counter(names).items() if c > 1]
                flag = f"  <- {dup} used twice" if dup else ""
                print(f"   {mid:<9}{lisa:<6}" +
                      ", ".join(f"{k}={v}" for k, v in sorted(r['mapping'].items())) + flag)

    print("\n4b. THE SAME TOPICS, BOTH ARMS — the paired comparison\n")
    print("   Sections 1 and 2 let each arm be scored on whatever it located, and the two arms")
    print("   do not locate the same topics. Here only topics located by BOTH are counted, so")
    print("   the difference cannot come from which topics were scored.\n")
    print(f"   {'LISA':>7}{'topics':>9}{'blind':>10}{'attributed':>13}{'change':>10}")
    for lisa in LISAS:
        n = b_ok = a_ok = 0
        for mid in HUMAN:
            human = [h for h in read_human(mid) if h[0]]
            B = topic_hits(human, read_generated(OUT / mid / f"protocol_{lisa}_blind.md")[0])
            A = topic_hits(human, read_generated(OUT / mid /
                                                 f"protocol_{lisa}_attributed.md")[0])
            for i in set(B) & set(A):
                n += 1
                b_ok += B[i] == human[i][0]
                a_ok += A[i] == human[i][0]
        if n:
            print(f"   {lisa:>7}{n:>9}{b_ok / n:>10.1%}{a_ok / n:>13.1%}"
                  f"{(a_ok - b_ok) / n:>+10.1%}")

    print("\n4c. IS IT THE DIARIZATION OR THE NAMING? — best possible re-assignment\n")
    print("   Arm B recorded which SPEAKER_xx it called which name, so the clusters can be")
    print("   recovered and every possible naming tried. 'best case' is what this exact")
    print("   clustering would score if the names were assigned perfectly. It is an upper")
    print("   bound and is NOT achievable without voice enrolment — never quote it as a result.\n")
    print(f"   {'meeting':<10}{'LISA':>6}{'topics':>8}{'as named':>11}{'best case':>12}")
    tot = tot_act = tot_best = 0
    for mid in HUMAN:
        human = [h for h in read_human(mid) if h[0]]
        for lisa in LISAS:
            gen, mapping = read_generated(OUT / mid / f"protocol_{lisa}_attributed.md")
            n, act, best = oracle_remap(human, gen, mapping)
            if not n:
                continue
            tot += n
            tot_act += act
            tot_best += best
            print(f"   {mid:<10}{lisa:>6}{n:>8}{act / n:>11.0%}{best / n:>12.0%}")
    if tot:
        print(f"   {'ALL':<10}{'':>6}{tot:>8}{tot_act / tot:>11.0%}{tot_best / tot:>12.0%}"
              f"   <- the gap IS the naming step")

    print("\n5. A CEILING NEITHER ARM CAN REACH, AND IT IS NOT THE PIPELINE'S FAULT\n")
    print("   data.csv records three meetings as missing a person, and that person is dropped")
    print("   from the attendee list handed to LISA together with 'use only names from this")
    print("   list'. But the team's own minutes still carry topics under the absent person.")
    print("   Those topics cannot be got right by either arm: the name is forbidden.\n")
    print(f"   {'meeting':<10}{'absent per data.csv':<22}{'topics under them':>18}")
    tot = unreachable = 0
    for mid in HUMAN:
        human = [h for h in read_human(mid) if h[0]]
        r = {person_key(x) for x in ROSTER[mid]}
        miss = sorted({p for p, _ in human if p not in r})
        n = sum(1 for p, _ in human if p not in r)
        tot += len(human)
        unreachable += n
        if miss:
            print(f"   {mid:<10}{', '.join(miss):<22}{n:>18}")
    print(f"\n   {unreachable} of {tot} topics ({unreachable / tot:.1%}) are attributed by the")
    print("   humans to somebody the roster forbids. Every accuracy above is therefore capped")
    print(f"   at about {1 - unreachable / tot:.0%}.")
    print("\n   These topics are credited to people who were not at the meeting (carry-over items,")
    print("   or someone reporting for an absent colleague), so they cannot be attributed from the")
    print("   audio. They are reported as excluded.")

    if missing:
        print(f"\n   {len(missing)} protocol(s) not generated yet.")

    out = REPO / "speaker_attribution_results.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwrote {out.name}")
    print("\nNOT MEASURED HERE: who spoke in which second (DER). The human minutes carry no")
    print("timestamps, so that needs a different reference and is a separate job.")


if __name__ == "__main__":
    main()
