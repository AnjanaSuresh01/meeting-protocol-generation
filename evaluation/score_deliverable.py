"""Scores the protocols of a protokoll.py run (outputs_deliverable/<tag>/) with the same instruments as the corpus arms.

  1. Headings: whether the '## <Name>' sections are the people who spoke, by the voice identities in
     goldset/speaker_identity.json; counted as named correctly, named wrongly, unnamed or duplicated.
     The name each voice label gets in run_pro.log is also compared with its identity, which catches
     two voices whose names are swapped.
  2. Attribution: ladder_split's topic matching of the topics of the reference minutes, for the run
     and for the shipped corpus arm of Canary-1B-v2 with LISA Pro, on the same topics with the same matcher.

    python score_deliverable.py v2   -> stdout only
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from evaluate_speaker_attribution import read_human, read_generated, person_key, bag, topic_hits
import ladder_split as L

tag = sys.argv[1] if len(sys.argv) > 1 else "v2"
D = REPO / "outputs_deliverable" / tag
ident = json.loads((REPO / "goldset/speaker_identity.json").read_text(encoding="utf-8"))["mapping"]

print(f"DELIVERABLE ({tag}) — HEADINGS")
h = Counter()
for mid in sorted(ident):
    p = D / mid / "protocol_pro.md"
    if not p.exists():
        continue
    spoke = {person_key(n) for n in ident[mid].values()}
    heads = [l[3:].strip() for l in p.read_text(encoding="utf-8").splitlines() if l.startswith("## ")]
    named = [x for x in heads if not re.match(r"(Sprecher \d+|Nicht zugeordnete)", x)]
    ok = [x for x in named if person_key(x) in spoke]
    bad = [x for x in named if person_key(x) not in spoke]
    unn = len(heads) - len(named)
    dup = len(named) - len({person_key(x) for x in named})
    h["heads"] += len(heads); h["ok"] += len(ok); h["bad"] += len(bad); h["unnamed"] += unn; h["dup"] += dup
    h["people"] += len(spoke); h["covered"] += len(spoke & {person_key(x) for x in named})
    print(f"  {mid:8s} {len(heads)} sections: named right {len(ok)}, wrong {bad or 0}, unnamed {unn}"
          f"{f', duplicated {dup}' if dup else ''}   (people who spoke: {len(spoke)})")
print(f"  TOTAL sections {h['heads']}: right {h['ok']}, wrong {h['bad']}, unnamed {h['unnamed']}, duplicated {h['dup']} | "
      f"people who spoke and got their own named section: {h['covered']}/{h['people']}")
# The heading check above only asks whether a heading is a person who spoke, so two voices whose
# names are swapped pass it. Each voice label's name is therefore compared with its identity.
vn = Counter(); swaps = []
for mid in sorted(ident):
    lp = D / mid / "run_pro.log"
    if not lp.exists():
        continue
    m = {}
    for line in lp.read_text(encoding="utf-8").splitlines():
        if "speaker names (voice, handover, topic):" in line:
            m = dict(re.findall(r"(SPEAKER_\d+)=([^,\n]+)", line))
    for lab, true in ident[mid].items():
        vn["n"] += 1
        if lab not in m:
            vn["unnamed"] += 1
        elif person_key(m[lab]) == person_key(true):
            vn["ok"] += 1
        else:
            swaps.append(f"{mid} {lab}: {m[lab].strip()} (is {true})")
print(f"  VOICE -> NAME against the voice identities: correct {vn['ok']}/{vn['n']}, wrong {len(swaps)}, "
      f"unnamed {vn['unnamed']}" + (f"  {swaps}" if swaps else ""))

print(f"\nATTRIBUTION, all topics of the reference minutes, topic matching as ladder_split")
hand = json.loads((REPO / "goldset/ladder_hand_labels.json").read_text(encoding="utf-8"))["labels"]
hl = {(x["meeting"], x["owner"], x.get("text", "")[:60]): x["hand_spoken"] for x in hand}
res = Counter()
for mid in L.MIDS:
    human = [x for x in read_human(mid) if x[0]]
    arms = {"deliverable": D / mid / "protocol_pro.md",
            "corpus shipped (Canary, Pro)": L.V10 / "canary_diar" / mid / "protocol_pro_fixed_carry.md"}
    got = {a: topic_hits(human, read_generated(p)[0], L.TH) if p.exists() else {} for a, p in arms.items()}
    for i, (owner, text) in enumerate(human):
        if len(bag(text)) < 3:
            continue
        for a, g in got.items():
            if i in g:
                res[(a, "loc")] += 1
                res[(a, "ok")] += g[i] == owner
        if all(i in g for g in got.values()):
            res["both"] += 1
            for a, g in got.items():
                res[(a, "both_ok")] += g[i] == owner
for a in ("deliverable", "corpus shipped (Canary, Pro)"):
    print(f"  {a:30s} located {res[(a,'loc')]:3d}  correct {res[(a,'ok')] / max(1, res[(a,'loc')]):.1%}   "
          f"| on the {res['both']} topics both locate: {res[(a,'both_ok')] / max(1, res['both']):.1%}")
