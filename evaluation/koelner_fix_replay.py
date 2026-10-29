"""Replays speaker naming, name flagging and glossary respelling with the original and a corrected Kölner Phonetik coder.

In koelner() the neighbor letter nxt/prv is "" at a word boundary, and in Python "" in "csz" is True. So a word-final
d/t is coded 8 (standard 2), a word-final c is coded 4 (standard 8) and a word-initial x is coded 8 (standard 48);
Schmidt gives 8628 instead of the standard 862. The corrected version guards every membership test with a non-empty
check (FIXES). It is compiled from the source of koelner() in entity_correct, entity_lexicon and handover_naming and
swapped in at run time; no module file is changed.

For both versions and for both recognizer trios of compare_asr_trio, each meeting's handover naming, name-check
flags and replayed speaker names are computed and the names are compared with the v10 run log; in addition, each
protocol of the series is respelled with its predecessor's glossary. The script prints every meeting and trio where
the two versions differ, and the totals of flag words and respelling substitutions.

    python koelner_fix_replay.py    -> goldset/koelner_fix_replay_2026-10-10.json
"""
import inspect
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import os
os.chdir(REPO)

import entity_correct
import entity_lexicon
import handover_naming
import compare_asr_trio as C
import protokoll as P
from handover_naming import name_speakers
from term_source import PREVIOUS, glossary_for_document

MODS = [entity_correct, entity_lexicon, handover_naming]
ORIG = {m: m.koelner for m in MODS}

FIXES = [
    ('nxt in "csz"', '(nxt != "" and nxt in "csz")'),
    ('nxt in "ahkloqrux"', '(nxt != "" and nxt in "ahkloqrux")'),
    ('nxt not in "ahkoqux"', '(nxt == "" or nxt not in "ahkoqux")'),
    ('nxt in "ahkoqux"', '(nxt != "" and nxt in "ahkoqux")'),
    ('prev in "ckq"', '(prev != "" and prev in "ckq")'),
    ('prv in "ckq"', '(prv != "" and prv in "ckq")'),
    ('prev in "sz"', '(prev != "" and prev in "sz")'),
    ('prv in "sz"', '(prv != "" and prv in "sz")'),
]


def fixed_version(mod):
    src = inspect.getsource(mod.koelner)
    for a, b in FIXES:
        # 'nxt not in' must be replaced before 'nxt in "ahkoqux"' can match inside it; use a regex on whole tokens
        src = re.sub(r"(?<![\w(])" + re.escape(a), b, src)
    ns = dict(mod.__dict__)
    exec(compile(src, f"<fixed {mod.__name__}>", "exec"), ns)
    return ns["koelner"], src


FIXED = {}
for m in MODS:
    FIXED[m], src = fixed_version(m)


def use(which):
    for m in MODS:
        m.koelner = ORIG[m] if which == "buggy" else FIXED[m]


for which in ("buggy", "fixed"):
    use(which)
    print(which, {m.__name__: (m.koelner("Schmidt"), m.koelner("Schmitt"), m.koelner("Bloc"), m.koelner("Xaver"))
                  for m in MODS})


def chain_protocols():
    """The Meeting_2026-*.md protocols of the series in date order."""
    files = sorted(REPO.glob("Meeting_2026-*.md"), key=lambda p: re.search(r"\d{4}-\d\d-\d\d", p.name).group(0))
    return files


def run(which):
    use(which)
    out = {}
    for mid in C.MEETINGS:
        main_tx = C.read(C.MAIN.format(mid=mid))
        roster = P.roster_from_document(REPO / PREVIOUS[mid])
        text = (C.DELIVERED / mid / "protocol_pro.md").read_text(encoding="utf-8")
        hist = [REPO / h for h in C.history(mid)]
        rec = {}
        for tag, models in C.TRIOS.items():
            others = [C.read(C.OTHER.format(model=m, mid=mid)) for m in models]
            hand = name_speakers([main_tx] + others, roster) if roster else {}
            rows, _ = P.names_to_check(main_tx, others, REPO / PREVIOUS[mid], hist, text)
            names, logged = C.replay_naming(mid, hand)
            rec[tag] = {"hand": hand, "flags": C.flag_lines(rows), "names": names, "logged": logged}
        out[mid] = rec
    # respelling safety test: each protocol of the series with its predecessor's glossary, no provenance guard
    prots = chain_protocols()
    subs = {}
    for prev, cur in zip(prots, prots[1:]):
        _, log = P.glossary_respell(cur.read_text(encoding="utf-8"), glossary_for_document(prev), None)
        subs[cur.name] = log
    out["_respell"] = subs
    return out


A = run("buggy")
B = run("fixed")
diff = 0
import json
summary = {}
for mid in C.MEETINGS:
    for tag in C.TRIOS:
        a, b = A[mid][tag], B[mid][tag]
        rec = {}
        if a["hand"] != b["hand"]:
            rec["hand"] = {"buggy": a["hand"], "fixed": b["hand"]}
        if a["names"] != b["names"]:
            rec["names"] = {"buggy": a["names"], "fixed": b["names"]}
        fa, fb = set(a["flags"]), set(b["flags"])
        if fa != fb:
            rec["flags_removed"] = sorted(x.split(" — ")[0] for x in fa - fb)
            rec["flags_added"] = sorted(x.split(" — ")[0] for x in fb - fa)
        rec["n_flag_words"] = (len(a["flags"]), len(b["flags"]))
        summary[f"{mid}/{tag}"] = rec
        if len(rec) > 1:
            diff += 1
            print(mid, tag, json.dumps({k: v for k, v in rec.items()}, ensure_ascii=False))
    print(mid, "| A names reproduce the v10 log with the buggy code:", A[mid]["A_evaluated"]["names"] == A[mid]["A_evaluated"]["logged"],
          "| with the fixed code:", B[mid]["A_evaluated"]["names"] == A[mid]["A_evaluated"]["logged"],
          "| B fixed names == v10 log:", B[mid]["B_proposed"]["names"] == A[mid]["A_evaluated"]["logged"])
print("respelling pairs:", len(A["_respell"]), "| substitutions buggy:", sum(len(v) for v in A["_respell"].values()),
      "fixed:", sum(len(v) for v in B["_respell"].values()))
tot = lambda R, tag: sum(len(R[m][tag]["flags"]) for m in C.MEETINGS)
print("flag words total, buggy A/B:", tot(A, "A_evaluated"), tot(A, "B_proposed"), "| fixed A/B:", tot(B, "A_evaluated"), tot(B, "B_proposed"))
json.dump(summary, open(str(REPO / "goldset" / "koelner_fix_replay_2026-10-10.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

def heard(R, tag):
    return {(m, x.split(" — ")[0]) for m in C.MEETINGS for x in R[m][tag]["flags"] if "gehört als" in x}
for lab, R in (("buggy", A), ("fixed", B)):
    a, b = heard(R, "A_evaluated"), heard(R, "B_proposed")
    print(f"{lab}: heard-as flags evaluated trio {len(a)}, final trio {len(b)}, in both {len(a & b)}")
