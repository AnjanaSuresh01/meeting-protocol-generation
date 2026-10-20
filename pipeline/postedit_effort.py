"""Bullet-level post-editing effort of protocol drafts against reference protocols.

content(), bullets() and covered() turn Markdown bullets into sets of content words and test
whether a bullet shares at least MATCH_TH of its words with any bullet of another list.
As a script, it compares TREE/<model>/<meeting>/protocol_<lisa><SUFFIX>.md with the reference
protocol of each meeting, prints ADD (reference bullets the draft misses), DELETE (draft
bullets the reference lacks) and KEEP per model and LISA model, and writes the rows to
goldset/<JSON>; a relative TREE and goldset/ are resolved from this file's folder.
The script mode needs evaluate_protocols.py, which is not part of this repository. Without it,
a run says so and exits with code 1.

    python postedit_effort.py [--tree TREE] [--suffix SUFFIX] [--json JSON]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

from thesis_pipeline.text_norm import normalize, NormConfig

REPO = Path(__file__).resolve().parent
CFG = NormConfig(casefold=True, strip_punct=True)
STOP = set("""der die das den dem des ein eine einer einem einen und oder aber auch noch nur
schon nicht kein keine ist sind war waren wird werden wurde wurden hat haben hatte hatten sich
mit von zu im in am an auf für als bei aus nach über um vor durch dass wenn weil ich er sie es
wir ihr man dann da so wie mehr sehr kann können soll sollen muss müssen übertrag aufgabe prüfen""".split())
MATCH_TH = 0.30


def content(s: str) -> set[str]:
    return {w for w in normalize(s, CFG).split() if w not in STOP and len(w) > 3}


def bullets(md: str) -> list[set[str]]:
    out = []
    for line in md.splitlines():
        s = line.strip()
        if not re.match(r"^[*-]\s+\S", s):
            continue
        c = content(s)
        if c:
            out.append(c)
    return out


def covered(item: set[str], pool: list[set[str]]) -> bool:
    return any(len(item & p) / len(item) >= MATCH_TH for p in pool if p)


def main() -> int:
    # evaluate_protocols is imported inside main() so that carry_forward, stale_flag and
    # style_exemplars can import the helpers above without it.
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default="outputs_v10",
                    help="protocol tree to score (default: the live outputs_v10)")
    ap.add_argument("--suffix", default="_fixed",
                    help="protocol variant: '_fixed' (shipping) or '' (raw model output)")
    ap.add_argument("--json", default="postedit_effort.json", help="output file name")
    args = ap.parse_args()
    try:
        from evaluate_protocols import REF, strip_backlog, OUT, LISA
    except ImportError:
        print("postedit_effort.py needs evaluate_protocols.py, which is not part of this repository.")
        return 1
    tree = Path(args.tree) if args.tree else OUT
    if not tree.is_absolute():
        tree = REPO / tree

    systems = sorted(d.name for d in tree.iterdir()
                     if d.is_dir() and any(d.glob(f"*/protocol_*{args.suffix}.md")))
    if not systems:
        print(f"no protocol_*{args.suffix}.md under {tree} — wrong tree or wrong --suffix")
        return 1

    raw = {m: (REPO / f).read_text(encoding="utf-8")
           for m, f in REF.items() if (REPO / f).exists()}
    refs = {}
    for m, t in raw.items():
        refs[m], _ = strip_backlog(t)

    print("POST-EDITING EFFORT — what is left for the human after the draft arrives")
    print("=" * 96)
    print(f"tree: {tree.name}    variant: protocol_<lisa>{args.suffix or ''}.md"
          f"{'  (RAW model output, not the shipping artefact)' if not args.suffix else ''}")
    print("ADD = the editor must write it (expensive)   DELETE = the editor must remove it (cheap)")
    print("Backlog excluded from the reference; the system is not meant to produce it.\n")
    print(f"{'model':17s}{'lisa':6s}{'n':>3}{'ref bullets':>12}{'draft bullets':>14}"
          f"{'ADD':>7}{'DELETE':>8}{'KEEP':>6}{'add rate':>10}")

    rows = []
    for model in systems:
        for lisa in LISA:
            adds = dels = keeps = nref = nhyp = 0
            n = 0
            for mid, ref in refs.items():
                p = tree / model / mid / f"protocol_{lisa}{args.suffix}.md"
                if not p.exists() or p.stat().st_size < 5:
                    continue
                n += 1
                rb = bullets(ref)
                hb = bullets(p.read_text(encoding="utf-8"))
                nref += len(rb)
                nhyp += len(hb)
                a = sum(1 for r in rb if not covered(r, hb))
                d = sum(1 for h in hb if not covered(h, rb))
                adds += a
                dels += d
                keeps += len(rb) - a
            if not n:
                continue
            rate = adds / nref if nref else 0.0
            rows.append({"model": model, "lisa": lisa, "n": n, "ref_bullets": nref,
                         "draft_bullets": nhyp, "add": adds, "delete": dels,
                         "keep": keeps, "add_rate": rate})
            print(f"{model:17s}{lisa:6s}{n:>3}{nref:>12}{nhyp:>14}"
                  f"{adds:>7}{dels:>8}{keeps:>6}{rate:>9.0%}")

    rows.sort(key=lambda r: r["add_rate"])
    print(f"\nRANKED BY THE EXPENSIVE OPERATION (lowest ADD rate = least writing left to do)")
    for i, r in enumerate(rows[:6], 1):
        print(f"  {i}. {r['model']:17s}{r['lisa']:6s} ADD {r['add_rate']:.0%}"
              f"   DELETE {r['delete']} lines")

    print("\n  Read the two columns separately. A draft that invents extra lines costs the editor")
    print("  a glance and a keystroke; a draft that omits something costs noticing that it is")
    print("  missing at all. They are not interchangeable and must not be averaged.")
    print("\n  NOT measured post-editing effort: the reference was written independently, so some")
    print("  ADDs are rewordings rather than defects. These are upper bounds on the real work.")

    out = REPO / "goldset" / args.json
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwrote -> {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
