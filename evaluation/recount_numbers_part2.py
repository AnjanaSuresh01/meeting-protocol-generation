"""Counts structure, name spelling, attribution, style, gold-set and speaker-naming figures of the protocol runs.

Each numbered section is computed from the files under outputs_deliverable/ and goldset/:

  13  bullet nesting in the v6 and v10 protocols (share of bullets that are sub-points of another bullet),
      the person and topic headings of v10, and the number of v10 run logs whose self-check reports that
      every structural rule passes
  14  occurrences of the correct and of the wrong spellings of the four names in `pairs` in the v10 protocols
  15  the person headings under which the terms "Personalrat" and "OlaPh" appear in the v10 protocols
  16  share of bare noun-phrase openers (style_flag.scan) among the language-model bullets (.neu.md) of v6,
      v8 and v10, with a Fisher exact test of v6 against v10 (bullets as units)
  17  composition of goldset/goldset_v10.json: clips per difficulty decile, clip length, meetings, resolution
  18  which step named each voice in the v10 run logs (voice profile, handover, topic)

    python recount_numbers_part2.py    -> stdout only
"""
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
MEET = ["march09", "april13", "april27", "may11", "may18", "june01", "june15", "june22"]
BUL = re.compile(r"^(\s*)[*-]\s+\S")


def section(t):
    print("\n" + "=" * 96 + f"\n{t}\n" + "=" * 96)


def prot(arm, mid, neu=False):
    p = REPO / f"outputs_deliverable/{arm}/{mid}/protocol_pro{'.neu' if neu else ''}.md"
    return p.read_text(encoding="utf-8") if p.exists() else None


def main():
    # 13 ----------------------------------------------------------------------------------
    section("13. STRUCTURE — are all bullet points on the same level?")
    from evaluate_speaker_attribution import md_indent_unit
    for arm in ("v6", "v10"):
        top = sub = 0
        for mid in MEET:
            t = prot(arm, mid)
            if t is None:
                continue
            unit = md_indent_unit(t.splitlines()) or 2   # needs a list of lines; a plain string gives the default 4
            for line in t.splitlines():
                m = BUL.match(line)
                if m:
                    if len(m.group(1).expandtabs(4)) >= unit:
                        sub += 1
                    else:
                        top += 1
        tot = top + sub
        print(f"   {arm:4s} final protocols: {tot} bullets under '## person / ### topic' headings; "
              f"{sub} of them are sub-points of another bullet ({100 * sub / tot:.1f}%)")
    heads = sum(len(re.findall(r"(?m)^### ", prot("v10", m))) for m in MEET)
    pers = sum(len(re.findall(r"(?m)^## (?!#)", prot("v10", m))) for m in MEET)
    print(f"   v10 levels used: {pers} person headings, {heads} topic headings, bullets, sub-bullets")
    ok = 0
    for mid in MEET:
        log = (REPO / f"outputs_deliverable/v10/{mid}/run_pro.log").read_text(
            encoding="utf-8", errors="replace")
        ok += "passes every structural rule" in log
    print(f"   self-check 'passes every structural rule': {ok}/8 meetings")

    # 14 ----------------------------------------------------------------------------------
    section("14. FOUR NAMES — occurrences across the 8 v10 protocols")
    pairs = [("MOSS-TTS", r"MOSS[- ]?TTS", r"\bMOS[- ]TTS"),
             ("Voxtral", r"Voxtral", r"VOX-?TRAAL|Voxtraal|Vox-Tral"),
             ("Magpie", r"Magpie", r"Mac-?Pipe|Mack-?Pipe|Magpipe"),
             ("Leuoth", r"Leuoth", r"\bLloyd\b")]
    for name, good, bad in pairs:
        g = b = 0
        where_b = []
        for mid in MEET:
            t = prot("v10", mid)
            g += len(re.findall(good, t))
            nb = len(re.findall(bad, t, re.I))
            b += nb
            if nb:
                where_b.append(mid)
        print(f"   {name:9s} correct {g:3d}   wrong form {b:2d}  {where_b if where_b else ''}")

    # 15 ----------------------------------------------------------------------------------
    section("15. TWO MIS-FILED ITEMS")
    from evaluate_speaker_attribution import person_key
    for kw, owner in (("Personalrat", "thomas"), ("OlaPh", "johannes")):
        c = Counter()
        for mid in MEET:
            who = None
            for line in prot("v10", mid).splitlines():
                if line.startswith("## ") and not line.startswith("###"):
                    who = person_key(line[3:])
                elif kw.lower() in line.lower() and who:
                    c[who] += 1
        tot = sum(c.values())
        print(f"   '{kw}': {c[owner]} of {tot} mentions under {owner}   {dict(c)}")

    # 16 ----------------------------------------------------------------------------------
    section("16. STYLE ACROSS VERSIONS — bare noun-phrase openers in LISA's NEW bullets")
    import style_flag as SF
    for arm, what in (("v6", "copy-forward, no style examples"),
                      ("v8", "+ sentences of earlier minutes as examples"),
                      ("v10", "final")):
        fl = tot = 0
        for mid in MEET:
            t = prot(arm, mid, neu=True)
            if t is None:
                continue
            fl += len(SF.scan(t))
            tot += sum(1 for ln in t.splitlines() if BUL.match(ln))
        print(f"   {arm:4s} {fl:4d} of {tot:5d} = {100 * fl / tot:5.2f}%   ({what})")
    from scipy.stats import fisher_exact

    def cnt(arm):
        fl = tot = 0
        for mid in MEET:
            t = prot(arm, mid, neu=True)
            fl += len(SF.scan(t)); tot += sum(1 for ln in t.splitlines() if BUL.match(ln))
        return fl, tot
    a, b = cnt("v6"), cnt("v10")
    p = fisher_exact([[a[0], a[1] - a[0]], [b[0], b[1] - b[0]]])[1]
    print(f"   v6 vs v10 (Fisher exact, bullets as units): p = {p:.2g}")

    # 17 ----------------------------------------------------------------------------------
    section("17. HOW THE GOLD SET WAS DRAWN — split according to the overall population")
    import json
    g = json.loads((REPO / "goldset/goldset_v10.json").read_text(encoding="utf-8"))["segments"]
    dec = Counter(s["decile"] for s in g)
    print(f"   clips per difficulty decile: {dict(sorted(dec.items()))}")
    dur = sorted(s["duration_s"] for s in g)
    print(f"   clip length: median {dur[len(dur) // 2]:.1f} s, range {dur[0]:.1f}-{dur[-1]:.1f} s")
    print(f"   meetings covered: {len(set(s['meeting_id'] for s in g))}")
    print(f"   resolution: {dict(Counter(s['resolution']['state'] for s in g))}")

    # 18 ----------------------------------------------------------------------------------
    section("18. SPEAKER NAMING — which step named each voice (v10 run logs)")
    tot = Counter()
    for mid in MEET:
        t = (REPO / f"outputs_deliverable/v10/{mid}/run_pro.log").read_text(
            encoding="utf-8", errors="replace")

        def grab(p):
            x = re.search(p + r": (.+)", t)
            return dict(y.split("=", 1) for y in x.group(1).strip().split(", ")) if x else {}
        v = grab("named by voice profile")
        tp = grab("named by topic from the previous document")
        fin = grab(r"speaker names \(voice, handover, topic\)")
        h = [k for k in fin if k not in v and k not in tp]
        tot.update(voice=len(v), handover=len(h), topic=len(tp))
    print(f"   {dict(tot)}  of {sum(tot.values())} voices")


if __name__ == "__main__":
    main()
