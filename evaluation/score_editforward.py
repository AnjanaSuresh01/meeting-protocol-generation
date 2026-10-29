"""Scores deliverable arms on post-edit effort, topic attribution and name spelling.

1. POST-EDIT, split. Each reference bullet is either NEW this week or CARRIED from last week
   (covered by the previous protocol). Any copy of last week's protocol gets the carried ones;
   the new ones are what the audio is for. ADD counts reference bullets the draft misses, DELETE
   counts draft bullets the reference does not contain. The Backlog (global and per person) and
   the reference's person-name lines are excluded from the reference, because the system does
   not produce the Backlog and writes names as '##' headings, not bullets. Topic headings are
   scored as bullets, because the reference writes its topics as bullets. postedit_effort's own
   bullet matcher (0.30) is used.
2. ATTRIBUTION on the same-topic ladder (ladder_split, Pro), Backlog topics removed, counting
   unlocated topics as wrong.
3. NAMES from goldset/name_reachability.json: right / only-wrong / absent / undecidable
   (a 'wrong form' that the reference protocols also use as a word decides nothing; see
   legitimate()).

The baseline row "last week alone" is the previous protocol with no audio at all.

    python score_editforward.py v5 v6 [v7 ...]      # folders under outputs_deliverable/

Prints to stdout only; goldset/ladder_split.json is restored after ladder_split has run.
"""
import contextlib
import io
import json
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import postedit_effort as PE                                     # noqa: E402
from evaluate_protocols import REF, strip_backlog               # noqa: E402
from term_source import PREVIOUS                                # noqa: E402
from carry_tag import read_prev                                 # noqa: E402


def no_backlog(t: str) -> str:
    out, skip = [], None
    for line in t.splitlines():
        ind = len(line) - len(line.lstrip())
        s = line.strip()
        if skip is not None and s and ind <= skip:
            skip = None
        if skip is not None:
            continue
        if re.match(r"^[*-]\s+[*_]*backlog", s, re.I):
            skip = ind
            continue
        if re.match(r"^[*-]\s+\*\*[^*]+\*\*\s*$", s):
            continue
        out.append(line)
    return "\n".join(out)


def previous_only(mid: str) -> str:
    t = read_prev(REPO / PREVIOUS[mid])
    if PREVIOUS[mid].endswith(".docx"):
        t = "\n".join("* " + l for l in t.splitlines() if l.strip())
    return no_backlog(strip_backlog(t)[0])


def arm_text(arm: str, mid: str):
    if arm == "prev":
        return previous_only(mid)
    p = REPO / "outputs_deliverable" / arm / mid / "protocol_pro.md"
    return re.sub(r"(?m)^#{3,4} ", "* ", p.read_text(encoding="utf-8")) if p.exists() else None


def postedit(arm: str) -> dict:
    refs = {m: no_backlog(strip_backlog((REPO / f).read_text(encoding="utf-8"))[0])
            for m, f in REF.items() if (REPO / f).exists()}
    AN = AC = N = C = D = G = 0
    for mid, ref in refs.items():
        t = arm_text(arm, mid)
        if t is None:
            continue
        pb, rb, hb = PE.bullets(previous_only(mid)), PE.bullets(ref), PE.bullets(t)
        for r in rb:
            miss = not PE.covered(r, hb)
            if PE.covered(r, pb):
                C += 1; AC += miss
            else:
                N += 1; AN += miss
        D += sum(not PE.covered(h, rb) for h in hb)
        G += len(hb)
    return {"add": AN + AC, "ref": N + C, "new": N, "new_missing": AN, "carried": C,
            "carried_missing": AC, "delete": D, "draft": G}


def ladder(arms: list[str]) -> dict:
    import ladder_split as L
    keep = REPO / "goldset" / "ladder_split.json"
    backup = keep.read_bytes() if keep.exists() else None
    L.ARMS = {a: "" for a in arms}
    L.ARM_SRC = {a: (REPO / "outputs_deliverable" / a, [""]) for a in arms}
    L.LISAS = ["pro"]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            L.main()
        d = json.loads(keep.read_text(encoding="utf-8"))
        from evaluate_speaker_attribution import person_key
        # The system does not produce the Backlog, and most carried topics an arm fails to locate
        # are Backlog entries. postedit_effort already excludes the Backlog; ladder_split does
        # not, so the summary is recomputed here from its rows without the Backlog topics.
        rows = [r for r in d["topics"]
                if not re.match(r"\s*[*_]*backlog", r["text"], re.I)]
        out: dict = {}
        for label, sel in (("ALL", lambda r: True), ("SPOKEN", lambda r: r["spoken"]),
                           ("UNSPOKEN (carried)", lambda r: not r["spoken"])):
            rs = [r for r in rows if sel(r)]
            out[label] = {}
            for a in arms:
                got = [r for r in rs if r.get(f"arm:{a}")]
                out[label][a] = {
                    "ok": sum(person_key(r[f"arm:{a}"]) == person_key(r["owner"]) for r in got),
                    "located": len(got), "total": len(rs)}
            # the no-audio baseline beside every arm: copying last week's owners attributes most
            # reference topics correctly, but it scores 0 on anything that was not in last
            # week's protocol.
            out[label]["last week alone (no audio)"] = {
                "ok": sum(r["last_week_ok"] for r in rs), "located": len(rs), "total": len(rs)}
        return out
    finally:
        if backup is not None:
            keep.write_bytes(backup)            # ladder_split overwrites its own file; restore it


def legitimate() -> set:
    """Squashed tokens that appear in the reference protocols themselves, so they are correct
    German or correct names, whatever else they may also be a mishearing of.

    goldset/name_reachability.json lists, per name, the forms some ASR system produced for it
    ('heard_as'). Some of these forms are words the reference protocols use themselves, for
    example 'LISA' (listed as a form of LUIS), 'H100' (of H200), 'Musik' (of M4-SKI) and 'wobei'
    (of WebUI), which is ordinary German. Document-level matching cannot tell the two uses
    apart; only the position in the audio could, and the protocol is not aligned to it. So these
    variants decide nothing and are excluded, and 'undecidable' is reported as its own column
    instead of being folded into either side."""
    import name_ab_score as N
    out = set()
    for f in REF.values():
        if (REPO / f).exists():
            out |= N.tokens((REPO / f).read_text(encoding="utf-8", errors="replace"))
    return out


def names(arm: str) -> dict:
    import name_ab_score as N
    ok_words = legitimate()
    rk = json.loads((REPO / "goldset" / "name_reachability.json").read_text(encoding="utf-8"))
    h = m = a = u = 0
    for r in rk["rows"]:
        p = REPO / "outputs_deliverable" / arm / r["meeting"] / "protocol_pro.md"
        if not p.exists():
            continue
        raw = p.read_text(encoding="utf-8", errors="replace")
        tok = N.tokens(raw)
        # a hyphenated compound carries the name: 'X-Stand' is the same evidence as 'X'
        tok |= {part for t in tok for part in re.split(r"[-/]", t) if part}
        # ... but a wrong form counts only where the protocol writes it as a name: a form
        # written entirely in lower case, such as 'wobei' for WebUI, is German, not a name.
        named = {N.E.squash(w) for w in re.findall(r"[\wÄÖÜäöüß-]+", raw)
                 if not w.islower()}
        named |= {N.E.squash(x) for w in named for x in re.split(r"[-/]", w) if x}
        variants = [x for x in r["heard_as"]
                    if N.E.squash(x) not in ok_words and N.E.squash(x) in named]
        dropped = len([x for x in r["heard_as"] if N.E.squash(x) not in variants])
        if N.E.squash(r["reference"]) in tok:
            h += 1
        elif any(N.E.squash(x) in tok for x in variants):
            m += 1
        elif dropped and any(N.E.squash(x) in tok for x in r["heard_as"]):
            u += 1
        else:
            a += 1
    return {"right": h, "wrong": m, "absent": a, "undecidable": u}


def main(arms: list[str]) -> int:
    print("POST-EDIT (8 meetings with reference minutes; Backlog and name lines excluded)")
    for a in ["prev"] + arms:
        r = postedit(a)
        label = "last week alone" if a == "prev" else a
        print(f"  {label:16s} ADD {r['add']}/{r['ref']} ({r['add'] / r['ref']:.1%})  | ref NEW "
              f"{r['new']}: missing {r['new_missing']} ({r['new_missing'] / r['new']:.1%})  | "
              f"CARRIED {r['carried']}: missing {r['carried_missing']} "
              f"({r['carried_missing'] / r['carried']:.1%})  | DELETE {r['delete']}/{r['draft']} "
              f"({r['delete'] / max(1, r['draft']):.1%})")
    print("\nATTRIBUTION, same-topic ladder, Pro — correct / all topics (unlocated = wrong)")
    s = ladder(arms)
    for k, v in s.items():
        print(f"  {k}")
        for a in list(arms) + ["last week alone (no audio)"]:
            x = v.get(a)
            if x:
                print(f"      {a:26s} {x['ok']}/{x['total']} = {x['ok'] / x['total']:.1%}   "
                      f"(located {x['located']}, {x['ok'] / max(1, x['located']):.1%} of located)")
    print("\nNAMES (31 names the minutes spell one way and the ASR another)")
    for a in arms:
        r = names(a)
        print(f"  {a:16s} right {r['right']}  only wrong {r['wrong']}  absent {r['absent']}  "
              f"undecidable {r['undecidable']}  | right when decided "
              f"{r['right'] / max(1, r['right'] + r['wrong']):.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
