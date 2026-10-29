"""Scores the post-edit comparison against reference protocols with last week's copied leaf bullets removed.

The delivered draft copies last week's protocol and inserts LISA's new items. The weekly reference
protocols repeat much of the previous week, so a plain copy of the previous protocol already scores
well. This script removes from each reference the leaf bullets that are copies of the previous
protocol (headlines are always kept) and measures how much of the remaining new content the draft,
LISA's own items and the previous protocol cover. It runs on the eight meetings of
evaluate_protocols.REF and pools them three ways: all eight, all eight without april13, and the three
meetings march09, april27 and june01. Before scoring it checks that score_editforward.postedit()
reproduces the stored counts in PUBLISHED and stops with exit code 1 if it does not. The script only
reads files; it makes no network or LISA call.

Reference: the reference protocol prepared as score_editforward.postedit() prepares it:
strip_backlog(), then no_backlog() (per-person and global Backlog dropped, bold person-name lines
dropped from scoring). The scored bullets are postedit_effort.bullets() of that text, in the same
order (asserted).

Tree: bullet nesting is read from the indentation of the reference after strip_backlog(), with the
person-name lines kept as roots so that dropping them cannot re-parent anything.
  leaf      a scored bullet with no child bullet. An empty placeholder bullet ('<br>', nothing
            left after markup removal) does not make its parent a non-leaf.
  headline  a person heading (bold name line; not a scored bullet) or a scored bullet that has
            at least one child bullet. Headlines are always kept.
  Rule L is the primary rule. Sensitivity rule T: a childless bullet directly under a person (a
  topic line with no detail) is also treated as a headline.

Stripping: each definition of "a copy from the previous week" is tested against the previous
protocol as score_editforward.previous_only() returns it (Backlog and name lines removed):
  S0  nothing stripped (the unstripped reference), for comparison.
  S1  literal copy: the leaf's normalized text is identical to the normalized text of some bullet
      of the previous protocol. Normalization (norm_s1): bullet marker removed, HTML entities
      decoded, HTML tags removed, markdown links reduced to their text, pipeline markers
      ([ÜBERTRAG]/[PRÜFEN]/[Aufgabe]) removed, emphasis characters * _ ` ~ removed, lower-cased,
      whitespace collapsed, trailing punctuation removed. A bullet's continuation lines (a wrapped
      line that is not itself a bullet) are part of its text. A bullet that is pure markup
      ('<br>', an inline '<img>') would normalize to '' and is compared by its literal markup.
  S2  the "carried" rule of postedit_effort: postedit_effort.covered(leaf, previous bullets), i.e.
      >= 30 % of the leaf's content words occur in one bullet of the previous protocol.
  S1C, S1CU  bounds from manual labels: each of the 181 leaves that S2 strips and S1 keeps has a
      label in HAND_LABELS; S1C also strips those labeled copies, S1CU also those labeled updates
      of last week's item. Rule L only.

Measures, per meeting and pooled:
  recall    of the remaining (new) leaves: share covered (postedit_effort.covered) by
            (a) the delivered draft protocol_pro.md, (b) LISA's own items protocol_pro.neu.md,
            (c) the baseline = previous protocol alone. Headlines are reported separately.
  precision share of bullets that match some bullet of the stripped reference (covered() in the
            reverse direction, as postedit_effort computes DELETE), for LISA's items and for the
            full draft; also against the remaining new leaves alone.
  owner     for new leaves the draft covers: whether a covering draft bullet stands under the same
            person (evaluate_speaker_attribution.person_key of the '##' heading against the bold
            name line of the reference). Counted two ways: any covering bullet, and the covering
            bullet with the best overlap.
  reading   under S2 every LISA match of a remaining leaf has a manual label (LISA_MATCH_LABELS),
            giving the S2 recall with false word matches removed (checks.S2_recall_by_reading).
Draft topic headings ('###', '####') are scored as bullets, as score_editforward.arm_text() does;
the same conversion is applied to the .neu.md file. The JSON also holds consistency checks and
random samples of stripped and kept leaves for april27 and june01.

Intervals: two-stage bootstrap over meetings, then over bullets within each drawn meeting (as in
wer_block_bootstrap.hierarchical_replicates), 10,000 draws, fixed seed, percentile interval.
Resampling binary bullets with replacement is drawn as one multinomial over the meeting's outcome
patterns, which keeps every flag of a bullet paired.

    python strip_copies_rescore.py    -> goldset/strip_copies_rescore_2026-10-07.json
"""
from __future__ import annotations

import html
import json
import random
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import postedit_effort as PE                                     # noqa: E402
import score_editforward as SE                                   # noqa: E402
from carry_tag import read_prev                                  # noqa: E402
from evaluate_protocols import REF, strip_backlog                # noqa: E402
from evaluate_speaker_attribution import person_key              # noqa: E402
from term_source import PREVIOUS                                 # noqa: E402

ARM = "v10"
OUT = REPO / "goldset" / "strip_copies_rescore_2026-10-07.json"
SEED = 20261007
N_BOOT = 10000
THREE = ["march09", "april27", "june01"]
FLAT = "april13"                       # Wiki PDF export: levels below the topic are flattened
SAMPLE_MEETINGS = ["april27", "june01"]
DEFS = {"L": ("S0", "S1", "S2", "S1C", "S1CU"), "T": ("S0", "S1", "S2")}
BUL = re.compile(r"^[*-]\s+\S")
MARKERS = re.compile(r"\[(ÜBERTRAG|PRÜFEN|Aufgabe)\]\s*", re.I)
PUBLISHED = {"v10": {"add": 325, "ref": 1183, "new": 459, "new_missing": 322, "carried": 724,
                     "carried_missing": 3, "delete": 1559, "draft": 2522},
             "prev": {"add": 459, "ref": 1183, "delete": 467, "draft": 1175}}
PUBLISHED_CI = {"add_v10": [22.3, 32.5], "add_prev": [31.3, 45.5], "delete_v10": [57.0, 66.4],
                "delete_prev": [32.3, 47.1], "new_missing_v10": [64.3, 76.4]}

# Manual labels for the 181 leaves where S1 and S2 disagree (S2 strips them, S1 keeps them), one
# reader. Each leaf was read next to its best-matching previous bullet:
#   C  copy: the same statement, reworded, shortened or with a trivial addition
#   U  update: last week's item carrying a new fact ("Prod-Update steht noch aus." ->
#      "Prod-Update am Fr vollzogen"; "Neuer Betriebsarzt wird Fr ausgewählt" -> "... wurde ...")
#   N  unrelated: matched only through a generic shared word ("Erste Tests erfolgreich" ~
#      "war aber nicht erfolgreich")
# Keys are line numbers in the reference file. load_meeting() asserts that the labeled lines are
# exactly the S2-only leaves, so a changed reference or matcher fails loudly instead of mislabeling.
# The labels define two variants: S1C = S1 + C, S1CU = S1 + C + U.
HAND_LABELS = {
    "march09": {23: "U", 26: "N", 29: "N", 38: "U", 39: "N", 42: "U", 79: "N", 83: "N", 91: "N", 94: "N", 97: "N", 98: "N", 101: "N", 103: "U", 105: "N", 115: "N", 121: "N", 126: "N", 145: "N", 147: "N", 156: "N", 209: "N", 211: "N", 222: "N", 229: "N", 231: "U", 235: "N", 237: "N"},
    "april13": {8: "N", 10: "U", 13: "N", 19: "N", 20: "N", 45: "N", 46: "N", 48: "N", 57: "N", 59: "C", 100: "N", 102: "U", 103: "N", 106: "N", 108: "N", 116: "N", 118: "N", 122: "N", 123: "C", 130: "N", 132: "N", 136: "N", 137: "U", 139: "U", 142: "U", 147: "N", 148: "N", 182: "U", 188: "N", 191: "N", 196: "N", 197: "N", 198: "N", 199: "C", 203: "N", 205: "N", 207: "N", 208: "U", 216: "N", 217: "C", 218: "N", 219: "U", 221: "N"},
    "april27": {11: "N", 14: "N", 18: "N", 24: "N", 136: "U", 172: "C", 176: "N", 178: "U", 189: "N", 249: "U", 250: "U", 251: "N", 256: "U", 260: "U", 269: "N", 295: "N", 303: "N", 307: "N", 308: "N", 309: "N", 324: "N", 327: "U", 332: "N", 335: "U", 344: "N"},
    "may11": {5: "U", 21: "N", 31: "N", 39: "U", 81: "N", 85: "C", 97: "C", 101: "N", 103: "N", 119: "N", 279: "N", 281: "N", 299: "N", 387: "N", 417: "N", 425: "U", 429: "N", 439: "N", 451: "U", 461: "C", 463: "C"},
    "may18": {5: "U", 7: "N", 11: "N", 21: "N", 35: "N", 211: "U", 225: "C", 245: "N", 315: "U", 337: "N", 383: "N", 391: "N", 447: "U", 453: "N", 503: "N"},
    "june01": {5: "N", 11: "N", 55: "U", 57: "N", 59: "N", 61: "N", 63: "N", 83: "N", 91: "N", 113: "N", 151: "U", 153: "N", 163: "N", 177: "N", 529: "N"},
    "june15": {5: "N", 17: "N", 31: "N", 33: "N", 37: "N", 191: "N", 209: "U", 403: "N", 405: "N", 413: "N", 483: "U", 495: "N", 503: "U", 513: "N"},
    "june22": {7: "N", 9: "N", 33: "N", 41: "N", 61: "N", 97: "N", 317: "N", 337: "N", 341: "N", 431: "N", 443: "N", 449: "N", 455: "N", 463: "N", 481: "U", 491: "N", 499: "N", 501: "N", 587: "U", 595: "U"},
}

# Manual labels for the 97 S2-new leaves that LISA's own items cover, one reader. Under S2 the
# draft can cover a new leaf only through LISA's items, so these matches are the whole of the
# system's measured contribution there. Each leaf was read next to its best LISA item:
#   G  genuine: the LISA item records the same fact (paraphrase allowed)
#   P  partial: same event or topic, the leaf's key fact missing
#      ("SARA ... ins Gespräch gebracht" ~ "Vortrag zu AI-Gigafactory-Projekt 'Blue Swan' gehört")
#   F  false: one generic shared word ("Gutes Networking" ~ "... für gutes Alignment ...")
# Only matches are labeled. Leaves the matcher misses are not labeled, so paraphrases the word
# matcher cannot see are not added back; the labels can only lower this count, not raise it.
LISA_MATCH_LABELS = {
    "march09": {3: "G", 10: "F", 11: "G", 12: "G", 18: "P", 20: "G", 25: "G", 82: "G", 85: "G", 87: "G", 93: "G", 95: "G", 96: "G", 99: "G", 117: "G", 128: "G", 144: "G", 151: "G", 216: "G", 219: "P", 223: "F", 240: "F"},
    "april13": {15: "F", 28: "F", 31: "F", 33: "G", 47: "G", 51: "G", 104: "G", 109: "F", 125: "G", 193: "P", 212: "G"},
    "april27": {185: "G", 273: "G", 302: "G", 325: "G"},
    "may11": {13: "G", 15: "G", 23: "G", 89: "G", 267: "G", 269: "G", 271: "G", 275: "P", 283: "G", 305: "G", 385: "G", 395: "G", 401: "G"},
    "may18": {217: "G", 237: "F", 329: "P", 335: "G", 339: "G", 355: "P", 377: "G", 387: "G", 395: "G", 441: "F"},
    "june01": {9: "G", 21: "F", 23: "G", 75: "G", 97: "G", 109: "G", 181: "G", 187: "P", 189: "G", 347: "F", 479: "F", 485: "P", 487: "F", 491: "G", 495: "G", 511: "G", 513: "G"},
    "june15": {11: "G", 25: "F", 215: "G", 227: "G", 489: "G", 491: "G"},
    "june22": {13: "P", 17: "G", 21: "G", 323: "G", 359: "G", 421: "F", 435: "G", 467: "F", 469: "F", 487: "F", 493: "G", 585: "G", 589: "G", 591: "P"},
}


# ----------------------------------------------------------------------------------- parsing
def indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def norm_s1(s: str) -> str:
    t = unicodedata.normalize("NFC", s.strip())
    t = re.sub(r"^[*-]\s+", "", t)
    t = html.unescape(t)
    t = re.sub(r"<[^>]*>", " ", t)
    t = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", t)
    t = MARKERS.sub("", t)
    t = re.sub(r"[*_`~]+", "", t)
    t = t.lower()
    t = re.sub(r"\s+", " ", t).strip()
    return re.sub(r"[\s.,;:!?…]+$", "", t)


def s1_key(s: str) -> str:
    """norm_s1, except that a bullet that is pure markup ('<br>', '<img src=.../2229>') keeps its
    literal markup as its identity. With the markup stripped, different inline images would all
    become '' and count as copies of one another."""
    t = norm_s1(s)
    if t:
        return t
    raw = re.sub(r"^[*-]\s+", "", unicodedata.normalize("NFC", s.strip()))
    return "markup:" + re.sub(r"\s+", " ", raw).strip().lower()


def display(s: str) -> str:
    t = re.sub(r"^[*-]\s+", "", s.strip())
    return re.sub(r"\s+", " ", t)


def no_backlog_states(lines: list[str]) -> list[str]:
    """score_editforward.no_backlog(), line by line: 'keep', 'backlog' (dropped with its block)
    or 'name' (a bold-only bullet line, dropped from scoring). ref_tree() asserts that the result
    matches score_editforward.no_backlog()."""
    out, skip = [], None
    for line in lines:
        i, s = indent(line), line.strip()
        if skip is not None and s and i <= skip:
            skip = None
        if skip is not None:
            out.append("backlog")
            continue
        if re.match(r"^[*-]\s+[*_]*backlog", s, re.I):
            skip = i
            out.append("backlog")
            continue
        if re.match(r"^[*-]\s+\*\*[^*]+\*\*\s*$", s):
            out.append("name")
            continue
        out.append("keep")
    return out


def strip_backlog_keep(lines: list[str]) -> list[int]:
    """Indices of the raw lines evaluate_protocols.strip_backlog() keeps (ref_tree() asserts the
    match), so every leaf can be reported with its line number in the reference file."""
    head = re.compile(r"^[*-]\s+[*_]*\s*backlog\s*[*_]*\s*:?\s*$", re.I)
    keep, skip = [], None
    for k, line in enumerate(lines):
        s, i = line.strip(), indent(line)
        if skip is not None:
            if s and i <= skip and s[:1] in "*-":
                skip = None
            else:
                continue
        if head.match(s):
            skip = i
            continue
        keep.append(k)
    return keep


def ref_tree(mid: str) -> tuple[list[dict], str]:
    raw = (REPO / REF[mid]).read_text(encoding="utf-8")
    a = strip_backlog(raw)[0]
    raw_lines = raw.splitlines()
    file_line = strip_backlog_keep(raw_lines)
    assert "\n".join(raw_lines[k] for k in file_line) == a, f"{mid}: strip_backlog replica diverges"
    lines = a.splitlines()
    states = no_backlog_states(lines)
    ref_text = "\n".join(l for l, s in zip(lines, states) if s == "keep")
    assert ref_text == SE.no_backlog(a), f"{mid}: no_backlog replica diverges"
    nodes: list[dict] = []
    for k, (line, st) in enumerate(zip(lines, states)):
        s = line.strip()
        if st == "backlog":
            continue
        if not BUL.match(s):
            if s and st == "keep" and nodes:
                nodes[-1]["cont"].append(s)
            continue
        nodes.append({"line": file_line[k] + 1, "indent": indent(line), "raw": s, "cont": [],
                      "name": st == "name", "content": PE.content(s) if st == "keep" else set()})
    stack: list[int] = []
    for j, n in enumerate(nodes):
        n["s1"] = s1_key(" ".join([n["raw"]] + n["cont"]))
        n["placeholder"] = not norm_s1(" ".join([n["raw"]] + n["cont"]))
        while stack and nodes[stack[-1]]["indent"] >= n["indent"]:
            stack.pop()
        n["parent"] = stack[-1] if stack else None
        stack.append(j)
    for n in nodes:
        n["n_child"] = n["n_child_any"] = 0
    for n in nodes:
        if n["parent"] is not None:
            nodes[n["parent"]]["n_child_any"] += 1
            if not n["placeholder"]:
                nodes[n["parent"]]["n_child"] += 1
    for n in nodes:
        anc, p = [], n["parent"]
        while p is not None:
            anc.append(p)
            p = nodes[p]["parent"]
        root = nodes[anc[-1]] if anc else n
        n["owner"] = person_key(display(root["raw"])) if root["name"] and root is not n else ""
        n["depth"] = len(anc) if (anc and nodes[anc[-1]]["name"]) else len(anc) + 1
        n["scored"] = (not n["name"]) and bool(n["content"])
        n["text"] = display(" ".join([n["raw"]] + n["cont"]))
    assert all(n["indent"] == 0 for n in nodes if n["name"]), f"{mid}: indented name line"
    scored = [n for n in nodes if n["scored"]]
    assert [n["content"] for n in scored] == PE.bullets(ref_text), f"{mid}: bullet set differs"
    return nodes, ref_text


def flat_nodes(md: str) -> list[dict]:
    """Every bullet line of a draft/previous text with its continuation lines and the '## '
    person it stands under. Scored ones are exactly postedit_effort.bullets() (asserted by the
    caller)."""
    out, owner = [], ""
    for line in md.splitlines():
        s = line.strip()
        if re.match(r"^#{1,6}\s", s):
            if s.startswith("## ") and not s.startswith("### "):
                owner = person_key(s[3:]) or s[3:].strip()
            out.append(None)                          # a heading ends any continuation
            continue
        if BUL.match(s):
            out.append({"raw": s, "cont": [], "owner": owner, "content": PE.content(s)})
        elif s and out and out[-1] is not None:
            out[-1]["cont"].append(s)
    out = [n for n in out if n is not None]
    for n in out:
        n["s1"] = s1_key(" ".join([n["raw"]] + n["cont"]))
        n["text"] = display(" ".join([n["raw"]] + n["cont"]))
    return out


def scored_flat(md: str) -> list[dict]:
    nodes = [n for n in flat_nodes(md) if n["content"]]
    assert [n["content"] for n in nodes] == PE.bullets(md)
    return nodes


def overlap(item: set, p: set) -> float:
    return len(item & p) / len(item) if item and p else 0.0


def best_match(item: set, pool: list[dict]) -> tuple[float, dict | None]:
    best, hit = -1.0, None
    for d in pool:
        o = overlap(item, d["content"])
        if o > best:
            best, hit = o, d
    return best, hit


# ----------------------------------------------------------------------------------- meeting
def load_meeting(mid: str) -> dict:
    nodes, _ = ref_tree(mid)
    prev_md = SE.previous_only(mid)
    prev_all = flat_nodes(prev_md)
    prev = scored_flat(prev_md)
    prev_full = flat_nodes(read_prev(REPO / PREVIOUS[mid]))          # with its Backlog
    draft = scored_flat(SE.arm_text(ARM, mid))
    neu_md = (REPO / "outputs_deliverable" / ARM / mid / "protocol_pro.neu.md").read_text(encoding="utf-8")
    neu = scored_flat(re.sub(r"(?m)^#{3,4} ", "* ", neu_md))
    prev_s1 = {n["s1"] for n in prev_all}
    prev_s1_full = {n["s1"] for n in prev_full}
    pc = [n["content"] for n in prev]
    labels = HAND_LABELS[mid]
    for n in nodes:
        if not n["scored"]:
            continue
        n["S0"] = False
        n["S1"] = n["s1"] in prev_s1
        n["S2"] = PE.covered(n["content"], pc)
        n["S1_full_prev"] = n["s1"] in prev_s1_full
        lab = labels.get(n["line"])
        disagree = n["S2"] and not n["S1"] and is_leaf(n, "L")
        assert (lab is not None) == disagree, f"{mid} line {n['line']}: hand label out of step"
        n["hand"] = lab
        n["S1C"] = n["S1"] or lab == "C"
        n["S1CU"] = n["S1"] or lab in ("C", "U")
    seen = {n["line"] for n in nodes if n["scored"] and n["hand"]}
    assert seen == set(labels), f"{mid}: labels on lines that are not S2-only leaves: {set(labels) - seen}"
    for d in draft:
        d["verbatim_prev"] = d["s1"] in prev_s1
    return {"mid": mid, "nodes": nodes, "prev": prev, "draft": draft, "neu": neu}


def is_leaf(n: dict, rule: str) -> bool:
    if n["n_child"]:
        return False
    return not (rule == "T" and n["depth"] == 1)


def evaluate(M: dict, definition: str, rule: str) -> tuple[dict, dict]:
    """Counts for one meeting plus the per-bullet flags the bootstrap needs."""
    scored = [n for n in M["nodes"] if n["scored"]]
    leaves = [n for n in scored if is_leaf(n, rule)]
    heads = [n for n in scored if not is_leaf(n, rule)]
    stripped = [n for n in leaves if n[definition]]
    new = [n for n in leaves if not n[definition]]
    strip_ids = {id(n) for n in stripped}
    kept_ref = [n["content"] for n in scored if id(n) not in strip_ids]
    new_c = [n["content"] for n in new]
    dc = [d["content"] for d in M["draft"]]
    nc = [d["content"] for d in M["neu"]]
    pc = [d["content"] for d in M["prev"]]

    rec = [(PE.covered(n["content"], dc), PE.covered(n["content"], nc), PE.covered(n["content"], pc))
           for n in new]
    hrec = [(PE.covered(n["content"], dc), PE.covered(n["content"], nc), PE.covered(n["content"], pc))
            for n in heads]
    pneu = [(PE.covered(h["content"], kept_ref), PE.covered(h["content"], new_c)) for h in M["neu"]]
    pdr = [(PE.covered(h["content"], kept_ref), PE.covered(h["content"], new_c)) for h in M["draft"]]

    def owner_check(pool: list[dict]) -> dict:
        any_ok = best_ok = n_cov = n_known = 0
        for n in new:
            cov = [(overlap(n["content"], d["content"]), k, d) for k, d in enumerate(pool)
                   if overlap(n["content"], d["content"]) >= PE.MATCH_TH]
            if not cov:
                continue
            n_cov += 1
            if not n["owner"]:
                continue
            n_known += 1
            any_ok += any(d["owner"] == n["owner"] for _, _, d in cov)
            best = max(cov, key=lambda x: (x[0], -x[1]))[2]          # highest overlap, first on ties
            best_ok += best["owner"] == n["owner"]
        return {"covered": n_cov, "with_owner": n_known, "same_owner_any": any_ok,
                "same_owner_best": best_ok}

    verb = [d for d in M["draft"] if d["verbatim_prev"]]
    verb_unmatched = sum(1 for d in verb if not PE.covered(d["content"], kept_ref))
    r = {
        "scored_bullets": len(scored), "leaves": len(leaves), "headlines": len(heads),
        "person_headings": sum(1 for n in M["nodes"] if n["name"]),
        "stripped": len(stripped), "new_leaves": len(new),
        "stripped_by_depth": _by_depth(stripped), "new_by_depth": _by_depth(new),
        "recall_new_leaves": {"draft": sum(x[0] for x in rec), "neu": sum(x[1] for x in rec),
                              "prev": sum(x[2] for x in rec), "n": len(rec)},
        # who covers each remaining leaf: d = draft, l = LISA's own items, p = previous protocol
        "recall_patterns": {f"d{int(a)}l{int(b)}p{int(c)}": sum(1 for x in rec if x == (a, b, c))
                            for a in (True, False) for b in (True, False) for c in (True, False)},
        "recall_headlines": {"draft": sum(x[0] for x in hrec), "neu": sum(x[1] for x in hrec),
                             "prev": sum(x[2] for x in hrec), "n": len(hrec)},
        "precision_neu": {"vs_stripped_ref": sum(x[0] for x in pneu),
                          "vs_new_leaves": sum(x[1] for x in pneu), "n": len(pneu)},
        "precision_draft": {"vs_stripped_ref": sum(x[0] for x in pdr),
                            "vs_new_leaves": sum(x[1] for x in pdr), "n": len(pdr),
                            "verbatim_copies_of_prev": len(verb),
                            "verbatim_copies_unmatched": verb_unmatched},
        "owner_draft": owner_check(M["draft"]),
        "owner_neu": owner_check(M["neu"]),
    }
    flags = {"recall": rec, "headlines": hrec, "prec_neu": pneu, "prec_draft": pdr}
    return r, flags


def _by_depth(ns: list[dict]) -> dict:
    out: dict[str, int] = {}
    for n in ns:
        out[str(n["depth"])] = out.get(str(n["depth"]), 0) + 1
    return dict(sorted(out.items()))


# ----------------------------------------------------------------------------------- bootstrap
def two_stage(groups: list[list[tuple]], seed: int = SEED, n_boot: int = N_BOOT) -> np.ndarray:
    """Replicate totals of each flag: meetings drawn with replacement, then the bullets of each
    drawn meeting with replacement (one multinomial over its outcome patterns). Returns an array
    (n_boot, n_flags + 1): flag sums and the item count."""
    pats = sorted({t for g in groups for t in g})          # caller guarantees at least one item
    k = len(pats[0])
    P = np.array([[int(v) for v in p] + [1] for p in pats], dtype=np.int64)
    C = np.array([[sum(1 for t in g if t == p) for p in pats] for g in groups], dtype=np.int64)
    rng = np.random.default_rng(seed)
    m = len(groups)
    out = np.zeros((n_boot, k + 1), dtype=np.int64)
    for b in range(n_boot):
        tot = np.zeros(len(pats), dtype=np.int64)
        for j in rng.integers(0, m, m):
            n = int(C[j].sum())
            if n:
                tot += rng.multinomial(n, C[j] / n)
        out[b] = tot @ P
    return out


def ci(values: np.ndarray) -> list[float]:
    v = np.sort(values)
    return [round(100 * float(v[int(0.025 * len(v))]), 1), round(100 * float(v[int(0.975 * len(v))]), 1)]


def boot_summary(groups: list[list[tuple]], names: list[str], diffs=()) -> dict:
    if not sum(len(g) for g in groups):
        return {}
    reps = two_stage(groups)
    n = reps[:, -1].astype(float)
    n[n == 0] = np.nan
    out = {}
    for j, name in enumerate(names):
        out[name] = ci(np.nan_to_num(reps[:, j] / n))
    for a, b in diffs:
        ia, ib = names.index(a), names.index(b)
        out[f"{a}-{b}"] = ci(np.nan_to_num((reps[:, ia] - reps[:, ib]) / n))
    return out


# ----------------------------------------------------------------------------------- harness
def harness_check(meetings: dict) -> dict:
    got = {"v10": SE.postedit(ARM), "prev": SE.postedit("prev")}
    ok = all(got["v10"][k] == v for k, v in PUBLISHED["v10"].items()) and \
        all(got["prev"][k] == v for k, v in PUBLISHED["prev"].items())
    # the same quantities with this script's bootstrap, for comparison with PUBLISHED_CI
    add_g, newm_g, del_d, del_p = [], [], [], []
    for mid, M in meetings.items():
        dc = [d["content"] for d in M["draft"]]
        pc = [d["content"] for d in M["prev"]]
        rb = [n["content"] for n in M["nodes"] if n["scored"]]
        add_g.append([(not PE.covered(r, dc), not PE.covered(r, pc)) for r in rb])
        newm_g.append([(not PE.covered(r, dc),) for r in rb if not PE.covered(r, pc)])
        del_d.append([(not PE.covered(h, rb),) for h in dc])
        del_p.append([(not PE.covered(h, rb),) for h in pc])
    boot = boot_summary(add_g, ["add_v10", "add_prev"], diffs=[("add_v10", "add_prev")])
    boot.update(boot_summary(newm_g, ["new_missing_v10"]))
    boot.update(boot_summary(del_d, ["delete_v10"]))
    boot.update(boot_summary(del_p, ["delete_prev"]))
    return {"reproduced_exactly": ok, "score_editforward_postedit": got, "published": PUBLISHED,
            "bootstrap_this_script": boot, "bootstrap_published": PUBLISHED_CI,
            "published_add_diff": [-14.8, -7.9]}


# ----------------------------------------------------------------------------------- samples
def samples(M: dict, definition: str, k: int = 10) -> dict:
    rng = random.Random(f"{SEED}-{M['mid']}-{definition}")
    leaves = [n for n in M["nodes"] if n["scored"] and is_leaf(n, "L")]
    out = {}
    for label, sel in (("stripped", [n for n in leaves if n[definition]]),
                       ("kept", [n for n in leaves if not n[definition]])):
        pick = sorted(rng.sample(range(len(sel)), min(k, len(sel))))
        rows = []
        for i in pick:
            n = sel[i]
            o, hit = best_match(n["content"], M["prev"])
            rows.append({"line": n["line"], "owner": n["owner"], "depth": n["depth"],
                         "leaf": n["text"], "identical_in_prev": n["S1"], "hand_label": n["hand"],
                         "best_prev_overlap": round(o, 2), "best_prev": hit["text"] if hit else ""})
        out[label] = rows
    return out


# ----------------------------------------------------------------------------------- main
def pct(a: int, n: int) -> str:
    return f"{100 * a / n:.1f}% ({a}/{n})" if n else "–"


def add_counts(rows: list[dict]) -> dict:
    """Sum the per-meeting count dicts (nested)."""
    def add(a: dict, b: dict) -> dict:
        out = dict(a)
        for k, v in b.items():
            out[k] = add(out.get(k, {}), v) if isinstance(v, dict) else out.get(k, 0) + v
        return out

    tot: dict = {}
    for r in rows:
        tot = add(tot, r)
    return tot


def main() -> int:
    meetings = {mid: load_meeting(mid) for mid in REF}
    hc = harness_check(meetings)
    print("HARNESS CHECK (published values, v10 Pro, 8 meetings)")
    print("  score_editforward.postedit:", hc["score_editforward_postedit"])
    print("  reproduced exactly:", hc["reproduced_exactly"])
    print("  two-stage CIs here:", hc["bootstrap_this_script"])
    print("  published CIs     :", PUBLISHED_CI, " ADD diff", hc["published_add_diff"])
    if not hc["reproduced_exactly"]:
        print("HARNESS DOES NOT REPRODUCE THE PUBLISHED VALUES — stopping.")
        return 1

    pools = {"all8": list(REF), "no_april13": [m for m in REF if m != FLAT], "three": THREE}
    result: dict = {"meta": {
        "date": "2026-10-07", "arm": ARM, "lisa": "pro",
        "reference": "evaluate_protocols.REF via strip_backlog + score_editforward.no_backlog",
        "previous": "term_source.PREVIOUS via score_editforward.previous_only",
        "draft": f"outputs_deliverable/{ARM}/<meeting>/protocol_pro.md (### -> bullet)",
        "lisa_items": f"outputs_deliverable/{ARM}/<meeting>/protocol_pro.neu.md (### -> bullet)",
        "matcher": f"postedit_effort.covered, MATCH_TH={PE.MATCH_TH}",
        "definitions": {
            "S0": "nothing stripped (comparison)",
            "S1": "leaf stripped if its normalised text is identical to a bullet of the previous protocol",
            "S2": "leaf stripped if postedit_effort.covered(leaf, previous-protocol bullets)",
            "S1C": "S1 plus the S2-only leaves read as copies (HAND_LABELS 'C'; one reader)",
            "S1CU": "S1 plus the S2-only leaves read as copies or updates (HAND_LABELS 'C' or 'U')"},
        "leaf_rules": {"L": "leaf = scored bullet with no child bullet (primary)",
                       "T": "as L, but childless bullets directly under a person are kept as headlines"},
        "bootstrap": f"two-stage (meetings, then bullets), {N_BOOT} draws, seed {SEED}, percentile",
        "pools": pools, "flattened_meeting": FLAT},
        "harness_check": hc, "per_meeting": {}, "pooled": {}, "checks": {}, "samples": {}}

    flags: dict = {}
    for rule in ("L", "T"):
        for d in DEFS[rule]:
            for mid, M in meetings.items():
                r, f = evaluate(M, d, rule)
                result["per_meeting"].setdefault(rule, {}).setdefault(d, {})[mid] = r
                flags[(rule, d, mid)] = f
            for pname, mids in pools.items():
                rows = [result["per_meeting"][rule][d][m] for m in mids]
                tot = add_counts(rows)
                fl = [flags[(rule, d, m)] for m in mids]
                tot["ci95"] = {
                    "recall_new_leaves": boot_summary([f["recall"] for f in fl],
                                                      ["draft", "neu", "prev"],
                                                      diffs=[("draft", "prev"), ("neu", "prev")]),
                    "recall_headlines": boot_summary([f["headlines"] for f in fl], ["draft", "neu", "prev"]),
                    "precision_neu": boot_summary([f["prec_neu"] for f in fl],
                                                  ["vs_stripped_ref", "vs_new_leaves"]),
                    "precision_draft": boot_summary([f["prec_draft"] for f in fl],
                                                    ["vs_stripped_ref", "vs_new_leaves"]),
                }
                result["pooled"].setdefault(rule, {}).setdefault(d, {})[pname] = tot

    # consistency checks
    chk: dict = {}
    s1_not_s2, s1_full_extra, placeholder_effect, placeholders = [], {}, 0, []
    for mid, M in meetings.items():
        sc = [n for n in M["nodes"] if n["scored"]]
        for n in sc:
            if is_leaf(n, "L") and n["S1"] and not n["S2"]:
                s1_not_s2.append({"meeting": mid, "line": n["line"], "leaf": n["text"]})
            if n["placeholder"]:
                placeholders.append({"meeting": mid, "line": n["line"], "raw": n["raw"],
                                     "S1": n["S1"], "S2": n["S2"]})
            if n["n_child"] == 0 and n["n_child_any"] > 0:
                placeholder_effect += 1
        s1_full_extra[mid] = sum(1 for n in sc if is_leaf(n, "L") and n["S1_full_prev"] and not n["S1"])
    p2 = result["pooled"]["L"]["S2"]["all8"]
    chk["S1_subset_of_S2_exceptions"] = s1_not_s2
    chk["S1_extra_if_previous_backlog_included"] = s1_full_extra
    chk["placeholder_bullets"] = placeholders
    chk["parents_whose_only_children_are_placeholders"] = placeholder_effect
    chk["S2_new_leaves_plus_new_headlines_equals_459"] = {
        "new_leaves": p2["new_leaves"],
        "new_headlines": p2["recall_headlines"]["n"] - p2["recall_headlines"]["prev"],
        "sum": p2["new_leaves"] + p2["recall_headlines"]["n"] - p2["recall_headlines"]["prev"]}
    chk["S2_new_missing_leaves_plus_new_missing_headlines_equals_322"] = None
    miss_h = 0
    for mid, M in meetings.items():
        dc = [d["content"] for d in M["draft"]]
        pc = [d["content"] for d in M["prev"]]
        for n in M["nodes"]:
            if n["scored"] and not is_leaf(n, "L") and not PE.covered(n["content"], pc) \
                    and not PE.covered(n["content"], dc):
                miss_h += 1
    chk["S2_new_missing_leaves_plus_new_missing_headlines_equals_322"] = {
        "new_leaves_missing": p2["recall_new_leaves"]["n"] - p2["recall_new_leaves"]["draft"],
        "new_headlines_missing": miss_h,
        "sum": p2["recall_new_leaves"]["n"] - p2["recall_new_leaves"]["draft"] + miss_h}
    hand: dict = {}
    for mid, M in meetings.items():
        dc = [d["content"] for d in M["draft"]]
        nc = [d["content"] for d in M["neu"]]
        for n in M["nodes"]:
            if n["scored"] and n["hand"]:
                h = hand.setdefault(n["hand"], {"n": 0, "draft": 0, "neu": 0, "by_meeting": {}})
                h["n"] += 1
                h["draft"] += PE.covered(n["content"], dc)
                h["neu"] += PE.covered(n["content"], nc)
                h["by_meeting"][mid] = h["by_meeting"].get(mid, 0) + 1
    chk["hand_labels_on_S2_only_leaves"] = hand

    # S2 recall using the manual label of every LISA match (LISA_MATCH_LABELS); the draft covers
    # no S2-new leaf that LISA's items do not (asserted), so its matches are a subset of the same
    # labeled set
    rd: dict = {}
    for pname, mids in pools.items():
        x = {"new_leaves_S2": 0, "lisa_G": 0, "lisa_P": 0, "lisa_F": 0,
             "draft_G": 0, "draft_P": 0, "draft_F": 0}
        for mid in mids:
            M = meetings[mid]
            nc = [d["content"] for d in M["neu"]]
            dc = [d["content"] for d in M["draft"]]
            labels = LISA_MATCH_LABELS.get(mid, {})
            seen = set()
            for n in M["nodes"]:
                if not (n["scored"] and is_leaf(n, "L") and not n["S2"]):
                    continue
                x["new_leaves_S2"] += 1
                in_draft = PE.covered(n["content"], dc)
                if PE.covered(n["content"], nc):
                    lab = labels[n["line"]]
                    seen.add(n["line"])
                    x["lisa_" + lab] += 1
                    x["draft_" + lab] += in_draft
                else:
                    assert not in_draft and n["line"] not in labels, f"{mid} {n['line']}: LISA labels out of step"
            assert seen == set(labels), f"{mid}: LISA labels on lines that are not LISA-covered S2 leaves"
        rd[pname] = x
    chk["S2_recall_by_reading"] = rd
    result["checks"] = chk

    for mid in SAMPLE_MEETINGS:
        result["samples"][mid] = {d: samples(meetings[mid], d) for d in ("S1", "S2")}

    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    report(result)
    print(f"\nwrote -> {OUT.relative_to(REPO)}")
    return 0


def report(res: dict) -> None:
    pm = res["per_meeting"]["L"]
    print("\nSTRIPPING (rule L: leaf = bullet without child bullet)")
    print("| meeting | scored bullets | headlines kept | person headings | leaves | stripped S1 | "
          "stripped S2 | left S1 | left S2 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for mid in REF:
        a, b = pm["S1"][mid], pm["S2"][mid]
        print(f"| {mid} | {a['scored_bullets']} | {a['headlines']} | {a['person_headings']} | "
              f"{a['leaves']} | {a['stripped']} | {b['stripped']} | {a['new_leaves']} | {b['new_leaves']} |")
    for d in ("S0", "S1", "S2"):
        print(f"\nPER MEETING {d}: recall of remaining leaves (draft / LISA / prev), precision vs stripped ref "
              "(LISA / draft), owner same (any/best of covered with owner)")
        print("| meeting | new leaves | draft | LISA | prev | prec LISA | prec draft | owner any | owner best |")
        print("|---|---|---|---|---|---|---|---|---|")
        for mid in REF:
            r = pm[d][mid]
            rr, pn, pd, ow = r["recall_new_leaves"], r["precision_neu"], r["precision_draft"], r["owner_draft"]
            print(f"| {mid} | {rr['n']} | {pct(rr['draft'], rr['n'])} | {pct(rr['neu'], rr['n'])} | "
                  f"{pct(rr['prev'], rr['n'])} | {pct(pn['vs_stripped_ref'], pn['n'])} | "
                  f"{pct(pd['vs_stripped_ref'], pd['n'])} | {ow['same_owner_any']}/{ow['with_owner']} | "
                  f"{ow['same_owner_best']}/{ow['with_owner']} |")
    for rule in ("L", "T"):
        print(f"\nPOOLED (rule {rule})")
        for pname in ("all8", "no_april13", "three"):
            for d in DEFS[rule]:
                r = res["pooled"][rule][d][pname]
                rr, hh, pn, pd = r["recall_new_leaves"], r["recall_headlines"], r["precision_neu"], r["precision_draft"]
                c = r["ci95"]
                print(f"  {pname:10s} {d}  leaves {r['leaves']} stripped {r['stripped']} left {rr['n']} | "
                      f"recall draft {pct(rr['draft'], rr['n'])} {c['recall_new_leaves'].get('draft')} "
                      f"LISA {pct(rr['neu'], rr['n'])} {c['recall_new_leaves'].get('neu')} "
                      f"prev {pct(rr['prev'], rr['n'])} {c['recall_new_leaves'].get('prev')} "
                      f"draft-prev {c['recall_new_leaves'].get('draft-prev')}")
                print(f"  {'':10s}     headlines {hh['n']}: draft {pct(hh['draft'], hh['n'])} "
                      f"LISA {pct(hh['neu'], hh['n'])} prev {pct(hh['prev'], hh['n'])} | "
                      f"prec LISA {pct(pn['vs_stripped_ref'], pn['n'])} {c['precision_neu'].get('vs_stripped_ref')} "
                      f"(new leaves only {pct(pn['vs_new_leaves'], pn['n'])} {c['precision_neu'].get('vs_new_leaves')}) "
                      f"prec draft {pct(pd['vs_stripped_ref'], pd['n'])} {c['precision_draft'].get('vs_stripped_ref')} "
                      f"(new leaves only {pct(pd['vs_new_leaves'], pd['n'])}) "
                      f"verbatim copies {pd['verbatim_copies_of_prev']}, unmatched {pd['verbatim_copies_unmatched']}")
                od, on = r["owner_draft"], r["owner_neu"]
                print(f"  {'':10s}     owner draft: covered {od['covered']} with owner {od['with_owner']} "
                      f"same any {od['same_owner_any']} best {od['same_owner_best']} | "
                      f"owner LISA: covered {on['covered']} with owner {on['with_owner']} "
                      f"same any {on['same_owner_any']} best {on['same_owner_best']}")
    print("\nCHECKS")
    for k, v in res["checks"].items():
        print(f"  {k}: {v}")
    print("\nSAMPLES")
    for mid, per in res["samples"].items():
        for d, sm in per.items():
            for label, rows in sm.items():
                print(f"\n  {mid} {d} {label}")
                for r in rows:
                    print(f"    [L{r['line']} {r['owner']} d{r['depth']} ident={r['identical_in_prev']} hand={r['hand_label']} "
                          f"ov={r['best_prev_overlap']}] {r['leaf']}")
                    print(f"        prev: {r['best_prev']}")


if __name__ == "__main__":
    raise SystemExit(main())
