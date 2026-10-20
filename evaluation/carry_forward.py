"""Carry last week's protocol forward into this week's protocol, without an LLM.

edit_forward() copies the previous protocol and inserts this week's new bullet groups under
their person and topic; protokoll.py uses it. carry_forward() is an older variant that appends
the previous items the new protocol does not cover, marked [ÜBERTRAG], under
"### Übertrag aus dem letzten Protokoll".

CUT is the word-bag Jaccard overlap at or above which two items count as the same topic.

Usage (prints the items carry_forward() would append; writes nothing):
    python carry_forward.py <new_protocol.md> <previous_protocol.md|.docx>
"""
from __future__ import annotations

import html
import re
from pathlib import Path

from carry_tag import MARK

CUT = 0.10
HEADING = "### Übertrag aus dem letzten Protokoll"
NAMES: dict[str, str] = {}  # person key -> name as written; set by previous_items

# Previous protocols may contain HTML markup (tags, entities, images).
# Leading whitespace of a body line is its nesting level; _PREFIX keeps it unchanged.
_IMG = re.compile(r"<img\b[^>]*>", re.I)
_TAG = re.compile(r"<[^>]+>")
_PREFIX = re.compile(r"^(\s*(?:[*-]\s+)?)(.*)$", re.S)


def clean_markup(s: str) -> str:
    """Remove HTML tags, HTML entities and **/__ markers from one line.

    Images become "[Bild]"; the bullet prefix and its indentation stay unchanged.
    """
    m = _PREFIX.match(s)
    prefix, text = m.group(1), m.group(2)
    text = _IMG.sub("[Bild]", text)
    text = _TAG.sub(" ", text)
    text = html.unescape(text)
    text = _TAG.sub(" ", text)  # decoded entities can form tags (&lt;br&gt;)
    text = re.sub(r"\*\*|__", "", text)
    return prefix + re.sub(r"[ \t]+", " ", text).strip()


def _has_text(line: str) -> bool:
    return bool(re.sub(r"^\s*[*-]\s*", "", line).strip())


def previous_items(path) -> list[tuple[str, str, list[str]]]:
    """Read the previous protocol (.docx, or Markdown bullets with bold names) as items.

    Each item is (person key, topic, sub-bullet lines); NAMES gets each person's name as written.
    """
    from evaluate_speaker_attribution import person_key, md_indent_unit
    p = Path(path)
    NAMES.clear()
    rows: list[tuple[str, int, str, bool]] = []  # level -1 person, 0 topic, 1+ sub-bullet
    if p.suffix.lower() == ".docx":
        import docx
        paras = []
        for para in docx.Document(str(p)).paragraphs:
            # Join soft line breaks so that one Word paragraph stays one line.
            t = " ".join(para.text.split())
            if not t:
                continue
            pr = para._p.pPr
            lvl = (pr.numPr.ilvl.val if pr is not None and pr.numPr is not None
                   and pr.numPr.ilvl is not None else None)
            paras.append((t, lvl, any(r.bold for r in para.runs)))
        # Person names are bold, either outside the list or at list level 0.
        base = -1 if any(b and l is None for _, l, b in paras) else 0
        for t, lvl, bold in paras:
            if bold and (lvl is None if base == -1 else lvl == 0):
                rows.append((t, -1, True))
            elif lvl is not None:
                rows.append((t, lvl - base - 1, False))
    else:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        unit = md_indent_unit(lines)
        for line in lines:
            m = re.match(r"(\s*)\*\s+(.*)", line)
            if not m or not m.group(2).strip():
                continue
            ind, t = len(m.group(1)), m.group(2).strip()
            rows.append((t, -1 if (ind == 0 and t.startswith("**")) else max(0, round(ind / unit) - 1), False))
    items, person, topic, body = [], None, None, []
    for t, lvl, _ in rows:
        if lvl == -1:
            if topic:
                items.append((person, topic, body))
            person, topic, body = person_key(t.strip("* ")), None, []
            NAMES.setdefault(person, html.unescape(t).strip("* _ 	 "))
        elif lvl == 0 and person:
            if topic:
                items.append((person, topic, body))
            topic, body = t, []
        elif person and topic:
            body.append("  " * (lvl - 1) + "* " + t)
    if topic:
        items.append((person, topic, body))
    return items


def carry_forward(text: str, previous) -> tuple[str, list[str]]:
    from evaluate_speaker_attribution import bag, person_key
    from attribution_baselines import jac
    units = [bag(u) for u in re.split(r"(?m)^(?=### |[*-] )", text) if u.strip()]
    parts = re.split(r"(?m)^(?=## (?!#))", text)
    keys = [person_key(s.split("\n", 1)[0][3:]) for s in parts[1:]]
    add: dict[int, list[str]] = {}
    log = []
    for owner, topic, body in previous_items(previous):
        if owner == "backlog" or re.sub(r"\W", "", topic).lower().startswith("backlog"):
            continue
        ib = bag(topic + " " + " ".join(body))
        if len(ib) < 3 or owner not in keys:
            continue
        if max((jac(ib, u) for u in units), default=0) >= CUT:
            continue
        clean = clean_markup
        ctop, cbody = clean(topic), [c for c in map(clean, body) if _has_text(c)]
        if not ctop.strip() and not cbody:
            continue
        add.setdefault(keys.index(owner), []).append(
            "\n".join([f"* {MARK} {ctop}"] + ["  " + b for b in cbody]))
        log.append(f"{owner}: {ctop[:60]}")
    if not add:
        return text, []
    out = parts[0]
    for i, sec in enumerate(parts[1:]):
        if i in add:
            sec = sec.rstrip("\n") + f"\n\n{HEADING}\n" + "\n".join(add[i]) + "\n\n"
        out += sec
    return out, log


def edit_forward(text: str, previous) -> tuple[str, dict]:
    """Copy the previous protocol without its Backlog and insert each new bullet group of `text`
    under its person's best-matching topic, or under its own heading.

    Groups with at least carry_tag.PREV of their distinctive words in the previous protocol are
    skipped; the returned dict counts new groups, skipped groups and the
    items read from the previous protocol (carried_items).
    """
    from evaluate_speaker_attribution import bag, person_key
    from attribution_baselines import jac
    import carry_tag as CT
    import postedit_effort as PE
    prev_text = CT.read_prev(Path(str(previous)))
    P = CT.terms(prev_text)
    # The Backlog, per person or global, is not protocol content and is never carried.
    items = [it for it in previous_items(previous) if it[0] != "backlog"
             and not re.sub(r"\W", "", it[1]).lower().startswith("backlog")]
    clean = clean_markup
    # person -> [[topic line, [body lines], [new groups]]]
    doc: dict[str, list] = {}
    for owner, topic, body in items:
        ctop = clean(topic)
        cbody = [c for c in map(clean, body) if _has_text(c)]
        if not ctop.strip():
            if not cbody:
                continue
            ctop, cbody = cbody[0].lstrip().lstrip("*- ").strip(), cbody[1:]
        doc.setdefault(owner, []).append([ctop, cbody, []])
    extra: dict[str, list] = {}          # person -> [(heading, group)]
    shown: dict[str, str] = {}
    n_new = n_old = 0
    for sec in re.split(r"(?m)^(?=## (?!#))", text)[1:]:
        head = sec.split("\n", 1)[0][3:].strip()
        who = person_key(head) or head
        shown.setdefault(who, head)
        for blk in re.split(r"(?m)^(?=### )", sec.split("\n", 1)[1] if "\n" in sec else ""):
            th = blk.split("\n", 1)[0] if blk.startswith("### ") else ""
            for g in re.split(r"(?m)^(?=[*-] )", blk):
                if not re.match(r"[*-] ", g):
                    continue
                w = CT.terms(g)
                if w and len(w & P) / len(w) >= CT.PREV:
                    n_old += 1
                    continue
                n_new += 1
                g = g.rstrip("\n")
                topics = doc.get(who, [])
                full = lambda i: bag(topics[i][0] + " " + " ".join(topics[i][1]))
                best = max(range(len(topics)), default=None,
                           key=lambda i: jac(bag(th + " " + g), full(i)))
                if best is not None and jac(bag(th + " " + g), full(best)) >= CUT:
                    topics[best][2].append(g)
                    continue
                # Second route: Jaccard is low for a short group against a long topic.
                # So headings are also compared, by content-word share (PE.MATCH_TH).
                hb = PE.content(th[4:] if th.startswith("### ") else th)
                if hb and topics:
                    hit = max(range(len(topics)),
                              key=lambda i: len(hb & PE.content(topics[i][0])) / len(hb))
                    if len(hb & PE.content(topics[hit][0])) / len(hb) >= PE.MATCH_TH:
                        topics[hit][2].append(g)
                        continue
                extra.setdefault(who, []).append((th or "### Neu", g))
    order = list(doc) + [w for w in extra if w not in doc]
    out = []
    for who in order:
        out.append(f"## {NAMES.get(who) or shown.get(who) or who}")
        for topic, body, new in doc.get(who, []):
            out.append(f"### {topic}")
            out += [("* " + b.lstrip("* ")) if not b.startswith(" ") else b for b in body]
            out += new
        last = None
        for th, g in extra.get(who, []):
            if th != last:
                out.append(th)
                last = th
            out.append(g)
        out.append("")
    return "\n".join(out) + "\n", {"new_groups": n_new, "already_last_week": n_old,
                                   "carried_items": len(items)}


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        sys.exit("usage: python carry_forward.py PROTOCOL.md PREVIOUS.md")
    t, lg = carry_forward(Path(sys.argv[1]).read_text(encoding="utf-8"), sys.argv[2])
    print("\n".join(lg) or "nothing carried")
