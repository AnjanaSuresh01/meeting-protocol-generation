"""Build the LLM prompt block that shows lines of the previous protocol as style examples.

The lines with the highest share of content words found in the transcript are chosen, and
the block tells the model to copy only their style, not their content. As a script, it
prints this block:

    python style_exemplars.py previous_protocol.md transcript.txt
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

N_EXEMPLARS = 12
MIN_WORDS, MAX_WORDS = 4, 16  # words per exemplar line


def his_bullets(previous) -> list[str]:
    """Return lines of MIN_WORDS to MAX_WORDS words from the previous protocol, without
    bullet or bold markers. A Backlog line and the lines indented below it are skipped.
    """
    import carry_tag as CT
    text = CT.read_prev(Path(str(previous)))
    out, skip = [], None
    for raw in text.splitlines():
        m = re.match(r"^(\s*)[*-]?\s*(.*)$", raw)
        if not m or not m.group(2).strip():
            continue
        ind, body = len(m.group(1)), re.sub(r"\*\*|__|\s+", " ", m.group(2)).strip()
        if skip is not None and ind <= skip:
            skip = None
        if re.match(r"[*_]*backlog", body, re.I):
            skip = ind
            continue
        if skip is not None or re.fullmatch(r"\*?\*?[A-ZÄÖÜ][\wäöüß.-]*( [A-ZÄÖÜ][\wäöüß.-]*)?\*?\*?",
                                            body):
            continue  # person-name line, not a sentence
        if MIN_WORDS <= len(body.split()) <= MAX_WORDS:
            out.append(body)
    return out


def pick(previous, transcript: str, n: int = N_EXEMPLARS) -> list[str]:
    """Return up to n lines with the highest share of content words found in the transcript.
    Lines with a digit get a small bonus; lines with the same first 28 characters count once.
    """
    from postedit_effort import content
    cand = his_bullets(previous)
    if not cand:
        return []
    heard = content(re.sub(r"\[[\d:]+\]\s+\S+?:", " ", transcript))
    scored = []
    for b in cand:
        c = content(b)
        if not c:
            continue
        sim = len(c & heard) / len(c)
        concrete = bool(re.search(r"\d", b))
        scored.append((sim + 0.1 * concrete, b))
    scored.sort(reverse=True)
    seen, out = set(), []
    for _, b in scored:
        key = b[:28].lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(b)
        if len(out) >= n:
            break
    return out


# Nested exemplars for protokoll.py --depth
N_GROUPS = 5


def his_groups(previous) -> list[list[tuple[int, str]]]:
    """Return bullets of MIN_WORDS to MAX_WORDS words with one to five leaf sub-bullets.
    Each group is [(0, bullet), (1, sub-bullet), ...]; Backlog subtrees are skipped.
    """
    import carry_tag as CT
    rows = []
    skip = None
    for raw in CT.read_prev(Path(str(previous))).splitlines():
        m = re.match(r"^(\s*)[*-]\s+(.*\S)", raw)
        if not m:
            continue
        ind = len(m.group(1).expandtabs(4))
        body = re.sub(r"\*\*|__|\s+", " ", m.group(2)).strip()
        if skip is not None and ind <= skip:
            skip = None
        if re.match(r"[*_]*backlog", body, re.I):
            skip = ind
            continue
        if skip is None:
            rows.append((ind, body))
    groups = []
    for i, (ind, body) in enumerate(rows):
        if len(body.split()) < MIN_WORDS or len(body.split()) > MAX_WORDS:
            continue
        kids = []
        for ind2, body2 in rows[i + 1:]:
            if ind2 <= ind:
                break
            kids.append((ind2, body2))
        if not kids or len(kids) > 5:
            continue
        child_ind = min(k for k, _ in kids)
        if any(k > child_ind for k, _ in kids):
            continue
        groups.append([(0, body)] + [(1, b) for _, b in kids])
    return groups


def pick_groups(previous, transcript: str, n: int = N_GROUPS) -> list[list[tuple[int, str]]]:
    """Return up to n groups with the highest share of content words found in the transcript.
    Groups whose first bullet has the same first 28 characters count once.
    """
    from postedit_effort import content
    heard = content(re.sub(r"\[[\d:]+\]\s+\S+?:", " ", transcript))
    scored = []
    for g in his_groups(previous):
        c = content(" ".join(b for _, b in g))
        if c:
            scored.append((len(c & heard) / len(c), g))
    scored.sort(key=lambda x: -x[0])
    seen, out = set(), []
    for _, g in scored:
        key = g[0][1][:28].lower()
        if key not in seen:
            seen.add(key)
            out.append(g)
        if len(out) >= n:
            break
    return out


def nested_block(previous, transcript: str) -> str:
    groups = pick_groups(previous, transcript)
    if not groups:
        return block(previous, transcript)
    lines = "\n".join(("  " * d) + f"* {b}" for g in groups for d, b in g)
    return ("=== SO SCHREIBT DER PROTOKOLLFÜHRER (nur STIL und GLIEDERUNG, nicht der Inhalt) ===\n"
            f"{lines}\n"
            "Schreibe deine Stichpunkte in genau diesem Ton: knapp, mit dem Verb oder der "
            "Tatsache voran, Zahlen und Eigennamen ausgeschrieben. Und gliedere wie er: Einzelheiten, "
            "Gründe und nächste Schritte zu einer Aussage stehen eingerückt darunter. Übernimm aus "
            "diesem Block KEINEN Inhalt — er zeigt nur, wie formuliert und gegliedert wird.\n"
            "=== ENDE ===\n\n")


def block(previous, transcript: str) -> str:
    ex = pick(previous, transcript)
    if not ex:
        return ""
    lines = "\n".join(f"* {b}" for b in ex)
    return ("=== SO SCHREIBT DER PROTOKOLLFÜHRER (nur der STIL, nicht der Inhalt) ===\n"
            f"{lines}\n"
            "Schreibe deine Stichpunkte in genau diesem Ton: knapp, mit dem Verb oder der "
            "Tatsache voran, Zahlen und Eigennamen ausgeschrieben. Übernimm aus diesem Block "
            "KEINEN Inhalt — er zeigt nur, wie formuliert wird.\n=== ENDE ===\n\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python style_exemplars.py PREVIOUS.md TRANSCRIPT.txt")
    print(block(sys.argv[1], Path(sys.argv[2]).read_text(encoding="utf-8", errors="replace")))
