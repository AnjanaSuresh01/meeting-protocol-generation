"""Mark carried bullets that were new in the previous protocol with [PRÜFEN].

A bullet is carried when it has at least carry_tag.MIN_TERMS distinctive words and the share
of them found in the previous protocol reaches carry_tag.PREV, and new when no line of the older protocol (the one before the
previous) matches it. [PRÜFEN] replaces a leading [ÜBERTRAG], so each line keeps one label.

Run as a script, it prints only the counts and writes no file:

    python stale_flag.py PROTOCOL.md PREVIOUS.md OLDER.md
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

MARK = "[PRÜFEN]"
_BULLET = re.compile(r"^(\s*[*-]\s+)(.*)$")


def mark(protocol: str, previous, older) -> tuple[str, int, int]:
    """Mark the protocol text; previous and older are paths to the two earlier protocols.

    Returns (text, n_marked, n_carried).
    """
    import carry_tag as CT
    import postedit_effort as PE
    P = CT.terms(CT.read_prev(Path(str(previous))))
    old_bullets = [c for c in (PE.content(l) for l in
                               CT.read_prev(Path(str(older))).splitlines()) if c]
    out, n_carried, n_marked = [], 0, 0
    for line in protocol.splitlines():
        m = _BULLET.match(line)
        if not m or MARK in line:
            out.append(line)
            continue
        body = m.group(2)
        plain = body.replace(CT.MARK, "").strip()
        # Terms come from the text without [ÜBERTRAG]: its label word would count as a
        # distinctive term and lower the share found in the previous protocol.
        w = CT.terms(plain)
        carried = len(w) >= CT.MIN_TERMS and len(w & P) / len(w) >= CT.PREV
        if carried:
            n_carried += 1
            c = PE.content(plain)
            if c and not PE.covered(c, old_bullets):
                # One label per line: drop the leading [ÜBERTRAG] before adding [PRÜFEN].
                if body.startswith(CT.MARK + " "):
                    body = body[len(CT.MARK) + 1:]
                line = f"{m.group(1)}{MARK} {body}"
                n_marked += 1
        out.append(line)
    return "\n".join(out) + ("\n" if protocol.endswith("\n") else ""), n_marked, n_carried


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("usage: python stale_flag.py PROTOCOL.md PREVIOUS.md OLDER.md")
    t, n, c = mark(Path(sys.argv[1]).read_text(encoding="utf-8"), sys.argv[2], sys.argv[3])
    print(f"{n}/{c} carried bullets marked {MARK}")
