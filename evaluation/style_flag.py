"""Flag protocol bullets that open with a nominalized noun and contain no finite verb.

A bullet is flagged when it has at least three words, its first word is capitalized and ends
in -ung, -ierung, -ation, -erung or -ens, and the spaCy German tagger finds no finite verb
(VVFIN, VAFIN, VMFIN) in it. Run as a script, it counts the flagged bullets of
outputs_deliverable/<ARM>/<meeting>/protocol_pro.md (relative to the working directory) for
eight fixed meetings, prints up to three per meeting, skips missing files and changes no file.

Usage:
    python style_flag.py [ARM]    # ARM defaults to v10
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import term_source as TS

_NOM = re.compile(r"^(?:[A-ZÄÖÜ][a-zäöüß]+)?[A-ZÄÖÜ][a-zäöüß]+(?:ung|ierung|ation|erung|ens)\b")
_FIN_TAGS = {"VVFIN", "VAFIN", "VMFIN"}


def flag(bullet: str, nlp=None) -> bool:
    """Return True if a bullet of 3+ words opens with a nominalized noun and has no finite verb."""
    if len(bullet.split()) < 3 or not _NOM.match(bullet):
        return False
    nlp = nlp or TS._nlp()
    if nlp is None:
        return True  # no tagger: the noun opener alone decides
    doc = nlp(bullet)
    return not any(t.tag_ in _FIN_TAGS for t in doc)


def scan(text: str, nlp=None) -> list[str]:
    nlp = nlp or TS._nlp()
    out = []
    for line in text.splitlines():
        m = re.match(r"^\s*[*-]\s+(.+)$", line)
        if not m:
            continue
        body = re.sub(r"\[(ÜBERTRAG|PRÜFEN|Bild)\]", "", m.group(1)).strip()
        if flag(body, nlp):
            out.append(body)
    return out


def main() -> int:
    arm = sys.argv[1] if len(sys.argv) > 1 else "v10"
    nlp = TS._nlp()
    MEET = ["march09", "april13", "april27", "may11", "may18", "june01", "june15", "june22"]
    print(f"STYLE FLAG — bare noun-phrase openers, arm {arm}")
    print("=" * 78)
    total = 0
    for mid in MEET:
        p = Path(f"outputs_deliverable/{arm}/{mid}/protocol_pro.md")
        if not p.exists():
            continue
        hits = scan(p.read_text(encoding="utf-8"), nlp)
        total += len(hits)
        if hits:
            print(f"  {mid:9s} {len(hits)} flagged")
            for h in hits[:3]:
                print(f"      {h[:88]}")
    print(f"\n  {total} bullets flagged for human review across {arm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
