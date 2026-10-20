"""Deterministic repair of the bullet and heading layout of generated Markdown protocols.

Marker lines ([ÜBERTRAG], [Aufgabe]) written as headings or bare text, indented bare lines and
'1.' enumerations become bullets (the number is dropped, except before a date word such as
'Juni'); a missing space after a bullet or heading marker is added; echoed prompt fences such as
'=== TRANSKRIPT ===' are blanked; echoed '[Person / Topic]' scaffolding on a marker line becomes
'Topic – '. Run as a script, it checks every outputs_v10/*/*/protocol_*.md next to this file.

    python normalize_structure.py            # report what would change
    python normalize_structure.py --apply    # rewrite files, write goldset/structure_repairs.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
TREE = REPO / "outputs_v10"
LOG = REPO / "goldset" / "structure_repairs.json"

# Case-insensitive, so '[Übertrag]' also counts as a marker.
# Marker spelling is checked in thesis_pipeline/protocol_validate.py, not here.
_MARKERED = re.compile(r"\[(?:ÜBERTRAG|UEBERTRAG|Aufgabe)\]", re.I)
_HEADING = re.compile(r"^\s*(#{2,4})\s+(.*)$")
_BULLET = re.compile(r"^\s*[*-]\s+\S")
_SEPARATOR = re.compile(r"^\s*([-*_]\s*){3,}$")
# A bullet without the space after its marker ('*Text', '-Text').
# Not '**bold**', '---' or a negative number such as '-20'.
_MISSING_SPACE_BULLET = re.compile(r"^[*-](?![*\s\d.,-])\S")
# An enumeration ('1. ', '2) ') used as a list marker.
# Only 1-2 digits, so a year such as '2026.' does not match.
_NUMBERED = re.compile(r"^\d{1,2}[.)]\s+")
# A day, period or floor number ('18. Juni', '3. Quartal', '2. OG') is content,
# so the number is kept.
_DATE_WORD = re.compile(r"^(?:Jan(?:uar)?|Feb(?:ruar)?|M[äa]r(?:z)?|Apr(?:il)?|Mai|Jun(?:i)?"
                        r"|Jul(?:i)?|Aug(?:ust)?|Sep(?:tember)?|Okt(?:ober)?|Nov(?:ember)?"
                        r"|Dez(?:ember)?|Quartal|Halbjahr|KW|Woche|Stock|OG|EG)\b", re.I)
# A heading without the space after its hashes ('###Title') does not render as a heading.
# Only 2-4 hashes, so '#hashtag' does not match.
_TIGHT_HEADING = re.compile(r"^(#{2,4})(?=[^\s#])")
# Echoed prompt scaffolding of the form '[Name / Topic] rest'.
_SCAFFOLD = re.compile(r"\[[^\]\[]{3,60}\s/\s[^\]\[]{0,60}\]\s*")


# A prompt section fence echoed into the output, such as '=== TRANSKRIPT ==='.
# It carries no content, so the line is blanked.
_ECHOED_FENCE = re.compile(r"^\s*={2,}[^=]{1,60}={2,}\s*$")


def repair_line(line: str) -> tuple[str, str | None]:
    """Repair one line; return (new_line, repair_name), or (line, None) if unchanged."""
    if not line.strip() or _SEPARATOR.match(line):
        return line, None

    if _ECHOED_FENCE.match(line):
        return "", "echoed-prompt-fence"

    tight = _TIGHT_HEADING.match(line.strip())
    if tight and not _HEADING.match(line):
        return f"{tight.group(1)} {line.strip()[len(tight.group(1)):].lstrip()}", "heading-spacing"

    m = _HEADING.match(line)
    if m and _MARKERED.search(m.group(2)):
        body = m.group(2).lstrip("*- ").strip()
        return f"* {body}", "heading->bullet"

    if not _HEADING.match(line) and not _BULLET.match(line):
        stripped = line.strip()
        # Keep the leading whitespace: it sets the bullet level.
        indent = line[:len(line) - len(line.lstrip())]
        if _MISSING_SPACE_BULLET.match(stripped):
            return f"{indent}* {stripped[1:].lstrip()}", "bullet-spacing"
        if _MARKERED.search(line):
            return f"{indent}* {stripped}", "bare->bullet"
        if _NUMBERED.match(stripped):
            rest = _NUMBERED.sub("", stripped, count=1)
            if _DATE_WORD.match(rest):
                return f"{indent}* {stripped}", "date->bullet"
            return f"{indent}* {rest}", "numbered->bullet"
        # Only an indented bare line becomes a bullet, because its indentation gives the level.
        # A flush-left bare line is left as it is.
        if indent:
            return f"{indent}* {stripped}", "indented->bullet"

    return line, None


def strip_scaffold(line: str) -> tuple[str, str | None]:
    """Replace '[Person / Topic] ' scaffolding on a marker line with 'Topic – '.

    Return (new_line, 'scaffold-stripped'), or (line, None) if unchanged.
    """
    if not _MARKERED.search(line):
        return line, None
    m = _SCAFFOLD.search(line)
    if not m:
        return line, None
    inner = m.group(0).strip().strip("[]")
    topic = inner.split("/", 1)[1].strip() if "/" in inner else ""
    replacement = f"{topic} – " if topic else ""
    new = line[:m.start()] + replacement + line[m.end():]
    return (new, "scaffold-stripped") if new != line else (line, None)


def repair(text: str) -> tuple[str, dict[str, int]]:
    out, counts = [], {}
    for line in text.splitlines():
        line, what = repair_line(line)
        if what:
            counts[what] = counts.get(what, 0) + 1
        line, what2 = strip_scaffold(line)
        if what2:
            counts[what2] = counts.get(what2, 0) + 1
        out.append(line)
    return "\n".join(out) + ("\n" if text.endswith("\n") else ""), counts


def main() -> int:
    apply = "--apply" in sys.argv
    log: dict[str, dict] = {}
    total = 0

    print("STRUCTURE NORMALISATION — deterministic layout repair")
    print("=" * 84)
    print("Layout only. Nothing is reworded, added or deleted; truncation is NOT repairable.\n")

    for p in sorted(TREE.glob("*/*/protocol_*.md")):
        text = p.read_text(encoding="utf-8")
        new, counts = repair(text)
        if not counts:
            continue
        rel = str(p.relative_to(TREE)).replace("\\", "/")
        log[rel] = counts
        total += sum(counts.values())
        print(f"  {rel:52s} {counts}")
        if apply:
            p.write_text(new, encoding="utf-8")

    if not log:
        print("  nothing to repair.")
    else:
        print(f"\n  {len(log)} file(s), {total} line(s) repaired"
              f"{'' if apply else ' — report only, re-run with --apply'}")

    if apply:
        LOG.parent.mkdir(exist_ok=True)
        LOG.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  wrote -> {LOG.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
