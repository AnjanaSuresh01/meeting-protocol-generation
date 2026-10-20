"""Checks the heading and bullet structure of the protocols in outputs_v10/<model>/<meeting>/.

Runs thesis_pipeline.protocol_validate on each file named in VARIANTS, adds checks for lines
outside the hierarchy, bullets with no topic and glued headings, and reports bullet nesting
without counting it as a failure. Reads outputs_v10/ next to this file, writes
protocol_validation_v10.json next to this file, exits with code 1 if any protocol fails and
with code 2 if no protocol file is found.

Usage:
    python validate_protocols_v10.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from thesis_pipeline.protocol_validate import validate_protocol

# A non-empty line that is not a heading, a bullet or a '---' separator is outside the hierarchy.
_HEADING = re.compile(r"^\s*#{2,4}\s+\S")
_BULLET = re.compile(r"^\s*[*-]\s+\S")
_SEPARATOR = re.compile(r"^\s*([-*_]\s*){3,}$")


def stray_content_lines(text: str) -> list[str]:
    return [l for l in text.splitlines()
            if l.strip()
            and not _HEADING.match(l)
            and not _BULLET.match(l)
            and not _SEPARATOR.match(l)]


_H2_ONLY = re.compile(r"^##\s+(?!#)\S")
_H3_ONLY = re.compile(r"^###\s+\S")

# Matches a '##' to '####' heading glued to text on the same line (Markdown does not render it).
# The preceding character must be neither whitespace nor '#', or '### Topic' would match.
_GLUED_HEADING = re.compile(r"[^\s#]#{2,4}\s+\S")


def glued_headings(text: str) -> list[str]:
    return [l for l in text.splitlines() if _GLUED_HEADING.search(l)]


def is_senat(text: str) -> bool:
    """Return True if the text has '## TOP n:' agenda headings (Senat format)."""
    return bool(re.search(r"^##\s+TOP\s+\d+\s*:", text, re.M))


def orphan_bullets(text: str) -> list[str]:
    """Return the bullets under a '##' heading that have no '###' topic above them.

    A Senat protocol returns an empty list: its bullets sit directly under '## TOP n:'
    headings, and it has no topic level.
    """
    if is_senat(text):
        return []
    out, have_topic = [], True
    for line in text.splitlines():
        if _H2_ONLY.match(line):
            have_topic = False
        elif _H3_ONLY.match(line):
            have_topic = True
        elif _BULLET.match(line) and not have_topic:
            out.append(line)
    return out


def bullet_depths(text: str) -> dict[int, int]:
    """Count bullets per indentation level (two spaces per level, level 0 is flush left)."""
    out: dict[int, int] = {}
    for line in text.splitlines():
        m = re.match(r"^( *)[*-]\s+\S", line)
        if m:
            lvl = len(m.group(1)) // 2
            out[lvl] = out.get(lvl, 0) + 1
    return out


# Minimum number of bullets for an unnested topic to count as a flat topic; a shorter topic
# needs no sub-structure.
FLAT_TOPIC_MIN_BULLETS = 6


def flat_topics(text: str) -> list[str]:
    """Return the '###' topics with at least FLAT_TOPIC_MIN_BULLETS bullets, all at level 0."""
    out, topic, depths = [], None, []
    def close():
        if topic and len(depths) >= FLAT_TOPIC_MIN_BULLETS and set(depths) == {0}:
            out.append(topic)
    for line in text.splitlines():
        if _H3_ONLY.match(line) or _H2_ONLY.match(line):
            close()
            topic = line.strip("# ").strip() if _H3_ONLY.match(line) else None
            depths = []
        else:
            m = re.match(r"^( *)[*-]\s+\S", line)
            if m:
                depths.append(len(m.group(1)) // 2)
    close()
    return out


REPO = Path(__file__).resolve().parent
TREE = REPO / "outputs_v10"
VARIANTS = ["protocol_flash.md", "protocol_pro.md",
            "protocol_flash_carry.md", "protocol_pro_carry.md",
            "protocol_flash_tracked.md", "protocol_pro_tracked.md",
            "protocol_flash_merged.md", "protocol_pro_merged.md",
            "protocol_flash_fixed.md", "protocol_pro_fixed.md",
            "protocol_flash_fixed_senat.md", "protocol_pro_fixed_senat.md",
            "protocol_pro_fixed_carry.md", "protocol_flash_fixed_carry.md",
            "protocol_pro_fixed_len.md", "protocol_flash_fixed_len.md"]


def main() -> int:
    report: dict[str, dict] = {}
    n_ok = n_bad = 0
    bad_rows = []

    for which in VARIANTS:
        for md in sorted(TREE.glob(f"*/*/{which}")):
            model, mid = md.parent.parent.name, md.parent.name
            text = md.read_text(encoding="utf-8")
            r = validate_protocol(text)
            rec = r.as_dict()
            stray = stray_content_lines(text)
            orphans = orphan_bullets(text)
            glued = glued_headings(text)
            flat = flat_topics(text)
            depths = bullet_depths(text)
            rec["stray_content_lines"] = len(stray)
            rec["orphan_bullets"] = len(orphans)
            rec["glued_headings"] = len(glued)
            rec["flat_topics"] = len(flat)
            rec["bullet_depths"] = depths
            rec["nested_share"] = (round(
                sum(v for k, v in depths.items() if k > 0) / sum(depths.values()), 3)
                if depths else 0.0)
            report.setdefault(which, {}).setdefault(model, {})[mid] = rec
            ok = r.ok and not stray and not orphans and not glued
            if ok:
                n_ok += 1
            else:
                n_bad += 1
                if stray:
                    r.problems.append(
                        f"{len(stray)} content line(s) outside the hierarchy "
                        f"(neither heading nor bullet), e.g. {stray[0].strip()[:34]!r}")
                if orphans:
                    r.problems.append(
                        f"{len(orphans)} bullet(s) directly under a person with no topic "
                        f"above, e.g. {orphans[0].strip()[:34]!r}")
                if glued:
                    r.problems.append(
                        f"{len(glued)} heading(s) glued to the end of a line and therefore "
                        f"invisible, e.g. ...{glued[0].strip()[-42:]!r}")
                bad_rows.append((which, model, mid, r))

    print("PROTOCOL STRUCTURE GATE — thesis_pipeline.protocol_validate over outputs_v10")
    print("=" * 88)
    print("Required format: '## Person' -> '### Topic' -> '* bullet', three levels, nothing flatter.\n")
    if n_ok + n_bad == 0:
        print(f"No protocol files found under {TREE}.")
        return 2

    if bad_rows:
        print(f"{'variant':26s}{'model':17s}{'meeting':10s}{'P':>3s}{'T':>5s}{'B':>5s}  problems")
        print("-" * 100)
        for which, model, mid, r in bad_rows:
            print(f"{which:26s}{model:17s}{mid:10s}{r.n_persons:>3d}{r.n_topics:>5d}"
                  f"{r.n_bullets:>5d}  {'; '.join(r.problems)[:44]}")
        print("-" * 88)
    else:
        print("   No violations.\n")

    total = n_ok + n_bad
    print(f"{n_ok} / {total} protocols pass the structure gate"
          f"{'' if n_bad == 0 else f' — {n_bad} FLAGGED'}.")

    # Nesting is reported only and does not change the pass/fail count.
    print("\nNESTING — measured, not assumed")
    print("-" * 88)
    for which, per_model in report.items():
        rows = [rec for md in per_model.values() for rec in md.values()]
        if not rows:
            continue
        nested = sum(r.get("nested_share", 0) for r in rows) / len(rows)
        flat = sum(r.get("flat_topics", 0) for r in rows)
        deep = sum(1 for r in rows if max((int(k) for k in r.get("bullet_depths", {0: 0})),
                                          default=0) >= 1)
        print(f"  {which:30s} {len(rows):3d} files | bullets below level 0: {nested:6.1%}"
              f" | files with any nesting: {deep:3d} | flat topics (>={FLAT_TOPIC_MIN_BULLETS}"
              f" bullets, no nesting): {flat}")

    out = REPO / "protocol_validation_v10.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote -> {out.name}")
    return 1 if n_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
