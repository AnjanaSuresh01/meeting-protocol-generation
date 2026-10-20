"""Mark every bullet that comes from last week's protocol and not from this meeting's audio.

A bullet is carried when fewer than SUPPORT of its distinctive words (at least 5 letters,
not ordinary German) occur in this meeting's transcript and at least PREV of them occur in
the previous protocol. It is also carried when every figure of two or more digits in it
occurs in the previous protocol and none occurs in the transcript. A number spoken as words
is not recognized, so such a bullet can be marked although it was said. Carried bullets get
the prefix [ÜBERTRAG]. Nothing is removed or reworded, and headings are never marked.

    python carry_tag.py protocol.md transcript.txt previous.md [--out tagged.md]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from term_source import GERMAN_COMMON, tokens, _clean

MARK = "[ÜBERTRAG]"
SUPPORT = 0.30
PREV = 0.60
MIN_TERMS = 2  # minimum distinctive words for the word rule; 3 removed more right marks than wrong ones


def terms(text: str) -> set[str]:
    return {_clean(t).lower() for t in tokens(text)
            if len(_clean(t)) >= 5 and _clean(t).lower() not in GERMAN_COMMON}


def numbers(text: str) -> set[str]:
    """Figures of two or more digits ('230', '8x' -> '8' is too short to count)."""
    return set(re.findall(r"(?<![\w.])\d{2,}(?:[.,]\d+)?(?![\w])", text))


def read_prev(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".docx":
        import docx
        return "\n".join(p.text for p in docx.Document(str(path)).paragraphs)
    if suffix == ".pdf":
        from term_source import read_document
        return read_document(str(path.resolve()))[0]
    return path.read_text(encoding="utf-8", errors="replace")


def tag(protocol: str, transcript: str, previous: str) -> tuple[str, int, int]:
    T, P = terms(transcript), terms(previous)
    # timestamps such as [00:58:30] are not spoken figures, so they are removed first
    NT = numbers(re.sub(r"\[\d\d:\d\d:\d\d\]", " ", transcript))
    NP = numbers(previous)
    out, n_bul, n_tag = [], 0, 0
    for line in protocol.splitlines():
        m = re.match(r"^(\s*[*-]\s+)(.*)$", line)
        if not m or MARK in line:
            out.append(line)
            continue
        n_bul += 1
        w = terms(m.group(2))
        nums = numbers(m.group(2))
        by_words = len(w) >= MIN_TERMS and len(w & T) / len(w) < SUPPORT and len(w & P) / len(w) >= PREV
        # every figure of the bullet is in last week's protocol and none was heard this week
        by_numbers = bool(nums) and nums <= NP and not (nums & NT)
        if by_words or by_numbers:
            line = f"{m.group(1)}{MARK} {m.group(2)}"
            n_tag += 1
        out.append(line)
    return "\n".join(out) + ("\n" if protocol.endswith("\n") else ""), n_bul, n_tag


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("protocol", type=Path)
    ap.add_argument("transcript", type=Path)
    ap.add_argument("previous", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    text, nb, nt = tag(a.protocol.read_text(encoding="utf-8"),
                       a.transcript.read_text(encoding="utf-8", errors="replace"),
                       read_prev(a.previous))
    (a.out or a.protocol).write_text(text, encoding="utf-8")
    print(f"{nt}/{nb} bullets marked {MARK}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
