"""Checks whether the content of a person section in the reference minutes occurs in the recording.

For each case in CASES (minutes file, meeting, person), the section under that person's bold top-level
bullet is split into topics: lines at nesting level 0 or 1 that do not match ABSENT_MARKERS are topics,
and every other line is a detail of the topic above. The distinctive terms of a topic and its details are the
tokens of at least 5 characters that are not in term_source.GERMAN_COMMON. A topic counts as audible when
at least half of its distinctive terms occur in the vocabulary of the meeting's transcripts from all six
recognizers (outputs_v10/<model>/<meeting>/transcript_speakers.txt, else transcript.txt). It prints the
share per topic and the number of audible topics per case and overall; topics whose terms occur in no
transcript cannot be produced from the audio by any system.

    python absent_reachability.py    -> stdout only
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from term_source import GERMAN_COMMON, tokens, _clean          # noqa: E402

TREE = REPO / "outputs_v10"
MODELS = ["canary_diar", "cohere", "granite41",
          "whisper_large_v3", "whisper_turbo", "whisper_turbo_de"]

# (reference minutes, meeting id, person whose section is checked)
CASES = [
    ("Meeting_2026-06-01.md", "june01", "Vincent Tischler"),
    ("Meeting_2026-06-15.md", "june15", "Thomas Weber"),
    ("Meeting_2026-06-22.md", "june22", "Johannes Wirth"),
]

ABSENT_MARKERS = re.compile(r"\b(urlaub|krank|nicht da|abwesend|entschuldigt)\b", re.I)


def person_section(path: Path, who: str) -> list[tuple[int, str]]:
    """[(indent level, text)] for the block under '**who**' in a markdown minutes file."""
    lines = path.read_text(encoding="utf-8").splitlines()
    first = who.split()[0]
    start = None
    for i, l in enumerate(lines):
        if re.match(r"^\*   \*\*", l) and first in l:
            start = i
            break
    if start is None:
        return []
    out = []
    widths = [len(l) - len(l.lstrip()) for l in lines if l.strip().startswith("*")]
    unit = min((w for w in widths if w > 0), default=4)
    for l in lines[start + 1:]:
        if re.match(r"^\*   \*\*", l):
            break
        if not l.strip():
            continue
        indent = len(l) - len(l.lstrip())
        body = re.sub(r"^\*\s+", "", l.strip())
        body = re.sub(r"^\*\*(.*?)\*\*$", r"\1", body).strip()
        out.append((indent // unit, body))
    return out


def distinctive(text: str) -> set[str]:
    return {_clean(t) for t in tokens(text)
            if len(_clean(t)) >= 5 and _clean(t).lower() not in GERMAN_COMMON}


def audible_vocab(mid: str) -> set[str]:
    vocab: set[str] = set()
    for m in MODELS:
        p = TREE / m / mid / "transcript_speakers.txt"
        if not p.exists() or not p.stat().st_size:
            p = TREE / m / mid / "transcript.txt"
        if p.exists() and p.stat().st_size:
            vocab |= {_clean(t).lower() for t in tokens(
                p.read_text(encoding="utf-8", errors="replace"))}
    return vocab


def main() -> int:
    print("IS THE ABSENT PERSON'S CONTENT IN THE RECORDING?")
    print("=" * 92)
    print("The reference minutes give a full section to people they mark as absent. This asks")
    print("whether an audio-based pipeline could have produced that section at all.\n")

    grand_reach = grand_total = 0
    for fname, mid, who in CASES:
        sec = person_section(REPO / fname, who)
        if not sec:
            print(f"  {mid}: could not locate the section for {who}")
            continue
        marker = next((t for _, t in sec if ABSENT_MARKERS.search(t)), None)
        vocab = audible_vocab(mid)

        # group into topics: level 0 and 1 lines are topics, deeper lines belong to the topic above
        topics: list[tuple[str, list[str]]] = []
        for lvl, body in sec:
            if lvl <= 1 and not ABSENT_MARKERS.search(body):
                topics.append((body, []))
            elif topics:
                topics[-1][1].append(body)

        print(f"  {mid}  {who}   (note: {marker!r})")
        reach = 0
        for topic, kids in topics:
            terms = distinctive(topic) | set().union(*(distinctive(k) for k in kids)) \
                if kids else distinctive(topic)
            terms = {t for t in terms if t}
            if not terms:
                continue
            hit = {t for t in terms if t.lower() in vocab}
            share = len(hit) / len(terms)
            ok = share >= 0.5
            reach += ok
            print(f"      {'AUDIBLE  ' if ok else 'NOT HEARD'} {share:5.0%} "
                  f"({len(hit):2d}/{len(terms):2d})  {topic[:58]}")
        grand_reach += reach
        grand_total += len(topics)
        print(f"      -> {reach}/{len(topics)} topics traceable in the audio\n")

    if grand_total:
        print("-" * 92)
        print(f"  ACROSS THE THREE ABSENT PEOPLE: {grand_reach}/{grand_total} topics "
              f"({grand_reach / grand_total:.0%}) have their content in the recording.")
        print("  The remainder cannot be produced from audio by ANY system, so coverage and")
        print("  'missing attendee' figures must exclude them or state that they include them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
