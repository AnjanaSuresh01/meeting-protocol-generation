"""Replace the one foreign-script word LISA is known to put into German protocols.

The only non-Latin word found in the generated protocols is the Russian "ранее" ("earlier"),
used where German has "zuvor". repair() replaces each known word with its German form and
counts it. Any other foreign script is left alone, so the self-check of protokoll.py still
fails on it. The list grows only from words actually found.

    python known_leaks.py      # self-test
"""
from __future__ import annotations

import re

KNOWN = {"ранее": "zuvor", "Ранее": "Zuvor"}
_PAT = re.compile(r"(?<!\w)(" + "|".join(map(re.escape, KNOWN)) + r")(?!\w)")


def repair(text: str) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}

    def sub(m: re.Match) -> str:
        counts[m.group(1)] = counts.get(m.group(1), 0) + 1
        return KNOWN[m.group(1)]

    return _PAT.sub(sub, text), counts


if __name__ == "__main__":
    t, c = repair("* Data Quality Report führte ранее bei großen Datensätzen zum Absturz\n* Ранее war es so")
    assert c == {"ранее": 1, "Ранее": 1} and "ранее" not in t and t.endswith("Zuvor war es so"), (t, c)
    t2, c2 = repair("* nichts Fremdes hier")
    assert c2 == {} and t2 == "* nichts Fremdes hier"
    print("known_leaks self-test passed")
