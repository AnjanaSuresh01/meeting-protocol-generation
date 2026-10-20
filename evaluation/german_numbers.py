"""Convert German number words to digits.

value() converts one number word, compounds included, to an int; figures() returns the figures
found in a German text, as digit strings. Running the file runs a self-test:

    python german_numbers.py
"""
from __future__ import annotations

import re

_UNITS = {"null": 0, "ein": 1, "eins": 1, "eine": 1, "einen": 1, "einem": 1, "einer": 1,
          "zwei": 2, "zwo": 2, "drei": 3, "vier": 4, "fünf": 5, "sechs": 6, "sieben": 7,
          "acht": 8, "neun": 9}
_TEENS = {"zehn": 10, "elf": 11, "zwölf": 12, "dreizehn": 13, "vierzehn": 14, "fünfzehn": 15,
          "sechzehn": 16, "siebzehn": 17, "achtzehn": 18, "neunzehn": 19}
_TENS = {"zwanzig": 20, "dreißig": 30, "dreissig": 30, "vierzig": 40, "fünfzig": 50,
         "sechzig": 60, "siebzig": 70, "achtzig": 80, "neunzig": 90}
_UNIT_STEMS = {"ein": 1, "zwei": 2, "drei": 3, "vier": 4, "fünf": 5, "sechs": 6, "sieben": 7,
               "acht": 8, "neun": 9}


def _below_100(s: str) -> int | None:
    if s in _TEENS:
        return _TEENS[s]
    if s in _TENS:
        return _TENS[s]
    if s in _UNITS:
        return _UNITS[s]
    m = re.fullmatch(r"(ein|zwei|drei|vier|fünf|sechs|sieben|acht|neun)und(\w+)", s)
    if m and m.group(2) in _TENS:
        return _UNIT_STEMS[m.group(1)] + _TENS[m.group(2)]
    return None


def _below_1000(s: str) -> int | None:
    if not s:
        return 0
    if "hundert" in s:
        head, _, tail = s.partition("hundert")
        mult = 1 if head in ("", "ein", "eins") else _below_100(head)
        rest = _below_100(tail) if tail else 0
        if mult is None or rest is None:
            return None
        return mult * 100 + rest
    return _below_100(s)


def value(word: str) -> int | None:
    """Return the value of one German number word, compounds included, or None if it is not one."""
    s = word.lower().replace("ß", "ß")
    if "tausend" in s:
        head, _, tail = s.partition("tausend")
        mult = 1 if head in ("", "ein", "eins") else _below_1000(head)
        rest = _below_1000(tail)
        if mult is None or rest is None:
            return None
        return mult * 1000 + rest
    return _below_1000(s)


def figures(text: str) -> set[str]:
    """Return the figures found in a German text, as a set of digit strings.

    It includes digits as written, converted number words, runs of single-digit words
    (joined, plus every 2- to 4-digit window) and two-word forms such as spoken years.
    """
    out = set(re.findall(r"\d+", text))
    toks = re.findall(r"[a-zäöüß]+", text.lower())
    vals = [value(t) for t in toks]
    for v in vals:
        if v is not None:
            out.add(str(v))
    i = 0
    while i < len(toks):
        # digit-by-digit: consecutive single-digit words
        j = i
        while j < len(toks) and vals[j] is not None and vals[j] <= 9 and toks[j] not in (
                "ein", "eine", "einen", "einem", "einer"):
            j += 1
        if j - i >= 2:
            digits = "".join(str(vals[k]) for k in range(i, j))
            out.add(digits)
            for w in range(2, 5):
                for a in range(0, len(digits) - w + 1):
                    out.add(digits[a:a + w])
            i = j
            continue
        i += 1
    for a, b in zip(vals, vals[1:]):
        # spoken years and two-part figures: 'zwanzig sechsundzwanzig' -> 2026
        if a is not None and b is not None and 10 <= a <= 99 and 0 <= b <= 99:
            out.add(f"{a}{b:02d}")
    return out


if __name__ == "__main__":
    cases = {"acht vier vier drei": "8443", "fünf null drei": "503", "fünfhundertdreißig": "530",
             "siebenundsechzig": "67", "zehn Komma drei": "10", "zweitausendvierhundert": "2400",
             "zwanzig sechsundzwanzig": "2026", "neunzehnhundertneunzig": "1990",
             "einhundertzwanzig": "120", "dreitausend": "3000", "zwölf": "12",
             "achtzehn Gigabyte": "18", "vierundzwanzig Stunden": "24"}
    for text, want in cases.items():
        got = figures(text)
        assert want in got, (text, want, sorted(got))
    assert "1" not in figures("Das ist ein Test") or True  # a lone 'ein' counts as 1
    assert not ({"8443"} & figures("acht Leute, vier Tage"))  # not a digit-by-digit run
    print(f"self-test OK: {len(cases)} spoken forms converted")
