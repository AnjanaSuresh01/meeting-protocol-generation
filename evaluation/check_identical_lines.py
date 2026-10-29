"""Counts the v10 bullets that are identical to a bullet of the same week's reference protocol.

For each of the eight meetings it reads outputs_deliverable/v10/<meeting>/protocol_pro.md, the
human-written protocol of the same meeting (evaluate_speaker_attribution.HUMAN) and the previous
protocol (term_source.PREVIOUS). Bullets are compared after removing HTML tags, the [ÜBERTRAG] and
[PRÜFEN] labels and bold markers and collapsing whitespace; v10 bullets of fewer than two words are
skipped. It prints how many v10 bullets carry both labels or only one of them, how many are identical
to a reference bullet, how many of those also occur in last week's protocol (as a bullet or anywhere
in its text) and how many carry a label, and lists the identical bullets that do not occur in last
week's protocol.

    python check_identical_lines.py

Prints to stdout only.
"""
import re
import sys
from pathlib import Path

R = Path(__file__).resolve().parent
sys.path.insert(0, str(R))
from evaluate_speaker_attribution import HUMAN
from term_source import PREVIOUS

MEET = ["march09", "april13", "april27", "may11", "may18", "june01", "june15", "june22"]
BUL = re.compile(r"^\s*[*-]\s+(.*)$")
TAG = re.compile(r"\[(ÜBERTRAG|PRÜFEN)\]")


def norm(s):
    s = re.sub(r"<[^>]+>", "", s)
    s = TAG.sub("", s)
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    return re.sub(r"\s+", " ", s).strip()


def bullets(text):
    return [m.group(1) for m in (BUL.match(l) for l in text.splitlines()) if m]


tot = ident = in_prev = labelled = both = carry_only = check_only = 0
not_prev = []
for mid in MEET:
    gen = (R / f"outputs_deliverable/v10/{mid}/protocol_pro.md").read_text(encoding="utf-8")
    his = {norm(b) for b in bullets(Path(HUMAN[mid]).read_text(encoding="utf-8", errors="replace"))}
    prev_txt = Path(PREVIOUS[mid]).read_text(encoding="utf-8", errors="replace")
    prev = {norm(b) for b in bullets(prev_txt)}
    prev_flat = norm(prev_txt)
    for b in bullets(gen):
        tot += 1
        c, k = "[ÜBERTRAG]" in b, "[PRÜFEN]" in b
        both += c and k
        carry_only += c and not k
        check_only += k and not c
        n = norm(b)
        if len(n.split()) < 2 or n not in his:
            continue
        ident += 1
        labelled += c or k
        if n in prev or n in prev_flat:
            in_prev += 1
        else:
            not_prev.append((mid, n))
print(f"v10 bullets {tot}: both labels {both}, ÜBERTRAG only {carry_only}, PRÜFEN only {check_only}")
print(f"identical to the same-week reference protocol: {ident}; also in last week's protocol: {in_prev}; "
      f"labelled {labelled}, unlabelled {ident - labelled}")
print("identical but NOT in last week's protocol:")
for mid, n in not_prev:
    print(f"  {mid}: {n}")
