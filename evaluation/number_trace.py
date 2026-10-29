"""Checks whether each figure in the bullets LISA wrote occurs in the transcript or the previous protocol.

The input per meeting is outputs_deliverable/<arm>/<meeting>/protocol_pro.neu.md, which holds only
the items LISA added; carried text is not in that file and is not checked. Meetings without the file
are skipped. Each figure of two to four digits on a bullet or heading line is classified, in order:
  digits   the same digit string occurs in the transcript
           (outputs_v11_final/canary_diar/<meeting>/transcript_speakers.txt) or the previous protocol
  spoken   it occurs there as German number words (german_numbers.figures)
  open     neither; printed with its meeting and line, for reading
An open figure is not necessarily invented: it can be correctly derived (a sum, a date written
differently), so only reading the printed lines decides.

    python number_trace.py v10 [v9 ...]    # arms; default v10

Prints to stdout only.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
MEET = ["march09", "april13", "april27", "may11", "may18", "june01", "june15", "june22"]
FIG = re.compile(r"(?<![\w.,])(\d{2,4})(?![\w.,])")


def main(arms: list[str]) -> int:
    import german_numbers as GN
    from carry_tag import read_prev
    from term_source import PREVIOUS
    for arm in arms:
        tot = {"digits": 0, "spoken": 0, "open": 0}
        opened = []
        for mid in MEET:
            p = REPO / f"outputs_deliverable/{arm}/{mid}/protocol_pro.neu.md"
            if not p.exists():
                continue
            tx = (REPO / f"outputs_v11_final/canary_diar/{mid}/transcript_speakers.txt").read_text(
                encoding="utf-8")
            src = tx + "\n" + read_prev(REPO / PREVIOUS[mid])
            digits = set(re.findall(r"\d+", src))
            spoken = GN.figures(src)
            for line in p.read_text(encoding="utf-8").splitlines():
                if not re.match(r"^\s*([*-]|#{2,4})\s", line):
                    continue
                for m in FIG.finditer(line):
                    v = m.group(1)
                    if v in digits:
                        tot["digits"] += 1
                    elif v in spoken:
                        tot["spoken"] += 1
                    else:
                        tot["open"] += 1
                        opened.append((mid, v, line.strip()[:120]))
        n = sum(tot.values())
        print(f"\n{arm}: {n} figures in LISA's own bullets — as digits {tot['digits']}, "
              f"as spoken words {tot['spoken']}, OPEN {tot['open']}")
        for mid, v, line in opened:
            print(f"   {mid:8s} {v:>5s}  {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or ["v10"]))
