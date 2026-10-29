"""Checks whether example text of the generation prompt reaches the generated protocols.

The examples of prompt rules 12, 14, 15, 15b and 2c (in STYLE_RULE, HEADING_RULE and PERSON_RULES) are
taken from this corpus: from reference minutes, transcripts and earlier system outputs. EXAMPLES holds
distinctive fragments of them, so that a match means the example and not only the topic. For every
fragment the script reports (1) which reference protocols (Meeting* files, .md preferred over .docx)
contain it, (2) in which meetings it occurs in the Canary-1B-v2 transcript (outputs_v11_final), and
(3) which bullets written by the language model (protocol_pro.neu.md, the model's output before the merge)
contain it, per arm and meeting. A copy counts as a leak when the meeting's own reference
(evaluate_protocols.REF) contains the fragment, the only case in which a copied example can raise a score.

    python prompt_leakage_check.py [arm ...]        (default: v10 v10flash v10pro_rep)    -> stdout only
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

MEET = ["march09", "april13", "april27", "may11", "may18", "june01", "june15", "june22"]

# (rule, example text). Distinctive fragments only, so a match means the example, not the topic.
EXAMPLES = [
    ("12", "Personalratssitzung"), ("12", "Schweinfurt"), ("12", "zweieinhalb Tage"),
    ("12", "2,5 Tage"),
    ("14", "Bernini"), ("14", "Bike Dance"), ("14", "externen Anbieters"),
    ("14", "20 Minuten für 5-7 Sekunden"), ("14", "RTX-4090"), ("14", "A100"),
    ("14", "Handelns"), ("14", "Flugs"),
    ("15", "Zahlreiche Modelle im Resilience Lab"), ("15", "Resilience Lab eingebunden"),
    ("15", "Fragebögen für das Ministerium gesichtet"), ("15", "Ministerium gesichtet"),
    ("15", "sieben bis acht Minuten"), ("15", "nicht praktikabel"),
    ("15b", "Datenbasis erweitert auf zusätzliche 50 Produkte"), ("15b", "50 Produkte"),
    ("15b", "Sicherheitsupdates eingespielt"), ("15b", "Port 0 statt 9000"),
    ("15b", "trotz Urlaub was zuliefern"), ("15b", "kann es in Brünn"), ("15b", "Brünn"),
    ("15b", "Workarounds für Produkt-PDFs"), ("15b", "Snippet-Erzeugung"),
    ("15b", "Einarbeitung der Reviews"), ("15b", "Vergleichskandidaten"),
    ("15b", "Sub-Words"),
    ("2c", "Strategie, Forschung und Organisation"), ("2c", "Neuroforge"),
    ("2c", "Munir aus Ingolstadt"), ("2c", "Frank aus Kempten"), ("2c", "IT-Forum"),
    ("2c", "Antrag Ministerium"), ("2c", "SARA Deployment"), ("2c", "PersonaPlex"),
    ("2c", "DNS-Eintrag"), ("2c", "Zertifikatsmanagement"), ("2c", "Sonstiges und Wochenplanung"),
    ("2c", "Weitere Tätigkeiten und Tools"),
]


def his_protocols() -> dict[str, str]:
    """Every reference protocol (Meeting* .md or .docx) as text, keyed by date; the .md is
    preferred when both exist."""
    from carry_tag import read_prev
    out = {}
    for p in sorted(REPO.glob("Meeting*")):
        m = re.search(r"(\d{4}-\d{2}-\d{2})", p.name)
        if not m or p.suffix not in (".md", ".docx"):
            continue
        if m.group(1) in out and p.suffix == ".docx":
            continue
        out[m.group(1)] = read_prev(p)
    return out


def main() -> int:
    from evaluate_protocols import REF
    arms = sys.argv[1:] or ["v10", "v10flash", "v10pro_rep"]
    his = his_protocols()
    ref_date = {mid: re.search(r"(\d{4}-\d{2}-\d{2})", REF[mid]).group(1) for mid in REF}
    transcripts = {mid: (REPO / f"outputs_v11_final/canary_diar/{mid}/transcript_speakers.txt")
                   .read_text(encoding="utf-8") for mid in MEET}
    print(f"reference protocols read: {len(his)}  ({', '.join(sorted(his))})")
    print("=" * 100)
    leaks = []
    for rule, ex in EXAMPLES:
        low = ex.lower()
        in_his = [d for d, t in his.items() if low in t.lower()]
        in_tx = [m for m in MEET if low in transcripts[m].lower()]
        row = []
        for arm in arms:
            for mid in MEET:
                p = REPO / f"outputs_deliverable/{arm}/{mid}/protocol_pro.neu.md"
                if not p.exists():
                    continue
                n = p.read_text(encoding="utf-8").lower().count(low)
                if n:
                    scored = mid in ref_date and ref_date[mid] in in_his
                    said = mid in in_tx
                    row.append(f"{arm}/{mid}x{n}{'*REF' if scored else ''}{'' if said else '!NOT-SAID'}")
                    if scored:
                        leaks.append((rule, ex, arm, mid, said))
        print(f"  rule {rule:4s} {ex[:44]:44s} ref:{','.join(d[5:] for d in in_his) or '-':18s}"
              f" said in:{','.join(in_tx) or '-':16s} LISA wrote: {' '.join(row) or '-'}")
    print("\n  *REF      = the meeting's own reference contains the example (a copy can raise its score)")
    print("  !NOT-SAID = the example text does not occur in that meeting's transcript")
    print(f"\n  copies that land in a meeting whose reference contains them: {len(leaks)}")
    for rule, ex, arm, mid, said in leaks:
        print(f"    rule {rule}: '{ex}' in {arm}/{mid} ({'also said in the meeting' if said else 'NOT said in the meeting'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
