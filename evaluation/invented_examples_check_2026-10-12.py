# -*- coding: utf-8 -*-
"""Checks whether the invented example phrases of prompt rules 12, 14, 15, 15b and 2c (generate_protocols.py) occur
in the 13 reference protocols or the 8 transcripts.

Function words occur in every protocol, so the check has two levels: (a) every quoted example phrase, verbatim and
case-insensitive, and (b) every word of 6 or more characters in those phrases. The reference protocols are loaded by
prompt_leakage_check; the transcripts are outputs_v11_final/canary_diar/<meeting>/transcript_speakers.txt.

    python invented_examples_check_2026-10-12.py   -> goldset/invented_examples_check_2026-10-12.json"""
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.stdout.reconfigure(encoding="utf-8")
import prompt_leakage_check as PL  # noqa: E402

MEET = ["march09", "april13", "april27", "may11", "may18", "june01", "june15", "june22"]
PHRASES = [
    "zweieinhalb Tage auf der Lieferantenmesse in Deggendorf",
    "Bildgenerator Lumora von Brite Lens getestet", "Bildgenerator eines externen Anbieters getestet",
    "4x L40S, 12 Minuten für 3 Sekunden Video", "Kleineres 12B-Modell statt größerem 72B A100 einsetzen",
    "Kleineres 12B-Modell statt des größeren 72B-Modells einsetzen", "Läuft vorerst Wandels und wird mit Kreisen gefahren",
    "Ganz schön viele Sensoren im Messlabor mit drin", "Zahlreiche Sensoren im Messlabor eingebunden",
    "Angebote fürs Ersatzteillager durchgeschaut", "Angebote für das Ersatzteillager gesichtet",
    "Mit Wasserzeichen angeschaltet: vier bis fünf Minuten Ladezeit, völlig indiskutabel",
    "Mit aktiviertem Wasserzeichen vier bis fünf Minuten Ladezeit, nicht praktikabel",
    "Implementierung von Filtern für Lieferantendaten", "Überprüfung der Exportfunktion", "Einarbeitung der Prüfregeln",
    "Auswahl der Testgeräte", "Ableitung neuer Schwellwerte",
    "Hat Messreihen erweitert auf zusätzliche 30 Geräte (je Labor A und B)", "Updates auf dem Druckserver eingespielt",
    "Backup lief auf Laufwerk D statt E => korrigiert", "Will diese Woche trotz Dienstreise was nachliefern.",
    "Positive Resonanz, darf es im November auf der Tagung zeigen.", "Fuhrpark, Wartung und Organisation",
    "Projekte Nordlicht, Kranich und Messtechnik", "Antrag mit Lea aus Passau und Timo aus Cham", "Hausmesse",
    "Antrag Landkreis", "Druckerumzug", "Landkreis-Antrag und Kranich", "Netzlaufwerk und Druckerwartung",
    "Sonstiges und Wochenplanung", "Weitere Tätigkeiten und Tools",
]

his = PL.his_protocols()
tx = {m: (REPO / f"outputs_v11_final/canary_diar/{m}/transcript_speakers.txt").read_text(encoding="utf-8") for m in MEET}
corpus = {**{f"protocol {k}": v for k, v in his.items()}, **{f"transcript {k}": v for k, v in tx.items()}}
low = {k: v.lower() for k, v in corpus.items()}
words_in = {k: set(re.findall(r"[a-zäöüß0-9-]+", v)) for k, v in low.items()}

phr = {p: [k for k, v in low.items() if p.lower() in v] for p in PHRASES}
content = sorted({w for p in PHRASES for w in re.findall(r"[a-zäöüß0-9-]+", p.lower()) if len(w) >= 6})
cw = {w: [k for k, s in words_in.items() if w in s] for w in content}
out = {"protocols": len(his), "transcripts": len(tx), "phrases": len(PHRASES),
       "phrases_found": {p: k for p, k in phr.items() if k},
       "content_words": len(content), "content_words_found": {w: k for w, k in cw.items() if k}}
print(json.dumps({k: (v if not isinstance(v, dict) else {a: len(b) for a, b in v.items()}) for k, v in out.items()},
                 ensure_ascii=False, indent=1))
(REPO / "goldset" / "invented_examples_check_2026-10-12.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                                                          encoding="utf-8")
