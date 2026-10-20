"""Prompt rules and the LISA client used by protokoll.py, plus a batch protocol generator.

As a script, it reads <tree>/<model>/<meeting>/transcript_speakers.txt (or transcript.txt) in
this folder, where <tree> is pipeline_outputs_extracted/outputs, outputs_v10 (--v10) or
outputs_v11_final (--v11). It writes protocol_<flash|pro|mini><suffix>.md next to each
transcript and skips outputs over 80 bytes that already exist.

Usage:
    python generate_protocols.py [--v10 | --v11] [--carry | --carry-tracked] [<meeting>] [<model>]
    python generate_protocols.py [--v10 | --v11] --fixed [--structure team|senat] [--markers]
                                 [--carry] [<meeting>] [<model>]
Further options: --with-backlog, --lisa flash|pro|mini, --mini, --suffix TEXT.
<meeting> is an id such as may11; <model> is a recognizer folder such as cohere.
"""

import os
import re
import sys
import json
import time
from pathlib import Path

from thesis_pipeline.env import load_env

def lisa_env() -> tuple[str, str]:
    """Return the LISA base URL and API key, loading .env first."""
    load_env()
    return os.environ["LISA_BASE_URL"].rstrip("/"), os.environ["LISA_API_KEY"]
LISA_MODELS = {"flash": "lisa-flash-03-2026", "pro": "lisa-pro-03-2026"}
from thesis_pipeline.protocol_validate import TRUNCATION_MARK                # noqa: E402

# LISA Mini is used only with --mini (after probe_mini() confirms that it answers) or --lisa mini.
LISA_MINI = {"mini": "mlx-community/qwen3.6-35b-a3b"}


def probe_mini(timeout: int = 120) -> str:
    """Return an empty string if LISA Mini answers, otherwise the error text."""
    try:
        r = lisa(LISA_MINI["mini"], "Antworte auf Deutsch.", "Sage genau: TEST OK",
                 timeout=timeout, retries=0)
        return "" if r.strip() else "empty response"
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {str(e)[:80]}"

REPO = Path(__file__).resolve().parent
# --with-backlog keeps the Backlog part of previous protocols and drops the rule against a
# Backlog section.
KEEP_BACKLOG = "--with-backlog" in sys.argv
if KEEP_BACKLOG:
    sys.argv.remove("--with-backlog")
NO_BACKLOG_RULE = "" if KEEP_BACKLOG else " Lege KEINEN Abschnitt oder Punkt 'Backlog' an."
USE_V10 = "--v10" in sys.argv
USE_V11 = "--v11" in sys.argv
if USE_V11:
    sys.argv.remove("--v11")
    OUTPUTS = REPO / "outputs_v11_final"
    MODELS = ["canary_diar"]
elif USE_V10:
    sys.argv.remove("--v10")
    OUTPUTS = REPO / "outputs_v10"
    # Folder names as written by recognizers/assemble_v9_transcripts.py.
    MODELS = ["canary_diar", "cohere", "granite41",
              "whisper_large_v3", "whisper_turbo", "whisper_turbo_de"]
else:
    OUTPUTS = REPO / "pipeline_outputs_extracted" / "outputs"
    MODELS = ["whisper", "whisper_turbo", "whisper_turbo_de", "granite41", "cohere"]
MEETINGS = ["march09", "april13", "april27", "may11", "may18", "june01", "june15", "june22"]
DATE = {"march09": "2026-03-09", "april13": "2026-04-13", "april27": "2026-04-27",
        "may11": "2026-05-11", "may18": "2026-05-18", "june01": "2026-06-01",
        "june15": "2026-06-15", "june22": "2026-06-22"}

# Carry-over (--carry): previous week's protocol per meeting, relative to this folder.
# april13 has no entry: its previous protocol exists only as a PDF.
PREV_PROTOCOL = {
    "march09": "Meeting-Protocol-FG-MM-AI_2026-03-02.docx",
    "april27": "Meeting_2026-04-20.md",
    "may11":   "Meeting_2026-05-04.md",
    "may18":   "Meeting_2026-05-11.md",
    "june01":  "Meeting_2026-05-18.md",
    "june15":  "Meeting_2026-06-08(1).md",
    "june22":  "Meeting_2026-06-15.md",
}

# --carry-tracked: the model may carry over only the verified items it is given.
CARRY_TRACKED_RULES = (
    "\n\nDU BEKOMMST EINE GEPRÜFTE LISTE OFFENER PUNKTE AUS DER VORWOCHE.\n"
    "5. Diese Punkte wurden bereits gegen das aktuelle Transkript geprüft und sind "
    "nachweislich weiterhin offen. Übernimm GENAU DIESE Punkte in das Protokoll, jeweils "
    "beim zuständigen Teilnehmer.\n"
    "6. JEDER übernommene Punkt ist ein STICHPUNKT auf der dritten Ebene, also eine Zeile "
    "der Form '* [ÜBERTRAG] <Text>', eingeordnet unter ein passendes '### <Thema>' des "
    "jeweiligen Teilnehmers. Ein Übertrag ist NIEMALS eine '###'-Themenüberschrift und "
    "niemals eine '##'-Überschrift. Lege für Überträge keine eigenen Themen an, sondern "
    "ordne sie unter die vorhandenen Themen ein.\n"
    "7. Markiere KEINE anderen Punkte mit '[ÜBERTRAG]'. Wenn die Liste leer ist, kommt im "
    "gesamten Protokoll kein '[ÜBERTRAG]' vor.\n"
    "8. Übernimm NUR den Text des Punktes. Die Felder 'Punkt N', 'Teilnehmer:' und 'Thema:' "
    "sind Angaben für dich und dürfen im Protokoll NICHT erscheinen — der Teilnehmer wird "
    "durch die '##'-Überschrift ausgedrückt, unter der du den Punkt einordnest."
)

# Maximum number of verified items put into the prompt; a longer list crowds out the meeting
# and can exhaust the output budget.
MAX_CARRIED = 15

# Carry-over rules for --carry, numbered 5 and 6 after the four rules of SYSTEM_PLAIN and
# SYSTEM_DIARIZED.
CARRY_RULES = (
    "\n\nDU BEKOMMST ZUSÄTZLICH DAS PROTOKOLL DER VORWOCHE.\n"
    "5. Gehe die offenen Punkte der Vorwoche durch und entscheide JEDEN einzeln anhand "
    "des aktuellen Transkripts:\n"
    "   - im Transkript als erledigt erkennbar -> NICHT übernehmen;\n"
    "   - im Transkript als weiterhin offen erkennbar oder erneut besprochen -> übernehmen "
    "und mit '[ÜBERTRAG]' markieren;\n"
    "   - im Transkript gar nicht erwähnt -> nur übernehmen, wenn es sich klar um eine "
    "laufende Aufgabe handelt, ebenfalls mit '[ÜBERTRAG]'.\n"
    "6. Übernimm KEINE Inhalte der Vorwoche, die im aktuellen Transkript weder erwähnt noch "
    "als laufend erkennbar sind. Das Protokoll der Vorwoche ist Kontext, keine Vorlage zum "
    "Abschreiben." + NO_BACKLOG_RULE
)

# Variant without the [ÜBERTRAG] marker (used by protokoll.py and by --fixed --carry without
# --markers); rules 17 to 19 follow LENGTH_RULE (rule 16).
CARRY_RULES_NOMARK = (
    "\n\nDU BEKOMMST ZUSÄTZLICH DAS PROTOKOLL DER VORWOCHE.\n"
    "17. Gehe die offenen Punkte der Vorwoche durch und entscheide JEDEN einzeln anhand des "
    "aktuellen Transkripts:\n"
    "   - im Transkript als erledigt erkennbar -> NICHT übernehmen;\n"
    "   - im Transkript als weiterhin offen erkennbar oder erneut besprochen -> übernehmen;\n"
    "   - im Transkript gar nicht erwähnt -> nur übernehmen, wenn es sich klar um eine "
    "laufende Aufgabe handelt.\n"
    "18. Übernommene Punkte werden als ganz normale Stichpunkte geschrieben, ohne Marker und "
    "ohne eckige Klammern, an der inhaltlich passenden Stelle unter der zuständigen Person.\n"
    "19. Übernimm KEINE Inhalte der Vorwoche, die im aktuellen Transkript weder erwähnt noch "
    "als laufend erkennbar sind. Das Protokoll der Vorwoche ist Kontext, keine Vorlage zum "
    "Abschreiben." + (NO_BACKLOG_RULE and NO_BACKLOG_RULE[:-1] + "; was dort stand und erneut "
                      "besprochen wird, gehört unter das passende Thema der zuständigen Person.")
)

BASE_TEAM = ["René Peinl", "Vincent Tischler", "Thomas Weber", "Johannes"]
ROSTER = {
    "march09": BASE_TEAM, "april13": BASE_TEAM, "april27": BASE_TEAM,
    "may11": BASE_TEAM + ["Osman"],
    "may18": BASE_TEAM + ["Osman"],
    "june01": ["René Peinl", "Thomas Weber", "Johannes", "Osman"],
    "june15": ["René Peinl", "Vincent Tischler", "Johannes", "Osman"],
    "june22": ["René Peinl", "Vincent Tischler", "Thomas Weber", "Osman"],
}

SYSTEM_PLAIN = (
    "Du bist ein Assistent, der aus einem deutschen Besprechungstranskript ein "
    "strukturiertes Protokoll erstellt.\n\n"
    "Aufgabe:\n"
    "1. Ordne die Redebeiträge den Teilnehmern aus der Teilnehmerliste zu (anhand von "
    "Anreden, Rollen und Themen). Verwende ausschließlich Namen aus der Teilnehmerliste. "
    "Wenn ein Beitrag keinem Teilnehmer eindeutig zuzuordnen ist, nutze 'Sprecher X' — "
    "erfinde keine Namen.\n"
    "2. Struktur GENAU so:\n"
    "   ## <Name des Teilnehmers>\n"
    "   ### <Thema>\n"
    "   * <Stichpunkt: Gesagtes / Beschluss / Aufgabe>\n"
    "   Beginne die Ausgabe SOFORT mit der ersten '##'-Überschrift — keine Einleitung, kein "
    "Titel, keine Wiederholung des Besprechungsdatums davor. Auf Ebene 1 ('##') stehen "
    "AUSSCHLIESSLICH Teilnehmernamen, keine weiteren Überschriften.\n"
    "3. Verwende ausschließlich Inhalte aus dem Transkript. Erfinde nichts.\n"
    "4. Markiere offene Aufgaben mit '[Aufgabe]', Übertragspunkte mit '[ÜBERTRAG]'. Diese "
    "Marker dürfen KEINEN zusätzlichen Text in den eckigen Klammern enthalten — schreibe "
    "genau '[Aufgabe]' bzw. '[ÜBERTRAG]' und den erklärenden Text danach, nicht darin."
)

# Team prompt for transcripts with resolved speaker names (--fixed and protokoll.py).
# Prompt examples must not be taken from the evaluated meetings.
SYSTEM_IDENTIFIED = (
    "Du bist ein Assistent, der aus einem deutschen Besprechungstranskript ein "
    "strukturiertes Protokoll erstellt.\n\n"
    "Das Transkript ist bereits nach Sprechern unterteilt und JEDER SPRECHER IST BEREITS "
    "NAMENTLICH KORREKT BEZEICHNET. Die Namen stammen aus einer Stimmanalyse, nicht aus dem "
    "Inhalt.\n\n"
    "Aufgabe:\n"
    "1. Übernimm die Sprecherzuordnung UNVERÄNDERT. Ordne nichts neu zu, auch wenn dir ein "
    "Thema zu einer anderen Person zu passen scheint. Erfinde keine Namen und führe keine "
    "Person ein, die im Transkript nicht als Sprecher auftritt. Ein Beitrag gehört zu der "
    "Person, deren Name vor ihm steht.\n"
    "2. Struktur GENAU so:\n"
    "   ## <Name des Teilnehmers>\n"
    "   ### <Thema>\n"
    "   * <Kernaussage zum Thema>\n"
    "     * <Detail, das die Kernaussage darüber erläutert>\n"
    "       * <noch feineres Detail, falls nötig>\n"
    "   Beginne die Ausgabe SOFORT mit der ersten '##'-Überschrift — keine Einleitung, kein "
    "Titel, keine Wiederholung des Besprechungsdatums davor. Auf Ebene 1 ('##') stehen "
    "AUSSCHLIESSLICH Teilnehmernamen, keine weiteren Überschriften. Jede Person, die im "
    "Transkript spricht, bekommt genau einen '##'-Abschnitt.\n"
    "2b. VERSCHACHTELE DIE STICHPUNKTE. Stichpunkte stehen NICHT alle auf derselben Ebene. "
    "Ein Stichpunkt, der einen anderen erläutert, begründet, aufzählt oder ein Beispiel dazu "
    "gibt, wird mit ZWEI zusätzlichen Leerzeichen eingerückt und gehört unter den Stichpunkt, "
    "auf den er sich bezieht. Eine neue Aussage beginnt wieder auf der äußeren Ebene. "
    "Nutze die Einrückung so tief wie der Inhalt es verlangt, üblicherweise zwei bis vier "
    "Ebenen. Beispiel:\n"
    "   * Neues Messgerät im Labor in Betrieb genommen\n"
    "     * Kalibrierung schlug beim ersten Versuch fehl\n"
    "     * Zweiter Versuch mit neuer Firmware erfolgreich\n"
    "       * Firmware-Version muss noch dokumentiert werden\n"
    "   * Lieferung des Zubehörs verzögert sich weiterhin\n"
    "   Eine flache Liste, in der jeder Stichpunkt auf derselben Ebene steht, ist FALSCH.\n"
    "3. Verwende ausschließlich Inhalte aus dem Transkript. Erfinde nichts. Zeitstempel und "
    "Sprechernamen in den Zeilen selbst gehören NICHT ins Protokoll, nur der Inhalt."
)

# Task markers: MARKERS_ON asks for [Aufgabe] and [ÜBERTRAG]; MARKERS_OFF, the default of
# --fixed, writes tasks as plain bullets.
MARKERS_ON = (
    "\n4. Markiere offene Aufgaben mit '[Aufgabe]', Übertragspunkte mit '[ÜBERTRAG]'. Diese "
    "Marker dürfen KEINEN zusätzlichen Text in den eckigen Klammern enthalten — schreibe "
    "genau '[Aufgabe]' bzw. '[ÜBERTRAG]' und den erklärenden Text danach, nicht darin."
)

MARKERS_OFF = (
    "\n4. Offene Aufgaben und Zuständigkeiten werden als ganz normale Stichpunkte geschrieben, "
    "mit der verantwortlichen Person im Satz: 'Erika Mustermann fragt beim Lieferanten wegen "
    "der Ersatzteile nach'. "
    "Verwende KEINE Marker und KEINE eckigen Klammern wie '[Aufgabe]' oder '[ÜBERTRAG]' — "
    "im Protokoll stehen keine eckigen Klammern."
)

# German-only rule; lisa() also asks again when lang_ok() rejects an answer.
GERMAN_ONLY = (
    "\n\nSPRACHE: Schreibe das gesamte Protokoll AUSSCHLIESSLICH auf Deutsch. Verwende keine "
    "chinesischen, kyrillischen, japanischen oder anderen nicht-lateinischen Schriftzeichen. "
    "Englische Fachbegriffe und Produktnamen sind erlaubt, ganze Sätze oder Satzteile in einer "
    "anderen Sprache als Deutsch sind es nicht."
)

# Non-Latin script: Cyrillic, Greek, CJK, Kana, Arabic.
_NONLATIN = re.compile(r"[Ѐ-ӿͰ-Ͽ一-鿿぀-ヿ؀-ۿ]")


def english_lean(text: str) -> float:
    """Share of English minus German function words in the text; -1.0 below 20 words."""
    from language_gate import EN, DE, WORD
    w = [x.lower() for x in WORD.findall(text)]
    if len(w) < 20:
        return -1.0  # too short to judge
    return (sum(x in EN for x in w) - sum(x in DE for x in w)) / len(w)


def lang_ok(text: str) -> bool:
    """True if the text has no non-Latin script and no more English than German function words."""
    return not _NONLATIN.search(text) and english_lean(text) <= 0.0


GLOSSARY_RULES = (
    "\n\n5. EIGENNAMEN. Du bekommst eine Liste der korrekten Schreibweisen von Modellen, "
    "Projekten, Partnern und Personen. Diese Liste wurde AUTOMATISCH aus dem Protokoll der "
    "Vorwoche gewonnen. Wenn im Transkript erkennbar eine verhörte Variante eines dieser Namen "
    "steht, schreibe im Protokoll die korrekte Form aus der Liste. "
    "Füge KEINE Namen aus der Liste ein, die im Transkript nicht vorkommen, und ersetze keine "
    "gewöhnlichen deutschen Wörter durch Namen aus der Liste."

)

# Rules 7 to 15: names and content.
PERSON_RULES = (
    "\n\n7. PERSONENNAMEN. Du bekommst eine Liste vollständiger Personennamen aus dem "
    "Vorwochenprotokoll. Die Spracherkennung verstümmelt Nachnamen häufig, während der Vorname "
    "erhalten bleibt. Wenn im Transkript ein Vorname aus dieser Liste mit einem abweichenden "
    "Nachnamen steht und der Kontext dieselbe Person meint, verwende den vollständigen Namen aus "
    "der Liste. Passt der Kontext nicht, lass den Namen unverändert."
    "\n8. UNKLARE NAMEN NICHT GLÄTTEN. Wenn ein Eigenname im Transkript unverständlich ist, "
    "übernimm ihn unverändert oder lass die Angabe weg. Erfinde NIEMALS eine Rolle, eine "
    "Funktion oder eine Umschreibung an seiner Stelle (also nicht 'beim Stellvertreter', wenn "
    "im Transkript ein unverständlicher Nachname steht). Ein stehengelassener Fehler ist "
    "erkennbar, ein geglätteter nicht."
    "\n9. KEINE DOPPELUNG. Jeder Inhalt gehört zu genau einer Person. Wiederhole einen Beitrag "
    "nicht unter einer zweiten Person, auch wenn sie daran beteiligt war."
    "\n10. KEINE ZUSÄTZLICHE GENAUIGKEIT. Übernimm Bezeichnungen, Versionsnummern und Titel "
    "genau so, wie sie im Transkript stehen. Ergänze keine Versionsnummern und keine Details "
    "aus deinem Weltwissen."
    "\n11. NUR DEUTSCH IM FLIESSTEXT. Einzelne englische Fachbegriffe sind erlaubt, aber keine "
    "englischen Funktionswörter mitten im deutschen Satz."
    "\n12. KONKRETE DETAILS ERHALTEN. Das Protokoll ist KEINE Kurzfassung. Übernimm jede "
    "konkrete Angabe, die im Transkript vorkommt: Zahlen, Mengen, Zeitangaben, Fristen, "
    "Versionen, Dateinamen, URLs, Orte und Namen. Schreibe 'zweieinhalb Tage auf der "
    "Lieferantenmesse in Deggendorf', nicht 'Bericht von einer Messe'. Ein Stichpunkt "
    "ohne konkrete Angabe ist meistens ein verlorener Stichpunkt.\n"
    "13. VOLLSTÄNDIGKEIT. Gehe den Beitrag jeder Person vollständig durch und lass kein Thema "
    "aus, das sie angesprochen hat — auch kurze Nebenpunkte, Zwischenstände und offene Fragen. "
    "Lieber ein Stichpunkt zu viel als ein fehlender."
    "\n14. KEIN STICHPUNKT, DER KEINEN SINN ERGIBT. Das Transkript ist eine automatische "
    "Spracherkennung und enthält verhörte Fachbegriffe, Produktnamen und Abkürzungen. "
    "Regel 12 verlangt konkrete Angaben — sie verlangt NICHT, einen Erkennungsfehler als "
    "sichere Tatsache ins Protokoll zu schreiben. Prüfe jeden Stichpunkt vor dem Schreiben: "
    "Ergibt er fachlich Sinn? Wenn ein Begriff im Kontext nicht zusammenpasst, schreibe die "
    "Angabe auf der Ebene, die du belegen kannst, statt eine falsche Genauigkeit zu behaupten. "
    "Kürze NICHT den Inhalt — nur die unsichere Bezeichnung. "
    "Beispiele: 'Bildgenerator Lumora von Brite Lens getestet (4x L40S, 12 Minuten für "
    "3 Sekunden Video)' -> 'Bildgenerator eines externen Anbieters getestet (4x L40S, 12 Minuten "
    "für 3 Sekunden Video)' — die Zahlen bleiben, der verhörte Firmenname verschwindet. "
    "'Kleineres 12B-Modell statt größerem 72B A100 einsetzen' -> 'Kleineres 12B-Modell statt des "
    "größeren 72B-Modells einsetzen' — eine Bezeichnung, die nicht in den Satz passt, wird "
    "weggelassen und NICHT durch eine bekannte Bezeichnung ersetzt. "
    "Wenn ein Satzteil komplett unverständlich ist ('Läuft vorerst Wandels und wird mit Kreisen "
    "gefahren'), lass diesen Satzteil weg und schreibe nur den verständlichen Rest. "
    "Erfinde niemals ein Modell, eine Version oder eine Firma, die es nicht gibt."
    "\n15. PROTOKOLLDEUTSCH, NICHT GESPROCHENE SPRACHE. Das Transkript ist gesprochene "
    "Sprache, das Protokoll ist geschriebene Sprache. Übernimm den INHALT, nicht die "
    "Formulierung. Der Fakt bleibt vollständig erhalten (Regel 12 gilt weiter), nur die "
    "Ausdrucksweise wird schriftsprachlich. Keine Füllwörter und keine Umgangssprache: "
    "halt, eben, irgendwie, quasi, sozusagen, eigentlich, ganz schön, ein bisschen, "
    "mit drin, da drinnen, dran, drauf, hin und her, gucken, kriegen, rumspielen. "
    "Benutze präzise Verben: 'aktiviert' statt 'angeschaltet', 'gesichtet' oder "
    "'geprüft' statt 'durchgeschaut', 'erhalten' statt 'gekriegt'. "
    "Jeder Stichpunkt ist eine vollständige, grammatisch korrekte deutsche Nominal- oder "
    "Verbalphrase. Beispiele: "
    "'Ganz schön viele Sensoren im Messlabor mit drin' -> 'Zahlreiche Sensoren im "
    "Messlabor eingebunden'; "
    "'Angebote fürs Ersatzteillager durchgeschaut' -> 'Angebote für das Ersatzteillager "
    "gesichtet'; "
    "'Mit Wasserzeichen angeschaltet: vier bis fünf Minuten Ladezeit, völlig indiskutabel' -> "
    "'Mit aktiviertem Wasserzeichen vier bis fünf Minuten Ladezeit, nicht praktikabel'. "
    "Beachte: die Zahlen und Namen bleiben in allen drei Beispielen vollständig erhalten."
)

SUGGEST_RULES = (
    "\n\n6. MÖGLICHE VERHÖRER. Du bekommst zusätzlich eine Liste von Wortpaaren der Form "
    "'im Transkript steht X - gemeint sein könnte Y'. Diese Paare klingen nur ähnlich; sie sind "
    "NICHT geprüft. Ersetze X durch Y AUSSCHLIESSLICH dann, wenn der Satz, in dem X steht, "
    "eindeutig von Y handelt. Im Zweifel lass X unverändert stehen. Ein gewöhnliches deutsches "
    "Wort wird NIEMALS ersetzt, auch wenn es in der Liste steht."
)

# Senat structure (--structure senat): agenda points instead of persons, with the speaker's
# name at the start of a bullet.
SYSTEM_SENAT = (
    "Du bist ein Assistent, der aus einem deutschen Sitzungstranskript ein Protokoll nach der "
    "Struktur einer Senatssitzung erstellt.\n\n"
    "Das Transkript ist nach Sprechern unterteilt und jeder Sprecher ist bereits namentlich "
    "korrekt bezeichnet. Die Namen stammen aus einer Stimmanalyse.\n\n"
    "Aufgabe:\n"
    "1. Gliedere das Protokoll nach TAGESORDNUNGSPUNKTEN, nicht nach Personen. Struktur GENAU so:\n"
    "   ## TOP <Nummer>: <Titel des Tagesordnungspunkts>\n"
    "   * <Stichpunkt>\n"
    "   Beginne die Ausgabe SOFORT mit der ersten '##'-Überschrift.\n"
    "2. Beginne einen Stichpunkt mit dem Namen der Person, die den Beitrag geleistet hat, wenn "
    "sich der Beitrag einer Person zuordnen lässt: '* <Name>: <Inhalt>'. Verwende dafür "
    "ausschliesslich die Namen aus dem Transkript; ordne nichts neu zu.\n"
    "3. Ein Tagesordnungspunkt kann drei Zeilen oder eine ganze Seite umfassen — richte die Länge "
    "nach dem Inhalt, nicht nach einem festen Schema.\n"
    "4. Verwende ausschliesslich Inhalte aus dem Transkript. Erfinde nichts. Zeitstempel und "
    "Sprechernamen der Transkriptzeilen gehören nicht als solche ins Protokoll.\n"
    "5. Verschachtele die Stichpunkte. Der Vortrag der zuständigen Person steht auf der äusseren "
    "Ebene; Rückfragen, Antworten und Wortmeldungen dazu werden mit zwei zusätzlichen "
    "Leerzeichen darunter eingerückt. Das Beispiel zeigt NUR die Form; Namen und Inhalt sind "
    "erfunden und dürfen nicht übernommen werden. Beispiel:\n"
    "   * Erika Mustermann: stellt das neue Messgerät im Labor vor\n"
    "     * Max Mustermann: fragt nach den Betriebskosten\n"
    "       * Erika Mustermann: Betriebskosten sind im Haushalt eingeplant\n"
)

SYSTEM_DIARIZED = (
    "Du bist ein Assistent, der aus einem deutschen Besprechungstranskript ein "
    "strukturiertes Protokoll erstellt.\n\n"
    "Das Transkript ist mit Sprecher-Labels (SPEAKER_00, SPEAKER_01, ...) und Zeitstempeln "
    "versehen. Dasselbe Label steht während der GESAMTEN Besprechung für dieselbe Person, "
    "auch wenn ihre Redebeiträge über das ganze Transkript verteilt sind.\n\n"
    "Aufgabe:\n"
    "1. Ordne JEDES Sprecher-Label GENAU EINEM Namen aus der Teilnehmerliste zu (anhand von "
    "Anreden, Rollen und Themen) und fasse ALLE Beiträge dieses Labels unter diesem einen "
    "Namen zusammen. Verwende ausschließlich Namen aus der Teilnehmerliste. Wenn ein Label "
    "keinem Teilnehmer eindeutig zuzuordnen ist, nutze 'Sprecher X' — erfinde keine Namen. "
    "Geh am Ende jedes Sprecher-Labels aus dem Transkript einzeln durch und stelle sicher, "
    "dass JEDES Label einem Abschnitt im Protokoll zugeordnet ist — keines darf fehlen.\n"
    "2. Struktur GENAU so:\n"
    "   ## <Name des Teilnehmers>\n"
    "   ### <Thema>\n"
    "   * <Stichpunkt: Gesagtes / Beschluss / Aufgabe>\n"
    "   Beginne die Ausgabe SOFORT mit der ersten '##'-Überschrift — keine Einleitung, kein "
    "Titel, keine Wiederholung des Besprechungsdatums davor. Auf Ebene 1 ('##') stehen "
    "AUSSCHLIESSLICH Teilnehmernamen, keine weiteren Überschriften.\n"
    "3. Verwende ausschließlich Inhalte aus dem Transkript. Erfinde nichts. Zeitstempel und "
    "Sprecher-Labels selbst gehören NICHT ins Protokoll, nur der Inhalt.\n"
    "4. Markiere offene Aufgaben mit '[Aufgabe]', Übertragspunkte mit '[ÜBERTRAG]'. Diese "
    "Marker dürfen KEINEN zusätzlichen Text in den eckigen Klammern enthalten — schreibe "
    "genau '[Aufgabe]' bzw. '[ÜBERTRAG]' und den erklärenden Text danach, nicht darin."
)


# Read on each call (defaults): LISA_TIMEOUT 600 s, LISA_CONNECT_TIMEOUT 45 s, LISA_RETRIES 2,
# LISA_MAX_TOKENS 16384. The timeout and retries arguments take precedence.


def lisa(model_id: str, system: str, user: str,
         timeout: int | None = None, retries: int | None = None,
         lang_gate: bool = True) -> str:
    """Send one chat request to LISA and return the answer text (an empty string if the last try
    returns no text). Set lang_gate=False for answers that are not German prose, such as JSON.
    """
    base, key = lisa_env()
    timeout = int(os.environ.get("LISA_TIMEOUT", "600")) if timeout is None else timeout
    retries = int(os.environ.get("LISA_RETRIES", "2")) if retries is None else retries
    # Reasoning is switched off by a request parameter, not by prompt text. With reasoning
    # on, the model can use the whole output budget for it and return empty content.
    body = ({"model": model_id, "temperature": 0.2,
        "max_tokens": int(os.environ.get("LISA_MAX_TOKENS", "16384")),
        "chat_template_kwargs": {"enable_thinking": os.environ.get("LISA_THINKING") == "1"},
        "messages": [{"role": "system", "content": "You are a helpful assistant."},
                     {"role": "user", "content": f"{system}\n\n{user}"}]})
    # LISA_REASONING_EFFORT replaces chat_template_kwargs with reasoning_effort, for models
    # that put the whole answer into reasoning_content when enable_thinking is false.
    if os.environ.get("LISA_REASONING_EFFORT"):
        body.pop("chat_template_kwargs")
        body["reasoning_effort"] = os.environ["LISA_REASONING_EFFORT"]
    body = json.dumps(body).encode()
    # Network retries (`retries`) and quality re-rolls (language, truncation) have separate
    # budgets; `attempt` counts network tries only.
    lang_rerolls = 3
    trunc_rerolls = 2
    attempt = 0
    # Failed connections have their own budget and do not use up `retries`.
    import requests
    connect_tries = 6
    while True:
        last = attempt >= retries
        t_req = time.time()
        try:
            r = requests.post(base + "/chat/completions", data=body,
                              timeout=(min(int(os.environ.get("LISA_CONNECT_TIMEOUT", "45")),
                                           timeout), timeout),
                              headers={"Authorization": "Bearer " + key,
                                       "Content-Type": "application/json"})
            r.raise_for_status()
            d = r.json()
            choice = d["choices"][0]
            txt = (choice["message"]["content"] or "").strip()
            cut = choice.get("finish_reason") == "length"
            if txt and (lang_ok(txt) or not lang_gate) and not cut:
                return txt
            if txt and cut and trunc_rerolls > 0:
                trunc_rerolls -= 1
                print("[truncated, retrying] ", end="", flush=True)
                time.sleep(8)
                continue
            if txt and lang_gate and not lang_ok(txt) and lang_rerolls > 0:
                lang_rerolls -= 1
                print("[non-German output, retrying] ", end="", flush=True)
                time.sleep(8)
                continue
            if txt:
                # Out of re-rolls: return the text. A truncated answer gets TRUNCATION_MARK,
                # which the structure check in protocol_validate rejects.
                if cut:
                    txt += "\n\n" + TRUNCATION_MARK
                return txt
        except requests.exceptions.ConnectionError:  # includes ConnectTimeout and SSLError
            if connect_tries > 0:
                connect_tries -= 1
                print("[no connection, retrying] ", end="", flush=True)
                time.sleep(15)
                continue
            if last:
                raise
        except requests.exceptions.ReadTimeout:
            # urllib3 reports a hanging TLS handshake as a read timeout; one well before the
            # read limit is retried like a failed connection.
            if time.time() - t_req < 0.9 * timeout and connect_tries > 0:
                connect_tries -= 1
                print("[handshake timed out, retrying] ", end="", flush=True)
                time.sleep(15)
                continue
            if last:
                raise
        except Exception:
            if last:
                raise
        if last:
            return ""
        attempt += 1
        time.sleep(8 * attempt)


# Talk time per speaker from the transcript timestamps, for LENGTH_RULE.
_TS = re.compile(r"^\[(\d\d):(\d\d):(\d\d)\]\s+([^:]{1,40}):", re.M)


def talk_minutes(transcript: str) -> dict[str, float]:
    """Minutes of speech per speaker from the [HH:MM:SS] turn timestamps.
    Each turn lasts until the next timestamp; the last turn counts 30 s.
    """
    rows = []
    for m in _TS.finditer(transcript):
        h, mi, s, who = m.groups()
        rows.append((int(h) * 3600 + int(mi) * 60 + int(s), who.strip()))
    out: dict[str, float] = {}
    for i, (t, who) in enumerate(rows):
        nxt = rows[i + 1][0] if i + 1 < len(rows) else t + 30
        out[who] = out.get(who, 0.0) + max(0, nxt - t) / 60.0
    return out


def talk_time_block(transcript: str) -> str:
    mins = talk_minutes(transcript)
    if len(mins) < 2:
        return ""
    total = sum(mins.values()) or 1.0
    rows = "\n".join(f"  {who}: {m:.0f} Minuten ({m / total:.0%} der Redezeit)"
                     for who, m in sorted(mins.items(), key=lambda kv: -kv[1]))
    return ("=== REDEZEIT JE PERSON (aus den Zeitstempeln gemessen) ===\n"
            f"{rows}\n=== ENDE ===\n\n")


# Register rule 15b: bullets start with what was done, not with a verbal noun.
STYLE_RULE = (
    "\n15b. SCHREIBE WIE ER, NICHT WIE EIN AMT. Beginne einen Stichpunkt mit dem, was getan "
    "wurde oder was der Fall ist — nicht mit einem Verbalsubstantiv. Falsch: 'Implementierung "
    "von Filtern für Lieferantendaten', 'Überprüfung der Exportfunktion', 'Einarbeitung der "
    "Prüfregeln', 'Auswahl der Testgeräte', 'Ableitung neuer Schwellwerte'. Richtig, im Ton "
    "seiner Protokolle: 'Hat Messreihen erweitert auf zusätzliche 30 Geräte (je Labor A und B)', "
    "'Updates auf dem Druckserver eingespielt', 'Backup lief auf Laufwerk D statt E => "
    "korrigiert', 'Will diese Woche trotz Dienstreise was nachliefern.', 'Positive Resonanz, "
    "darf es im November auf der Tagung zeigen.' Ein Stichpunkt ist kurz — meistens fünf bis zehn "
    "Wörter. Der Inhalt bleibt dabei vollständig (Regel 12): kürze die Verpackung, nicht die "
    "Angaben."
)


# STYLE_RULE, HEADING_RULE and DEPTH_RULE are added only by protokoll.py (DEPTH_RULE with
# --depth); main() below does not use them.
DEPTH_RULE = (
    "\n\n2d. UNTERPUNKTE. Unter einer '###'-Überschrift steht jede Aussage als Stichpunkt ('* '). "
    "Einzelheiten, Gründe, Zahlen und nächste Schritte, die zu EINER Aussage gehören, schreibst du "
    "als eingerückte Unterpunkte (zwei Leerzeichen, dann '* ') direkt unter diese Aussage — nicht "
    "als eigenen Stichpunkt und nicht als eigene Überschrift. Eine neue '###'-Überschrift gibt es "
    "nur für einen neuen Gegenstand, nicht für die Einzelheiten eines Gegenstands. Beispiel:\n"
    "### Serverumzug\n"
    "* Umzug auf das neue System abgeschlossen\n"
    "  * Zwei Dienste laufen noch auf dem alten System\n"
    "  * Abschaltung nächste Woche geplant")


HEADING_RULE = (
    "\n2c. EIN THEMA PRO ÜBERSCHRIFT. Eine '###'-Überschrift benennt GENAU EINEN Gegenstand. "
    "Fasse nicht mehrere Gegenstände zu einer Sammelüberschrift zusammen: 'Fuhrpark, Wartung "
    "und Organisation' ist keine Überschrift, sondern drei, und 'Projekte Nordlicht, Kranich und "
    "Messtechnik' sind drei Projekte. Wenn eine Person über fünf Dinge berichtet, schreibe "
    "fünf '###'-Überschriften — auch wenn unter einer davon nur ein einziger Stichpunkt steht. "
    "Das gilt NICHT für Überschriften, in denen 'und' zu EINEM Gegenstand gehört: '### Antrag "
    "mit Lea aus Passau und Timo aus Cham' ist ein Antrag und bleibt eine Überschrift. "
    "Die Überschrift benennt die Sache möglichst kurz: 'Hausmesse', 'Antrag Landkreis', "
    "'Druckerumzug'."
    " Auch ZWEI Gegenstände sind zu viel: 'Landkreis-Antrag und Kranich' sind zwei "
    "Überschriften, nicht eine, und 'Netzlaufwerk und Druckerwartung' ebenso. Sammelnde "
    "Überschriften wie 'Sonstiges und Wochenplanung' oder 'Weitere Tätigkeiten und Tools' gibt "
    "es nicht — benenne stattdessen jede Sache einzeln."
)


LENGTH_RULE = (
    "\n16. LÄNGE NACH REDEZEIT. Du bekommst die gemessene Redezeit jeder Person. Der Umfang "
    "eines Abschnitts richtet sich danach, wie lange über das Thema gesprochen wurde: wer "
    "lange geredet hat, bekommt einen längeren Abschnitt, wer kurz geredet hat, einen "
    "kürzeren. Ein Punkt, über den fünf Minuten gesprochen wurde, sind drei bis vier Zeilen; "
    "ein Punkt über eine halbe Stunde kann eine ganze Seite sein. Das ist eine Richtgröße, "
    "keine Quote — erfinde nichts, um eine Länge zu erreichen, und lass nichts weg, um sie "
    "einzuhalten."
)


# Input builder for --fixed
TURN = re.compile(r"^(\[\d\d:\d\d:\d\d\]\s*)(SPEAKER_\d+)(:\s*)", re.M)
_IDENT_P = REPO / "goldset" / "speaker_identity.json"
IDENTITY = (json.loads(_IDENT_P.read_text(encoding="utf-8"))["mapping"]
            if _IDENT_P.exists() else {})


def fixed_input(raw: str, mid: str):
    """Resolve speaker labels to names and correct entity spellings in a transcript.
    Returns (text, glossary, suggestions, correction log, named count, unnamed count),
    where the counts are numbers of distinct speaker labels.
    """
    from entity_correct import correct, for_meeting

    apply_map, suggest, gloss = for_meeting(mid)
    text, log = correct(raw, apply_map)

    names = IDENTITY.get(mid, {})
    named, anon = set(), set()

    def sub(m):
        who = names.get(m.group(2))
        (named if who else anon).add(m.group(2))
        return m.group(1) + (who or m.group(2)) + m.group(3)

    return TURN.sub(sub, text), gloss, suggest, log, len(named), len(anon)


# A line that starts with the word Backlog, after an optional bullet or bold marker.
_BACKLOG = re.compile(r"^\s*(?:[*-]\s+)?\**\s*backlog\b", re.I)


def previous_text(p: Path) -> str:
    """Return a previous protocol (.md or .docx) as prompt text, an empty string if it is missing.
    Backlog lines and the lines nested below them are left out unless --with-backlog is given.
    """
    if not p.exists():
        return ""
    if KEEP_BACKLOG:
        if p.suffix.lower() == ".docx":
            from docx import Document
            return "\n".join(par.text for par in Document(str(p)).paragraphs if par.text.strip())
        return p.read_text(encoding="utf-8", errors="replace")
    if p.suffix.lower() == ".docx":
        from term_source import read_document
        _, rows = read_document(str(p))                  # [(level 0/1/2, text)]
    else:
        rows = [(len(r) - len(r.lstrip()), r) for r in
                p.read_text(encoding="utf-8", errors="replace").splitlines()]
    out, skip = [], None
    for depth, line in rows:
        if not line.strip():
            continue
        if skip is not None and depth > skip:
            continue
        skip = None
        if _BACKLOG.match(line.lstrip("# ")):
            skip = depth
            continue
        out.append(line)
    return "\n".join(out)


def read_prev_protocol(mid: str) -> str:
    name = PREV_PROTOCOL.get(mid)
    return previous_text(REPO / name) if name else ""


def verified_open_items(model: str, mid: str, lisa_key: str = "pro") -> list[dict] | None:
    """Return the items marked as carried in goldset/carryover_tracked.json for this meeting.
    Falls back to another model's record for the same meeting; None if there is no record.
    """
    p = REPO / "goldset" / "carryover_tracked.json"
    if not p.exists():
        return None
    tracked = json.loads(p.read_text(encoding="utf-8"))
    rec = (tracked.get(f"{model}/{mid}/{lisa_key}")
           or next((v for k, v in tracked.items()
                    if len(k.split("/")) > 1 and k.split("/")[1] == mid), None))
    if not rec or "items" not in rec:
        return None
    return [it for it in rec["items"] if it.get("carried")]


def main():
    args = sys.argv[1:]
    fixed = "--fixed" in args
    if fixed:
        args.remove("--fixed")
    structure = "team"
    if "--structure" in args:
        i = args.index("--structure")
        structure = args[i + 1]
        del args[i:i + 2]
        if structure not in ("team", "senat"):
            raise SystemExit("--structure must be 'team' or 'senat'")
    markers = "--markers" in args
    if markers:
        args.remove("--markers")
    suffix_extra = ""
    if "--suffix" in args:
        i = args.index("--suffix")
        suffix_extra = args[i + 1]
        del args[i:i + 2]
    models_used = dict(LISA_MODELS)
    if "--mini" in args:
        args.remove("--mini")
        why = probe_mini()
        if why:
            print(f"LISA Mini requested but not serving ({why}) — continuing without it.\n")
        else:
            models_used.update(LISA_MINI)
    # --lisa limits the run to one LISA model, for example when another one is unavailable.
    if "--lisa" in args:
        i = args.index("--lisa")
        want = args[i + 1]
        del args[i:i + 2]
        pool = {**LISA_MODELS, **LISA_MINI}
        if want not in pool:
            raise SystemExit(f"--lisa must be one of {', '.join(pool)}")
        models_used = {want: pool[want]}
    carry_tracked = "--carry-tracked" in args
    if carry_tracked:
        args.remove("--carry-tracked")
    carry = "--carry" in args
    if carry:
        args.remove("--carry")
    only_meeting = next((a for a in args if a in MEETINGS), None)
    only_model = next((a for a in args if a in MODELS), None)
    base, _ = lisa_env()
    print(f"LISA {base}  | tree: {OUTPUTS.name}  | "
          f"models: {only_model or f'all {len(MODELS)}'}  | meetings: {only_meeting or 'all 8'}\n")
    for model in MODELS:
        if only_model and model != only_model:
            continue
        for mid in MEETINGS:
            if only_meeting and mid != only_meeting:
                continue
            tpath = OUTPUTS / model / mid / "transcript_speakers.txt"
            used_speakers = True
            if not tpath.exists() or tpath.stat().st_size == 0:
                tpath = OUTPUTS / model / mid / "transcript.txt"
                used_speakers = False
            if not tpath.exists() or tpath.stat().st_size == 0:
                print(f"{model:16s} {mid:9s} SKIP (no transcript)")
                continue
            transcript = tpath.read_text(encoding="utf-8", errors="replace")
            system = SYSTEM_DIARIZED if used_speakers else SYSTEM_PLAIN
            roster = ", ".join(ROSTER.get(mid, BASE_TEAM))
            prev = read_prev_protocol(mid) if carry else ""
            block = ""
            if fixed:
                if not used_speakers:
                    print(f"{model:16s} {mid:9s} SKIP (--fixed needs transcript_speakers.txt)")
                    continue
                transcript, gloss, suggest, log, n_named, n_anon = fixed_input(transcript, mid)
                from term_source import people_for
                people = people_for(mid)
                base = SYSTEM_SENAT if structure == "senat" else SYSTEM_IDENTIFIED
                system = (base + (MARKERS_ON if markers else MARKERS_OFF)
                          + GLOSSARY_RULES + SUGGEST_RULES + PERSON_RULES + LENGTH_RULE
                          + GERMAN_ONLY)
                # After fixed_input, so talk time is counted per resolved name.
                block = talk_time_block(transcript)
                block += ("=== KORREKTE EIGENNAMEN (automatisch aus dem Vorwochenprotokoll) ===\n"
                          + ", ".join(gloss) + "\n=== ENDE ===\n\n")
                if people:
                    block += ("=== VOLLSTÄNDIGE PERSONENNAMEN AUS DEM VORWOCHENPROTOKOLL ===\n"
                              + ", ".join(sorted(people.values()))
                              + "\n=== ENDE ===\n\n")
                if suggest:
                    block += ("=== MÖGLICHE VERHÖRER (ungeprüft, nur bei passendem Kontext "
                              "ersetzen) ===\n"
                              + "; ".join(f"{a} -> {b}" for a, b in suggest)
                              + "\n=== ENDE ===\n\n")
                if carry:
                    if not prev:
                        print(f"{model:16s} {mid:9s} SKIP (--carry needs a previous protocol)")
                        continue
                    system = system + (CARRY_RULES if markers else CARRY_RULES_NOMARK)
                    block += (f"=== PROTOKOLL DER VORWOCHE ===\n{prev}\n"
                              f"=== ENDE VORWOCHE ===\n\n")
                print(f"{model:16s} {mid:9s} prep[{structure}"
                      + ("+carry" if carry else "") + f"]: {n_named} Sprecher benannt"
                      + (f", {n_anon} anonym" if n_anon else "")
                      + f", {len(gloss)} Begriffe, {len(people)} Personen, "
                      + f"{sum(log.values())} Korrekturen, {len(suggest)} Vorschläge"
                      + (f" ({', '.join(f'{k} x{v}' for k, v in log.most_common(3))})"
                         if log else ""))
            elif carry:
                if not prev:
                    print(f"{model:16s} {mid:9s} SKIP (no previous protocol for carry-over)")
                    continue
                system = system + CARRY_RULES
                block = f"=== PROTOKOLL DER VORWOCHE ===\n{prev}\n=== ENDE VORWOCHE ===\n\n"
            elif carry_tracked:
                verified = verified_open_items(model, mid)
                if verified is None:
                    print(f"{model:16s} {mid:9s} SKIP (not tracked — run carryover_track.py)")
                    continue
                system = system + CARRY_TRACKED_RULES
                # Items with lexical support come first.
                verified = sorted(
                    verified, key=lambda it: it.get("support") != "lexical")[:MAX_CARRIED]
                # Person and topic go in labeled fields, not in a bracket prefix,
                # which the model copied into the protocol.
                listing = "\n\n".join(
                    f"Punkt {n}:\n  Teilnehmer: {it['person'] or 'unbekannt'}\n"
                    + (f"  Thema: {it['topic']}\n" if it.get("topic") else "")
                    + f"  Text: {it['text']}"
                    for n, it in enumerate(verified, 1)) or "(keine)"
                block = (f"=== GEPRÜFTE OFFENE PUNKTE AUS DER VORWOCHE "
                         f"({len(verified)} Stück) ===\n{listing}\n=== ENDE ===\n\n")
            user = (f"Besprechungsdatum: {DATE[mid]}\nTeilnehmerliste: {roster}\n\n"
                    + block
                    + f"=== TRANSKRIPT ===\n{transcript}\n=== ENDE ===")
            for key, model_id in models_used.items():
                suffix = (("_fixed" + ("_senat" if structure == "senat" else "")
                           + ("_carry" if carry else "")) if fixed
                          else "_carry" if carry else "_tracked" if carry_tracked else "")
                suffix += suffix_extra
                out = OUTPUTS / model / mid / f"protocol_{key}{suffix}.md"
                if out.exists() and out.stat().st_size > 80:  # over 80 bytes counts as done
                    continue
                src = ("named+gloss" if fixed else
                       ("diar" if used_speakers else "plain")
                       + ("+carry" if carry else "+tracked" if carry_tracked else ""))
                print(f"{model:16s} {mid:9s} {key:5s} ({src}, {len(transcript.split()):5d}w in"
                      f"{f', prev {len(prev.split())}w' if prev else ''}) ... ",
                      end="", flush=True)
                t0 = time.time()
                try:
                    protocol = lisa(model_id, system, user)
                    # Empty or non-German answers are not written, so a later run
                    # retries the missing file.
                    if not protocol.strip():
                        print(f"EMPTY (no response) {round(time.time()-t0):3d}s")
                        continue
                    if not lang_ok(protocol):
                        print(f"REFUSED (non-German after re-rolls) "
                              f"{round(time.time()-t0):3d}s")
                        continue
                    out.write_text(protocol, encoding="utf-8")
                    print(f"OK {len(protocol.split()):4d}w  {round(time.time()-t0):3d}s")
                except Exception as e:
                    print(f"ERROR {type(e).__name__}: {str(e)[:80]}")


if __name__ == "__main__":
    main()
