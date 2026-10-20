"""Deterministic rule modules of pipeline/.

The expected values are the output of the code that produced the thesis numbers.
Quirks (e.g. numbers() also counting the digits of a timestamp) are kept on
purpose: the end-to-end behavior is what those numbers rest on.
"""
from __future__ import annotations

import pytest

from conftest import ROOT, folder_on_path, load

CT = load("pipeline", "carry_tag")
KL = load("pipeline", "known_leaks")
IN = load("pipeline", "invented_names")
NS = load("pipeline", "normalize_structure")
SF = load("pipeline", "stale_flag")


# ---------------------------------------------------------------- carry_tag

def test_carry_tag_terms():
    t = "Wir haben den Prototyp Atlas vorgestellt und die Kalibrierung besprochen"
    assert sorted(CT.terms(t)) == ["atlas", "besprochen", "kalibrierung", "prototyp", "vorgestellt"]


def test_carry_tag_numbers():
    assert sorted(CT.numbers("Budget 510 Euro, 58 Anträge, 8x, 570,3")) == ["510", "570,3", "58"]
    # raw call (no timestamp stripping): every ':'-separated group counts
    assert sorted(CT.numbers("[00:58:30] 58 Anträge und 570")) == ["00", "30", "570", "58"]


def test_carry_tag_tag():
    protocol = (
        "## Projekt Atlas\n"
        "* Der Prototyp wurde kalibriert und die Sensordaten geprüft\n"
        "* Wir planen ein Sommerfest im Juli\n"
        "* Neue Zahlen: 510 Euro für Material\n"
        "* 570 Euro für Lizenzen\n"
        "- [ÜBERTRAG] etwas bereits Markiertes\n"
        "\n"
        "* Kurz\n"
    )
    transcript = ("Wir haben den Prototyp Atlas vorgestellt und die Kalibrierung besprochen. "
                  "Kein Sommerfest erwähnt.")
    previous = "Der Prototyp wurde letzte Woche kalibriert. Budget 510 Euro. 570 Euro für Lizenzen."
    out, n_bul, n_tag = CT.tag(protocol, transcript, previous)
    assert (n_bul, n_tag) == (5, 2)
    assert out == (
        "## Projekt Atlas\n"
        "* Der Prototyp wurde kalibriert und die Sensordaten geprüft\n"
        "* Wir planen ein Sommerfest im Juli\n"
        "* [ÜBERTRAG] Neue Zahlen: 510 Euro für Material\n"
        "* [ÜBERTRAG] 570 Euro für Lizenzen\n"
        "- [ÜBERTRAG] etwas bereits Markiertes\n"
        "\n"
        "* Kurz\n"
    )


def test_carry_tag_preserves_trailing_newline_state():
    out, _, _ = CT.tag("## X\n* nichts Besonderes hier", "", "")
    assert not out.endswith("\n")


# ---------------------------------------------------------------- known_leaks

def test_known_leaks_repair():
    txt, counts = KL.repair(
        "* Data Quality Report führte ранее bei großen Datensätzen zum Absturz\n"
        "* Ранее war es so\n"
        "* nichts Fremdes hier"
    )
    assert counts == {"ранее": 1, "Ранее": 1}
    assert txt == (
        "* Data Quality Report führte zuvor bei großen Datensätzen zum Absturz\n"
        "* Zuvor war es so\n"
        "* nichts Fremdes hier"
    )


def test_known_leaks_leaves_other_foreign_script_alone():
    txt, counts = KL.repair("* текст bleibt stehen")
    assert (txt, counts) == ("* текст bleibt stehen", {})


# ---------------------------------------------------------------- invented_names

def test_invented_names_find_and_apply():
    text = "Herr Mueller berichtet über den Stand. Stefan Wulf war nicht da."
    transcript = "Sebastian Miller berichtet über den Stand der Arbeiten."
    previous = "## Team\n* Alter Punkt"
    found = IN.find(text, transcript, previous)
    assert found == {
        "Mueller": ("Miller", "not in either source; the audio says 'Miller' (0.77)"),
        "Wulf": (None, "not in either source and nothing near it in the audio"),
    }
    new, changed, flagged = IN.apply(text, transcript, previous)
    assert new == "Herr Miller berichtet über den Stand. Stefan Wulf war nicht da."
    assert changed == {"Mueller": "Miller"}
    assert flagged == [("Wulf", "not in either source and nothing near it in the audio")]


def test_invented_names_skips_names_present_in_previous():
    text = "Valentin Atlas hat den Bericht geschrieben."
    found = IN.find(text, "", "## Alt\n* Atlas hat den Bericht geschrieben")
    assert found == {}


# ---------------------------------------------------------------- normalize_structure

@pytest.mark.parametrize("line,expected", [
    ("[ÜBERTRAG] Foo", ("* [ÜBERTRAG] Foo", "bare->bullet")),
    ("### * [ÜBERTRAG] Foo", ("* [ÜBERTRAG] Foo", "heading->bullet")),
    ("### [ÜBERTRAG] Foo", ("* [ÜBERTRAG] Foo", "heading->bullet")),
    ("###Atlas und API-Entwicklung", ("### Atlas und API-Entwicklung", "heading-spacing")),
    ("*Gastvortrag Datenbanken", ("* Gastvortrag Datenbanken", "bullet-spacing")),
    ("-20% Verbesserung", ("-20% Verbesserung", None)),
    ("18. Juni Termin mit dem Partner", ("* 18. Juni Termin mit dem Partner", "date->bullet")),
    ("3. Quartal geplant", ("* 3. Quartal geplant", "date->bullet")),
    ("1. Nur mit Testbildern", ("* Nur mit Testbildern", "numbered->bullet")),
    ("  Vortrag beim Stadtverein gehalten (18 Personen)",
     ("  * Vortrag beim Stadtverein gehalten (18 Personen)", "indented->bullet")),
    ("=== ÜBERTRAG AUS VORWOCHE ===", ("", "echoed-prompt-fence")),
    ("  ", ("  ", None)),
    ("---", ("---", None)),
    ("**bold** bleibt", ("**bold** bleibt", None)),
])
def test_normalize_structure_repair_line(line, expected):
    assert NS.repair_line(line) == expected


def test_normalize_structure_strip_scaffold():
    line = "* [ÜBERTRAG] [Zoe / Atlas Bericht] Der Prototyp läuft"
    assert NS.strip_scaffold(line) == ("* [ÜBERTRAG] Atlas Bericht – Der Prototyp läuft",
                                       "scaffold-stripped")
    no_marker = "* [Max Muster / Projekt Website] text ohne Marker"
    assert NS.strip_scaffold(no_marker) == (no_marker, None)


def test_normalize_structure_repair_document():
    doc = (
        "# Protokoll\n"
        "### [ÜBERTRAG] Alter Punkt\n"
        "* [ÜBERTRAG] [Zoe / Atlas Bericht] Der Prototyp läuft\n"
        "18. Juni Termin\n"
        "* Normal\n"
    )
    new, counts = NS.repair(doc)
    assert new == (
        "# Protokoll\n"
        "* [ÜBERTRAG] Alter Punkt\n"
        "* [ÜBERTRAG] Atlas Bericht – Der Prototyp läuft\n"
        "* 18. Juni Termin\n"
        "* Normal\n"
    )
    assert counts == {"heading->bullet": 1, "scaffold-stripped": 1, "date->bullet": 1}


# ---------------------------------------------------------------- stale_flag

def test_stale_flag_mark(tmp_path):
    prev = tmp_path / "prev.md"
    older = tmp_path / "older.md"
    prev.write_text("## Team\n* Der Prototyp wurde kalibriert und geprüft\n"
                    "* Standortfrage bleibt offen\n", encoding="utf-8")
    older.write_text("## Team\n* Der Prototyp wurde kalibriert und geprüft\n", encoding="utf-8")
    draft = (
        "## Team\n"
        "* [ÜBERTRAG] Der Prototyp wurde kalibriert und geprüft\n"
        "* [ÜBERTRAG] Standortfrage bleibt offen\n"
        "* Etwas völlig Neues aus dem Audio\n"
        "- [PRÜFEN] schon markiert\n"
    )
    with folder_on_path(ROOT / "pipeline"):
        out, n_marked, n_carried = SF.mark(draft, prev, older)
    assert (n_marked, n_carried) == (1, 2)
    assert out == (
        "## Team\n"
        "* [ÜBERTRAG] Der Prototyp wurde kalibriert und geprüft\n"
        "* [PRÜFEN] Standortfrage bleibt offen\n"
        "* Etwas völlig Neues aus dem Audio\n"
        "- [PRÜFEN] schon markiert\n"
    )
