"""Behavior-freeze tests: entity/naming modules (pipeline)."""
from __future__ import annotations

from conftest import ROOT, folder_on_path, load

EC = load("pipeline", "entity_correct")
HN = load("pipeline", "handover_naming")
NC = load("pipeline", "name_check")


# ---------------------------------------------------------------- entity_correct

def test_entity_squash_and_koelner():
    assert EC.squash("Mösbauer") == "moesbauer"
    assert EC.squash("M4-SKI") == "m4ski"
    assert EC.squash("ISIS.de") == "isisde"
    assert EC.koelner("Mösbauer") == "6817"
    assert EC.koelner("Okafor") == "0437"
    assert EC.koelner("Zoé") == "8"


def test_entity_unusual_caps():
    assert [EC._UNUSUAL_CAPS(s) for s in
            ["ISIS", "M4SKI", "Forum", "Dienst", "Klasse", "Ab", "iisys"]] == \
           [True, False, False, False, False, False, False]


def test_entity_correct_compound_and_host():
    mapping = {"isis": {"canonical": "iisys"}, "m4ski": {"canonical": "M4-SKI"}}
    new, log = EC.correct("Wo ist ISIS? ISIS-Webseite und ISIS.de sind bei M4SKI.", mapping)
    assert new == "Wo ist iisys? iisys-Webseite und iisys.de sind bei M4SKI."
    assert dict(log) == {"ISIS -> iisys": 1, "ISIS -> iisys (compound)": 1,
                         "ISIS -> iisys (host)": 1}


# ---------------------------------------------------------------- handover_naming

def test_handover_koelner_and_score():
    assert HN.koelner("Brandl") == "17625"
    assert HN.koelner("Müller") == "657"
    assert round(HN.score("Ilkar", "Ilkan"), 4) == 0.7667
    assert round(HN.score("Walentyn", "Valentin"), 4) == 0.9167
    assert round(HN.score("Jensen", "Valentin"), 4) == 0.4392
    assert round(HN.score("Zoé", "Zoe"), 4) == 0.8889


def test_handover_best_name():
    window = "Lena machst du den Bericht? dann Lena bitte. Valentin kannst du übernehmen?"
    assert HN.best_name(window, ["Lena", "Valentin", "Zoé"]) == ("Lena", 1.0, "Lena")


def test_handover_report_order():
    turns = [
        (0.0, "SPEAKER_00", "ich eröffne die sitzung heute"),
        (300.0, "SPEAKER_01", "mein bericht " + "wort " * 60),
        (900.0, "SPEAKER_02", "kurzer zwischenruf"),
        (1200.0, "SPEAKER_02", "mein zweiter bericht " + "text " * 60),
        (1800.0, "SPEAKER_00", "ich schließe die sitzung"),
    ]
    assert HN.report_order(turns) == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"]


def test_handover_name_speakers_no_handover():
    tx = ("[00:00:00] SPEAKER_00: ich eröffne die sitzung heute\n"
          "[00:05:00] SPEAKER_01: mein bericht beginnt\n"
          "[00:15:00] SPEAKER_00: danke Valentin\n")
    assert HN.name_speakers([tx], ["Valentin", "Lena"]) == {}


# ---------------------------------------------------------------- name_check

def test_name_check_cluster():
    votes = [("canary", "Mösbauer"), ("cohere", "Mösbaur"),
             ("granite", "Mösbauer"), ("whisper", "Nösbauer")]
    clusters = NC._cluster(votes)
    assert [dict(c) for c in clusters] == [{"Mösbauer": 2, "Mösbaur": 1, "Nösbauer": 1}]


def test_name_check_as_markdown():
    rows = [
        {"as": None, "majority": "Mösbauer", "variants": {"Mösbauer": 3, "Mösbaur": 1}},
        {"as": ["Brenntal"], "why": "klingt wie ein Name aus einem früheren Protokoll"},
    ]
    assert NC.as_markdown(rows) == (
        '\n## Namen zum Prüfen\n'
        '*Die Schreibweise dieser Wörter im Protokoll ist unsicher: entweder sind sich die '
        'Spracherkennungssysteme uneinig ("gehört als"), oder das Wort klingt wie ein Name '
        'aus einem früheren Protokoll ("klingt wie").*\n'
        '\n'
        '* **Mösbauer** — gehört als: Mösbauer (3), Mösbaur (1)\n'
        '* **Brenntal** — klingt wie ein Name aus einem früheren Protokoll\n'
    )
    assert NC.as_markdown([]) == ""
