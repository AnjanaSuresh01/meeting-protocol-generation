"""Behavior-freeze tests: protocol_repair, text_norm, protocol_validate (pipeline)."""
from __future__ import annotations

from conftest import ROOT, folder_on_path, load

PR = load("pipeline", "protocol_repair")
TN = load("pipeline", "thesis_pipeline.text_norm")
PV = load("pipeline", "thesis_pipeline.protocol_validate")


# ---------------------------------------------------------------- protocol_repair

def test_protocol_repair_is_team_arm():
    assert [PR.is_team_arm(n) for n in
            ["protocol_pro_fixed.md", "protocol_pro_fixed_carry.md",
             "protocol_pro_fixed_senat.md", "protocol_pro.md"]] == [True, True, False, False]


DOC = ("# Protokoll\n"
       "## Zoé\n"
       "### Thema A\n"
       "* Bullet 1\n"
       "## Lena\n"
       "### Thema B\n"
       "* Bullet 2\n"
       "## SPEAKER_05\n"
       "### Thema A\n"
       "* Bullet 3")


def test_protocol_repair_split_blocks():
    pre, blocks = PR.split_blocks(DOC)
    assert pre == "# Protokoll\n"
    assert [h for h, _ in blocks] == ["Zoé", "Lena", "SPEAKER_05"]
    assert blocks[0][1] == "## Zoé\n### Thema A\n* Bullet 1\n"


def test_protocol_repair_repair():
    with folder_on_path(ROOT / "pipeline"):
        new, changes = PR.repair(DOC, {"lena": 100.0, "zoe": 200.0})
    assert changes == {
        "unnamed_cluster_renamed": ["SPEAKER_05"],
        "reordered": {"was": ["Zoé", "Lena", "Nicht zugeordnete Beiträge"],
                      "now": ["Lena", "Zoé", "Nicht zugeordnete Beiträge"]},
    }
    assert new == ("# Protokoll\n"
                   "## Lena\n### Thema B\n* Bullet 2\n"
                   "## Zoé\n### Thema A\n* Bullet 1\n"
                   "## Nicht zugeordnete Beiträge\n### Thema A\n* Bullet 3\n")


def test_protocol_repair_duplicate_topics():
    assert PR.duplicate_topics(DOC) == ["themaa"]


# ---------------------------------------------------------------- text_norm

def test_text_norm_canonicalize_markers():
    demo = "Er berichtete. [UEBERTRAG] Rueckstand offen. [Übertrag] nochmal."
    assert TN.canonicalize_markers(demo) == \
        "Er berichtete. [ÜBERTRAG] Rückstand offen. [ÜBERTRAG] nochmal."


def test_text_norm_check_encoding():
    assert TN.check_encoding("Grüße aus Hof. Straße ok.").as_dict() == {
        "n_chars": 25, "n_replacement_chars": 0, "n_mojibake_hits": 0,
        "is_nfc": True, "healthy": True}
    assert TN.check_encoding("groÃŸe GrÃ¼ÃŸe").as_dict() == {
        "n_chars": 14, "n_replacement_chars": 0, "n_mojibake_hits": 3,
        "is_nfc": True, "healthy": False}
    assert TN.check_encoding("kaputt � hier").n_replacement_chars == 1


def test_text_norm_fix_mojibake():
    assert TN.fix_mojibake("groÃŸe GrÃ¼ÃŸe") == "große Grüße"
    assert TN.fix_mojibake("Grüße bleiben") == "Grüße bleiben"


def test_text_norm_normalize():
    assert TN.normalize("Er sagte: „Hallo, Welt!“ äh um 3,5 % und-so.") == \
        "Er sagte Hallo Welt um 3 5 % und so"
    assert TN.normalize("Er sagte: „Hallo, Welt!“ äh", TN.NormConfig(casefold=True)) == \
        "er sagte hallo welt"
    assert TN.normalize("drei und fünf", TN.NormConfig(normalize_numbers=True)) == "3 und 5"
    assert TN.tokens("Äh, zwei Wörter!") == ["zwei", "Wörter"]


# ---------------------------------------------------------------- protocol_validate

def test_protocol_validate_good():
    good = "## Zoé\n### Thema\n" + "* " + "wort " * 45 + "ende.\n"
    r = PV.validate_protocol(good)
    assert r.ok is True
    assert r.as_dict() == {
        "ok": True, "n_persons": 1, "n_topics": 1, "n_bullets": 1,
        "hierarchy_ok": True, "truncated": False, "bad_markers": 0,
        "encoding_ok": True,
        "problems": ["only ONE person section (possible degenerate-transcript symptom, cf. C5)"]}


def test_protocol_validate_truncated():
    r = PV.validate_protocol("## Zoé\n### Thema\n* Das Projekt läuft sehr gut und")
    assert r.ok is False and r.truncated is True
    assert r.problems == [
        "only ONE person section (possible degenerate-transcript symptom, cf. C5)",
        "ends mid-sentence (truncation / finish_reason=length?): "
        "...'* Das Projekt läuft sehr gut und'",
        "near-empty protocol (11 words)"]


def test_protocol_validate_bad_marker():
    r = PV.validate_protocol("## Zoé\n### Thema\n" + "* [UEBERTRAG] " + "wort " * 45 + "ende.\n")
    assert r.ok is False and r.bad_markers == 1


def test_protocol_validate_small_and_normalize():
    r = PV.validate_protocol("## Zoé\n* kurz\n")
    assert r.ok is True and r.problems == [
        "only ONE person section (possible degenerate-transcript symptom, cf. C5)",
        "near-empty protocol (4 words)"]
    assert PV.normalize_protocol("[Uebertrag] text") == "[ÜBERTRAG] text"
