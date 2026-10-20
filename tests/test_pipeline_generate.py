"""Pure helpers of generate_protocols.

generate_protocols reads LISA_* only when a call is made. The dummy values below
guard against an accidental call; no test here makes one.
"""
from __future__ import annotations

import os

os.environ.setdefault("LISA_BASE_URL", "http://localhost:0")
os.environ.setdefault("LISA_API_KEY", "dummy")

from conftest import load  # noqa: E402

GP = load("pipeline", "generate_protocols")

DE = "Wir haben das Protokoll heute besprochen und die Ergebnisse der Sitzung festgehalten."
EN = ("We have discussed the protocol today and the results of the meeting were noted "
      "and the team will follow up on the items and then we will see what happens next.")


def test_generate_english_lean():
    assert GP.english_lean(DE) == -1.0            # < 20 words: not judgeable
    assert round(GP.english_lean(EN), 4) == 0.4839
    assert GP.english_lean("zu kurz") == -1.0


def test_generate_lang_ok():
    assert GP.lang_ok(DE) is True
    assert GP.lang_ok(EN) is False
    assert GP.lang_ok("текст " * 30) is False


def test_generate_talk_minutes_and_block():
    tx = ("[00:00:00] SPEAKER_00: guten morgen zusammen\n"
          "[00:01:00] SPEAKER_01: mein bericht heute\n"
          "[00:06:00] SPEAKER_00: danke\n")
    assert GP.talk_minutes(tx) == {"SPEAKER_00": 1.5, "SPEAKER_01": 5.0}
    assert GP.talk_time_block(tx) == (
        "=== REDEZEIT JE PERSON (aus den Zeitstempeln gemessen) ===\n"
        "  SPEAKER_01: 5 Minuten (77% der Redezeit)\n"
        "  SPEAKER_00: 2 Minuten (23% der Redezeit)\n"
        "=== ENDE ===\n\n")
    assert GP.talk_time_block("[00:00:00] SPEAKER_00: text") == ""
