"""Flags protocol bullets that treat two opposite speech-processing terms as one step or as successive steps.

TTS (Text-to-Speech) and STT/ASR (Speech-to-Text) are opposite processes, so a bullet such as
"Nächste Schritte: ERG und TTS anschauen, dann Speech-to-Text" names one step twice under two terms,
which is neither a wrong number, an invented name nor a duplicate sentence. PAIRS lists such pairs of
opposite terms. The table holds general speech-processing vocabulary, like the GERMAN_COMMON word list of
term_source.py, and no names taken from the meetings.

A bullet is flagged when it contains a term from both sides of a pair and a sequencing word (dann,
danach, anschließend, sowie, und). Bullets that contrast the two terms ("und nicht", "nicht ... sondern")
or report a mix-up ("vermisch", "durcheinander", ...) are not flagged. Each bullet is checked on its own;
adjacent bullets are not combined.

    python acronym_swap_check.py v10           # protocols of outputs_deliverable/v10, stdout only
    python acronym_swap_check.py --selftest    # checks the flag on built-in examples
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# (side A acronym, side A spelled out, side B acronym, side B spelled out, what confusing them means)
PAIRS = [
    (r"\bTTS\b", r"Text-?to-?Speech|Sprachsynthese",
     r"\bSTT\b|\bASR\b", r"Speech-?to-?Text|Spracherkennung",
     "TTS (Text-to-Speech) and STT/ASR (Speech-to-Text) are opposite directions"),
]

_SEQ = re.compile(r"\b(dann|danach|anschließend|sowie|und)\b", re.I)
# "X und nicht Y" / "nicht X sondern Y" is a contrast, the opposite of confusing the two, e.g.
# "ist aber für ASR und nicht für TTS" as a faithful paraphrase of "ist für ASA und nicht für TTS".
_CONTRAST = re.compile(r"\bund\s+nicht\b|\bnicht\b.{0,20}\bsondern\b", re.I)
# Excludes a sentence that reports an existing mix-up of the categories rather than confusing
# them itself, e.g. a paraphrase of the spoken "TTS, LLM und ASR... sind jetzt eben alle wild
# vermischt", where the speaker describes a disorganized overview.
_REPORTED_CONFUSION = re.compile(r"vermisch|durcheinander|unübersichtlich|uneindeutig", re.I)


def flag(text: str) -> list[tuple[str, str]]:
    """[(bullet, why)] for every bullet whose sentence treats two opposite terms as the same
    step or the same thing, e.g. 'ERG und TTS anschauen, dann Speech-to-Text'."""
    out = []
    for line in text.splitlines():
        m = re.match(r"^\s*[*-]\s+(.+)$", line)
        if not m:
            continue
        body = re.sub(r"\[(ÜBERTRAG|PRÜFEN|Bild)\]", "", m.group(1)).strip()
        if _REPORTED_CONFUSION.search(body) or _CONTRAST.search(body):
            continue                      # reporting confusion, or correctly contrasting the
            #                                two, is not the same as confusing them
        for a1, a2, b1, b2, why in PAIRS:
            has_a = re.search(a1, body) or re.search(a2, body)
            has_b = re.search(b1, body) or re.search(b2, body)
            if has_a and has_b and _SEQ.search(body):
                out.append((body, why))
    return out


def main() -> int:
    arm = sys.argv[1] if len(sys.argv) > 1 else "v10"
    MEET = ["march09", "april13", "april27", "may11", "may18", "june01", "june15", "june22"]
    print(f"ACRONYM-SWAP CHECK — arm {arm}")
    print("=" * 78)
    total = 0
    for mid in MEET:
        p = Path(f"outputs_deliverable/{arm}/{mid}/protocol_pro.md")
        if not p.exists():
            continue
        hits = flag(p.read_text(encoding="utf-8"))
        total += len(hits)
        for body, why in hits:
            print(f"  {mid:9s} {body[:80]}")
            print(f"            -> {why}")
    print(f"\n  {total} instance(s) found")
    return 0


def _self_test() -> None:
    """Checks flag() on built-in examples: the confused bullet must fire, the contrast and
    reported-confusion bullets must not."""
    text = "* Nächste Schritte: ERG und TTS anschauen, dann Speech-to-Text\n"
    hits = flag(text)
    assert len(hits) == 1, f"self-test failed: expected 1 hit, got {hits}"
    # a sentence that correctly CONTRASTS the two must not fire
    clean = "* TTS für Ausgabe geplant, Speech-to-Text bleibt bei Whisper\n"
    hits2 = flag(clean)
    assert len(hits2) == 0, f"self-test failed on the contrast case: {hits2}"
    # two bullets from the v8 output that must not fire
    fp1 = "* Eigener harter Datensatz ist aber für ASR und nicht für TTS\n"
    fp2 = "* Aktuelle Übersicht unübersichtlich durch Vermischung von TTS, LLM und ASR\n"
    assert not flag(fp1), f"regression: contrast case now flagged: {flag(fp1)}"
    assert not flag(fp2), f"regression: reported-confusion case now flagged: {flag(fp2)}"
    print("self-test OK: fires on the reported example, silent on both real false positives found")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _self_test()
    else:
        raise SystemExit(main())
