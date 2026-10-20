"""German text normalization for word error rates, checks for broken UTF-8 text, and the
canonical carry-over marker [ÜBERTRAG].

Usage (prints a short demo):
    python text_norm.py
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, asdict

# Canonical markers

CANONICAL_CARRYOVER = "[ÜBERTRAG]"
CANONICAL_BACKLOG_HEADING = "Rückstand / Offene Punkte"

_CARRYOVER_VARIANTS = re.compile(
    r"\[\s*(?:UEBERTRAG|UBERTRAG|ÜBERTRAG|Übertrag|Ue?bertrag|UE?BERTRAG)\s*\]",
    re.IGNORECASE,
)
_BACKLOG_VARIANTS = re.compile(
    r"R(?:ü|ue|u)ckstand", re.IGNORECASE,
)


def canonicalize_markers(text: str) -> str:
    """Rewrite spelling variants of the carry-over marker and of 'Rückstand'
    to their canonical umlaut forms.
    """
    text = _CARRYOVER_VARIANTS.sub(CANONICAL_CARRYOVER, text)
    text = _BACKLOG_VARIANTS.sub("Rückstand", text)
    return text


# Encoding checks and repair

# UTF-8 umlauts and ß decoded as cp1252 (Ã¼ Ã¶ Ã¤ ÃŸ Ã„ Ã– Ãœ)
# or as latin-1 (Ã followed by the C1 control U+0084, U+0096, U+009C or U+009F).
_MOJIBAKE = re.compile(r"Ã[¤¶¼Ÿ„–œ\x84\x96\x9c\x9f]")
_REPLACEMENT = "�"  # U+FFFD replacement character


@dataclass
class EncodingReport:
    n_chars: int
    n_replacement_chars: int  # U+FFFD characters (decode loss)
    n_mojibake_hits: int  # Ã¼-style mojibake matches
    is_nfc: bool
    healthy: bool

    def as_dict(self):
        return asdict(self)


def check_encoding(text: str) -> EncodingReport:
    """Count U+FFFD characters and Ã¼-style mojibake and check the NFC form, without
    changing the text. The healthy flag is True when both counts are 0;
    is_nfc does not affect it.
    """
    n_repl = text.count(_REPLACEMENT)
    n_moji = len(_MOJIBAKE.findall(text))
    is_nfc = unicodedata.is_normalized("NFC", text)
    return EncodingReport(
        n_chars=len(text),
        n_replacement_chars=n_repl,
        n_mojibake_hits=n_moji,
        is_nfc=is_nfc,
        healthy=(n_repl == 0 and n_moji == 0),
    )


def fix_mojibake(text: str) -> str:
    """Repair Ã¼-style mojibake by encoding the text as cp1252 and decoding it as UTF-8.
    Returns NFC text, unrepaired if no mojibake is found or the round trip fails.
    """
    if not _MOJIBAKE.search(text):
        return unicodedata.normalize("NFC", text)
    try:
        repaired = text.encode("cp1252", errors="strict").decode("utf-8", errors="strict")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return unicodedata.normalize("NFC", text)
    return unicodedata.normalize("NFC", repaired)


def to_nfc(text: str) -> str:
    """Return the NFC form, so precomposed and combining-mark umlauts compare equal."""
    return unicodedata.normalize("NFC", text)


# German normalizer for WER

# Hesitation tokens, removed when remove_filled_pauses is True.
# 呃 and 嗯 are Chinese hesitation marks that one recognizer writes for German speech.
_FILLED_PAUSES = {"äh", "ähm", "öh", "hm", "hmm", "mhm", "ah", "eh", "呃", "嗯"}

_NUM_WORDS = {
    "null": "0", "eins": "1", "zwei": "2", "drei": "3", "vier": "4",
    "fünf": "5", "sechs": "6", "sieben": "7", "acht": "8", "neun": "9",
    "zehn": "10", "elf": "11", "zwölf": "12",
}

_PUNCT = re.compile(r"[.,;:!?\"'()\[\]{}«»„“”‚‘’�…—–\-/]")
_WS = re.compile(r"\s+")


@dataclass
class NormConfig:
    """Options for normalize(); the defaults keep case and umlauts."""
    casefold: bool = False
    umlaut_ascii: bool = False  # ü->ue, ö->oe, ä->ae, ß->ss
    strip_punct: bool = True
    remove_filled_pauses: bool = True
    normalize_numbers: bool = False  # number words null to zwölf -> digits
    collapse_ws: bool = True

    def as_dict(self):
        return asdict(self)


_UMLAUT_MAP = str.maketrans({"ü": "ue", "ö": "oe", "ä": "ae", "ß": "ss",
                             "Ü": "Ue", "Ö": "Oe", "Ä": "Ae"})


def normalize(text: str, cfg: NormConfig | None = None) -> str:
    """Normalize text with the options in cfg (default NormConfig()).
    Use the same cfg for hypothesis and reference.
    """
    cfg = cfg or NormConfig()
    text = to_nfc(text)
    if cfg.casefold:
        text = text.casefold()
    if cfg.umlaut_ascii:
        text = text.translate(_UMLAUT_MAP)
    if cfg.strip_punct:
        text = _PUNCT.sub(" ", text)
    tokens = text.split()
    if cfg.remove_filled_pauses:
        tokens = [t for t in tokens if t.casefold() not in _FILLED_PAUSES]
    if cfg.normalize_numbers:
        tokens = [_NUM_WORDS.get(t.casefold(), t) for t in tokens]
    text = " ".join(tokens)
    if cfg.collapse_ws:
        text = _WS.sub(" ", text).strip()
    return text


def tokens(text: str, cfg: NormConfig | None = None) -> list[str]:
    """Return the whitespace tokens of the normalized text."""
    return normalize(text, cfg).split()


if __name__ == "__main__":
    demo = "Er berichtete über die Sitzung. [UEBERTRAG] Rueckstand offen. äh das war's."
    print("input     :", demo)
    print("canonical :", canonicalize_markers(demo))
    print("normalized:", normalize(demo, NormConfig(casefold=True)))
    print("encoding  :", check_encoding(demo).as_dict())
