"""Find person names in the protocol that occur in neither the transcript nor the previous document.

A person mention is a title (Herr, Frau, Prof., Dr.) or a known first name, followed by a
surname. If the surname is in neither source, the transcript is searched for a word with the
same Kölner Phonetik code and a similarity of at least SIM. If one exists, it replaces the
invented form, because that is what was said (for example Mueller -> Miller). If none exists,
the name is left as it is and reported. Only the transcript and the previous document are used.
"""
from __future__ import annotations

import difflib
import re

import entity_correct as E

KNOWN_FIRST = (r"Sebastian|Thomas|Vincent|Johannes|Ren[ée]|Osman|Noah|Artem|Vinayak|Rassul"
               r"|Frank|Munir|Jens|Stefan|Natalie|Ron|Marcus|Alexander|Jonas|Jan|Florian"
               r"|J[üu]rgen|Hartmut|Lia|Olaf|Erika|Anna|Lena|Beatrix|Nicklas")
PERSON = re.compile(rf"\b(?:(?:Herrn?|Frau|Prof\.?|Dr\.?)\s+([A-ZÄÖÜ][a-zäöüß]{{2,}})"
                    rf"|({KNOWN_FIRST})\s+([A-ZÄÖÜ][a-zäöüß]{{2,}}))\b")
SIM = 0.60  # minimum similarity; a germanized spelling stays close to the one heard


def _surnames_in(text: str) -> set[str]:
    return {m.group(1) or m.group(3) for m in PERSON.finditer(text)}


def find(text: str, transcript: str, previous: str) -> dict[str, tuple[str | None, str]]:
    """surname -> (replacement or None, why). Present in neither source = invented."""
    tx_low, pv_low = transcript.lower(), previous.lower()
    heard = {w for w in re.findall(r"\b[A-ZÄÖÜ][a-zäöüß]{2,}\b", transcript)}
    out: dict[str, tuple[str | None, str]] = {}
    for sur in _surnames_in(text):
        if sur.lower() in tx_low or sur.lower() in pv_low:
            continue  # found in a source, so not invented
        best, score = None, 0.0
        for cand in heard:
            if cand == sur:
                continue
            r = difflib.SequenceMatcher(None, sur.lower(), cand.lower()).ratio()
            # the Kölner Phonetik code must match too, so a short name cannot capture an unrelated word
            if r > score and E.koelner(E.squash(sur)) == E.koelner(E.squash(cand)):
                best, score = cand, r
        if best and score >= SIM:
            out[sur] = (best, f"not in either source; the audio says '{best}' ({score:.2f})")
        else:
            out[sur] = (None, "not in either source and nothing near it in the audio")
    return out


def apply(text: str, transcript: str, previous: str) -> tuple[str, dict, list]:
    """Restore what was said where the audio supports it. Returns (text, changed, flagged)."""
    found = find(text, transcript, previous)
    changed, flagged = {}, []
    for sur, (repl, why) in found.items():
        if repl:
            text = re.sub(rf"(?<![\wÄÖÜäöü]){re.escape(sur)}(?![\wÄÖÜäöü])", repl, text)
            changed[sur] = repl
        else:
            flagged.append((sur, why))
    return text, changed, flagged
