"""Derive the names and terms of one prior document, such as last week's protocol or an invitation
letter, from its spelling, its heading levels and a German part-of-speech tagger, each with the
reason it was admitted. glossary_for_document() and people_for_document() take a document path;
glossary_for(), targets_for() and people_for() take a meeting key of PREVIOUS. Relative paths are
resolved against the folder of this file.

    python term_source.py [MEETING]         # term count per meeting (MEETING: a key of PREVIOUS)
    python term_source.py --why [MEETING]   # also lists the terms by reason, at most 40 per reason
    python term_source.py --compare         # all terms against an optional hand-written name list
"""

from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent

# The previous protocol of each meeting, as a file name in the folder of this file.
PREVIOUS = {
    "march09": "Meeting_2026-03-02.md",
    "april13": "Meeting_2026-04-08.md",
    "april27": "Meeting_2026-04-20.md",
    "may11":   "Meeting_2026-05-04.md",
    "may18":   "Meeting_2026-05-11.md",
    "june01":  "Meeting_2026-05-18.md",
    "june15":  "Meeting_2026-06-08(1).md",
    "june22":  "Meeting_2026-06-15.md",
}

# The protocol before PREVIOUS, for the stale-item check ([PRÜFEN]). march09 and april13 have no
# entry: march09 has no older protocol, and the protocol before april13's previous one is a month
# older, so it would mark month-old items as new.
OLDER = {
    "april27":  "Meeting_2026-04-13.md",
    "may11":    "Meeting_2026-04-27.md",
    "may18":    "Meeting_2026-05-04.md",
    "june01":   "Meeting_2026-05-11.md",
    "june15":   "Meeting_2026-06-01.md",
    "june22":   "Meeting_2026-06-08(1).md",
}

# Common German words in lower case. Rules 2 to 6 of extract() never admit them: German
# capitalizes every noun, so a capital letter alone does not mark a name.
GERMAN_COMMON = set("""
der die das den dem des ein eine einen einem eines und oder aber wenn dann also sowie
ist sind war waren wird werden wurde wurden hat haben hatte hatten kann können konnte
muss müssen soll sollen will wollen darf dürfen mag mögen
nicht nur noch schon auch sehr mehr weniger viel viele wenig alle alles jeder jede jedes
für von zu mit bei aus nach vor über unter durch gegen ohne um an auf in im am beim zum zur
ich du er sie es wir ihr man sich mein dein sein unser euer
woche wochen tag tage tagen monat jahr jahre stunde stunden minute minuten
problem probleme thema themen frage fragen antwort punkt punkte teil teile ende anfang
arbeit aufgabe aufgaben termin termine treffen meeting besprechung protokoll
modell modelle daten datei dateien system systeme server version versionen update updates
test tests fehler fehlern ergebnis ergebnisse beispiel beispiele
neue neuer neues alte alter altes gut gute guter besser beste
status backlog stand info infos hinweis hinweise
montag dienstag mittwoch donnerstag freitag samstag sonntag
januar februar märz april mai juni juli august september oktober november dezember
paper papers projekt projekte student studenten kollege kollegen
""".split()) | set("""
nett nette netter nettes bereits voran west ost nord sued schueler schuelern kennen kennt
gemixt gemacht gesagt gesehen geht steht laeuft lief liefen bleibt blieb kommt kam gibt gab
nimmt nahm setzt setzte stellt stellte zeigt zeigte findet fand macht machte sagt sagte
schon eben halt etwa circa ungefaehr genau wirklich eigentlich vielleicht sicher klar
gross grosse grosser kleines kleine klein lang lange kurz kurze hoch hohe niedrig
erste erster erstes zweite zweiter dritte letzte letzter naechste naechster
richtig falsch falsche falscher gleich gleiche anders andere anderen
zusammen einzeln jeweils immer nie oft selten manchmal wieder weiter zurueck
oben unten vorne hinten links rechts innen aussen
moeglich moegliche noetig noetige wichtig wichtige einfach einfache schwer schwierig
ganze ganzen ganz teil teile teilweise stueck stuecke seite seiten
punkt punkte frage fragen antwort antworten grund gruende folge folgen
woche wochen tage stunde stunden zeit zeiten mal male
leute personen person mensch menschen kunde kunden partner
sache sachen ding dinge art arten weise wege weg
stand staende schritt schritte ziel ziele plan plaene
""".split())

_CAMEL = re.compile(r"[a-zäöüß][A-ZÄÖÜ]")
_ALNUM = re.compile(r"(?=.*[A-Za-zÄÖÜäöü])(?=.*\d)")
_HYPHEN_CAPS = re.compile(r"^[A-ZÄÖÜ][\wäöüß]*-[A-ZÄÖÜ0-9]")
_TOKEN = re.compile(r"[A-Za-zÄÖÜäöüß0-9][\wÄÖÜäöüß\.\-]*")


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text)


def _clean(tok: str) -> str:
    return tok.strip(".,;:()[]\"'")


def orthographic_reason(tok: str) -> str | None:
    t = _clean(tok)
    if len(t) < 3:
        return None
    if _CAMEL.search(t):
        return "internal capital"
    if t.isupper() and len(t) >= 3 and t.isalpha():
        return "all caps"
    if _ALNUM.match(t) and re.search(r"[A-ZÄÖÜ]", t):
        return "letters + digits"
    if _HYPHEN_CAPS.match(t):
        return "hyphenated capitals"
    return None


def read_document(name: str) -> tuple[str, list[tuple[int, str]]]:
    """Read a Markdown, .docx or .pdf document and return (text, [(level, line), ...]), with level 0
    for a person or agenda heading, 1 for a topic and 2 for a detail.

    A relative name is taken from the folder of this file; a missing file, a PDF that pypdf cannot
    read or a file with a known binary signature gives an empty text and no lines.
    """
    p = REPO / name
    if not p.exists():
        return "", []
    if p.suffix.lower() == ".docx":
        import docx
        paras = []
        for par in docx.Document(str(p)).paragraphs:
            t = par.text.strip()
            if not t:
                continue
            pr = par._p.pPr
            lvl = (pr.numPr.ilvl.val if pr is not None and pr.numPr is not None
                   and pr.numPr.ilvl is not None else None)
            bold = any(r.bold for r in par.runs)
            paras.append((lvl, bold, t))
        bold_lvls = [l for l, b, _ in paras if b]
        person_lvl = None if None in bold_lvls else (min(bold_lvls) if bold_lvls else None)
        lines = []
        for lvl, bold, t in paras:
            if bold and lvl == person_lvl:
                lines.append((0, t))
            elif lvl == (0 if person_lvl is None else person_lvl + 1):
                lines.append((1, t))
            else:
                lines.append((2, t))
        return "\n".join(t for _, _, t in paras), lines
    if p.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
            txt = "\n".join(_drop_running_lines([(pg.extract_text() or "")
                                                 for pg in PdfReader(str(p)).pages]))
        except Exception:
            return "", []
        # PDF exports mark the level with a bullet character, not with indentation.
        # An unindented line without a marker continues the row above it.
        BULLET = {"•": 0, "◦": 1, "▪": 2, "▫": 2, "‣": 2, "·": 1}
        rows: list[tuple[int, str]] = []
        last = 0
        for raw in txt.splitlines():
            s = raw.strip()
            if not s:
                continue
            if s[0] in BULLET:
                last = BULLET[s[0]]
                body = s[1:].strip()
                if body:
                    rows.append((last, body))
                continue
            if rows and len(raw) - len(raw.lstrip()) == 0 and not re.match(r"^[A-ZÄÖÜ].{0,3}$", s):
                lvl, prev = rows[-1]
                rows[-1] = (lvl, f"{prev} {s}".strip())
                continue
            rows.append((0 if len(raw) - len(raw.lstrip()) < 2 else 1, s))
        if not any(l == 0 for l, _ in rows):
            # No level-0 row: fall back to the indentation of each line.
            rows = [(0 if len(r) - len(r.lstrip()) < 2 else 1, r.strip())
                    for r in txt.splitlines() if r.strip()]
        return txt, rows
    # Any other file is read as text. Known binary formats are refused, because read_text() with
    # errors="replace" would return their bytes as text instead of failing.
    head = p.read_bytes()[:8]
    if head[:4] in (b"%PDF", b"PK\x03\x04", b"\xd0\xcf\x11\xe0", b"{\\rt"):
        return "", []
    txt = p.read_text(encoding="utf-8", errors="replace")
    lines = []
    from evaluate_speaker_attribution import md_indent_unit
    unit = md_indent_unit(txt.splitlines())  # spaces per nesting level
    for raw in txt.splitlines():
        if not raw.strip():
            continue
        indent = len(raw) - len(raw.lstrip())
        m = re.match(r"\*\s+(.*)", raw.strip())
        body = m.group(1).strip() if m else raw.strip()
        body = re.sub(r"^\*\*(.*?)\*\*$", r"\1", body).strip()
        lines.append((0 if indent == 0 else (1 if indent <= unit + 1 else 2), body))
    return txt, lines


def _mid_line_caps(lines: list[tuple[int, str]]) -> set[str]:
    """Return the tokens that appear capitalized somewhere other than as the first token of a line.

    The first word of a line is capitalized whatever it is, so only these capitals count.
    """
    out = set()
    for _, line in lines:
        toks = tokens(line)
        for t in toks[1:]:
            c = _clean(t)
            if c and c[0].isupper():
                out.add(c)
    return out


# Signals from the German spaCy model
_NLP = None
_NLP_TRIED = False


def _nlp():
    """Load the spaCy model de_core_news_sm once; return None if it cannot be loaded."""
    global _NLP, _NLP_TRIED
    if not _NLP_TRIED:
        _NLP_TRIED = True
        try:
            import spacy
            _NLP = spacy.load("de_core_news_sm")
        except Exception:
            _NLP = None
    return _NLP


def _plain(lines: list[tuple[int, str]]) -> str:
    """Join the lines, each ending in a period, so that NER sees sentence boundaries."""
    return "\n".join(l.strip().rstrip(".") + "." for _, l in lines if l.strip())


# The tagger judges each candidate inside this neutral sentence, not where it stands: at the
# start of a line it often reads a name as a verb.
_FRAME = "Wir haben das Thema {} gestern ausfuehrlich besprochen."
_FRAME_CACHE: dict[str, str] = {}


def _frame_pos(tok: str) -> str:
    if tok not in _FRAME_CACHE:
        nlp = _nlp()
        if nlp is None:
            return "?"
        doc = nlp(_FRAME.format(tok))
        _FRAME_CACHE[tok] = next((t.pos_ for t in doc if t.text == tok), "?")
    return _FRAME_CACHE[tok]


def _lowercase_marked(text: str, lines: list[tuple[int, str]]) -> set[str]:
    """Return lower-case words of four or more letters, not in GERMAN_COMMON, that the tagger
    reads as PROPN or X inside _FRAME (an empty set without the spaCy model).
    """
    if _nlp() is None:
        return set()
    cands = {c for c in (_clean(t) for t in tokens(text))
             if len(c) >= 4 and c.islower() and c.isalpha() and c not in GERMAN_COMMON}
    return {c for c in cands if _frame_pos(c) in ("PROPN", "X")}


def person_candidates(name: str) -> set[str]:
    """Return the capitalized tokens of the PER entities that spaCy finds in a document, plus each
    capitalized word after Herr, Frau, Prof. or Dr.

    NER also tags ordinary words as people, so these candidates are kept out of extract().
    """
    text, lines = read_document(name)
    nlp = _nlp()
    if not text or nlp is None:
        return set()
    out = set()
    for ent in nlp(_plain(lines)[:100000]).ents:
        if ent.label_ != "PER":
            continue
        for tok in tokens(ent.text):
            t = _clean(tok)
            if len(t) >= 4 and t[0].isupper() and t.lower() not in GERMAN_COMMON:
                out.add(t)
    for m in re.finditer(r"\b(?:Herr|Frau|Prof\.?|Dr\.?)\s+([A-ZÄÖÜ][a-zäöüß\-]{2,})", text):
        out.add(m.group(1))
    return out


def extract(name: str) -> dict[str, str]:
    """Return {term: reason} for one document, found by the six rules below."""
    text, lines = read_document(name)
    if not text:
        return {}
    terms: dict[str, str] = {}
    midline = _mid_line_caps(lines)

    # 1. Spelling marks (orthographic_reason), anywhere in the document.
    for tok in tokens(text):
        why = orthographic_reason(tok)
        if why:
            terms.setdefault(_clean(tok), why)

    # 2. Capitalized tokens on the top level: persons in a protocol, agenda items in an invitation.
    #    In a Markdown or PDF file without indentation or bullets, every line is on the top level.
    for lvl, line in lines:
        if lvl != 0:
            continue
        for tok in tokens(line):
            t = _clean(tok)
            if len(t) >= 4 and t[0].isupper() and t.lower() not in GERMAN_COMMON:
                terms.setdefault(t, "names the document is organised by")

    # 3. Capitalized tokens in topic headings (level 1) that also appear mid-line or carry a
    #    spelling mark. This also admits ordinary nouns: German capitalizes nouns mid-sentence too.
    for lvl, line in lines:
        if lvl != 1:
            continue
        for tok in tokens(line):
            t = _clean(tok)
            if (len(t) >= 4 and t[0].isupper() and t.lower() not in GERMAN_COMMON
                    and (t in midline or orthographic_reason(t))):
                terms.setdefault(t, "capitalised in a topic heading")

    # 4. Parts of dotted host names with three or more parts.
    for tok in tokens(text):
        t = _clean(tok)
        if re.fullmatch(r"[A-Za-z0-9][\w\-]*(\.[A-Za-z0-9][\w\-]*){2,}", t):
            for part in t.split(".")[:-1]:               # drop the TLD
                if len(part) >= 4 and part.lower() not in GERMAN_COMMON:
                    terms.setdefault(part, "part of a host name")

    # 5. Capitalized words used at least twice in the body (level 2 and deeper) that also appear
    #    mid-line.
    body = Counter()
    for lvl, line in lines:
        if lvl < 2:
            continue
        for tok in tokens(line):
            t = _clean(tok)
            if (len(t) >= 4 and t[0].isupper() and t.lower() not in GERMAN_COMMON
                    and t in midline):
                body[t] += 1
    for t, c in body.items():
        if c >= 2:
            terms.setdefault(t, f"used {c}x in the body")

    # 6. Lower-case words that the tagger reads as proper nouns (_lowercase_marked).
    for t in _lowercase_marked(text, lines):
        terms.setdefault(t, "lower case where German capitalises")
    return terms


# Reasons that make a term a correction target in targets_for(); the glossary functions return
# every term. "lower case where German capitalises" is left out on purpose: it also admits
# English words, which a rewriter would then put in place of German ones.
_STRONG = ("internal capital", "all caps", "letters + digits", "hyphenated capitals",
           "names the document is organised by", "part of a host name",
           "capitalised in a topic heading")


def targets_for(meeting: str) -> list[str]:
    if meeting not in PREVIOUS:
        return []
    return sorted(t for t, why in extract(PREVIOUS[meeting]).items() if why in _STRONG)


_FULLNAME = re.compile(r"\b([A-ZÄÖÜ][a-zäöüß]{2,})\s+([A-ZÄÖÜ][a-zäöüß\-]{2,})\b")


def people_for(meeting: str) -> dict[str, str]:
    """Return {first name: full name} for the people that a meeting's PREVIOUS document names in
    full, as people_for_document() does for any document. A first name seen with more than one
    surname is left out.
    """
    if meeting not in PREVIOUS:
        return {}
    return people_for_document(PREVIOUS[meeting])


# Full names from people_for() and people_for_document() are a hint for the prompt, not a
# rewrite rule: no offline test tells a German surname from a German noun.


# Document-keyed API: the *_for_document() functions take a document path instead of a meeting key.
def glossary_for_document(doc: Path | str) -> list[str]:
    return sorted(extract(str(doc)))


def _drop_running_lines(pages: list[str], top: int = 2, bottom: int = 3,
                        share: float = 0.5) -> list[str]:
    """Remove running headers and footers from the text of PDF pages.

    A line among the first `top` or last `bottom` non-empty lines of a page is removed when, with
    digits and spaces normalized, it recurs at a page edge on at least `share` of the pages and on
    two pages or more.
    """
    if len(pages) < 2:
        return pages
    from collections import Counter

    def norm(s: str) -> str:
        return re.sub(r"\d+", "#", " ".join(s.split()))

    seen: Counter = Counter()
    for pg in pages:
        ls = [l for l in pg.splitlines() if l.strip()]
        seen.update({norm(l) for l in ls[:top] + ls[-bottom:]})
    need = max(2, -(-int(share * 100) * len(pages) // 100))
    running = {k for k, n in seen.items() if n >= need}
    out = []
    for pg in pages:
        ls = pg.splitlines()
        idx = [i for i, l in enumerate(ls) if l.strip()]
        edge = set(idx[:top] + idx[-bottom:])
        out.append("\n".join(l for i, l in enumerate(ls) if not (i in edge and norm(l) in running)))
    return out


def attendees_for_document(doc: Path | str) -> list[str]:
    """Return the level-0 lines of a document, without bullets and bold marks, as a list of names.

    Lines with "://" or a digit, lines that start with "[" or not with a capital letter, and lines
    of more than four words are skipped.
    """
    _, lines = read_document(str(doc))
    out: list[str] = []
    for lvl, line in lines:
        if lvl != 0:
            continue
        s = re.sub(r"^[\s•*+-]+", "", line).strip()
        s = re.sub(r"^\*\*(.*?)\*\*$", r"\1", s).strip()
        if not s or "://" in s or s.startswith("["):
            continue
        if any(c.isdigit() for c in s) or len(s.split()) > 4:
            continue
        if not s[0].isupper():
            continue
        if s not in out:
            out.append(s)
    return out


def people_for_document(doc: Path | str) -> dict[str, str]:
    text, lines = read_document(str(doc))
    if not text:
        return {}
    by_first: dict[str, set[str]] = defaultdict(set)
    for lvl, line in lines:
        if lvl <= 1:
            for first, last in _FULLNAME.findall(line):
                if first.lower() not in GERMAN_COMMON and last.lower() not in GERMAN_COMMON:
                    by_first[first].add(last)
    for m in re.finditer(r"\b(?:Herr|Frau|Prof\.?|Dr\.?)\s+" + _FULLNAME.pattern, text):
        by_first[m.group(1)].add(m.group(2))
    return {f: f"{f} {next(iter(ls))}" for f, ls in by_first.items() if len(ls) == 1}


def glossary_for(meeting: str) -> list[str]:
    """Return the sorted terms of the meeting's PREVIOUS document, or [] for an unknown meeting."""
    return sorted(extract(PREVIOUS[meeting])) if meeting in PREVIOUS else []


def main() -> None:
    args = sys.argv[1:]
    if "--compare" in args:
        try:
            from entity_lexicon import ALL_NAMES
        except ImportError:
            print("--compare needs entity_lexicon.py (the retired hand-written list), "
                  "which is not part of this repository.")
            return
        print("AUTOMATIC EXTRACTION vs THE RETIRED HAND-WRITTEN LIST")
        print("=" * 92)
        union = set()
        for mid in PREVIOUS:
            union |= set(glossary_for(mid))
        low = {t.lower() for t in union}
        hit = [n for n in ALL_NAMES if n.lower() in low
               or any(n.lower() in t.lower() or t.lower() in n.lower() for t in union)]
        print(f"  terms found automatically across the {len(PREVIOUS)} previous documents : "
              f"{len(union)}")
        print(f"  hand-written names recovered automatically                : "
              f"{len(hit)}/{len(ALL_NAMES)}")
        miss = [n for n in ALL_NAMES if n not in hit]
        print(f"  not recovered: {', '.join(miss) if miss else '(none)'}")
        print("\n  A name the previous document never mentions CANNOT be recovered, and should")
        print("  not be — that is the transferability rule doing its job, not a defect.")
        return

    why = "--why" in args
    only = next((a for a in args if a in PREVIOUS), None)
    print("VOCABULARY FROM THE PREVIOUS DOCUMENT ALONE")
    print("=" * 92)
    print("No hand-written names. Each term is justified by the document it came from.\n")
    for mid in ([only] if only else PREVIOUS):
        terms = extract(PREVIOUS[mid])
        print(f"  {mid:<10} <- {PREVIOUS[mid]:<44} {len(terms)} terms")
        if why:
            by = defaultdict(list)
            for t, w in sorted(terms.items()):
                by[w].append(t)
            for w, ts in by.items():
                print(f"       [{w}] {', '.join(ts[:40])}")
            print()


if __name__ == "__main__":
    main()
