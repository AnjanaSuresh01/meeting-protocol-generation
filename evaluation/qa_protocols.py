"""Runs a layered quality check over every generated protocol of one or more deliverable arms.

For each meeting it reads the arm's outputs_deliverable/<arm>/<meeting>/protocol_pro.md, the
Canary-1B-v2 transcript (outputs_v11_final/canary_diar/<meeting>/transcript_speakers.txt) and
the previous protocol (term_source.PREVIOUS), and reports each finding as critical or warning,
with the evidence. It is a diagnostic only and changes no file it reads.

Layers, in the order in which defects propagate through the pipeline:
  1  TRANSCRIPT   empty turns, ASR loops (a phrase of 6 or more words repeated 3 or more
                  times in one turn), turns that look English.
  2  STRUCTURE    missing '##' person or '###' topic headings, raw SPEAKER_xx labels, headings
                  glued onto a bullet, leaked HTML, unknown bracket markers, a person with two
                  sections, an entirely flat bullet list, bullets with no topic heading.
  3  CONTENT      repeated words, near-duplicate bullets, English sentences, figures that
                  appear in neither source, bullets that end unfinished.
  4  NAMES        known misspellings, the same entity spelled two ways in one document, and
                  persons whose surname appears in neither source.
  5  REFERENCE    against the human-written minutes (Meeting_<date>.md) where they exist:
                  section order and the set of people.
  6  STYLE        bare noun-phrase bullet openers (style_flag.py) and terms that stand for
                  their own opposite (acronym_swap_check.py).

Pitfalls the checks avoid:
  * German capitalizes every noun, so a capitalized word is not treated as a name. Only
    title-led names (Herr/Frau/Prof/Dr + X) and a known first name + surname are matched.
  * Spoken German says numbers as words: "acht vier vier drei" is 8443, "zehn Komma drei" is
    10,3. A digit missing from the transcript is therefore not evidence of invention.
  * Paraphrase is not hallucination. A bullet whose words are absent from the source is usually
    the model's own nominalization ("Beantragung", "Behebung"), not a fabrication.
  * "Stimmt ... zu" and "sicherte ... zu" are correct German separable verbs, not truncation.
  * A '###' heading with no bullets also occurs in the human-written minutes, at a rate of
    2.9-9.7 %, so it is not flagged.

    python qa_protocols.py v10
    python qa_protocols.py v9 v10          # two arms side by side (default arm: v9)

Writes outputs_deliverable/qa_report.json and prints the findings per arm, severity and layer.
"""
from __future__ import annotations

import difflib
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent
MEET = ["march09", "april13", "april27", "may11", "may18", "june01", "june15", "june22"]
HUMAN = {"april27": "Meeting_2026-04-27.md", "may11": "Meeting_2026-05-11.md",
         "may18": "Meeting_2026-05-18.md", "june01": "Meeting_2026-06-01.md",
         "june15": "Meeting_2026-06-15.md", "june22": "Meeting_2026-06-22.md"}

NOM = re.compile(r"^(?:[A-ZÄÖÜ][a-zäöüß]+)?[A-ZÄÖÜ][a-zäöüß]+(?:ung|ierung|ation|erung|ens)\b")
# reliable name shapes only (see the module docstring)
KNOWN_FIRST = (r"Sebastian|Thomas|Vincent|Johannes|Ren[ée]|Osman|Noah|Artem|Vinayak|Rassul|Frank"
               r"|Munir|Jens|Stefan|Natalie|Ron|Marcus|Alexander|Jonas|Jan|Florian|J[üu]rgen")
PERSON = re.compile(rf"\b(?:(?:Herrn?|Frau|Prof\.?|Dr\.?)\s+([A-ZÄÖÜ][a-zäöüß]{{2,}})"
                    rf"|({KNOWN_FIRST})\s+([A-ZÄÖÜ][a-zäöüß]{{2,}}))")
# (misspelling, spelling used in the human-written minutes). Both forms in one document is
# critical; the misspelling alone is a warning. "Pick"/"Pico" is deliberately not a pair: in the
# transcripts "Pick" is part of the term "Frame Pick Tool", a correct word, not a mishearing of
# the Pico headset (see name_flag_context_check.py). The pair whose two forms share a first name
# is inferred from the shared context, not confirmed from the audio.
NAME_PAIRS = [("Pau SM", "PowSM"), ("Vilo", "Wilo"), ("Heikon", "HAICON"), ("Meincker", "Mainka"),
              ("Lloyd", "Leuoth"), ("Sebastian Miller", "Sebastian Leuoth"), ("Olaf", "OlaPh"),
              ("Gwen", "Qwen"), ("Redik", "Redig"), ("Feintuning", "Finetuning")]


def rd(p: Path) -> str:
    p = Path(p)
    if not p.exists():
        return ""
    if p.suffix == ".docx":
        import docx
        return "\n".join(x.text for x in docx.Document(str(p)).paragraphs)
    return p.read_text(encoding="utf-8", errors="replace")


def bullets(text: str) -> list[tuple[str, str, int, str]]:
    """Return (person, topic, indent level, body) for every bullet, with marker tags removed from the body."""
    out, who, top = [], None, None
    for line in text.splitlines():
        if line.startswith("## ") and not line.startswith("###"):
            who, top = line[3:].strip(), None
        elif line.startswith("### "):
            top = line[4:].strip()
        else:
            m = re.match(r"^(\s*)[*-]\s+(\S.*)$", line)
            if m:
                out.append((who, top, len(m.group(1)) // 2,
                            re.sub(r"\[(ÜBERTRAG|PRÜFEN|Bild)\]", "", m.group(2)).strip()))
    return out


def layer1_transcript(mid: str, F: list) -> None:
    tx = rd(REPO / f"outputs_v11_final/canary_diar/{mid}/transcript_speakers.txt")
    if not tx:
        F.append((mid, "TRANSCRIPT", "critical", "no transcript on disk", ""))
        return
    turns = re.findall(r"\[[\d:]+\]\s+(\S+):\s*(.*)", tx)
    empty = sum(1 for _, t in turns if not t.strip())
    if empty:
        F.append((mid, "TRANSCRIPT", "warn", f"{empty} empty turns of {len(turns)}", ""))
    # ASR looping: the same 6+ word phrase repeated 3+ times inside one turn
    for who, t in turns:
        for m in re.finditer(r"((?:\b\w+\b[ ,]+){6,}?)\1{2,}", t):
            F.append((mid, "TRANSCRIPT", "critical", "ASR loop (phrase repeated 3x)",
                      m.group(0)[:90]))
            break
    en = [t for _, t in turns
          if len(re.findall(r"\b(the|and|is|are|with|that|this|have|will|thank you)\b", t, re.I)) >= 3]
    if en:
        F.append((mid, "TRANSCRIPT", "warn", f"{len(en)} turns look English", en[0][:90]))


def layer2_structure(mid: str, arm: str, text: str, F: list) -> None:
    people = re.findall(r"(?m)^## (.+)$", text)
    topics = re.findall(r"(?m)^### (.+)$", text)
    if not people:
        F.append((mid, "STRUCTURE", "critical", "no '##' person sections at all", ""))
        return
    if not topics:
        F.append((mid, "STRUCTURE", "critical", "no '###' topic headings (the required third level)", ""))
    if re.search(r"(?m)^## SPEAKER_\d+", text):
        F.append((mid, "STRUCTURE", "critical", "raw diarization label printed as a person", ""))
    for m in re.finditer(r"[^\s#]#{2,4}\s", text):
        F.append((mid, "STRUCTURE", "critical", "heading glued onto a bullet",
                  text[max(0, m.start() - 50):m.end() + 40].replace("\n", " ")))
    # HTML tags and entities (&amp;, &gt;, <br>, <img ...>) that leaked from the exported
    # minutes. A placeholder inside backticks, such as `<image1>` naming a prompt keyword, is
    # content and is not counted.
    _spans = [(mm.start(), mm.end()) for mm in re.finditer("`[^`]*`", text)]
    for m in re.finditer(r"<[a-zA-Z][^>]*>|&(?:quot|amp|nbsp|gt|lt);", text):
        if any(a <= m.start() and m.end() <= b for a, b in _spans):
            continue                      # inside backticks: content, not markup
        F.append((mid, "STRUCTURE", "critical", "HTML leaked from the reference export",
                  text[max(0, m.start() - 40):m.end() + 40].replace("\n", " ")))
    for m in re.finditer(r"\[(?!ÜBERTRAG\]|PRÜFEN\]|Bild\])([A-ZÄÖÜ][^\]]{0,30})\]", text):
        F.append((mid, "STRUCTURE", "warn", "invented bracket marker", m.group(0)))
    dup = [p for p, c in Counter(people).items() if c > 1]
    if dup:
        F.append((mid, "STRUCTURE", "critical", "a person has two sections", str(dup)))
    bs = bullets(text)
    if bs and not any(lvl >= 1 for _, _, lvl, _ in bs):
        F.append((mid, "STRUCTURE", "critical", "every bullet is flat (no nesting)", ""))
    # a bullet under a person heading but before any topic heading does not fit the target format
    stray = sum(1 for who, top, _, _ in bs if who and top is None)
    if stray:
        F.append((mid, "STRUCTURE", "warn", f"{stray} bullets sit under a person with no topic", ""))


def layer3_content(mid: str, arm: str, text: str, prev_text: str, F: list) -> None:
    tx = rd(REPO / f"outputs_v11_final/canary_diar/{mid}/transcript_speakers.txt")
    src = (tx + "\n" + prev_text)
    bs = bullets(text)

    for _, _, _, b in bs:
        m = re.search(r"\b(\w{3,})\s+\1\b", b, re.I)
        if m:
            F.append((mid, "CONTENT", "warn", "word repeated", b[:90]))

    seen: list[str] = []
    for _, _, _, b in bs:
        if len(b.split()) < 5:
            continue
        for prev in seen:
            if difflib.SequenceMatcher(None, b.lower(), prev.lower()).ratio() > 0.88:
                F.append((mid, "CONTENT", "warn", "near-duplicate bullet", b[:90]))
                break
        seen.append(b)

    for _, _, _, b in bs:
        if len(re.findall(r"\b(the|and|with|is a|will be|has been|thank you)\b", b, re.I)) >= 2:
            F.append((mid, "CONTENT", "critical", "English sentence in a German protocol", b[:90]))

    # numbers: a figure of 2-4 digits is traced when the source has it as digits or as spoken
    # German words (german_numbers: "acht vier vier drei" -> 8443, "fünfhundertdreißig" -> 530).
    # number_trace.py applies the same test.
    import german_numbers as GN
    src_digits = set(re.findall(r"\d+", src))
    src_spoken = GN.figures(src)
    for _, _, _, b in bs:
        for nm in re.finditer(r"(?<![\w.,])(\d{2,4})(?![\w.,])", b):
            v = nm.group(1)
            if v in src_digits or v in src_spoken:
                continue
            F.append((mid, "CONTENT", "warn", f"figure '{v}' not found in either source "
                      "(neither as digits nor as spoken words) — read it", b[:90]))

    # truncation: the bullet ends on a conjunction or article, or on '-', '(' or ','. Separable-verb
    # particles ('zu', 'an', 'auf', 'mit', 'vor', 'nach') are correct German endings and are not listed.
    for _, _, _, b in bs:
        if re.search(r"\b(und|oder|die|der|das|dem|den|des|eine|einen|einem)\s*$", b, re.I) \
                or b.endswith(("-", "(", ",")):
            F.append((mid, "CONTENT", "warn", "bullet ends unfinished", b[:90]))


def layer4_names(mid: str, arm: str, text: str, prev_text: str, F: list) -> None:
    tx = rd(REPO / f"outputs_v11_final/canary_diar/{mid}/transcript_speakers.txt")
    src = (tx + "\n" + prev_text).lower()
    for wrong, right in NAME_PAIRS:
        w = len(re.findall(rf"\b{re.escape(wrong)}\b", text))
        r = len(re.findall(rf"\b{re.escape(right)}\b", text))
        if w and r:
            F.append((mid, "NAMES", "critical",
                      f"'{wrong}' ×{w} AND '{right}' ×{r} — one entity, two spellings", ""))
        elif w:
            F.append((mid, "NAMES", "warn", f"'{wrong}' ×{w} (reference: '{right}')", ""))
    for m in PERSON.finditer(text):
        parts = [x for x in m.groups() if x]
        sur = parts[-1]
        if sur.lower() not in src:
            F.append((mid, "NAMES", "critical", f"person '{' '.join(parts)}' in neither source", ""))


def layer5_reference(mid: str, arm: str, text: str, F: list) -> None:
    if mid not in HUMAN or not (REPO / HUMAN[mid]).exists():
        return
    from evaluate_speaker_attribution import person_key
    his = [person_key(re.sub(r"<[^>]+>", "", x.group(1)).strip("* "))
           for x in re.finditer(r"(?m)^\*\s+\*\*(.+?)\*\*", rd(REPO / HUMAN[mid]))]
    ours = [person_key(x) for x in re.findall(r"(?m)^## (.+)$", text)]
    ours_f = [o for o in ours if o in his]
    his_f = [h for h in his if h in ours]
    if ours_f != his_f:
        # number of person pairs in the wrong relative order
        idx = {p: i for i, p in enumerate(ours_f)}
        bad = sum(1 for i in range(len(his_f)) for j in range(i + 1, len(his_f))
                  if his_f[i] in idx and his_f[j] in idx and idx[his_f[i]] > idx[his_f[j]])
        F.append((mid, "REFERENCE", "warn" if bad <= 1 else "critical",
                  f"section order differs from the reference ({bad} pair(s) inverted)",
                  f"reference {his_f} / draft {ours_f}"))
    missing = [h for h in his if h not in ours]
    if missing:
        F.append((mid, "REFERENCE", "warn", f"people in the reference but not in the draft: {missing}", ""))


def layer6_style(mid: str, arm: str, text: str, F: list) -> None:
    """Flag bare noun-phrase bullet openers and terms that stand for their own opposite.

    A bullet counts as a bare noun-phrase opener when it opens with a nominalized noun and
    style_flag.py finds no finite verb (spaCy VVFIN/VAFIN/VMFIN) anywhere in it, so a sentence
    such as "Installation pausiert bis...", where the nominalized word is the subject of a real
    verb, is not flagged. Openers are warnings, since they are a style defect, not a correctness
    one; nothing is rewritten. A term that stands for its own opposite (acronym_swap_check.py)
    is critical."""
    import style_flag as SF
    hits = SF.scan(text)
    if hits:
        F.append((mid, "STYLE", "warn",
                  f"{len(hits)} bare noun-phrase opener(s) (the reference has 1.24 % of these)",
                  hits[0][:90]))

    import acronym_swap_check as AC
    for body, why in AC.flag(text):
        F.append((mid, "STYLE", "critical", "a term used to mean its own opposite", f"{body[:80]}  ({why})"))


def audit(arm: str) -> list:
    from term_source import PREVIOUS
    F: list = []
    for mid in MEET:
        p = REPO / "outputs_deliverable" / arm / mid / "protocol_pro.md"
        if not p.exists():
            F.append((mid, "RUN", "critical", "protocol missing — the run did not finish", ""))
            continue
        text = rd(p)
        prev_text = rd(REPO / PREVIOUS[mid])
        layer1_transcript(mid, F)
        layer2_structure(mid, arm, text, F)
        layer3_content(mid, arm, text, prev_text, F)
        layer4_names(mid, arm, text, prev_text, F)
        layer5_reference(mid, arm, text, F)
        layer6_style(mid, arm, text, F)
    return F


def main() -> int:
    arms = sys.argv[1:] or ["v9"]
    allf = {}
    for arm in arms:
        F = audit(arm)
        allf[arm] = F
        crit = [f for f in F if f[2] == "critical"]
        warn = [f for f in F if f[2] == "warn"]
        print(f"\n{'=' * 92}\nARM {arm}:  {len(crit)} critical, {len(warn)} warnings\n{'=' * 92}")
        for sev in ("critical", "warn"):
            rows = [f for f in F if f[2] == sev]
            if not rows:
                continue
            print(f"\n--- {sev.upper()} ({len(rows)})")
            for layer in ("RUN", "TRANSCRIPT", "STRUCTURE", "CONTENT", "NAMES", "REFERENCE", "STYLE"):
                sub = [r for r in rows if r[1] == layer]
                if not sub:
                    continue
                print(f"  [{layer}]")
                for mid, _, _, what, ev in sub[:14]:
                    print(f"     {mid:9s} {what}")
                    if ev:
                        print(f"               {ev[:96]}")
                if len(sub) > 14:
                    print(f"     … and {len(sub) - 14} more")
    out = REPO / "outputs_deliverable" / "qa_report.json"
    out.write_text(json.dumps(
        {"built": "2026-09-29", "arms": arms,
         "findings": {a: [{"meeting": m, "layer": l, "severity": s, "what": w, "evidence": e}
                          for m, l, s, w, e in f] for a, f in allf.items()}},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwrote -> {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
