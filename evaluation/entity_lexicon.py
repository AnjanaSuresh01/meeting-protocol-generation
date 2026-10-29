"""Builds the domain glossary of canonical names and a variant -> canonical map for misheard names.

It provides two things:

  1. The glossary: the canonical spellings in CANONICAL (AI models, projects, partners, people)
     that occur in the team's own protocol documents (Meeting_2026-*.md, Meeting-Protocol-*.docx).
     The minutes serve as free supervision from an artifact the team already keeps, like the
     attendance record used in speaker_identity.py.

  2. A corrector: a variant -> canonical map in which every entry was observed at an aligned audio
     position, written by one system where another system wrote the name correctly, plus a
     phonetic pass (Kölner Phonetik) for names no system spells correctly. correct() applies the
     map to a text. The hypotheses are read from goldset/clip_asr_v10/<model>__v10.json.

Leave-one-meeting-out (LOMO): the minutes are also the evaluation reference, so a glossary built
from a meeting's own minutes and then used on that meeting would leak the reference.
glossary(exclude_meeting=...) therefore drops that meeting's own minutes (OWN_MINUTES) and builds
from the other protocol documents. Names recur week to week, so little is lost. In deployment no
exclusion applies.

Variant filter: string similarity alone is not enough, because ordinary German words are
string-similar to short English names ("aber" ~ BERT, "so" ~ ITSO, "weil" ~ Wilo) and no threshold
separates them. A variant is admitted only if all of these hold:

    observed   it stood at an aligned position where another system wrote the name
    similar    squashed similarity to the canonical >= SIM (it is a mishearing, not a synonym)
    unwritten  the team never writes that string in its own minutes. This stop-list keeps
               ordinary German safe: "so", "hier", "Thema", "Leute" and "Projekt" are all
               written in the minutes, so none of them can ever be rewritten to a name.
    long       >= 4 characters
    distinct   it is not itself a canonical name: GemmaX is never "corrected" to Gemma
    not a      it is not a truncation or a compound of the name. "Project" for OpenProject and
    fragment   "Populi" for VoxPopuli are partial recognitions and expanding them is a guess;
               "Personaplex-Server" carries a word the canonical does not. Containment either way
               is rejected unless the two differ by at most one character and neither is
               hyphenated, which keeps "Sarah" -> SARA.
    unambiguous  no other canonical name matches the string better than the anchored one

Every admitted entry is written to goldset/entity_variants.json with its counts, so the map can be
audited line by line.

    python entity_lexicon.py            # build, print the map, write the JSON
    python entity_lexicon.py --show     # also print rejected variants and the phonetic pass
"""

from __future__ import annotations

import difflib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent
G = REPO / "goldset"
MODELS = ["canary_diar", "cohere", "granite41",
          "whisper_large_v3", "whisper_turbo", "whisper_turbo_de"]

SIM = 0.70          # squashed similarity to the canonical name
MINLEN = 4

# Which protocol document belongs to which audio meeting — the file excluded under LOMO.
OWN_MINUTES = {
    "march09": "Meeting-Protocol-FG-MM-AI_2026-03-09.docx",
    "april27": "Meeting_2026-04-27.md",
    "may11":   "Meeting_2026-05-11.md",
    "may18":   "Meeting_2026-05-18.md",
    "june01":  "Meeting_2026-06-01.md",
    "june15":  "Meeting_2026-06-15.md",
    "june22":  "Meeting_2026-06-22.md",
}

# ---------------------------------------------------------------------------------------
# The canonical names, each taken from the minutes. Two- and three-letter acronyms are
# deliberately absent: they collide with German function words and cannot be corrected
# safely.
# ---------------------------------------------------------------------------------------
CANONICAL: dict[str, list[str]] = {
    "ai_model": [
        "SparkTTS", "ByT5", "GemmaX", "Gemma", "VoxCPM", "NeuTTS", "DeepSeek", "Qwen",
        "QwenImage", "Whisper", "Parakeet", "Conformer", "LlamaIndex", "Ollama", "GGUF",
        "LoRA", "Z-Image", "Granite", "Canary", "Kokoro", "Orpheus", "ChatGPT",
    ],
    "project": [
        "M4-SKI", "SARA", "LISA", "OlaPh", "ResilienceLab", "SITUATE", "LIFA", "JCRG",
        "TORTILLA", "REDIG", "PersonaPlex", "ObjectCount", "ParseBench", "ModelHub",
        "Re-LAION", "VoxPopuli", "FineVision", "JupyterHub", "Keycloak", "OpenProject",
        "OpenXR", "RealityScan", "EmergentTTS",
    ],
    "partner": [
        "Kreul", "Matra", "Wilo", "Witt-Gruppe", "Leuoth", "iisys", "ITSO", "Dienstbier",
        "VISAPP", "Müller-Czygan", "Wieland-Reß", "Wülfert",
    ],
    # People, so that misheard names of the persons a protocol credits can be corrected. Every
    # entry below is taken from the team's own minutes. Names under four characters and ones
    # that collide with ordinary German words are deliberately left out: they cannot be
    # corrected safely and a wrong correction costs more than a miss.
    "person": [
        "Peinl", "Tischler", "Wirth", "Osman", "Görner", "Schaller", "Mainka", "Artem",
        "Nestel", "Edelmann", "Rosner", "Hoffmann", "Jayan", "Suresh", "Kannan", "Vinayak",
    ],
}
ALL_NAMES = sorted({n for v in CANONICAL.values() for n in v})
GROUP = {n: g for g, v in CANONICAL.items() for n in v}


_UML = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss",
                      "Ä": "ae", "Ö": "oe", "Ü": "ue"})


_KOELN = {**{c: "0" for c in "aeijouyäöü"}, **{c: "1" for c in "bp"},
          **{c: "2" for c in "dt"}, **{c: "3" for c in "fvw"},
          **{c: "4" for c in "gkq"}, "l": "5", **{c: "6" for c in "mn"},
          "r": "7", **{c: "8" for c in "sz"}, "c": "4", "x": "48", "h": ""}


def koelner(word: str) -> str:
    """Kölner Phonetik, the German soundex. Two strings that a German ear hears the same get
    the same code.

    It reaches names that alignment cannot. mine_variants() finds variants at positions where
    one system spelled a name correctly; a name every system misspells has no such position.
    Phonetics needs no correct spelling anywhere: 'OlaPh' and 'Olaf' both code to 053, so the
    corpus can be searched for what a name sounds like instead of what some system typed.
    Chosen over Soundex/Metaphone because those are built for English and this corpus is
    German speech.
    """
    w = re.sub(r"[^a-z]", "", word.translate(_UML).lower())
    out = []
    for i, ch in enumerate(w):
        nxt = w[i + 1] if i + 1 < len(w) else ""
        prv = w[i - 1] if i else ""
        if ch == "p":
            code = "3" if nxt == "h" else "1"
        elif ch in "dt":
            code = "8" if nxt in "csz" else "2"
        elif ch == "c":
            if not prv:
                code = "4" if nxt in "ahkloqrux" else "8"
            elif prv in "sz":
                code = "8"
            else:
                code = "4" if nxt in "ahkoqux" else "8"
        elif ch == "x":
            code = "8" if prv in "ckq" else "48"
        else:
            code = _KOELN.get(ch, "")
        for c in code:
            if not out or out[-1] != c:
                out.append(c)
    return (out[0] if out else "") + "".join(c for c in out[1:] if c != "0")


def squash(s: str) -> str:
    """Comparable letters+digits, with German transliterated rather than deleted.

    Umlauts and ß are transliterated (ä -> ae, ö -> oe, ü -> ue, ß -> ss) before everything
    except letters and digits is removed, so 'Größe' and 'Groesse' squash to the same string
    instead of one losing letters and looking like a truncation of the other."""
    return re.sub(r"[^a-z0-9]", "", s.translate(_UML).lower())


def tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-zÄÖÜäöüß0-9][\wÄÖÜäöüß\.\-]*", text)


def doc_text(name: str) -> str:
    p = REPO / name
    if not p.exists():
        return ""
    if p.suffix.lower() == ".docx":
        import docx
        return "\n".join(x.text for x in docx.Document(str(p)).paragraphs)
    return p.read_text(encoding="utf-8", errors="replace")


def _forms(text: str) -> set[str]:
    tk = [squash(t) for t in tokens(text)]
    out = set(tk)
    for n in (2, 3):
        out |= {"".join(tk[i:i + n]) for i in range(len(tk) - n + 1)}
    out.discard("")
    return out


def glossary(exclude_meeting: str | None = None) -> list[str]:
    """Canonical names evidenced in the team's protocol documents, minus, under LOMO, the
    minutes of the meeting being processed."""
    skip = OWN_MINUTES.get(exclude_meeting or "", "")
    docs = sorted(REPO.glob("Meeting_2026-*.md")) + sorted(REPO.glob("Meeting-Protocol-*.docx"))
    text = "\n".join(doc_text(p.name) for p in docs if p.name != skip)
    f, tk = _forms(text), [squash(t) for t in tokens(text)]
    out = []
    for n in ALL_NAMES:
        e = squash(n)
        if e in f or (len(e) >= 5 and any(s.startswith(e) or s.endswith(e) for s in tk)):
            out.append(n)
    return out


# ---------------------------------------------------------------------------------------
# variant mining
# ---------------------------------------------------------------------------------------
def _segment_hyps() -> dict[str, dict[str, str]]:
    out = {}
    for m in MODELS:
        p = G / "clip_asr_v10" / f"{m}__v10.json"
        if p.exists():
            out[m] = json.loads(p.read_text(encoding="utf-8"))
    return out


def mine_variants(verbose: bool = False) -> dict:
    """Every token that stood where another system wrote a name, with the counts that justify
    admitting or rejecting it."""
    import jiwer

    hyps = _segment_hyps()
    if not hyps:
        raise SystemExit("goldset/clip_asr_v10/ not found — cannot mine variants")
    esq = {n: squash(n) for n in ALL_NAMES}
    longest_first = sorted(ALL_NAMES, key=lambda n: -len(esq[n]))
    canon_set = set(esq.values())

    # Counts are keyed on the squashed form, because that is what correct() looks up: surface
    # forms that squash alike ("M-V-S-K-I", "MVSKI") share one count and pass or fail the
    # filter together.
    corpus = Counter()
    for h in hyps.values():
        for t in h.values():
            for tok in tokens(t):
                corpus[squash(tok)] += 1

    # The stop-list: every word the team writes in their own protocol documents. A string that
    # appears here is legitimate German (or a name already spelled right) and is never rewritten.
    docs = sorted(REPO.glob("Meeting_2026-*.md")) + sorted(REPO.glob("Meeting-Protocol-*.docx"))
    written = {squash(t) for p in docs for t in tokens(doc_text(p.name))}
    written.discard("")

    observed: dict[str, Counter] = defaultdict(Counter)
    right: Counter = Counter()
    seg_ids = sorted(set().union(*(set(v) for v in hyps.values())))
    for sid in seg_ids:
        raws = {m: tokens(hyps[m].get(sid, "")) for m in hyps}
        norms = {m: [squash(t) for t in raws[m]] for m in raws}
        anchor = {}
        for m, nt in norms.items():
            for i, s in enumerate(nt):
                # longest canonical first: a token reading "gemmax" anchors GemmaX, never Gemma
                for n in longest_first:
                    if s == esq[n]:
                        anchor.setdefault(n, (m, i))
                        break
        for n, (pm, pi) in anchor.items():
            pivot = " ".join(norms[pm])
            for m in norms:
                if m == pm:
                    right[n] += 1
                    continue
                other = " ".join(norms[m])
                if not other.strip() or not pivot.strip():
                    continue
                tok = None
                for ch in jiwer.process_words([pivot], [other]).alignments[0]:
                    if ch.ref_start_idx <= pi < ch.ref_end_idx:
                        j = ch.hyp_start_idx + (pi - ch.ref_start_idx)
                        if ch.type == "equal":
                            tok = raws[m][j] if j < len(raws[m]) else None
                        elif ch.type == "substitute":
                            tok = raws[m][j] if j < len(raws[m]) else None
                        break
                if tok is None:
                    continue
                if squash(tok) == esq[n]:
                    right[n] += 1
                else:
                    observed[n][tok] += 1

    admitted, rejected = {}, []
    for n, ctr in observed.items():
        # fold surface forms onto the squashed key the corrector actually uses
        folded: dict[str, tuple[str, int]] = {}
        for var, cnt in ctr.items():
            sv = squash(var)
            if not sv:
                continue
            prev_var, prev_cnt = folded.get(sv, (var, 0))
            folded[sv] = (var if cnt >= prev_cnt else prev_var, prev_cnt + cnt)
        for sv, (var, cnt) in folded.items():
            sn = esq[n]
            sim = difflib.SequenceMatcher(None, sn, sv).ratio()
            freq = corpus.get(sv, cnt)
            share = cnt / freq if freq else 0.0
            why = []
            if len(sv) < MINLEN:
                why.append("too short")
            if sim < SIM:
                why.append(f"similarity {sim:.2f}<{SIM}")
            if sv in written:
                why.append("the team writes this word")
            if sv in canon_set:
                why.append("is itself a canonical name")
            if (sv in sn or sn in sv) and not (
                    abs(len(sv) - len(sn)) <= 1 and "-" not in var and "-" not in n):
                why.append("truncation or compound, not a mis-hearing")
            # Ambiguity guard: if some OTHER canonical name matches this string better than the
            # one it was anchored to, the correction would be a coin flip. "GEMA-X" is closer to
            # GemmaX than to Gemma, so rewriting it to Gemma would delete a real distinction.
            best = max(ALL_NAMES, key=lambda o: difflib.SequenceMatcher(
                None, esq[o], sv).ratio())
            if best != n and difflib.SequenceMatcher(None, esq[best], sv).ratio() > sim:
                why.append(f"closer to {best}")
            if why:
                rejected.append((n, var, cnt, freq, sim, "; ".join(why)))
                continue
            prev = admitted.get(var)
            if prev is None or cnt > prev["count"]:
                admitted[var] = {"canonical": n, "count": cnt, "corpus_freq": freq,
                                 "similarity": round(sim, 3), "share": round(share, 3),
                                 "group": GROUP[n]}
    if verbose:
        print(f"\nREJECTED ({len(rejected)}) — the filter doing its job\n")
        for n, var, cnt, freq, sim, why in sorted(rejected, key=lambda r: -r[2])[:30]:
            print(f"  {var:<22} as {n:<14} seen {cnt:>3}x of {freq:>5} — {why}")
    # ---- PHONETIC PASS: the names alignment can never see -------------------------------
    # A name every system mangles has no correctly-spelled anchor, so the loop above cannot
    # find it. Here the corpus is searched for tokens that sound like a glossary name instead.
    # The same three guards apply, and they matter more here because phonetic equality is a
    # looser relation than string similarity: the team must never write the token, it must be
    # rare in speech, and it must not already be a canonical name. A fourth guard requires the
    # name itself to be essentially absent from the corpus, so that this pass only fires where
    # the aligned pass had nothing to work with.
    # Phonetic equality on its own is far too loose: it matches German words that merely sound
    # like short product names ("aussieht" for iisys, "kaufen" for Qwen). Sounding alike is
    # necessary but not sufficient; the spelling has to be close as well (PHON_SIM).
    PHON_MAXFREQ = 60
    # 0.65 rather than 0.70: "olaf" vs "olaph" scores 0.667, and OlaPh is wrong in all six
    # systems and therefore unreachable by alignment. A threshold that drops it to keep a few
    # German words out is the wrong trade when those words can be listed in PHON_DENY.
    PHON_SIM = 0.65
    # Tokens that pass both tests and are still ordinary German. An explicit, auditable
    # exception list holds them out, instead of a tighter threshold that would also drop real
    # corrections.
    PHON_DENY = {"schlauer", "geeinigt", "projekts", "rollen", "holen", "allein", "wenige",
                 "schueler", "nettes", "prakt", "kannen", "yellow", "source", "panel",
                 "gewinn", "kaufen", "fange", "modus", "werte", "warte", "worte", "vierte",
                 "zuerst", "aussieht", "tshirt", "whatsapp", "maing", "kennen", "gemixt"}
    codes: dict[str, list[str]] = defaultdict(list)
    for n in ALL_NAMES:
        c = koelner(n)
        if len(c) >= 3:
            codes[c].append(n)
    phonetic = {}
    for sv, freq in corpus.items():
        if len(sv) < MINLEN or sv in written or sv in canon_set or sv in admitted:
            continue
        if freq > PHON_MAXFREQ:
            continue
        cand = codes.get(koelner(sv))
        if not cand:
            continue
        n = min(cand, key=lambda x: abs(len(esq[x]) - len(sv)))
        sn = esq[n]
        if corpus.get(sn, 0) > 3:            # the aligned pass could already reach this name
            continue
        if sv in sn or sn in sv or sv in PHON_DENY:
            continue
        if abs(len(sv) - len(sn)) > 3:
            continue
        sim = difflib.SequenceMatcher(None, sn, sv).ratio()
        if sim < PHON_SIM:
            continue
        phonetic[sv] = {"canonical": n, "count": freq, "corpus_freq": freq,
                        "similarity": round(sim, 3),
                        "share": 1.0, "group": GROUP[n], "via": "phonetic",
                        "code": koelner(n)}
    if verbose:
        print(f"\nPHONETIC PASS admitted {len(phonetic)}\n")
        for k, v in sorted(phonetic.items(), key=lambda kv: -kv[1]["count"])[:30]:
            print(f"  {k:<22} -> {v['canonical']:<14} x{v['count']:<4} code {v['code']}")

    for k, v in phonetic.items():
        admitted.setdefault(k, v)

    return {"map": admitted, "right": dict(right),
            "params": {"SIM": SIM, "MINLEN": MINLEN, "stop_list": len(written),
                       "phonetic": len(phonetic), "PHON_MAXFREQ": PHON_MAXFREQ}}


def load_variants() -> dict[str, str]:
    p = G / "entity_variants.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text(encoding="utf-8"))
    return {squash(v): rec["canonical"] for v, rec in d["map"].items()}


_PUNCT = re.compile(r"^([^\wÄÖÜäöüß]*)(.*?)([^\wÄÖÜäöüß]*)$", re.S)


def correct(text: str, mapping: dict[str, str], allow: set[str] | None = None
            ) -> tuple[str, Counter]:
    """Rewrite known mishearings to the canonical spelling, punctuation preserved.

    `allow` restricts corrections to a given glossary (used for leave-one-meeting-out): a name
    the glossary does not contain is left exactly as the ASR produced it."""
    log: Counter = Counter()
    out = []
    for word in text.split(" "):
        m = _PUNCT.match(word)
        pre, core, post = (m.group(1), m.group(2), m.group(3)) if m else ("", word, "")
        canon = mapping.get(squash(core)) if core else None
        if canon and (allow is None or canon in allow) and squash(core) != squash(canon):
            log[f"{core} -> {canon}"] += 1
            out.append(pre + canon + post)
        else:
            out.append(word)
    return " ".join(out), log


def main() -> None:
    show = "--show" in sys.argv
    print("ENTITY LEXICON")
    print("=" * 92)
    full = glossary()
    print(f"\nglossary, all documents: {len(full)} names")
    for g in CANONICAL:
        names = [n for n in full if GROUP[n] == g]
        print(f"  {g:<10} {len(names):>2}  {', '.join(names)}")

    print("\nleave-one-meeting-out sizes (what generation is actually allowed to use)")
    for mid in OWN_MINUTES:
        lomo = glossary(mid)
        lost = sorted(set(full) - set(lomo))
        print(f"  {mid:<10}{len(lomo):>3} names" + (f"   drops: {', '.join(lost)}" if lost else ""))

    res = mine_variants(verbose=show)
    out = G / "entity_variants.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\nADMITTED VARIANT -> CANONICAL ({len(res['map'])} entries)\n")
    print(f"  {'variant':<24}{'canonical':<16}{'seen':>6}{'corpus':>8}{'sim':>7}{'share':>8}")
    for var, r in sorted(res["map"].items(), key=lambda kv: -kv[1]["count"]):
        print(f"  {var:<24}{r['canonical']:<16}{r['count']:>6}{r['corpus_freq']:>8}"
              f"{r['similarity']:>7.2f}{r['share']:>8.0%}")
    print(f"\nwrote -> {out.relative_to(REPO)}")
    print("\nEvery row above is a position where one system wrote the name and another wrote this")
    print("instead. Run with --show to see what the filter threw out and why.")


if __name__ == "__main__":
    main()
