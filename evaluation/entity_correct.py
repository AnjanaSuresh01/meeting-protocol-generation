"""Spelling correction of names and terms, plus squash and Kölner Phonetik helpers.

build_map() reads one meeting's previous document (term_source), its clip decodes in
goldset/clip_asr_v10/ and its transcripts in outputs_v10/ (paths relative to this folder) and
returns an "apply" and a "suggest" map. Aligned variants and phonetic matches of
orthographically marked terms go into "apply", which correct() applies to text; other phonetic
matches go into "suggest", because ordinary German words often sound like short names and no
similarity threshold separates the two.

Usage (--why prints every map entry):
    python entity_correct.py [--why]            # all meetings; writes goldset/entity_maps_auto.json
    python entity_correct.py <meeting> [--why]  # one meeting, a key of term_source.PREVIOUS
"""

from __future__ import annotations

import difflib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from term_source import (
    GERMAN_COMMON, PREVIOUS, extract, targets_for, glossary_for, tokens, read_document,
    person_candidates,
)

REPO = Path(__file__).resolve().parent
G = REPO / "goldset"
MODELS = ["canary_diar", "cohere", "granite41",
          "whisper_large_v3", "whisper_turbo", "whisper_turbo_de"]

_INTERNAL_CAP = re.compile(r"[a-zäöüß][A-ZÄÖÜ]")


def _UNUSUAL_CAPS(form: str) -> bool:
    """True if the word (3+ characters, punctuation stripped) is all-letter ALL CAPS or has a
    capital after a lowercase letter.
    """
    core = form.strip(".,;:()[]\"'")
    if len(core) < 3 or not any(c.isalpha() for c in core):
        return False
    return (core.isupper() and core.isalpha()) or bool(_INTERNAL_CAP.search(core))


SIM = 0.70  # aligned pass: minimum SequenceMatcher ratio
PHON_SIM = 0.65  # phonetic pass: minimum SequenceMatcher ratio
MINLEN = 4

_UML = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss",
                      "Ä": "ae", "Ö": "oe", "Ü": "ue"})
_KOELN = {**{c: "0" for c in "aeijouy"}, **{c: "1" for c in "bp"},
          **{c: "2" for c in "dt"}, **{c: "3" for c in "fvw"},
          **{c: "4" for c in "gkq"}, "l": "5", **{c: "6" for c in "mn"},
          "r": "7", **{c: "8" for c in "sz"}, "c": "4", "x": "48", "h": ""}


def squash(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.translate(_UML).lower())


def koelner(word: str) -> str:
    w = re.sub(r"[^a-z]", "", word.translate(_UML).lower())
    out: list[str] = []
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


def _hyps(mid: str) -> dict[str, dict[str, str]]:
    out = {}
    for m in MODELS:
        p = G / "clip_asr_v10" / f"{m}__v10.json"
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            out[m] = {k: v for k, v in d.items() if k.startswith(mid + "_")}
    return out


def _stoplist(mid: str) -> set[str]:
    """Squashed generic German words plus every word of the meeting's previous document."""
    text, _ = read_document(PREVIOUS[mid])
    return {squash(w) for w in GERMAN_COMMON} | {squash(t) for t in tokens(text)} - {""}


_TREE = REPO / "outputs_v10"


def _full_transcripts(mid: str) -> dict[str, str]:
    """Return each recognizer's full transcript of the meeting from outputs_v10/, preferring
    transcript_speakers.txt over transcript.txt.
    """
    out = {}
    for m in MODELS:
        p = _TREE / m / mid / "transcript_speakers.txt"
        if not p.exists() or p.stat().st_size == 0:
            p = _TREE / m / mid / "transcript.txt"
        if p.exists() and p.stat().st_size:
            out[m] = p.read_text(encoding="utf-8", errors="replace")
    return out


def build_map(mid: str, verbose: bool = False) -> dict[str, dict]:
    """Return {"apply": map, "suggest": map} for one meeting, or {} without terms or clip decodes.

    Each map goes from squashed variant to {canonical, count, via, similarity}.
    """
    import jiwer

    terms = targets_for(mid)
    # NER person names carry no orthographic mark. They become targets here; only the aligned
    # pass can apply them, the phonetic pass only suggests them.
    person_only = sorted(person_candidates(PREVIOUS[mid]) - set(terms))

    # Lower-case terms of the previous document become targets too. Only the phonetic pass
    # requires their variant to show unusual capitalization.
    lower_marked = {t for t, w in extract(PREVIOUS[mid]).items()
                    if w == "lower case where German capitalises" and t.islower()}
    terms = list(terms) + person_only + sorted(lower_marked - set(terms) - set(person_only))
    if not terms:
        return {}
    esq = {t: squash(t) for t in terms}
    esq = {t: s for t, s in esq.items() if len(s) >= MINLEN}
    stop = _stoplist(mid)

    hyps = _hyps(mid)
    if not hyps:
        return {}
    corpus: Counter = Counter()
    # Surface spellings per squashed token; `corpus` is case-folded, and the phonetic pass checks
    # the original case.
    raw_forms: dict[str, Counter] = defaultdict(Counter)
    for h in hyps.values():
        for t in h.values():
            for tok in tokens(t):
                corpus[squash(tok)] += 1
                raw_forms[squash(tok)][tok] += 1

    # full_corpus counts the full transcripts (phonetic pass); `corpus` counts the clips.
    # `consensus` holds tokens nearly every system writes in one clip; the aligned pass skips them.
    full_corpus: Counter = Counter()
    for t in _full_transcripts(mid).values():
        for tok in tokens(t):
            full_corpus[squash(tok)] += 1
            raw_forms[squash(tok)][tok] += 1
    if not full_corpus:
        full_corpus = corpus

    agree_floor = max(2, len(hyps) - 1)
    consensus: set[str] = set()
    for sid in sorted(set().union(*(set(v) for v in hyps.values()))):
        per_system = [set(squash(t) for t in tokens(h.get(sid, ""))) for h in hyps.values()]
        counts: Counter = Counter()
        for s in per_system:
            counts.update(s)
        consensus |= {t for t, c in counts.items() if c >= agree_floor}

    # A target may occur at most DISTINCT_MAXFREQ times in the clips, since frequent words are
    # ordinary German; orthographically marked terms are exempt.
    DISTINCT_MAXFREQ = 15
    why = extract(PREVIOUS[mid])
    marked = {"internal capital", "all caps", "letters + digits", "part of a host name"}
    esq = {t: s for t, s in esq.items()
           if why.get(t) in marked or corpus.get(s, 0) <= DISTINCT_MAXFREQ}
    canon_set = set(esq.values())
    longest_first = sorted(esq, key=lambda t: -len(esq[t]))

    # Aligned pass
    observed: dict[str, Counter] = defaultdict(Counter)
    seg_ids = sorted(set().union(*(set(v) for v in hyps.values())))
    for sid in seg_ids:
        raws = {m: tokens(hyps[m].get(sid, "")) for m in hyps}
        norms = {m: [squash(t) for t in raws[m]] for m in raws}
        anchor = {}
        for m, nt in norms.items():
            for i, s in enumerate(nt):
                for n in longest_first:
                    if s == esq[n]:
                        anchor.setdefault(n, (m, i))
                        break
        for n, (pm, pi) in anchor.items():
            pivot = " ".join(norms[pm])
            for m in norms:
                if m == pm:
                    continue
                other = " ".join(norms[m])
                if not other.strip() or not pivot.strip():
                    continue
                tok = None
                for ch in jiwer.process_words([pivot], [other]).alignments[0]:
                    if ch.ref_start_idx <= pi < ch.ref_end_idx:
                        j = ch.hyp_start_idx + (pi - ch.ref_start_idx)
                        if ch.type in ("equal", "substitute"):
                            tok = raws[m][j] if j < len(raws[m]) else None
                        break
                if tok is not None and squash(tok) != esq[n]:
                    observed[n][squash(tok)] += 1

    out: dict[str, dict] = {}

    def admit(sv: str, n: str, cnt: int, via: str, sim: float) -> None:
        out.setdefault(sv, {"canonical": n, "count": cnt, "via": via,
                            "similarity": round(sim, 3)})

    # A variant said more than ALIGNED_VARIANT_MAXFREQ times in the clips is an ordinary word,
    # and correct() would rewrite every occurrence of it.
    ALIGNED_VARIANT_MAXFREQ = 12
    for n, ctr in observed.items():
        sn = esq[n]
        for sv, cnt in ctr.items():
            if len(sv) < MINLEN or sv in stop or sv in canon_set:
                continue
            if sv in sn or sn in sv:
                continue
            if corpus.get(sv, 0) > ALIGNED_VARIANT_MAXFREQ or sv in consensus:
                continue
            sim = difflib.SequenceMatcher(None, sn, sv).ratio()
            # Equal Kölner Phonetik codes (2+ digits) lower the similarity floor from SIM to 0.45.
            phon_equal = len(koelner(n)) >= 2 and koelner(n) == koelner(sv)
            if sim < (0.45 if phon_equal else SIM):
                continue
            best = max(esq, key=lambda o: difflib.SequenceMatcher(None, esq[o], sv).ratio())
            if best != n and difflib.SequenceMatcher(None, esq[best], sv).ratio() > sim:
                continue
            admit(sv, n, cnt, "aligned+phonetic" if phon_equal and sim < SIM else "aligned", sim)

    # Phonetic pass
    codes: dict[str, list[str]] = defaultdict(list)
    for n, s in esq.items():
        c = koelner(n)
        if len(c) >= 3:
            codes[c].append(n)
    # PHON_MAXFREQ is the variant frequency cap for the clip corpus; phon_cap scales it to
    # the size of the full transcripts.
    PHON_MAXFREQ = 12
    scale = max(1.0, sum(full_corpus.values()) / max(1, sum(corpus.values())))
    phon_cap = int(round(PHON_MAXFREQ * scale))
    phon: dict[str, dict] = {}
    for sv, freq in full_corpus.items():
        if len(sv) < MINLEN or sv in stop or sv in canon_set or sv in out:
            continue
        cand = codes.get(koelner(sv))
        if not cand:
            continue
        n = max(cand, key=lambda x: difflib.SequenceMatcher(None, esq[x], sv).ratio())
        # Marked canonicals skip the cap: a name that is spoken often is still a name.
        if freq > phon_cap and why.get(n) not in marked:
            continue
        sn = esq[n]
        if corpus.get(sn, 0) > 3 or sv in sn or sn in sv or abs(len(sv) - len(sn)) > 3:
            continue
        sim = difflib.SequenceMatcher(None, sn, sv).ratio()
        if sim < PHON_SIM:
            continue
        # Another term vetoes the match if it is spelled closer and sounds the same, is closer
        # by more than AMBIGUITY_MARGIN, or contains or is contained in the variant.
        AMBIGUITY_MARGIN = 0.15
        best = max(esq, key=lambda o: difflib.SequenceMatcher(None, esq[o], sv).ratio())
        if best != n:
            bs = difflib.SequenceMatcher(None, esq[best], sv).ratio()
            competes = koelner(best) == koelner(sv)
            if (bs > sim and competes) or bs > sim + AMBIGUITY_MARGIN \
                    or squash(esq[best]) in sv or sv in squash(esq[best]):
                continue
        # Apply the match when the canonical is orthographically marked, otherwise only suggest it.
        # A lower-case canonical is skipped unless a variant spelling has unusual capitalization.
        if n in lower_marked:
            if not any(_UNUSUAL_CAPS(form) for form in raw_forms.get(sv, {})):
                continue
            marked_ok = True
        else:
            marked_ok = why.get(n) in {"internal capital", "all caps", "letters + digits",
                                       "part of a host name"}
        (out if marked_ok else phon)[sv] = {
            "canonical": n, "count": freq,
            "via": "phonetic" + ("+marked" if marked_ok else ""),
            "similarity": round(sim, 3)}

    if verbose:
        print(f"\n  {mid}: {len(terms)} terms from {PREVIOUS[mid]}")
        print(f"  APPLIED deterministically ({len(out)}) — aligned evidence:\n")
        for sv, r in sorted(out.items(), key=lambda kv: -kv[1]["count"]):
            print(f"    {sv:<22} -> {r['canonical']:<18} x{r['count']:<4} "
                  f"sim {r['similarity']:.2f}")
        print(f"\n  SUGGESTED to the summariser ({len(phon)}) — phonetic only, never applied "
              f"by this module:\n")
        for sv, r in sorted(phon.items(), key=lambda kv: -kv[1]["count"]):
            print(f"    {sv:<22} ~  {r['canonical']:<18} x{r['count']:<4} "
                  f"sim {r['similarity']:.2f}")
    return {"apply": out, "suggest": phon}


_PUNCT = re.compile(r"^([^\wÄÖÜäöüß]*)(.*?)([^\wÄÖÜäöüß]*)$", re.S)


def _correct_core(core: str, mapping: dict[str, dict], log: Counter) -> str:
    """Correct one token, or the parts of a hyphenated compound or dotted host name."""
    rec = mapping.get(squash(core)) if core else None
    if rec and squash(core) != squash(rec["canonical"]):
        log[f"{core} -> {rec['canonical']}"] += 1
        return rec["canonical"]
    # Split on hyphens or dots into at most four parts of two or more characters; dotted
    # parts must all be alphanumeric.
    for sep in ("-", "."):
        if sep not in core:
            continue
        parts = core.split(sep)
        if len(parts) > 4 or any(not p or len(p) < 2 for p in parts):
            continue
        if sep == "." and not all(p.isalnum() for p in parts):
            continue
        new, hit = [], False
        for p in parts:
            r = mapping.get(squash(p))
            if r and squash(p) != squash(r["canonical"]):
                log[f"{p} -> {r['canonical']} ({'host' if sep == '.' else 'compound'})"] += 1
                new.append(r["canonical"])
                hit = True
            else:
                new.append(p)
        if hit:
            return sep.join(new)
    return core


def correct(text: str, mapping: dict[str, dict]) -> tuple[str, Counter]:
    log: Counter = Counter()
    out = []
    for word in text.split(" "):
        m = _PUNCT.match(word)
        pre, core, post = (m.group(1), m.group(2), m.group(3)) if m else ("", word, "")
        fixed = _correct_core(core, mapping, log)
        out.append(word if fixed == core else pre + fixed + post)
    return " ".join(out), log


_CACHE = G / "entity_maps_auto.json"


def load_all(rebuild: bool = False) -> dict[str, dict]:
    if _CACHE.exists() and not rebuild:
        return json.loads(_CACHE.read_text(encoding="utf-8"))
    maps = {mid: build_map(mid) for mid in PREVIOUS}
    _CACHE.write_text(json.dumps(maps, ensure_ascii=False, indent=1), encoding="utf-8")
    return maps


def for_meeting(mid: str) -> tuple[dict[str, dict], list[tuple[str, str]], list[str]]:
    """Return one meeting's apply map, its sorted (heard, canonical) suggestions and its prompt
    glossary, using the cached maps.
    """
    m = load_all().get(mid, {})
    apply_map = m.get("apply", {})
    suggest = [(sv, r["canonical"]) for sv, r in m.get("suggest", {}).items()]
    suggest.sort(key=lambda kv: kv[0])
    return apply_map, suggest, glossary_for(mid)


def main() -> None:
    why = "--why" in sys.argv
    only = next((a for a in sys.argv[1:] if a in PREVIOUS), None)
    print("AUTOMATIC ENTITY CORRECTION — vocabulary from the previous document alone")
    print("=" * 96)
    print("No hand-written names anywhere in this pipeline.\n")
    maps = {}
    ta = ts = 0
    for mid in ([only] if only else PREVIOUS):
        m = build_map(mid, verbose=why)
        maps[mid] = m
        if not m:
            print(f"  {mid:<10} skipped: previous document or goldset/clip_asr_v10/ transcripts missing")
            continue
        ta += sum(r["count"] for r in m["apply"].values())
        ts += sum(r["count"] for r in m["suggest"].values())
        if not why:
            top = sorted(m["apply"].items(), key=lambda kv: -kv[1]["count"])[:5]
            print(f"  {mid:<10}{len(targets_for(mid)):>4} terms | apply {len(m['apply']):>3}"
                  f" | suggest {len(m['suggest']):>3}   "
                  + ", ".join(f"{k}->{r['canonical']}" for k, r in top))
    if not any(maps.values()):
        raise SystemExit("no meeting had data, nothing written")
    if not only:
        _CACHE.write_text(json.dumps(maps, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n  applied deterministically : {ta} substitutions")
        print(f"  suggested to the LLM      : {ts} occurrences")
        print(f"  wrote -> {_CACHE.relative_to(REPO)}")


if __name__ == "__main__":
    main()
