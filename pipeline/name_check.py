"""Find the names in a protocol whose spelling is uncertain: words that the recognizers spell
differently for the same audio, and words that sound like a term of the previous document but
are spelled differently. As a script, it lists them for each meeting's protocols under
outputs_v10/ and writes goldset/name_check.json; --append also writes each non-empty list to
namen_zum_pruefen.md beside the meeting's protocols.

    python name_check.py [meeting] [--model MODEL] [--min-systems N]
                         [--max-agreement X] [--top N] [--append]
"""

from __future__ import annotations

import argparse
import json
import difflib
import re
from collections import Counter, defaultdict
from pathlib import Path

import entity_correct as E
from term_source import glossary_for, PREVIOUS

REPO = Path(__file__).resolve().parent
V10 = REPO / "outputs_v10"
OUT = REPO / "goldset" / "name_check.json"

# A capitalized token of four or more characters. German capitalizes all nouns, so this
# matches many ordinary words; the checks below do the filtering.
CAP = re.compile(r"\b[A-ZÄÖÜ][\wäöüßA-ZÄÖÜ-]{3,}\b")

# Frequent words that appear capitalized at the start of a sentence; they are never names.
SKIP = {
    "aber", "also", "dann", "das", "dass", "der", "die", "ein", "eine", "und", "oder", "wenn",
    "wir", "was", "wie", "weil", "nicht", "noch", "nur", "schon", "sehr", "über", "aufgabe",
    "übertrag", "ja", "nein", "gut", "okay", "genau", "halt", "eben", "man", "ich", "sie",
    "jetzt", "hier", "dort", "immer", "wieder", "mal", "doch", "sich", "ist", "sind", "haben",
}


def _cluster(votes: list[tuple[str, str]], thresh: float = 0.70) -> list[Counter]:
    """Group (system, token) votes for one position into Counters of similar spellings;
    each system votes once per group.
    """
    import difflib
    clusters: list[dict] = []
    for system, tok in votes:
        sq = E.squash(tok)
        for c in clusters:
            if difflib.SequenceMatcher(None, c["rep"], sq).ratio() >= thresh:
                if system not in c["systems"]:
                    c["systems"].add(system)
                    c["votes"][tok] += 1
                break
        else:
            clusters.append({"rep": sq, "systems": {system}, "votes": Counter({tok: 1})})
    return [c["votes"] for c in clusters]


def flags(mid: str, min_systems: int = 2) -> list[dict]:
    """Return the recognizers' spelling disagreements on one meeting's clips
    (entity_correct._hyps), lowest agreement first.
    """
    hyps = E._hyps(mid)
    if not hyps:
        return []
    known = {g.lower() for g in glossary_for(mid)}
    out: list[dict] = []
    for sid in sorted(set().union(*(set(v) for v in hyps.values()))):
        votes: list[tuple[str, str]] = []
        for system, segs in hyps.items():
            seen: set[str] = set()
            for tok in CAP.findall(segs.get(sid, "")):
                if tok.lower() in SKIP or len(tok) < 4:
                    continue
                sq = E.squash(tok)
                if sq in seen:
                    continue
                seen.add(sq)
                votes.append((system, tok))
        for v in _cluster(votes):
            n = sum(v.values())
            if n < min_systems or len(v) < 2:
                continue
            # Variants that differ only in case, umlaut spelling or punctuation are one spelling.
            if len({E.squash(x) for x in v}) < 2:
                continue
            # Names that the previous document's glossary contains are left to entity_correct.
            if any(x.lower() in known for x in v):
                continue
            top, n_top = v.most_common(1)[0]
            out.append({"meeting": mid, "segment": sid, "majority": top,
                        "agreement": round(n_top / n, 2), "systems": n,
                        "variants": dict(v.most_common())})
    out.sort(key=lambda r: (r["agreement"], -r["systems"]))
    return out


def uncertain_from_transcripts(per_system: list[list[str]]) -> list[dict]:
    """Return spelling disagreements between recognizers.

    per_system holds one list of segment texts per recognizer; index i must be the same audio
    in every list.
    """
    if len(per_system) < 2:
        return []
    n_seg = min(len(s) for s in per_system)
    out = []
    for i in range(n_seg):
        votes: list[tuple[str, str]] = []
        for si, segs in enumerate(per_system):
            seen: set[str] = set()
            for tok in CAP.findall(segs[i]):
                if tok.lower() in SKIP or len(tok) < 4:
                    continue
                sq = E.squash(tok)
                if sq in seen:
                    continue
                seen.add(sq)
                votes.append((str(si), tok))
        for v in _cluster(votes):
            n = sum(v.values())
            if n < 2 or len(v) < 2 or len({E.squash(x) for x in v}) < 2:
                continue
            top, n_top = v.most_common(1)[0]
            out.append({"majority": top, "agreement": round(n_top / n, 2), "systems": n,
                        "variants": dict(v.most_common()), "why": "Systeme uneinig"})
    out.sort(key=lambda r: (r["agreement"], -r["systems"]))
    return out


def _docx_text(p: Path) -> str:
    import docx
    return "\n".join(x.text for x in docx.Document(str(p)).paragraphs)


def uncertain_single(text: str, prev_doc: Path, history: list[Path] | None = None) -> list[dict]:
    """Flag words in text that sound like a term of prev_doc or the history documents (same
    Kölner code) but are spelled differently.

    Pass only documents dated before the meeting.
    """
    from term_source import glossary_for_document, GERMAN_COMMON
    docs = [prev_doc] + [h for h in (history or []) if h != prev_doc]
    terms: list[str] = []
    for d in docs:
        if d.exists():
            terms += [t for t in glossary_for_document(d) if t not in terms]
    if not terms:
        return []
    prev_words: set = set()
    for d in docs:
        if not d.exists():
            continue
        if d.suffix.lower() == ".docx":
            txt = _docx_text(d)
        elif d.suffix.lower() == ".pdf":
            from term_source import read_document
            txt = read_document(str(d.resolve()))[0]
        else:
            txt = d.read_text(encoding="utf-8", errors="replace")
        prev_words |= {E.squash(w) for w in re.findall(r"[^\W\d_]{3,}", txt)}
    canon = {E.squash(x) for x in terms}

    # Index the terms by Kölner code. Two-digit codes are included; the spelling check below
    # is stricter for them.
    codes: dict[str, list[str]] = defaultdict(list)
    for term in terms:
        c = E.koelner(E.squash(term))
        if len(c) >= 2:
            codes[c].append(term)
    seen: set[str] = set()
    out = []
    for tok in CAP.findall(text):
        sq = E.squash(tok)
        if (tok.lower() in SKIP or len(tok) < 4 or sq in canon or sq in seen
                or tok.lower() in GERMAN_COMMON):
            continue
        # A word that the documents contain is a real word, not a mishearing.
        if sq in prev_words:
            continue
        cand = codes.get(E.koelner(sq))
        if not cand:
            continue
        # Choose the candidate by letter similarity, the same score the threshold below uses.
        near = max(cand, key=lambda x: difflib.SequenceMatcher(None, E.squash(x), sq).ratio())
        nsq = E.squash(near)
        if nsq in sq or sq in nsq:
            continue        # a truncation or an inflection, not a rival spelling
        # Similarity needed: 0.75 for a two-digit code (weak evidence), 0.65 otherwise.
        if difflib.SequenceMatcher(None, nsq, sq).ratio() < (
                0.75 if len(E.koelner(sq)) < 3 else 0.65):
            continue
        seen.add(sq)
        out.append({"majority": tok, "agreement": 0.0, "systems": 1,
                    "variants": {tok: 1}, "canonical": near,
                    "why": f"klingt wie '{near}' aus dem vorherigen Dokument"})
    return out


def reaching_protocol(mid: str, rows: list[dict], model: str = "canary_diar") -> list[dict]:
    """Keep the flags with a variant in the meeting's protocol_flash_fixed.md or
    protocol_pro_fixed.md, and add the vote counts of the printed spelling.
    """
    texts = {}
    for lisa in ("flash", "pro"):
        p = V10 / model / mid / f"protocol_{lisa}_fixed.md"
        if p.exists():
            texts[lisa] = p.read_text(encoding="utf-8", errors="replace")
    if not texts:
        return []
    hits = []
    for r in rows:
        where = [l for l, t in texts.items()
                 for v in r["variants"] if re.search(rf"\b{re.escape(v)}\b", t)]
        if where:
            present = sorted({v for t in texts.values() for v in r["variants"]
                              if re.search(rf"\b{re.escape(v)}\b", t)})
            # contradicted_by: how many more votes the top variant has than the best
            # variant printed in the protocol.
            top_votes = max(r["variants"].values())
            printed_votes = max((r["variants"].get(v, 0) for v in present), default=0)
            hits.append({**r, "in_protocol": sorted(set(where)), "as": present,
                         "printed_votes": printed_votes, "top_votes": top_votes,
                         "contradicted_by": top_votes - printed_votes})
    return hits


def list_for_meeting(mid: str, model: str = "canary_diar",
                    max_agreement: float = 0.75, top: int = 10) -> list[dict]:
    """Return the ranked list of names to check for one meeting, capped at top: cross-system
    disagreements and previous-document near-misses that appear in its protocols.
    """
    rows = reaching_protocol(mid, flags(mid, 2), model)
    rows = [r for r in rows if r["agreement"] <= max_agreement]
    prev = PREVIOUS.get(mid)
    if prev and Path(prev).exists():
        p = V10 / model / mid / "transcript.txt"
        if p.exists():
            near = uncertain_single(p.read_text(encoding="utf-8", errors="replace"), Path(prev))
            texts = [(V10 / model / mid / f"protocol_{l}_fixed.md") for l in ("flash", "pro")]
            body = "\n".join(q.read_text(encoding="utf-8", errors="replace")
                              for q in texts if q.exists())
            seen = {E.squash(v) for r in rows for v in r["variants"]}
            for n in near:
                tok = n["majority"]
                if E.squash(tok) in seen or not re.search(rf"\b{re.escape(tok)}\b", body):
                    continue
                rows.append({**n, "meeting": mid, "as": [tok]})
    # Disagreements rank by how far the printed spelling is outvoted, then by agreement.
    # Near-misses get up to half of the slots.
    dis = sorted([r for r in rows if r.get("systems", 1) >= 2],
                 key=lambda r: (-r.get("contradicted_by", 0), r["agreement"]))
    near = [r for r in rows if r.get("systems", 1) < 2]
    half = max(1, top // 2)
    picked = dis[:top - min(half, len(near))] + near[:half]
    return picked[:top]


def as_markdown(rows: list[dict]) -> str:
    """Render the rows as the German 'Namen zum Prüfen' Markdown block (empty for no rows)."""
    if not rows:
        return ""
    # Two row kinds: "gehört als" = the recognizers disagree; "klingt wie" = the word
    # sounds like a term from an earlier protocol.
    lines = ["", "## Namen zum Prüfen",
             "*Die Schreibweise dieser Wörter im Protokoll ist unsicher: entweder sind sich die "
             "Spracherkennungssysteme uneinig (\"gehört als\"), oder das Wort klingt wie ein Name "
             "aus einem früheren Protokoll (\"klingt wie\").*", ""]
    for r in rows:
        shown = r["as"][0] if r.get("as") else r["majority"]
        if r.get("why", "").startswith("klingt wie"):
            lines.append(f"* **{shown}** — {r['why']}")
        else:
            variants = ", ".join(f"{v} ({n})" for v, n in r["variants"].items())
            lines.append(f"* **{shown}** — gehört als: {variants}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("meeting", nargs="?", help="limit to one meeting")
    ap.add_argument("--model", default="canary_diar")
    ap.add_argument("--min-systems", type=int, default=2)
    # Agreement is the share of a cluster's systems that wrote the majority spelling. The default 0.75
    # keeps names with a 4-of-6 majority (0.67), which still need checking; --top keeps the list short.
    ap.add_argument("--max-agreement", type=float, default=0.75,
                    help="only report names at or below this cross-system agreement")
    ap.add_argument("--top", type=int, default=10,
                    help="cap the per-protocol list; a list nobody scans is worth nothing")
    ap.add_argument("--append", action="store_true",
                    help="write the 'Namen zum Prüfen' list to namen_zum_pruefen.md beside each protocol")
    a = ap.parse_args()

    meetings = [a.meeting] if a.meeting else [m for m in E.PREVIOUS]
    print("NAMES TO CHECK — recognizer disagreements and near-misses of names in earlier protocols")
    print("=" * 92)
    print("Flagged, never rewritten: a guessed name is worse than a misspelled one.\n")

    store: dict[str, list[dict]] = {}
    for mid in meetings:
        try:
            rows = flags(mid, a.min_systems)
        except Exception as e:
            print(f"  {mid:<9} skipped ({type(e).__name__})")
            continue
        hit = reaching_protocol(mid, rows, a.model)
        # 'shown' is the ranked, capped list; 'all' keeps every cross-system disagreement that reached a
        # protocol, before the agreement filter and the cap.
        shown = list_for_meeting(mid, a.model, a.max_agreement, a.top)
        store[mid] = {"shown": shown, "all": hit}
        print(f"  {mid:<9} {len(rows):>3} disagreements, {len(hit):>3} reached the protocol, "
              f"{len(shown):>2} listed (agreement <= {a.max_agreement:.0%})")
        hit = shown
        for r in hit[:6]:
            variants = ", ".join(f"{v}x{n}" for v, n in r["variants"].items())
            print(f"      {str(r['as']):<26} agreement {r['agreement']:.0%}  [{variants}]")
        # Written beside the protocol, not into it: protocol parsers read its "##" heading
        # as a person.
        if a.append and hit:
            side = V10 / a.model / mid / "namen_zum_pruefen.md"
            side.write_text(as_markdown(hit).lstrip("\n"), encoding="utf-8")

    OUT.write_text(json.dumps(store, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  wrote -> {OUT.relative_to(REPO)}")
    if a.append:
        print("  wrote namen_zum_pruefen.md beside each protocol")
    print("\n  WHAT THIS CANNOT DO — two limits, both real:")
    print("  1. When every system hears something different (may11 gave five spellings of one")
    print("     surname) there is no majority to trust. The disagreement is")
    print("     still the useful signal: it says the name is unreliable, which is what a")
    print("     reader needs to know.")
    print("  2. A name every system gets wrong the SAME way is invisible here. 'MOS-TTS' for")
    print("     MOSS-TTS is exactly that: the decoders agree, and they are all wrong. Cross-")
    print("     system evidence cannot reach it, and neither can the previous document when")
    print("     the name first occurs in this meeting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
