"""Constrained entity correction of the carry-arm protocols, after the RECOVER method.

RECOVER ("Robust Entity Correction via agentic Orchestration of hypothesis Variants for
Evidence-based Recovery", arXiv:2603.16411) treats several ASR hypotheses as evidence because they
make complementary errors: an entity one decoder garbles another often gets right. Candidates are
retrieved from an entity phrase list, and every proposed replacement must pass deterministic
verification before it is applied:

    1. the replacement must be an entry of the entity list
    2. case-only changes are discarded
    3. normalized Levenshtein similarity must clear a threshold       (citeva -> cytiva yes,
                                                                        star -> cytiva no)
    4. no generic word rewrites
    5. overlapping edits are skipped

Here the hypotheses are the six ASR systems on the same segment, and the entity list is built only
from earlier protocol documents. No language model proposes candidates: a replacement Y for a
printed token X is proposed only if some system wrote Y in a segment cluster where another wrote X.
Check 3 is difflib's SequenceMatcher ratio on squashed forms (SIM = 0.75). Check 4 is three guards:
X is not a word that at least four systems wrote in the same segment, the length ratio of Y to X
lies strictly between 0.8 and 1.25, and neither form contains the other at either edge (inflection
or compounding, not a mishearing). X is never replaced if the entity list contains it, and a
printed form with two different candidates is left unchanged. The vote count is not used; the
entity list decides, as in RECOVER.

Rules (--rule, default history; recover and history take effect only with --apply):
    recover   last week's protocol only (term_source.PREVIOUS)
    history   every protocol of the series dated before the meeting (prior_documents)
    audio     Path 2, a dry run: candidates from the current audio's six decodes alone, gated by
              model-family agreement and phonetic similarity (audio_path); --apply is refused
The measurement table also shows the majority rule (the cluster's most-voted spelling, no list) as
a control, each rule at similarity thresholds 0.65, 0.75 and 0.85.

Scoring uses the reference minutes of the same meeting (evaluate_speaker_attribution.HUMAN), which
the corrector never reads:
    confirmed   Y is in the minutes of this meeting, X is not
    refuted     X is in the minutes, Y is not
    open        neither decides it
plus a safety run: the corrector applied to the minutes themselves, where every change is by
construction a false positive. Edits are counted over the team-arm protocol_*_fixed_carry.md files
in outputs_v10/<model>/<meeting>/.

    python entity_recover.py                          # measurement table, stdout only
    python entity_recover.py --apply                  # also rewrite the carry protocols, history list
    python entity_recover.py --apply --rule recover   # ... last week's protocol only
    python entity_recover.py --rule audio             # Path 2 dry run

--apply rewrites the protocol_*_fixed_carry.md files in place and logs the edits to
goldset/entity_recover.json; --rule audio writes goldset/entity_recover_audio_dryrun.json.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import entity_correct as E                                              # noqa: E402
import name_check as NC                                                 # noqa: E402
from term_source import PREVIOUS, read_document                        # noqa: E402
from evaluate_speaker_attribution import HUMAN                         # noqa: E402
from protocol_repair import is_team_arm                                 # noqa: E402

V10 = REPO / "outputs_v10"
LOG = REPO / "goldset" / "entity_recover.json"
SIM = 0.75                    # RECOVER check 3; swept below
WORD = re.compile(r"[\wÄÖÜäöüß][\wÄÖÜäöüß\-]*")


def nfc(s: str) -> str:
    import unicodedata
    return unicodedata.normalize("NFC", s)


def words(text: str) -> set[str]:
    return {nfc(w).lower() for w in WORD.findall(text)}


MEETING_DATE = {"march09": "2026-03-09", "april13": "2026-04-13", "april27": "2026-04-27",
                "may11": "2026-05-11", "may18": "2026-05-18", "june01": "2026-06-01",
                "june15": "2026-06-15", "june22": "2026-06-22"}
DOC_DATE = re.compile(r"(2026-\d\d-\d\d)")


def prior_documents(mid: str) -> list[str]:
    """Every protocol of this meeting series dated strictly before the meeting.

    A file in the repository folder counts if its name contains "Meeting" and a date
    (YYYY-MM-DD) earlier than the meeting date, and its suffix is .md or .docx. This widens the
    entity list from last week's protocol to the series' own record: a name absent from last
    week's protocol but written in an earlier one becomes reachable. Every name in that record
    was written by the same group before the meeting, with no manual curation, so the wider list
    adds no outside knowledge."""
    cut = MEETING_DATE[mid]
    docs = []
    for p in sorted(REPO.glob("*")):
        m = DOC_DATE.search(p.name)
        if (m and m.group(1) < cut and p.suffix in (".md", ".docx")
                and ("Meeting" in p.name)):
            docs.append(p.name)
    return docs


def entity_list(mid: str, history: bool = False) -> dict[str, str]:
    """squashed -> spelling. Tokens of four or more characters that are capitalized, contain a
    digit or contain a hyphen. German capitalizes every noun, so the list is wide on purpose,
    like RECOVER's phrase lists; the hypotheses narrow it, not the list.

    history=False  the previous document alone (term_source.PREVIOUS)
    history=True   every protocol of the series written before this meeting (prior_documents)"""
    names = prior_documents(mid) if history else [PREVIOUS.get(mid)]
    out = {}
    for name in names:
        if not name:
            continue
        text, _ = read_document(name)
        for w in WORD.findall(text):
            if len(w) >= 4 and (w[0].isupper() or any(c.isdigit() for c in w) or "-" in w):
                out.setdefault(E.squash(w), w)
    return out


def clusters(mid: str) -> list[Counter]:
    """Per-segment clusters of rival spellings across the six systems: same audio position,
    never merged across segments, because merging across segments chains unrelated spellings
    into one cluster."""
    hyps = E._hyps(mid)
    out = []
    for sid in sorted(set().union(*(set(v) for v in hyps.values()))):
        votes = []
        for system, segs in hyps.items():
            seen = set()
            for tok in NC.CAP.findall(segs.get(sid, "")):
                sq = E.squash(tok)
                if tok.lower() in NC.SKIP or sq in seen:
                    continue
                seen.add(sq)
                votes.append((system, tok))
        out.extend(v for v in NC._cluster(votes) if len(v) >= 2)
    return out


def agreed_words(mid: str, k: int = 4) -> set[str]:
    """Squashed tokens that at least k of the six systems wrote in the same segment.

    RECOVER's check 4 forbids generic word rewrites, and without a German lexicon (none ships
    offline; see entity_correct.py) the only evidence that a token is a real word somebody said
    is that the decoders agree on it. Without this guard, real words such as Dateien -> Daten or
    Mission -> Vision can be rewritten, changing their meaning, and still score as confirmed
    whenever the replacement occurs somewhere in the minutes. A misheard entity is sporadic; a
    word four decoders agree on was said."""
    hyps = E._hyps(mid)
    out = set()
    for sid in set().union(*(set(v) for v in hyps.values())):
        c = Counter()
        for segs in hyps.values():
            c.update({E.squash(t) for t in WORD.findall(segs.get(sid, ""))})
        out |= {w for w, n in c.items() if n >= k}
    return out


def proposals(mid: str, rule: str, sim: float) -> dict[str, str]:
    """printed spelling (lowercase) -> replacement, for one meeting.

    rule 'recover'   replacement must be in the entity list from last week's protocol (RECOVER
                     check 1) and the printed token must not be;
    rule 'history'   the same, with the entity list from every earlier protocol of the series;
    rule 'majority'  the control: the cluster's most-voted spelling, no list.
    """
    constrained = rule != "majority"
    ents = entity_list(mid, history=(rule == "history")) if constrained else {}
    said = agreed_words(mid) if constrained else set()
    cand: dict[str, set[str]] = {}
    for votes in clusters(mid):
        forms = list(votes)
        if rule == "majority":
            top, n_top = votes.most_common(1)[0]
            targets = [top] if n_top >= 2 else []
        else:
            targets = [f for f in forms if E.squash(f) in ents]
        for y in targets:
            for x in forms:
                sx, sy = E.squash(x), E.squash(y)
                if sx == sy or nfc(x).lower() == nfc(y).lower():      # check 2: case/punct only
                    continue                                          # (and ü vs u+U+0308)
                # A mishearing replaces sounds; it does not delete a fifth of the word. The
                # length ratio also stops qualifier deletions that differ in spelling and so
                # slip past the edge test below (IDP-Webseite -> Website, Ground-up -> Grund,
                # Nachtwissenschaften -> Wissenschaft).
                if constrained and not (0.8 < len(sy) / max(len(sx), 1) < 1.25):
                    continue
                if constrained and sx in ents:                  # the document uses X too
                    continue
                if constrained and sx in said:                  # check 4: X is a word the
                    continue                                          # decoders agree was said
                if difflib.SequenceMatcher(None, sx, sy).ratio() < sim:   # check 3
                    continue
                # check 4, continued: one form containing the other at either edge is inflection
                # or compounding, not a mishearing. Both edges are tested because German puts
                # the qualifier in front: a prefix-only test would let "SARA-Deployment" become
                # "Deployment" and "KI-Allianz" become "Allianz", deleting the qualifier.
                if (sx.startswith(sy) or sy.startswith(sx)
                        or sx.endswith(sy) or sy.endswith(sx)):
                    continue
                spelled = ents.get(sy, y) if constrained else y
                cand.setdefault(x.lower(), set()).add(spelled)
    # ambiguity: two different replacements for one printed form -> no decision
    return {x: next(iter(ys)) for x, ys in cand.items() if len(ys) == 1}


# ---------------------------------------------------------------------------------------------
# PATH 2: AUDIO-ONLY RECOVERY (dry run only, --rule audio). Path 1 (rule 'recover') needs the
# candidate in last week's document; Path 2 handles names that document does not contain, and
# may use only the current audio's six decodes. Gates, in order:
#   G1 candidate is a spelling some system wrote at this aligned occurrence (same segment
#      cluster), never a name from a list, an LLM or an older protocol
#   G2 at least 2 model families wrote exactly that spelling; the three Whisper variants are
#      one family, however many of them agree
#   G3 it strictly beats every rival spelling in the cluster (by family count)
#   G4 phonetic/orthographic similarity to the printed form >= 0.62 (handover_naming.score,
#      Kölner Phonetik + edit ratio; a spelling comparison, not acoustic evidence)
#   G5 lexical: the candidate is not ordinary German (GERMAN_COMMON) and not lower case
#   G6 the printed form is not in last week's document (a word the permitted context uses is
#      never overwritten) and not a word 4+ decoders agree was said
#   plus the Path-1 shape checks (length ratio 0.8-1.25, no prefix/suffix inflection)
# Anything that fails a gate is flagged with the reason, never changed.
# ---------------------------------------------------------------------------------------------
FAMILY = {"whisper_large_v3": "whisper", "whisper_turbo": "whisper", "whisper_turbo_de": "whisper",
          "canary_diar": "canary", "cohere": "cohere", "granite41": "granite"}
PHON_MIN = 0.62          # the handover_naming threshold, reused unchanged
MIN_FAMILIES = 2


def _audio_clusters(mid: str) -> list[tuple[str, list[tuple[str, str]]]]:
    """(segment id, [(system, token), ...]) per cluster of rival spellings — the same grouping
    as clusters() (edit ratio >= 0.70 on squashed forms) but keeping which system wrote each
    spelling."""
    hyps = E._hyps(mid)
    out = []
    for sid in sorted(set().union(*(set(v) for v in hyps.values()))):
        groups: list[dict] = []
        for system, segs in hyps.items():
            seen = set()
            for tok in NC.CAP.findall(segs.get(sid, "")):
                sq = E.squash(tok)
                if tok.lower() in NC.SKIP or sq in seen:
                    continue
                seen.add(sq)
                for g in groups:
                    if difflib.SequenceMatcher(None, g["rep"], sq).ratio() >= 0.70:
                        if system not in {s for s, _ in g["votes"]}:
                            g["votes"].append((system, tok))
                        break
                else:
                    groups.append({"rep": sq, "votes": [(system, tok)]})
        out.extend((sid, g["votes"]) for g in groups if len({E.squash(t) for _, t in g["votes"]}) >= 2)
    return out


def audio_path(mid: str) -> tuple[dict[str, str], list[dict]]:
    """Path 2 for one meeting: (AUTO map printed->candidate, decision records incl. FLAGs)."""
    from handover_naming import score
    from term_source import GERMAN_COMMON
    prev = entity_list(mid)                         # last week's document, for guard G6 only
    said = agreed_words(mid)
    common = {E.squash(w) for w in GERMAN_COMMON}
    path1 = proposals(mid, "recover", SIM)          # Path 1 takes precedence
    records, cand = [], {}
    for sid, votes in _audio_clusters(mid):
        fam: dict[str, set] = {}
        spell: dict[str, str] = {}
        for system, tok in votes:
            k = nfc(tok).lower()
            fam.setdefault(k, set()).add(FAMILY.get(system, system))
            spell.setdefault(k, tok)
        ranked = sorted(fam.items(), key=lambda kv: -len(kv[1]))
        top_k, top_f = ranked[0]
        rival = max((len(f) for k, f in ranked[1:]), default=0)
        support = {spell[k]: sorted(f) for k, f in fam.items()}
        base = {"meeting": mid, "segment": sid, "support": support}
        if len(top_f) < MIN_FAMILIES:
            records.append({**base, "decision": "FLAG", "reason": "G2: no spelling from 2+ model families"})
            continue
        if len(top_f) <= rival:
            records.append({**base, "decision": "FLAG", "reason": "G3: tie with a rival spelling"})
            continue
        y = spell[top_k]
        sy = E.squash(y)
        if sy in common or y == y.lower():
            records.append({**base, "candidate": y, "decision": "FLAG", "reason": "G5: ordinary German word"})
            continue
        for k, _ in ranked[1:]:
            x = spell[k]
            sx = E.squash(x)
            if sx == sy or x.lower() in path1:
                continue
            why = None
            if score(x, y) < PHON_MIN:
                why = f"G4: phonetic similarity {score(x, y):.2f} < {PHON_MIN}"
            elif sx in prev:
                why = "G6: printed form is in last week's document"
            elif sx in said:
                why = "G6: 4+ decoders agree the printed form was said"
            elif not (0.8 < len(sy) / max(len(sx), 1) < 1.25):
                why = "shape: length ratio"
            elif sx.startswith(sy) or sy.startswith(sx) or sx.endswith(sy) or sy.endswith(sx):
                why = "shape: inflection/compound, not a mishearing"
            rec = {**base, "printed": x, "candidate": y, "families": sorted(top_f),
                   "decision": "FLAG" if why else "AUTO", "reason": why or "all gates passed",
                   "candidate_source": "current_audio", "evidence_source": "current_audio"}
            records.append(rec)
            if not why:
                cand.setdefault(x.lower(), set()).add(y)
    auto = {x: next(iter(ys)) for x, ys in cand.items() if len(ys) == 1}
    for r in records:                                # an ambiguous printed form is not corrected
        if r["decision"] == "AUTO" and r["printed"].lower() not in auto:
            r["decision"], r["reason"] = "FLAG", "ambiguous: printed form maps to two candidates"
    return auto, records


def audio_report() -> dict:
    """Dry run of Path 2 over every meeting: nothing is written to any protocol."""
    tally, safety, forms, records = Counter(), Counter(), Counter(), []
    for mid in HUMAN:
        mp, recs = audio_path(mid)
        records += recs
        truth = minutes_words(mid)
        _, fp = apply(read_document(HUMAN[mid])[0], mp)
        safety["edits"] += len(fp)
        for model_dir in sorted(p for p in V10.iterdir() if p.is_dir()):
            for md in sorted((model_dir / mid).glob("protocol_*_fixed_carry.md")):
                if not is_team_arm(md.name):
                    continue
                _, edits = apply(md.read_text(encoding="utf-8", errors="replace"), mp)   # read only
                for x, y in edits:
                    v = verdict(x, y, truth)
                    tally[v] += 1
                    forms[(mid, x, y, v)] += 1
    return {"tally": tally, "safety": safety, "forms": forms, "records": records}


def apply(text: str, mp: dict[str, str]) -> tuple[str, list[tuple[str, str]]]:
    if not mp:
        return text, []
    edits = []
    pat = re.compile(r"(?<![\w-])(" + "|".join(re.escape(k) for k in
                                                sorted(mp, key=len, reverse=True))
                     + r")(?![\w-])", re.I)

    def sub(m):
        edits.append((m.group(1), mp[m.group(1).lower()]))
        return mp[m.group(1).lower()]
    return pat.sub(sub, text), edits                                  # check 5: left-to-right


def minutes_words(mid: str) -> set[str]:
    return words(read_document(HUMAN[mid])[0])


def verdict(x: str, y: str, truth: set[str]) -> str:
    xi, yi = x.lower() in truth, y.lower() in truth
    if yi and not xi:
        return "confirmed"
    if xi and not yi:
        return "refuted"
    return "open"


def measure(rule: str, sim: float, write: bool = False) -> dict:
    tally, per_form, safety, log = Counter(), Counter(), Counter(), {}
    for mid in HUMAN:
        mp = proposals(mid, rule, sim)
        truth = minutes_words(mid)
        _, fp = apply(read_document(HUMAN[mid])[0], mp)
        safety["edits"] += len(fp)
        for model_dir in sorted(p for p in V10.iterdir() if p.is_dir()):
            for md in sorted((model_dir / mid).glob("protocol_*_fixed_carry.md")):
                if not is_team_arm(md.name):
                    continue
                text = md.read_text(encoding="utf-8", errors="replace")
                new, edits = apply(text, mp)
                for x, y in edits:
                    v = verdict(x, y, truth)
                    tally[v] += 1
                    per_form[(mid, x, y, v)] += 1
                if edits:
                    log[f"{model_dir.name}/{mid}/{md.name}"] = edits
                    if write:
                        md.write_text(new, encoding="utf-8")
    return {"tally": tally, "forms": per_form, "safety": safety, "log": log}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--rule", default="history", choices=["recover", "history", "audio"],
                    help="which entity list --apply uses; see the module docstring. "
                         "'audio' = Path 2, report only")
    a = ap.parse_args()

    if a.rule == "audio":
        if a.apply:
            print("REFUSED: Path 2 (--rule audio) is dry-run only until its report has been reviewed.")
            return 2
        r = audio_report()
        t = r["tally"]
        dec = t["confirmed"] + t["refuted"]
        print("PATH 2 — AUDIO-ONLY RECOVERY, DRY RUN (no protocol is written)")
        print("=" * 96)
        print(f"  protocol edits it WOULD make: {sum(t.values())}   confirmed {t['confirmed']}   "
              f"refuted {t['refuted']}   open {t['open']}   precision* "
              f"{(t['confirmed'] / dec if dec else float('nan')):.0%}   safety FP {r['safety']['edits']}")
        print("\n  would-be corrections (distinct):")
        for (mid, x, y, v), n in sorted(r["forms"].items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"     {v:9} {mid:8} {x:24} -> {y:24} x{n}")
        dec_c = Counter(x["decision"] for x in r["records"])
        why = Counter(x["reason"].split(":")[0] for x in r["records"] if x["decision"] == "FLAG")
        print(f"\n  decisions over {len(r['records'])} rival-spelling cases: {dict(dec_c)}   flag reasons {dict(why)}")
        out = REPO / "goldset" / "entity_recover_audio_dryrun.json"
        out.write_text(json.dumps({"tally": dict(t), "safety_edits": r["safety"]["edits"],
                                   "records": r["records"]}, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        print(f"  full decision records -> {out.relative_to(REPO)}")
        return 0

    print("CONSTRAINED ENTITY CORRECTION — after RECOVER (arXiv:2603.16411)")
    print("=" * 96)
    print("Hypotheses: six ASR systems per segment. Entity list: the previous document alone.")
    print("Scored against the same meeting's minutes, which the corrector never reads.\n")
    print(f"  {'rule':10}{'sim':>5}{'edits':>7}{'confirmed':>11}{'refuted':>9}{'open':>6}"
          f"{'precision*':>12}{'safety FP':>11}")
    print("  " + "-" * 70)
    results = {}
    for rule in ("majority", "recover", "history"):
        for sim in (0.65, 0.75, 0.85):
            r = measure(rule, sim)
            results[(rule, sim)] = r
            t = r["tally"]
            dec = t["confirmed"] + t["refuted"]
            prec = t["confirmed"] / dec if dec else float("nan")
            print(f"  {rule:10}{sim:>5.2f}{sum(t.values()):>7}{t['confirmed']:>11}"
                  f"{t['refuted']:>9}{t['open']:>6}{prec:>12.0%}{r['safety']['edits']:>11}")
    print("\n  * precision over DECIDABLE edits (confirmed + refuted). 'open' edits are ones the"
          "\n    minutes cannot settle. safety FP = edits the rule makes to the reference minutes.")

    r = results[("recover", SIM)]
    print(f"\n  RECOVER at the pre-set threshold {SIM}: distinct corrections")
    for (mid, x, y, v), n in sorted(r["forms"].items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"     {v:9} {mid:8} {x:24} -> {y:24} x{n}")
    rm = results[("majority", SIM)]
    print(f"\n  the rejected majority rule at the same threshold, refuted edits it would make:")
    for (mid, x, y, v), n in sorted(rm["forms"].items(), key=lambda kv: -kv[1]):
        if v == "refuted":
            print(f"     {mid:8} {x:24} -> {y:24} x{n}")

    if a.apply:
        r = measure(a.rule, SIM, write=True)
        LOG.write_text(json.dumps({"rule": a.rule, "sim": SIM, "tally": dict(r["tally"]),
                                   "protocols": r["log"]}, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        print(f"\n  APPLIED to protocol_*_fixed_carry.md  log -> {LOG.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
