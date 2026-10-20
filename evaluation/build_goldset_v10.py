"""Builds goldset/goldset_v10.json, the scoreable gold set, from the rated references, the draw
file, the recognizer hypotheses and the exclusion list of asr_eval_scope.py.

Each clip record holds the reference text as rated, its tier, resolution, scoring eligibility,
seeding flags and the hypotheses of the six recognizers. With --peinl, a second-opinion export
is first merged into the rated records and saved to goldset/goldset_v9_after_second_opinion.json,
which later runs without --peinl read. Do not run --peinl again: it starts from the 150 rated
records and would drop the two replacement clips that only the stored merge holds.
meta.noise_floor holds the small first reader-agreement floor; score_v10.py uses
goldset/reference_reliability.json when it exists.

    python build_goldset_v10.py [--out OUT.json]
    python build_goldset_v10.py --peinl EXPORT.json [--out OUT.json]
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

import asr_eval_scope as SCOPE
from thesis_pipeline.text_norm import NormConfig, normalize

G = pathlib.Path(__file__).resolve().parent / "goldset"

RATED = G / "goldset_v9_2026-09-04.json"
# --peinl saves its merge here; later runs without --peinl read this file when it exists.
# RATED stays unchanged as the snapshot before the second opinion.
MERGED = G / "goldset_v9_after_second_opinion.json"
DRAW = G / "goldset_v9_draw.json"
FLOOR = G / "rater_noise_floor.json"
OUT = G / "goldset_v10.json"

# Hypothesis folders in order of preference. Each system comes from the first folder that
# has it and is never mixed across folders; meta.panel.hypotheses_from records the choice.
HYP_DIRS = [(G / "clip_asr_v10", "v10"), (G / "clip_asr_v9", "v9")]

# System code names; each matches the prefix of its hypothesis file <name>__<tag>.json.
PANEL = ["canary_diar", "whisper_large_v3", "whisper_turbo", "whisper_turbo_de",
         "cohere", "granite41"]

# Same normalization as score_v10.py, so `identical_to` marks the clips where a system
# scores a WER of zero.
CFG = NormConfig(casefold=True)

# Size of the segment population the gold set was drawn from, and its number of deciles.
POPULATION = 940
N_DECILES = 10


def _load(p: pathlib.Path):
    return json.loads(p.read_text(encoding="utf-8"))


def panel_hypotheses() -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    """Return system -> {segment_id: text} and system -> the hypothesis file it came from."""
    out: dict[str, dict[str, str]] = {}
    whence: dict[str, str] = {}
    for d, tag in HYP_DIRS:
        if not d.exists():
            continue
        for f in sorted(d.glob(f"*__{tag}.json")):
            name = f.name.split("__")[0]
            if name not in PANEL or name in out:
                continue
            h = _load(f)
            out[name] = {k: (v.get("text", "") if isinstance(v, dict) else v)
                         for k, v in h.items()}
            whence[name] = f"{d.name}/{f.name}"
    return out, whence


# ---------------------------------------------------------------------------
# Second-opinion merge
# ---------------------------------------------------------------------------

def merge_peinl(rated: list[dict], export_path: pathlib.Path) -> list[dict]:
    """Merge a second-opinion export (a JSON list of answers per clip) into the rated records.

    Unusable clips become excluded, unclear clips stay provisional, changed clips are
    confirmed on the reviewer's text, other answered clips are confirmed unchanged, and
    unanswered clips stay as they are.
    """
    ret = {r["segment_id"]: r for r in _load(export_path)}
    who = next((r.get("reviewer") or "" for r in ret.values() if r.get("reviewer")), "Peinl")
    n = collections.Counter()

    for rec in rated:
        r = ret.get(rec["segment_id"])
        if r is None or not r.get("answered"):
            n["not_answered"] += 1
            continue
        prov = rec.setdefault("provenance", {})
        prov["second_opinion_by"] = who
        if r.get("note"):
            prov.setdefault("second_opinion_note", r["note"])

        if r.get("unusable"):
            rec["tier"] = "excluded"
            rec["tier_why"] = f"{who} ruled the clip unusable on the second opinion."
            # The reference text stays unchanged; a rewrite by the reviewer is kept in provenance.
            if r.get("changed_by_reviewer"):
                prov["second_opinion_text_discarded"] = r["reviewer_text"]
            n["excluded"] += 1
        elif r.get("still_unclear"):
            rec["tier"] = "provisional"
            rec["tier_why"] = (f"{who} listened and could not settle it either. Two German "
                               "readers disagree or abstain, so it is not a reference.")
            # Marks a clip the reviewer heard but could not settle, unlike an unanswered clip.
            prov["second_opinion_unresolved"] = True
            n["still_unclear"] += 1
        elif r.get("changed_by_reviewer"):
            prov["anjana_text"] = r.get("anjana_text", rec["reference_text"])
            rec["reference_text"] = r["reviewer_text"]
            rec["tier"] = "confirmed"
            rec["tier_why"] = f"Corrected by {who} on the second opinion."
            rec["open_questions"] = []
            n["corrected"] += 1
        else:
            rec["tier"] = "confirmed"
            rec["tier_why"] = f"Certified by {who} on the second opinion; text unchanged."
            rec["open_questions"] = []
            n["confirmed"] += 1

    print(f"\nMERGED {export_path.name} — reviewer: {who}")
    for k in ("confirmed", "corrected", "still_unclear", "excluded", "not_answered"):
        print(f"  {k:<16}{n[k]:>5}")
    return rated


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def resolution_of(rec: dict) -> dict:
    """Return the clip's resolution state, who settled it and who is still awaited.

    Settled clips (confirmed or excluded) also carry `readers`, the number of readers who
    heard them.
    """
    tier, prov = rec["tier"], rec.get("provenance", {})
    by = prov.get("second_opinion_by")
    readers = 2 if by else 1
    if tier == "excluded":
        return {"state": "excluded", "settled_by": by or prov.get("rater", "Anjana"),
                "readers": readers, "awaiting": None}
    if tier == "confirmed":
        return {"state": "confirmed", "settled_by": by or prov.get("rater", "Anjana"),
                "readers": readers, "awaiting": None}
    if prov.get("second_opinion_unresolved"):
        return {"state": "unresolved", "settled_by": None, "awaiting": None,
                "why": f"{by} heard it too and could not settle it. Nothing is owed on this "
                       "clip; it stays out of the score permanently."}
    return {"state": "pending_second_opinion", "settled_by": None,
            "awaiting": by or "a second German reader"}


def build(rated: list[dict], draw: dict[str, dict], hyps: dict[str, dict[str, str]]) -> list[dict]:
    bad = SCOPE.excluded()
    out = []

    for rec in rated:
        sid = rec["segment_id"]
        d = draw.get(sid, {})
        prov = rec.get("provenance", {})
        res = resolution_of(rec)
        ref_n = normalize(rec["reference_text"], CFG)

        if res["state"] == "excluded":
            reason = f"{res['settled_by']} ruled the clip unusable"
        elif sid in bad:
            reason = bad[sid]
        else:
            reason = None

        mh = {m: hyps[m][sid] for m in PANEL if m in hyps and sid in hyps[m]}

        out.append({
            "segment_id": sid,
            "meeting_id": rec["meeting_id"],
            "speaker": rec["speaker"],
            "start_s": rec["start_s"],
            "end_s": rec["end_s"],
            "duration_s": rec["duration_s"],
            "clip": f"goldset/clips_v9/clips/{sid}.mp3",

            "decile": rec["decile"],
            "difficulty": rec["difficulty"],
            "words_per_s": rec["words_per_s"],
            "n_words": rec["n_words"],

            "reference_text": rec["reference_text"],
            "tier": rec["tier"],
            "tier_why": rec["tier_why"],
            "open_questions": rec.get("open_questions", []),
            "resolution": res,

            "scoring": {
                "eligible": reason is None and res["state"] == "confirmed",
                # Upper bound: True for confirmed and pending clips, as if every pending clip were confirmed;
                # excluded and unresolved clips never count.
                "eligible_when_provisional_resolved":
                    reason is None and res["state"] != "unresolved",
                "exclude_reason": reason,
                "boundary_why": d.get("boundary_why"),
                "boundary_votes": rec.get("provenance", {}).get("boundary_votes"),
                "boundary_sil_s": rec.get("provenance", {}).get("boundary_sil_s"),
                "long_form_risk": bool(d.get("long_form_risk")),
            },

            "bias": {
                "seed_model": prov.get("seed_model"),
                "reference_is_seed_verbatim":
                    normalize(prov.get("seed_text", ""), CFG) == ref_n,
                "rater_edited": not prov.get("rater_left_seed_unchanged", False),
                "identical_to": sorted(m for m, t in mh.items() if normalize(t, CFG) == ref_n),
            },

            "model_hypotheses": mh,
            "provenance": prov,
        })
    return out


def meta(segments: list[dict], hyps: dict, whence: dict, rated_src: pathlib.Path = RATED) -> dict:
    tiers = collections.Counter(s["tier"] for s in segments)
    states = collections.Counter(s["resolution"]["state"] for s in segments)
    elig = [s for s in segments if s["scoring"]["eligible"]]
    ident = collections.Counter(m for s in segments for m in s["bias"]["identical_to"])
    floor = _load(FLOOR)["headline"]["cross_rater_peinl_vs_anjana"] if FLOOR.exists() else None

    per_dec = collections.Counter(s["decile"] for s in segments)
    per_dec_elig = collections.Counter(s["decile"] for s in elig)

    pending = states.get("pending_second_opinion", 0)
    two = sum(1 for s in elig if s["resolution"].get("readers") == 2)
    one = len(elig) - two
    # Clips the reviewer heard. Provenance holds the first reader's text only where the
    # reviewer changed it, so `rewritten` counts the rewrites.
    heard = [s for s in segments if s["resolution"].get("readers") == 2]
    certified = [s for s in heard if s["resolution"]["state"] == "confirmed"]
    rewritten = sum(1 for s in certified if "anjana_text" in s.get("provenance", {}))
    dropped = len(heard) - len(certified)

    if pending:
        first = (
            f"Score `scoring.eligible` only. {pending} clips are still waiting on a second "
            "German reader, and the ones that went unresolved are the hard ones, so the "
            "eligible set is decile-skewed towards easy audio and understates every system. "
            "Report a headline word error rate only when `pending_second_opinion` reaches "
            "zero; until then the eligible set supports relative statements under the caveat.")
    else:
        first = (
            "Score `scoring.eligible` only. Every clip has now been settled and the eligible "
            f"set is the drawn sample minus {tiers.get('excluded', 0)} unusable clips, so it is "
            "no longer decile-skewed — check `per_decile_eligible` below rather than assuming "
            "it. Two things still stand between this file and a headline word error rate. "
            "First, the panel is incomplete: a ranking over the systems present is a ranking "
            "over the systems that seeded the reference. Second, reading depth is uneven — "
            f"{two} references were heard by two German readers and {one} by one. The second "
            f"reader heard {len(heard)} clips: he rewrote {rewritten} of the {len(certified)} "
            f"he certified and ruled the other {dropped} unusable. Those {len(heard)} were "
            "selected as the doubtful ones so that rate does not transfer, but nobody has "
            "measured what a second ear would do to the rest. Split by `resolution.readers` "
            "before quoting a number, and print the noise floor beside it.")

    return {
        "version": "v10",
        "built_from": [rated_src.name, DRAW.name, *sorted(set(whence.values())),
                       "asr_eval_scope.py", FLOOR.name],
        "read_this_first": first,
        "seeding_declaration": (
            "Every reference in this file was seeded from consensus/canary_diar+whisper_turbo "
            "and edited by a human. Those two systems, and whisper_turbo_de as the same family "
            "as one of them, are therefore partly scored against text they helped write. "
            "`bias.identical_to` names the clips where a system's hypothesis IS the reference "
            "after normalisation, i.e. where it scores zero for free. Split the word error rate "
            "by `bias.rater_edited` before quoting any comparison, and give the ranking on the "
            "systems that did not seed."),
        "normalisation": {"applied_to": "the identical_to and seed_verbatim flags only; the "
                                        "reference text is stored verbatim", **CFG.as_dict()},
        "population": {
            "n": POPULATION,
            "definition": "v9 segments with an agreed boundary, inside the loosened band, one "
                          "speaker, difficulty computable (draw_goldset_v9.py)",
            "deciles": N_DECILES,
            "per_decile_approx": POPULATION // N_DECILES,
            "note": "Deciles are equal-size tenths and 15 were drawn from each, so sample "
                    "weights are uniform by construction. They stop being uniform once clips "
                    "become ineligible — weight each decile by per_decile / per_decile_eligible "
                    "when the eligible set is not flat.",
        },
        "counts": {
            "segments": len(segments),
            "by_tier": dict(tiers),
            "by_state": dict(states),
            "eligible_now": len(elig),
            "eligible_when_resolved":
                sum(1 for s in segments if s["scoring"]["eligible_when_provisional_resolved"]),
            "per_decile": {str(d): per_dec.get(d, 0) for d in range(1, N_DECILES + 1)},
            "per_decile_eligible": {str(d): per_dec_elig.get(d, 0) for d in range(1, N_DECILES + 1)},
            "minutes_eligible": round(sum(s["duration_s"] for s in elig) / 60, 1),
        },
        "reading_depth": {
            "eligible_two_readers": two,
            "eligible_one_reader": one,
            "second_reader_heard": len(heard),
            "second_reader_certified": len(certified),
            "second_reader_rewrote": rewritten,
            "second_reader_ruled_unusable": dropped,
            "use": "Split every word error rate on `resolution.readers`. The one-reader clips "
                   "are the ones nobody raised a question about, not the ones checked twice; "
                   "the gap between the two groups is the closest thing this set has to an "
                   "estimate of its own residual reference error.",
        },
        "panel": {
            "systems": PANEL,
            # Counted over these segments only; a hypothesis file can hold more
            # clips than the gold set.
            "hypotheses_present": {m: sum(1 for s in segments if m in s["model_hypotheses"])
                                   for m in PANEL},
            "hypotheses_in_file": {m: len(hyps.get(m, {})) for m in PANEL if m in hyps},
            "hypotheses_from": whence,
            "missing": [m for m in PANEL if m not in hyps],
            "blocker": "A word error rate over an incomplete panel is not a comparison. The "
                       "missing systems are the only ones that did not seed the reference, so "
                       "they are also the only way to separate the v9 segmentation gain from "
                       "the seeding advantage.",
            "reference_is_own_output": dict(ident),
        },
        "noise_floor": {
            **(floor or {}),
            "use": "Print beside every word error rate. Differences smaller than this are not "
                   "distinguishable from two competent humans hearing the same audio.",
            "caveat": "n is single digits and both readings are by the two people who built the "
                      "set. A floor, never 'the human error rate'.",
        },
        "merge_slot": {
            "how": "python build_goldset_v10.py --peinl <second_opinion_export.json>",
            "expects": "the export written by build_second_opinion_app.py",
            "rule": {"unusable": "excluded", "still_unclear": "stays provisional",
                     "changed_by_reviewer": "confirmed on his text",
                     "confirmed": "confirmed on her text",
                     "not answered": "untouched"},
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--peinl", help="second-opinion export to merge before building")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    if a.peinl:
        rated = merge_peinl(_load(RATED), pathlib.Path(a.peinl))
        MERGED.write_text(json.dumps(rated, ensure_ascii=False, indent=1), encoding="utf-8")
        src = MERGED
        print(f"  wrote {MERGED.name} — plain rebuilds now read this, not {RATED.name}")
    else:
        src = MERGED if MERGED.exists() else RATED
        rated = _load(src)
        print(f"reading rated text from {src.name}")

    draw = {s["segment_id"]: s for s in _load(DRAW)}
    hyps, whence = panel_hypotheses()
    segments = build(rated, draw, hyps)
    m = meta(segments, hyps, whence, src)

    out = pathlib.Path(a.out)
    out.write_text(json.dumps({"meta": m, "segments": segments}, ensure_ascii=False, indent=1),
                   encoding="utf-8")

    c = m["counts"]
    print(f"\nGOLD SET v10 — {out}\n")
    print(f"  {'state':<26}{'n':>5}{'minutes':>10}")
    for k, v in c["by_state"].items():
        mins = sum(s["duration_s"] for s in segments if s["resolution"]["state"] == k) / 60
        print(f"  {k:<26}{v:>5}{mins:>10.1f}")
    print(f"  {'TOTAL':<26}{c['segments']:>5}"
          f"{sum(s['duration_s'] for s in segments) / 60:>10.1f}")

    print(f"\n  scoreable now                 {c['eligible_now']:>5}"
          f"   ({c['minutes_eligible']} min)")
    print(f"  scoreable once resolved       {c['eligible_when_resolved']:>5}")

    print(f"\n  decile   drawn  eligible   <- the cost of scoring before the open clips are resolved")
    for d in range(1, N_DECILES + 1):
        k = str(d)
        print(f"  {d:>6}{c['per_decile'][k]:>8}{c['per_decile_eligible'][k]:>10}")

    print(f"\n  panel hypotheses on these clips")
    for mdl in PANEL:
        n = m["panel"]["hypotheses_present"][mdl]
        own = m["panel"]["reference_is_own_output"].get(mdl, 0)
        src = m["panel"]["hypotheses_from"].get(mdl, "")
        flag = f"   {src.split('/')[0]}" if n else "   <- MISSING, cluster"
        seed = f"   reference is its own output on {own}" if own else ""
        print(f"    {mdl:<20}{n:>5} / {len(segments)}{flag}{seed}")

    nf = m["noise_floor"].get("disagreement_rate_pooled")
    if nf:
        print(f"\n  human-human floor             {nf}"
              f"   ({m['noise_floor'].get('words_compared')} words)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
