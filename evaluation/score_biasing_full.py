"""Scores the full-meeting Canary biasing run in a sweep JSON file: the plain arm beam4_alpha0.0
against the boosted arm beam4_alpha1.0, with the per-meeting bias terms from
dist/cluster_biasing_full/manifest_biasing_full.json.
The five pass/fail rules cover gold-clip WER and term precision with the strict matcher of
score_sweep (R1, R2), correct spellings of the name errors that
goldset/name_reachability.json marks as "offered" (R3), segments turned
English (R4), and the share of new list terms that replace a similar plain-arm word (R5).
All results go to goldset/bias_full_scores.json, with ENABLE only if all five rules pass.

    python score_biasing_full.py bias_full/sweep_full.json
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import score_sweep as SW
from language_gate import EN, DE, WORD, MIN_WORDS, MARGIN

MAN = REPO / "dist/cluster_biasing_full/manifest_biasing_full.json"
PLAIN, BOOST = "beam4_alpha0.0", "beam4_alpha1.0"


def english(t: str) -> bool:
    w = [x.lower() for x in WORD.findall(t)]
    return len(w) >= MIN_WORDS and (sum(x in EN for x in w) - sum(x in DE for x in w)) / len(w) > MARGIN


def has(text: str, term: str) -> bool:
    return re.search(rf"(?<![\w-]){re.escape(term)}(?![\w-])", text, re.I) is not None


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    sweep = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    assert PLAIN in sweep and BOOST in sweep, f"expected arms {PLAIN} and {BOOST}"
    man = json.loads(MAN.read_text(encoding="utf-8"))
    terms = man["bias_terms"]
    txt = {arm: {k: v["text"] for k, v in sweep[arm].items() if "text" in v} for arm in (PLAIN, BOOST)}
    both = sorted(set(txt[PLAIN]) & set(txt[BOOST]))
    changed = [k for k in both if txt[PLAIN][k] != txt[BOOST][k]]
    print(f"segments decoded in both arms: {len(both)} of {man['n']}   changed by boosting: {len(changed)}")
    if not changed:
        print("BOOSTING WAS INERT — nothing to score. Check cell 5 on the cluster.")
        return 2
    res = {"n_both": len(both), "n_changed": len(changed)}

    # R1/R2: gold clips, strict matcher
    gold = json.loads((REPO / "goldset/goldset_v10.json").read_text(encoding="utf-8"))
    refs = {s["segment_id"]: s for s in gold["segments"] if s["scoring"]["eligible"]}
    sub = {arm: {k: v for k, v in sweep[arm].items() if k in refs} for arm in (PLAIN, BOOST)}
    g = SW.score(sub, refs, terms, SW.strict)
    p, b = g[PLAIN], g[BOOST]
    r1 = b["wer"] - p["wer"] <= 0.010
    r2 = p["precision"] - b["precision"] <= 0.05
    print(f"\nGOLD ({p['n']} clips)   plain WER {p['wer']:.4f}  boosted {b['wer']:.4f}   "
          f"recall {p['recall']:.1%} -> {b['recall']:.1%}   precision {p['precision']:.1%} -> {b['precision']:.1%}   "
          f"F1 {p['f1']:.3f} -> {b['f1']:.3f}")
    print(f"   gold-clip sweep for comparison: F1 0.882 -> 0.924 at the same setting")
    print(f"   R1 WER not worse by > 0.010 : {'PASS' if r1 else 'FAIL'}")
    print(f"   R2 precision drop <= 5 pp   : {'PASS' if r2 else 'FAIL'}")
    res["gold"] = {"plain": p, "boosted": b, "R1": r1, "R2": r2}

    # R3: offered name errors
    rows = [r for r in json.loads((REPO / "goldset/name_reachability.json").read_text(encoding="utf-8"))["rows"]
            if r["verdict"] == "offered"]
    print(f"\nOFFERED NAME ERRORS ({len(rows)}: correct spelling was in the previous document)")
    gain = 0
    r3rows = []
    for r in rows:
        mid, ref = r["meeting"], r["reference"]
        seg = [k for k in both if k.startswith(mid + "_")]
        c = {arm: sum(has(txt[arm][k], ref) for k in seg) for arm in (PLAIN, BOOST)}
        w = {arm: sum(any(has(txt[arm][k], h) for h in r["heard_as"]) for k in seg) for arm in (PLAIN, BOOST)}
        gain += c[BOOST] - c[PLAIN]
        r3rows.append({**r, "right": c, "wrong": w})
        print(f"   {mid:8} {ref:14} right {c[PLAIN]:2d} -> {c[BOOST]:2d}   "
              f"wrong ({'/'.join(r['heard_as'])[:24]}) {w[PLAIN]:2d} -> {w[BOOST]:2d}")
    r3 = gain > 0
    print(f"   R3 correct spellings gained: {gain:+d} : {'PASS' if r3 else 'FAIL'}")
    res["offered"] = {"rows": r3rows, "gain": gain, "R3": r3}

    # R4: segments turned English
    new_en = [k for k in both if english(txt[BOOST][k]) and not english(txt[PLAIN][k])]
    r4 = not new_en
    print(f"\nR4 segments turned English by boosting: {len(new_en)} : {'PASS' if r4 else 'FAIL'}")
    for k in new_en[:5]:
        print(f"   {k}: {txt[BOOST][k][:100]}")
    res["R4"], res["new_english"] = r4, new_en

    # R5: replacement or insertion
    from handover_naming import score
    new_terms = []
    for k in changed:
        mid = k.split("_")[0]
        words = re.findall(r"[\wÄÖÜäöüß][\wÄÖÜäöüß\-]*", txt[PLAIN][k])
        for t in terms.get(mid, []):
            if has(txt[BOOST][k], t) and not has(txt[PLAIN][k], t):
                best = max(((score(w.replace("-", ""), t.replace("-", "")), w) for w in words), default=(0.0, ""))
                new_terms.append({"segment": k, "term": t, "kind": "replacement" if best[0] >= 0.62 else "insertion",
                                  "replaced": best[1] if best[0] >= 0.62 else None,
                                  "plain": txt[PLAIN][k], "boosted": txt[BOOST][k]})
    ins = [x for x in new_terms if x["kind"] == "insertion"]
    share = 1 - len(ins) / len(new_terms) if new_terms else 1.0
    r5 = share >= 0.80
    print(f"\nR5 list terms written only by the boosted arm: {len(new_terms)}   replacements "
          f"{len(new_terms) - len(ins)}  insertions {len(ins)}   ({share:.0%} replacements) : "
          f"{'PASS' if r5 else 'FAIL'} — read every one")
    for x in sorted(new_terms, key=lambda x: x["kind"])[:60]:
        tag = f"{x['replaced']} -> {x['term']}" if x["replaced"] else f"INSERTED {x['term']}"
        print(f"   {x['kind']:11} {x['segment']}  {tag}\n      plain  : {x['plain'][:110]}\n      boosted: {x['boosted'][:110]}")
    res["R5"], res["new_terms"] = r5, new_terms

    ok = all([r1, r2, r3, r4, r5])
    res["verdict"] = "ENABLE" if ok else "DO NOT ENABLE"
    print(f"\nVERDICT: {res['verdict']}   (R1 {r1}, R2 {r2}, R3 {r3}, R4 {r4}, R5 {r5})")
    out = REPO / "goldset" / "bias_full_scores.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
