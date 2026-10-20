"""Score a beam size x boosting strength (alpha) decoding sweep on the eligible gold-set clips.

For each config it reports recall, precision and F1 of the bias terms of each clip's meeting
under a strict whole-token match and a substring match, plus corpus WER, and compares each
biased config with the alpha 0.0 control at the same beam.

Reads dist/cluster_biasing/manifest_biasing.json and goldset/goldset_v10.json relative to this
file's folder and writes goldset/sweep_scores.json there.

    python score_sweep.py sweep.json

sweep.json maps each config tag (beam<N>_alpha<X>) to {segment_id: {"text": ...}}.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import jiwer                                                     # noqa: E402

from thesis_pipeline.text_norm import NormConfig, normalize      # noqa: E402

CFG = NormConfig(casefold=True)


def sq(s: str) -> str:
    return "".join(c for c in s.lower() if c.isalnum())


def strict(text: str, term: str) -> bool:
    return sq(term) in {sq(t) for t in text.split()}


def substring(text: str, term: str) -> bool:
    return sq(term) in sq(text)


def order(tags):
    return sorted(tags, key=lambda t: (int(t.split("_")[0][4:]), float(t.split("alpha")[1])))


def score(sweep, refs, terms, matcher):
    """Return per-config bias-term TP/FP/FN, recall, precision, F1 and corpus WER.

    Only clips in refs with a decoded text count; a config without any is left out.
    """
    out = {}
    for tag in order(sweep):
        tp = fp = fn = 0
        refs_txt, hyps_txt = [], []
        for sid, v in sweep[tag].items():
            if sid not in refs or "text" not in v:
                continue
            ref, hyp = refs[sid]["reference_text"], v["text"]
            for t in terms.get(refs[sid]["meeting_id"], []):
                in_ref, in_hyp = matcher(ref, t), matcher(hyp, t)
                tp += in_ref and in_hyp
                fn += in_ref and not in_hyp
                fp += in_hyp and not in_ref
            refs_txt.append(ref)
            hyps_txt.append(hyp)
        if not refs_txt:
            continue
        wer = float(jiwer.process_words([normalize(x, CFG) for x in refs_txt],
                                        [normalize(x, CFG) for x in hyps_txt]).wer)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[tag] = {"n": len(refs_txt), "tp": tp, "fp": fp, "fn": fn,
                    "recall": rec, "precision": prec, "f1": f1, "wer": wer}
    return out


def show(rows, label):
    print(f"\n{label}")
    print("-" * 86)
    print(f"{'config':20}{'recall':>18}{'forced in':>11}{'prec':>9}{'F1':>8}{'WER':>10}")
    for tag, v in rows.items():
        star = "  <- plain" if tag.endswith("alpha0.0") else ""
        rec = f"{v['tp']}/{v['tp'] + v['fn']} ({v['recall']:.1%})"
        print(f"{tag:20}{rec:>18}{v['fp']:>11}{v['precision']:>8.1%}"
              f"{v['f1'] * 100:>8.1f}{v['wer']:>10.4f}{star}")


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python score_sweep.py sweep.json")
        return 1
    sweep = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    man = json.loads((REPO / "dist/cluster_biasing/manifest_biasing.json").read_text(encoding="utf-8"))
    terms = man["bias_terms"]
    gold = json.loads((REPO / "goldset/goldset_v10.json").read_text(encoding="utf-8"))
    refs = {s["segment_id"]: s for s in gold["segments"] if s["scoring"]["eligible"]}

    print("BEAM SIZE x BOOSTING STRENGTH")
    print("=" * 86)

    both = {"substring": score(sweep, refs, terms, substring),
            "strict": score(sweep, refs, terms, strict)}

    show(both["substring"], "PRIMARY — substring match (survives 'ASR-Leaderboard', 'Persona Plex')")
    show(both["strict"], "SENSITIVITY — strict token match (penalises German compounding)")

    print("\n" + "=" * 86)
    print("GAIN OVER THE PLAIN CONTROL AT THE SAME BEAM  (primary matcher)")
    print(f"{'config':20}{'recall':>12}{'prec':>12}{'F1':>10}{'WER':>11}   verdict")
    print("-" * 86)
    rows = both["substring"]
    best = None
    for tag, v in rows.items():
        beam = tag.split("_")[0]
        base = rows.get(f"{beam}_alpha0.0")
        if not base or tag == f"{beam}_alpha0.0":
            continue
        d_rec = (v["recall"] - base["recall"]) * 100
        d_prec = (v["precision"] - base["precision"]) * 100
        d_f1 = (v["f1"] - base["f1"]) * 100
        d_wer = v["wer"] - base["wer"]
        if d_f1 > 0:
            verdict = "BETTER — recall gain survives precision"
        elif d_rec > 0:
            verdict = "over-biasing — recall bought with false insertions"
        else:
            verdict = "no recall gain"
        print(f"{tag:20}{d_rec:>+11.1f}pp{d_prec:>+11.1f}pp{d_f1:>+9.1f}{d_wer:>+11.4f}   {verdict}")
        if best is None or v["f1"] > rows[best]["f1"]:
            best = tag

    overall = max(rows, key=lambda t: rows[t]["f1"])
    print(f"\n  best F1 among biased settings : {best}  (F1 {rows[best]['f1'] * 100:.1f})")
    print(f"  best F1 overall               : {overall}  (F1 {rows[overall]['f1'] * 100:.1f})")
    if overall.endswith("alpha0.0"):
        print("  ** the plain control wins — biasing did not pay off at any strength tested")

    print("\n  Read precision beside recall. A setting that lifts recall while precision falls is")
    print("  over-biasing: the decoder forces listed words into places they do not belong, and a")
    print("  confidently wrong name costs more to post-edit than a missing one.")

    (REPO / "goldset" / "sweep_scores.json").write_text(
        json.dumps(both, indent=1), encoding="utf-8")
    print("\nwrote -> goldset/sweep_scores.json  (both matchers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
