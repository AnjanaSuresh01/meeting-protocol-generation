"""Scores new-item coverage of draft protocols with a cross-encoder reranker instead of shared words.

A new item is a bullet of the reference protocol that is not in the previous week's protocol
(coverage_semantic.new_items). A bi-encoder embedding (coverage_semantic.py) compares two vectors
made separately; a cross-encoder reads both bullets together, the usual retrieve-then-rerank
setup for deciding whether two texts say the same thing. The reranker is BAAI/bge-reranker-v2-m3
(XLM-RoBERTa based, multilingual), a model family different from the Qwen-based LISA models that
wrote the drafts; jina-reranker-v3 is built on Qwen3 and is therefore not used.

For each new bullet (with its parent topic as context), the candidates are the K_EMB = 8 draft
bullets most similar by embedding plus the K_LEX = 4 with the most shared content words; the
reranker scores each (reference, draft) pair, and the bullet counts as recorded if word overlap
(postedit_effort.covered) already says so or the best reranker score is at least TAU. Scores are
cached in goldset/rerank_cache_bge_v2_m3.json.

--calibrate chooses TAU on the april27 hand labels (dev; highest agreement, ties broken by how
close the number of covered bullets is to the hand count) and tests it once on the held-out
june22 hand labels, both on the v6 drafts. TAU is accepted only if the held-out agreement reaches
ACCEPT_MIN = 0.70, set in advance to clearly exceed the 0.58 agreement of word overlap. Without
--calibrate, the script scores "prev" (last week's protocol alone) and the given arms (default
v10) on every meeting of evaluate_protocols.REF with the calibrated TAU.

    python coverage_rerank.py --calibrate     -> goldset/coverage_rerank_calibration_2026-10-01.json
    python coverage_rerank.py v10 [v11 ...]   -> goldset/coverage_rerank_2026-10-01.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import coverage_semantic as CS                                       # noqa: E402

RERANKER = "BAAI/bge-reranker-v2-m3"
CAL = REPO / "goldset" / "coverage_rerank_calibration_2026-10-01.json"
OUT = REPO / "goldset" / "coverage_rerank_2026-10-01.json"
SCORE_CACHE = REPO / "goldset" / "rerank_cache_bge_v2_m3.json"
K_EMB, K_LEX = 8, 4
ACCEPT_MIN = 0.70

_rr = None
_scores: dict[str, float] = {}


def _load() -> None:
    if SCORE_CACHE.exists() and not _scores:
        _scores.update(json.loads(SCORE_CACHE.read_text(encoding="utf-8")))


def rerank(pairs: list[tuple[str, str]]) -> list[float]:
    global _rr
    _load()
    keys = [hashlib.sha1((a + "␞" + b).encode("utf-8")).hexdigest() for a, b in pairs]
    todo = [i for i, k in enumerate(keys) if k not in _scores]
    if todo:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        if _rr is None:
            _rr = (AutoTokenizer.from_pretrained(RERANKER),
                   AutoModelForSequenceClassification.from_pretrained(RERANKER).eval())
        tok, model = _rr
        for s in range(0, len(todo), 16):
            idx = todo[s:s + 16]
            with torch.no_grad():
                b = tok([pairs[i][0] for i in idx], [pairs[i][1] for i in idx], padding=True,
                        truncation=True, max_length=256, return_tensors="pt")
                logit = model(**b).logits.view(-1).float()
                prob = torch.sigmoid(logit).tolist()
            for i, p in zip(idx, prob):
                _scores[keys[i]] = float(p)
        SCORE_CACHE.write_text(json.dumps(_scores), encoding="utf-8")
    return [_scores[k] for k in keys]


def best_scores(items: list[dict], draft: list[dict]) -> list[float]:
    if not items or not draft:
        return [0.0] * len(items)
    a = CS.embed([i["ctx"] for i in items])
    b = CS.embed([d["ctx"] for d in draft])
    sims = a @ b.T
    out = []
    for r, it in enumerate(items):
        cand = set(np.argsort(-sims[r])[:K_EMB].tolist())
        if it["content"]:
            ov = [len(it["content"] & d["content"]) / len(it["content"]) for d in draft]
            cand |= set(np.argsort(-np.array(ov))[:K_LEX].tolist())
        sc = rerank([(it["ctx"], draft[j]["ctx"]) for j in sorted(cand)])
        out.append(max(sc) if sc else 0.0)
    return out


def lexical(items, draft) -> list[bool]:
    import postedit_effort as PE
    pool = [d["content"] for d in draft]
    return [bool(i["content"]) and PE.covered(i["content"], pool) for i in items]


def calibrate() -> dict:
    dev_i, dev_y = CS._hand_items("april27")
    tst_i, tst_y = CS._hand_items("june22")
    dev_d = CS.bullets_ctx(CS.draft_md("v6", "april27"))
    tst_d = CS.bullets_ctx(CS.draft_md("v6", "june22"))
    dev_s, tst_s = best_scores(dev_i, dev_d), best_scores(tst_i, tst_d)
    dev_l, tst_l = lexical(dev_i, dev_d), lexical(tst_i, tst_d)
    acc = lambda p, y: sum(a == b for a, b in zip(p, y)) / len(y)
    grid = [round(x / 100, 2) for x in range(1, 100)]
    best = max((acc([l or s >= t for l, s in zip(dev_l, dev_s)], dev_y),
                -abs(sum(l or s >= t for l, s in zip(dev_l, dev_s)) - sum(dev_y)), t) for t in grid)
    tau = best[2]
    pred = [l or s >= tau for l, s in zip(tst_l, tst_s)]
    tp = sum(p and y for p, y in zip(pred, tst_y))
    fp = sum(p and not y for p, y in zip(pred, tst_y))
    fn = sum((not p) and y for p, y in zip(pred, tst_y))
    test_acc = acc(pred, tst_y)
    res = {"reranker": RERANKER, "tau": tau, "accept_min": ACCEPT_MIN,
           "accepted": test_acc >= ACCEPT_MIN,
           "dev_april27": {"n": len(dev_y), "hand_covered": sum(dev_y), "lexical_covered": sum(dev_l),
                           "lexical_agreement": round(acc(dev_l, dev_y), 3),
                           "rerank_covered": sum(l or s >= tau for l, s in zip(dev_l, dev_s)),
                           "rerank_agreement": round(best[0], 3)},
           "test_june22": {"n": len(tst_y), "hand_covered": sum(tst_y), "lexical_covered": sum(tst_l),
                           "lexical_agreement": round(acc(tst_l, tst_y), 3),
                           "rerank_covered": sum(pred), "rerank_agreement": round(test_acc, 3),
                           "precision": round(tp / max(1, tp + fp), 3), "recall": round(tp / max(1, tp + fn), 3),
                           "false_covered": [i["text"] for i, p, y in zip(tst_i, pred, tst_y) if p and not y],
                           "false_missing": [i["text"] for i, p, y in zip(tst_i, pred, tst_y) if (not p) and y]}}
    CAL.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    return res


def score_arms(arms: list[str], tau: float) -> dict:
    from evaluate_protocols import REF
    out = {}
    for arm in ["prev"] + arms:
        n = lc = rc = 0
        per = {}
        for mid in REF:
            md = CS.draft_md(arm, mid)
            if md is None:
                continue
            items = CS.new_items(mid)
            d = CS.bullets_ctx(md)
            lx = lexical(items, d)
            sc = best_scores(items, d)
            rr = [l or s >= tau for l, s in zip(lx, sc)]
            per[mid] = {"new": len(items), "lexical": sum(lx), "rerank": sum(rr)}
            n += len(items); lc += sum(lx); rc += sum(rr)
        out[arm] = {"new": n, "lexical_covered": lc, "rerank_covered": rc,
                    "lexical_missed_pct": round(100 * (n - lc) / n, 1),
                    "rerank_missed_pct": round(100 * (n - rc) / n, 1), "per_meeting": per}
    return out


def main() -> int:
    if "--calibrate" in sys.argv:
        print(json.dumps(calibrate(), indent=2, ensure_ascii=False))
        return 0
    cal = json.loads(CAL.read_text(encoding="utf-8"))
    if not cal["accepted"]:
        print(f"NOT ACCEPTED: held-out agreement {cal['test_june22']['rerank_agreement']} < {ACCEPT_MIN}."
              " Scores below are for the record only.")
    arms = [a for a in sys.argv[1:] if not a.startswith("-")] or ["v10"]
    res = score_arms(arms, cal["tau"])
    for arm, r in res.items():
        label = "last week alone" if arm == "prev" else arm
        print(f"  {label:16s} new {r['new']}: missed by word overlap {r['new'] - r['lexical_covered']} "
              f"({r['lexical_missed_pct']} %), missed by the reranker {r['new'] - r['rerank_covered']} "
              f"({r['rerank_missed_pct']} %)")
    OUT.write_text(json.dumps({"tau": cal["tau"], "accepted": cal["accepted"], "arms": res}, indent=2,
                              ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
