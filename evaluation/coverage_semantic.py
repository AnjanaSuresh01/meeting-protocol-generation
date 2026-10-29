"""Counts the new reference items a draft records, matched by meaning as well as by words.

The word-overlap rule of postedit_effort.covered() counts a reference bullet as recorded only if
30 % of its content words occur in one generated bullet, so a paraphrase counts as missing. Here
each bullet is embedded together with its parent topic (short bullets such as "Scheint jetzt
endlich zu gehen" carry little meaning on their own) with the multilingual embedding model
IBM granite-embedding-311m-multilingual-r2 (Apache-2.0, German supported, CLS pooling), which
comes from a different model family than the generator. A reference bullet counts as recorded if
its nearest draft bullet reaches cosine similarity THETA, or if the word-overlap rule already
counts it. New items are the reference bullets that the previous protocol does not cover, the
same set as in score_editforward; the previous protocol alone is scored as a baseline row.

THETA is calibrated on the hand labels of the v6 drafts
(goldset/new_bullet_coverage_<meeting>_handcheck.json): it is the value on a grid from 0.70 to
0.95 (step 0.005) with the best agreement on april27 (dev), ties broken by how close the predicted
number of recorded bullets is to the hand count, and it is tested once on june22 (held out).
Completeness as recall of the reference's key facts follows Zhou et al., arXiv 2606.15974, who
also used a judge from a different model family than the systems scored.

    python coverage_semantic.py --calibrate     # choose THETA on april27, test it on june22
    python coverage_semantic.py v10 [v11 ...]   # new-item coverage of deliverable arms

--calibrate writes goldset/coverage_semantic_calibration_2026-10-01.json; scoring reads that file
and writes goldset/coverage_semantic_2026-10-01.json. Embeddings are cached in
goldset/emb_cache_granite_r2.npz.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
MODEL_ID = "ibm-granite/granite-embedding-311m-multilingual-r2"
CACHE = REPO / "goldset" / "emb_cache_granite_r2.npz"
OUT = REPO / "goldset" / "coverage_semantic_2026-10-01.json"
MARKERS = re.compile(r"\[(ÜBERTRAG|PRÜFEN|Aufgabe)\]\s*", re.I)

_model = None
_cache: dict[str, np.ndarray] = {}


def _load_cache() -> None:
    if CACHE.exists() and not _cache:
        z = np.load(CACHE)
        for k in z.files:
            _cache[k] = z[k]


def _save_cache() -> None:
    np.savez_compressed(CACHE, **_cache)


def embed(texts: list[str]) -> np.ndarray:
    global _model
    _load_cache()
    keys = [hashlib.sha1(t.encode("utf-8")).hexdigest() for t in texts]
    todo = [i for i, k in enumerate(keys) if k not in _cache]
    if todo:
        import torch
        from transformers import AutoModel, AutoTokenizer
        if _model is None:
            _model = (AutoTokenizer.from_pretrained(MODEL_ID), AutoModel.from_pretrained(MODEL_ID).eval())
        tok, model = _model
        for s in range(0, len(todo), 32):
            batch = [texts[i] for i in todo[s:s + 32]]
            with torch.no_grad():
                b = tok(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
                e = torch.nn.functional.normalize(model(**b)[0][:, 0].float(), dim=1).numpy()   # weights load as bf16
            for j, i in enumerate(todo[s:s + 32]):
                _cache[keys[i]] = e[j].astype(np.float32)
        _save_cache()
    return np.stack([_cache[k] for k in keys]) if keys else np.zeros((0, 768), np.float32)


def clean(s: str) -> str:
    s = MARKERS.sub("", s)
    s = re.sub(r"[*_`]+", "", s)
    return re.sub(r"\s+", " ", s).strip()


def bullets_ctx(md: str) -> list[dict]:
    """Every bullet that postedit_effort.bullets() counts, in the same order, with its text and
    its parent topic. '##' person headings are not bullets; '###' topic headings are (the
    scorer converts them, because the reference protocol writes its topics as bullets)."""
    import postedit_effort as PE
    out, stack = [], []                      # stack of (indent, text)
    for line in md.splitlines():
        s = line.strip()
        if s.startswith("## ") and not s.startswith("### "):
            stack = []
            continue
        if not re.match(r"^[*-]\s+\S", s):
            continue
        c = PE.content(s)
        ind = len(line) - len(line.lstrip())
        text = clean(re.sub(r"^[*-]\s+", "", s))
        while stack and stack[-1][0] >= ind:
            stack.pop()
        parent = stack[-1][1] if stack else ""
        stack.append((ind, text))
        if c:
            out.append({"text": text, "parent": parent, "content": c,
                        "ctx": (parent + " — " + text) if parent else text})
    return out


def draft_md(arm: str, mid: str) -> str | None:
    import score_editforward as SE
    return SE.arm_text(arm, mid)


def his_md(mid: str) -> str:
    import score_editforward as SE
    from evaluate_protocols import REF, strip_backlog
    return SE.no_backlog(strip_backlog((REPO / REF[mid]).read_text(encoding="utf-8"))[0])


def new_items(mid: str) -> list[dict]:
    """Reference bullets the previous protocol does not cover (score_editforward's NEW set)."""
    import postedit_effort as PE
    import score_editforward as SE
    pb = PE.bullets(SE.previous_only(mid))
    return [b for b in bullets_ctx(his_md(mid)) if not PE.covered(b["content"], pb)]


def best_sim(items: list[dict], draft: list[dict]) -> list[float]:
    if not items or not draft:
        return [0.0] * len(items)
    a = embed([i["ctx"] for i in items])
    b = embed([d["ctx"] for d in draft])
    return [float(x) for x in (a @ b.T).max(axis=1)]     # plain floats: JSON-safe comparisons


def decide(items, draft, theta: float) -> list[bool]:
    import postedit_effort as PE
    sims = best_sim(items, draft)
    hb = [d["content"] for d in draft]
    return [bool((i["content"] and PE.covered(i["content"], hb)) or s >= theta)
            for i, s in zip(items, sims)]


def _hand(mid: str):
    x = json.loads((REPO / f"goldset/new_bullet_coverage_{mid}_handcheck.json").read_text(encoding="utf-8"))
    return [(v["his"], v["verdict"] == "covered") for v in x["verdicts"]]


def _hand_items(mid: str) -> tuple[list[dict], list[bool]]:
    """The hand-labeled bullets, located in the reference protocol to carry their parent topic."""
    import postedit_effort as PE
    mine = bullets_ctx(his_md(mid))
    items, labels = [], []
    for his, lab in _hand(mid):
        h = clean(his)
        hit = next((b for b in mine if b["text"] == h), None) \
            or next((b for b in mine if h[:40] and b["text"].startswith(h[:40])), None)
        if hit is None:
            hit = {"text": h, "parent": "", "content": PE.content(h), "ctx": h}
        items.append(hit)
        labels.append(lab)
    return items, labels


def calibrate() -> dict:
    import postedit_effort as PE
    res = {}
    dev_i, dev_y = _hand_items("april27")
    tst_i, tst_y = _hand_items("june22")
    dev_d = bullets_ctx(draft_md("v6", "april27"))
    tst_d = bullets_ctx(draft_md("v6", "june22"))
    dev_s, tst_s = best_sim(dev_i, dev_d), best_sim(tst_i, tst_d)
    lex = lambda items, d: [bool(i["content"]) and PE.covered(i["content"], [x["content"] for x in d])
                            for i in items]   # a bullet with no content word cannot match by words
    dev_lex, tst_lex = lex(dev_i, dev_d), lex(tst_i, tst_d)

    def acc(pred, y):
        return sum(p == t for p, t in zip(pred, y)) / len(y)

    grid = [round(float(x), 3) for x in np.arange(0.70, 0.951, 0.005)]   # plain floats
    scored = []
    for th in grid:
        pred = [l or s >= th for l, s in zip(dev_lex, dev_s)]
        scored.append((acc(pred, dev_y), -abs(sum(pred) - sum(dev_y)), th))
    best = max(scored)
    theta = best[2]
    tst_pred = [l or s >= theta for l, s in zip(tst_lex, tst_s)]
    tp = sum(p and y for p, y in zip(tst_pred, tst_y))
    fp = sum(p and not y for p, y in zip(tst_pred, tst_y))
    fn = sum((not p) and y for p, y in zip(tst_pred, tst_y))
    res = {"theta": theta,
           "dev_april27": {"n": len(dev_y), "hand_covered": sum(dev_y),
                           "lexical_covered": sum(dev_lex), "lexical_agreement": round(acc(dev_lex, dev_y), 3),
                           "semantic_covered": sum(l or s >= theta for l, s in zip(dev_lex, dev_s)),
                           "semantic_agreement": round(best[0], 3)},
           "test_june22": {"n": len(tst_y), "hand_covered": sum(tst_y),
                           "lexical_covered": sum(tst_lex), "lexical_agreement": round(acc(tst_lex, tst_y), 3),
                           "semantic_covered": sum(tst_pred), "semantic_agreement": round(acc(tst_pred, tst_y), 3),
                           "semantic_precision": round(tp / max(1, tp + fp), 3),
                           "semantic_recall": round(tp / max(1, tp + fn), 3),
                           "false_covered": [i["text"] for i, p, y in zip(tst_i, tst_pred, tst_y) if p and not y],
                           "false_missing": [i["text"] for i, p, y in zip(tst_i, tst_pred, tst_y) if (not p) and y]}}
    return res


def score_arms(arms: list[str], theta: float) -> dict:
    from evaluate_protocols import REF
    out = {}
    for arm in ["prev"] + arms:
        n = lex_c = sem_c = 0
        per = {}
        for mid in REF:
            md = draft_md(arm, mid)
            if md is None:
                continue
            items = new_items(mid)
            d = bullets_ctx(md)
            import postedit_effort as PE
            lexc = [PE.covered(i["content"], [x["content"] for x in d]) for i in items]
            semc = decide(items, d, theta)
            per[mid] = {"new": len(items), "lexical": sum(lexc), "semantic": sum(semc)}
            n += len(items); lex_c += sum(lexc); sem_c += sum(semc)
        out[arm] = {"new": n, "lexical_covered": lex_c, "semantic_covered": sem_c,
                    "lexical_missed_pct": round(100 * (n - lex_c) / n, 1),
                    "semantic_missed_pct": round(100 * (n - sem_c) / n, 1), "per_meeting": per}
    return out


def main() -> int:
    if "--calibrate" in sys.argv:
        r = calibrate()
        print(json.dumps(r, indent=2, ensure_ascii=False))
        OUT.with_name("coverage_semantic_calibration_2026-10-01.json").write_text(
            json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
        return 0
    cal = json.loads(OUT.with_name("coverage_semantic_calibration_2026-10-01.json").read_text(encoding="utf-8"))
    arms = [a for a in sys.argv[1:] if not a.startswith("-")] or ["v10"]
    res = score_arms(arms, cal["theta"])
    print(f"NEW ITEMS RECORDED (theta {cal['theta']}, calibrated on april27, tested on june22)")
    for arm, r in res.items():
        label = "last week alone" if arm == "prev" else arm
        print(f"  {label:16s} new {r['new']}: missed by word overlap {r['new'] - r['lexical_covered']} "
              f"({r['lexical_missed_pct']} %), missed by meaning {r['new'] - r['semantic_covered']} "
              f"({r['semantic_missed_pct']} %)")
    OUT.write_text(json.dumps({"theta": cal["theta"], "model": MODEL_ID, "arms": res}, indent=2,
                              ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
