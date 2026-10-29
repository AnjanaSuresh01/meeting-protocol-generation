"""Builds and scores a blind reading test of whether two generated drafts record the new items of the reference protocol.

The comparison is made by reading because no automatic matcher agreed closely enough with the hand labels
of the held-out meeting june22 (word overlap 0.58, IBM granite-embedding-r2 0.58, bge-reranker-v2-m3 0.667):

  1. For each hand-labeled new item of the reference protocol (april27: 31, june22: 69), the four
     bullets of each arm's draft that are most similar to it are listed with their parent topic. The
     twelve nearest bullets by embedding (coverage_semantic) are reranked by the cross-encoder
     (coverage_rerank), so paraphrases surface.
  2. The two arms appear as X and Y, with the side shuffled per item by a fixed seed; the key is
     written to a separate file that is opened only after every item has been judged.
  3. The reader enters covered or missing per item and side in
     goldset/blind_read_verdicts_<meeting>_2026-10-01.json.
  4. --score unblinds the judgments and reports per arm how many items are recorded, with an exact
     McNemar test on the paired judgments.

One reader judges both arms, so this is a paired comparison under one reader, not an inter-rater
measurement. The reader sees only the candidate bullets, so an item recorded elsewhere in the draft
can be missed in both arms alike, and a more detailed arm may be recognizable by its style.

    python blind_coverage_read.py --build april27 v14ctrl v14seg
        -> goldset/blind_read_sheet_april27_2026-10-01.md and goldset/blind_read_key_april27_2026-10-01.json
    python blind_coverage_read.py --score    -> stdout only
    --tag NAME (in either mode) appends _NAME to the three file names.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import coverage_rerank as CR                                         # noqa: E402
import coverage_semantic as CS                                       # noqa: E402

G = REPO / "goldset"
SEED = 20261001
# --tag NAME keeps a second comparison apart from the first (file names end in _2026-10-01_NAME)
TAG = ("_" + sys.argv[sys.argv.index("--tag") + 1]) if "--tag" in sys.argv else ""


def files(mid: str) -> tuple[Path, Path, Path]:
    return (G / f"blind_read_sheet_{mid}_2026-10-01{TAG}.md", G / f"blind_read_key_{mid}_2026-10-01{TAG}.json",
            G / f"blind_read_verdicts_{mid}_2026-10-01{TAG}.json")


def top(item: dict, draft: list[dict], k: int = 4) -> list[str]:
    import numpy as np
    a = CS.embed([item["ctx"]])
    b = CS.embed([d["ctx"] for d in draft])
    order = list(np.argsort(-(a @ b.T)[0])[:12])
    scores = CR.rerank([(item["ctx"], draft[j]["ctx"]) for j in order])
    best = [j for _, j in sorted(zip(scores, order), reverse=True)[:k]]
    return [draft[j]["ctx"] for j in best]


def build(mid: str, arms: list[str]) -> int:
    sheet, keyf, _ = files(mid)
    rng = random.Random(f"{SEED}-{mid}{TAG}")
    items, _ = CS._hand_items(mid)
    drafts = {a: CS.bullets_ctx(CS.draft_md(a, mid)) for a in arms}
    lines, key = [f"# Blind read, {mid} — does each draft record the new reference item? (X / Y randomized)\n"], {}
    for n, it in enumerate(items, 1):
        sides = arms[:]
        rng.shuffle(sides)
        key[str(n)] = {"meeting": mid, "his": it["text"], "X": sides[0], "Y": sides[1]}
        lines.append(f"\n### {n}. {it['ctx']}\n")
        for lab, arm in (("X", sides[0]), ("Y", sides[1])):
            lines.append(f"{lab}:")
            lines += [f"  - {c}" for c in top(it, drafts[arm])]
    sheet.write_text("\n".join(lines) + "\n", encoding="utf-8")
    keyf.write_text(json.dumps(key, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(items)} items -> {sheet.relative_to(REPO)} (key in {keyf.name}: do not open before reading)")
    return 0


def score() -> int:
    from scipy.stats import binomtest
    got: dict[str, dict[str, bool]] = {}
    for keyf in sorted(G.glob(f"blind_read_key_*_2026-10-01{TAG}.json")):
        mid = keyf.name.split("_")[3]
        verf = files(mid)[2]
        if not verf.exists():
            print(f"  {mid}: no judgments yet")
            continue
        key = json.loads(keyf.read_text(encoding="utf-8"))
        ver = json.loads(verf.read_text(encoding="utf-8"))
        for n, v in ver.items():
            for side in ("X", "Y"):
                got.setdefault(key[n][side], {})[f"{mid}:{n}"] = v[side] == "covered"
    arms = sorted(got)
    common = sorted(set.intersection(*(set(got[a]) for a in arms)))
    per_mid = sorted({c.split(":")[0] for c in common})
    for a in arms:
        c = sum(got[a][n] for n in common)
        print(f"  {a:10s} records {c} of {len(common)} new reference items ({100 * c / len(common):.1f} %)"
              f"   [{', '.join(f'{m}: {sum(got[a][n] for n in common if n.startswith(m))}' for m in per_mid)}]")
    b = sum(got[arms[0]][n] and not got[arms[1]][n] for n in common)
    c = sum(got[arms[1]][n] and not got[arms[0]][n] for n in common)
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    print(f"  only {arms[0]}: {b}, only {arms[1]}: {c}, exact McNemar p = {p:.4f}")
    return 0


if __name__ == "__main__":
    if "--build" in sys.argv:
        rest = [a for a in sys.argv[1:] if not a.startswith("-") and a != TAG.lstrip("_")]
        raise SystemExit(build(rest[0], rest[1:]))
    raise SystemExit(score())
