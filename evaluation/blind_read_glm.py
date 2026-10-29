"""Repeats the blind coverage reading of the 100 new reference items with further reader models on LISA.

Each item is one bullet of a reference protocol. The blind sheets
(goldset/blind_read_sheet_<meeting>_2026-10-01<tag>.md, april27 and june22, plain and "_think")
pair it with four candidate bullets from each of two draft protocols, labeled X and Y with
randomized sides. For each draft the reader decides whether a candidate records the same fact,
in any wording ("covered"), or not ("missing"); a match in wording or level of detail is not
required. The reader sees only the sheets; the key files that map X and Y to arms are opened
only when scoring. v14ctrl stands on both sheets of a meeting with re-randomized sides, so
every reader also reads it twice.

BLIND_READER selects the reader: "glm" (LISA Pro, GLM 5.3 Flash, reasoning effort from
GLM_REASONING_EFFORT, default "low"; the default reader) or "kolibri" (LISA Mini, Kolibri 1,
trained on German and English only). BLIND_RUN=N writes a repeat run on identical input to
separate *_runN files. LISA_BASE_URL and LISA_API_KEY must be set. Temperature is 0.

--score unblinds and compares the reader with the first reader's judgments
(goldset/blind_read_verdicts_<meeting>_2026-10-01<tag>.json, from the blind_coverage_read.py reading):
agreement and Cohen's kappa per arm and pooled, the arm comparisons by exact McNemar test, and
the reader's self-consistency on v14ctrl. --retest compares the repeat runs of one reader.
--score-all gives the pairwise agreement of the three readers on the 300 judgments of v14ctrl,
v14seg and v14think.

    python blind_read_glm.py --selftest     # one invented item; checks the API route only (stdout)
    python blind_read_glm.py --read         # all 4 sheets, resumable
                                            #   -> goldset/blind_read_verdicts_<meeting>_2026-10-01<tag>_<reader>.json
    python blind_read_glm.py --score        # also the default without a flag
                                            #   -> goldset/blind_read_<reader>_score_2026-10-28.json
    python blind_read_glm.py --retest       # -> goldset/blind_read_<reader>_retest_2026-10-28.json
    python blind_read_glm.py --score-all    # -> goldset/blind_read_three_readers_2026-10-28.json

Settings and the decision rule set before the run: goldset/blind_read_glm_preregistration_2026-10-28.md
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
G = REPO / "goldset"
# Labels as listed by LISA's /api/models.
# GLM reasoning effort defaults to "low": at "medium" an item can spend the whole 8,192-token
# budget on reasoning and return no answer (see the amendment in the preregistration file).
READERS = {"glm": {"model": "lisa-pro", "label": "GLM 5.3 Flash (320B A18B)",
                   "effort": os.environ.get("GLM_REASONING_EFFORT", "low")},
           "kolibri": {"model": "lisa-mini-2026-10", "label": "Kolibri 1 (78B A3.5B, Aleph Alpha)",
                       "effort": None}}
READER = os.environ.get("BLIND_READER", "glm")
MODEL, EFFORT = READERS[READER]["model"], READERS[READER]["effort"]
# BLIND_RUN=N repeats a reader on identical input and writes separate *_runN files (see retest()).
RUN = os.environ.get("BLIND_RUN", "")
SUFFIX = READER + (f"_run{RUN}" if RUN else "")
SHEETS = [(mid, tag) for tag in ("", "_think") for mid in ("april27", "june22")]

PROMPT = """You compare German meeting minutes. Below is one bullet from the reference minutes of a meeting, followed by the candidate bullets of two draft minutes of the same meeting, labeled X and Y.

Every line has the form "heading — bullet". The text after the last " — " is the bullet. The text before it is the heading it stands under and only gives context. A line without " — " is a bullet on its own.

For each draft separately, decide whether at least one of its listed candidates records the same fact as the reference bullet.
- covered: a candidate states the core fact of the reference bullet, in any wording. More or less detail is fine if the core statement is the same.
- missing: no candidate states it. A candidate that only shares the topic or some words but says something different does not count.
Judge X and Y independently. Use only the listed candidates.

Answer with JSON only, no other text:
{"X": "covered" or "missing", "Y": "covered" or "missing", "why_X": "<one short sentence>", "why_Y": "<one short sentence>"}

REFERENCE:
{item}

X:
{x}

Y:
{y}"""


def out_file(mid: str, tag: str, reader: str = SUFFIX) -> Path:
    return G / f"blind_read_verdicts_{mid}_2026-10-01{tag}_{reader}.json"


def parse_sheet(mid: str, tag: str) -> list[dict]:
    text = (G / f"blind_read_sheet_{mid}_2026-10-01{tag}.md").read_text(encoding="utf-8")
    items, cur, side = [], None, None
    for line in text.splitlines():
        m = re.match(r"^### (\d+)\. (.*)$", line)
        if m:
            cur = {"n": m.group(1), "item": m.group(2).strip(), "X": [], "Y": []}
            items.append(cur)
            side = None
        elif line in ("X:", "Y:") and cur is not None:
            side = line[0]
        elif line.startswith("  - ") and cur is not None and side:
            cur[side].append(line[4:])
    for it in items:
        assert it["X"] and it["Y"], f"{mid}{tag} item {it['n']}: empty side"
    return items


def call(prompt: str, tries: int = 4) -> tuple[dict, dict]:
    """Send one prompt to LISA and return (parsed answer, metadata), retrying on errors and unparsable answers.

    web_search is switched off explicitly because LISA Pro lists it as a default feature;
    enable_thinking is not sent because GLM then returns its whole answer as reasoning and the
    content stays empty (see generate_protocols.lisa)."""
    import requests
    base, key = os.environ["LISA_BASE_URL"].rstrip("/"), os.environ["LISA_API_KEY"]
    body = {"model": MODEL, "temperature": 0, "max_tokens": 8192,
            "features": {"web_search": False}, "stream": False,
            "messages": [{"role": "user", "content": prompt}]}
    if EFFORT:                                           # GLM only; Kolibri is not a reasoning model
        body["reasoning_effort"] = EFFORT
    for k in range(tries):
        try:
            t0 = time.time()
            r = requests.post(base + "/chat/completions", json=body, timeout=(45, 600),
                              headers={"Authorization": "Bearer " + key})
            r.raise_for_status()
            d = r.json()
            ch = d["choices"][0]
            txt = (ch["message"].get("content") or "").strip()
            m = re.search(r"\{.*\}", txt, re.S)
            v = json.loads(m.group(0)) if m else {}
            if v.get("X") in ("covered", "missing") and v.get("Y") in ("covered", "missing"):
                meta = {"s": round(time.time() - t0, 1), "finish": ch.get("finish_reason"),
                        "usage": d.get("usage"), "model": d.get("model")}
                return v, meta
            print(f"[unparsable answer, retry {k + 1}] {txt[:120]!r}", flush=True)
        except Exception as e:                                   # noqa: BLE001 — retried, then raised
            print(f"[{type(e).__name__}, retry {k + 1}] {str(e)[:120]}", flush=True)
        time.sleep(10 * (k + 1))
    raise RuntimeError("no valid judgment after retries")


def fmt(lines: list[str]) -> str:
    return "\n".join(f"- {c}" for c in lines)


def read_all() -> int:
    for mid, tag in SHEETS:
        f = out_file(mid, tag)
        done = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
        items = parse_sheet(mid, tag)
        for it in items:
            if it["n"] in done:
                continue
            v, meta = call(PROMPT.replace("{item}", it["item"]).replace("{x}", fmt(it["X"]))
                           .replace("{y}", fmt(it["Y"])))
            done[it["n"]] = {"X": v["X"], "Y": v["Y"], "why_X": v.get("why_X", ""),
                             "why_Y": v.get("why_Y", ""), **meta}
            f.write_text(json.dumps(done, indent=1, ensure_ascii=False), encoding="utf-8")
            print(f"{mid}{tag} {it['n']}/{len(items)}: X {v['X']}, Y {v['Y']} ({meta['s']} s)", flush=True)
    return 0


def selftest() -> int:
    """Check the API route with an invented item that occurs nowhere in the corpus: X records it, Y does not."""
    v, meta = call(PROMPT.replace("{item}", "Druckerraum — Neuer Toner für den Drucker im 2. Stock bestellt")
                   .replace("{x}", fmt(["Druckerraum — Toner für Drucker im zweiten Stock nachbestellt",
                                        "Mensa — Speiseplan für nächste Woche hängt aus"]))
                   .replace("{y}", fmt(["Druckerraum — Drucker im 2. Stock zeigt Papierstau",
                                        "Mensa — Speiseplan für nächste Woche hängt aus"])))
    print(json.dumps({"verdict": v, **meta}, ensure_ascii=False, indent=1))
    ok = v["X"] == "covered" and v["Y"] == "missing"
    print("selftest", "PASSED" if ok else "FAILED (expected X covered, Y missing)")
    return 0 if ok else 1


def kappa(a: list[bool], b: list[bool]) -> tuple[float, float]:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return po, (po - pe) / (1 - pe) if pe < 1 else float("nan")


def score() -> int:
    from scipy.stats import binomtest
    first: dict[tuple[str, str], bool] = {}            # (arm, sheet:item) -> covered
    second: dict[tuple[str, str], bool] = {}
    for mid, tag in SHEETS:
        key = json.loads((G / f"blind_read_key_{mid}_2026-10-01{tag}.json").read_text(encoding="utf-8"))
        v1 = json.loads((G / f"blind_read_verdicts_{mid}_2026-10-01{tag}.json").read_text(encoding="utf-8"))
        v2 = json.loads(out_file(mid, tag).read_text(encoding="utf-8"))
        assert set(v1) == set(v2) == set(key), f"{mid}{tag}: item sets differ"
        for n in key:
            for side in ("X", "Y"):
                arm, cell = key[n][side], f"{mid}:{n}"
                sheet_arm = arm + ("@think" if tag and arm == "v14ctrl" else "")
                first[(sheet_arm, cell)] = v1[n][side] == "covered"
                second[(sheet_arm, cell)] = v2[n][side] == "covered"
    res: dict = {"reader_2": f"{MODEL} ({READERS[READER]['label']}), reasoning_effort={EFFORT}, temperature 0"}
    print(res["reader_2"])
    arms = sorted({a for a, _ in first})
    pooled_a, pooled_b = [], []
    for arm in arms:
        cells = sorted(c for a, c in first if a == arm)
        a = [first[(arm, c)] for c in cells]
        b = [second[(arm, c)] for c in cells]
        po, k = kappa(a, b)
        only1 = sum(x and not y for x, y in zip(a, b))
        only2 = sum(y and not x for x, y in zip(a, b))
        res[arm] = {"items": len(cells), "reader_1": sum(a), "reader_2": sum(b), "agree": round(po, 3),
                    "kappa": round(k, 3), "covered_only_reader_1": only1, "covered_only_reader_2": only2}
        print(f"  {arm:14s} reader 1 {sum(a):3d}  reader 2 {sum(b):3d} of {len(cells)}"
              f"   agreement {po:.2f}  kappa {k:.2f}   (only 1: {only1}, only 2: {only2})")
        if not arm.endswith("@think"):
            pooled_a += a
            pooled_b += b
    po, k = kappa(pooled_a, pooled_b)
    res["pooled_v14ctrl_v14seg_v14think"] = {"verdicts": len(pooled_a), "agree": round(po, 3), "kappa": round(k, 3)}
    print(f"  pooled over the three arms ({len(pooled_a)} judgments): agreement {po:.2f}  kappa {k:.2f}")
    # arm comparisons on reader 2's judgments (exact McNemar test)
    for other in ("v14seg", "v14think"):
        base = "v14ctrl" if other == "v14seg" else "v14ctrl@think"
        cells = sorted(c for a, c in second if a == other)
        b_ = sum(second[(base, c)] and not second[(other, c)] for c in cells)
        c_ = sum(second[(other, c)] and not second[(base, c)] for c in cells)
        p = binomtest(b_, b_ + c_, 0.5).pvalue if b_ + c_ else 1.0
        res[f"reader_2_{base}_vs_{other}"] = {"only_ctrl": b_, f"only_{other}": c_, "p_mcnemar": round(p, 4)}
        print(f"  reader 2: {base} vs {other}: only ctrl {b_}, only {other} {c_}, exact McNemar p = {p:.4f}")
    cells = sorted(c for a, c in second if a == "v14ctrl")
    same = sum(second[("v14ctrl", c)] == second[("v14ctrl@think", c)] for c in cells)
    res["reader_2_self_consistency_v14ctrl"] = {"same": same, "of": len(cells)}
    print(f"  reader 2 read v14ctrl twice (two sheets, re-randomized sides): same judgment on {same} of {len(cells)}")
    (G / f"blind_read_{SUFFIX}_score_2026-10-28.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


def retest() -> int:
    """Compare every pair of runs of the same reader on identical input: same judgment per item and side.

    Run 1 has no suffix; runs 2, 3, ... are whichever *_runN files exist."""
    runs = {"1": READER}
    for k in range(2, 10):
        if out_file("april27", "", f"{READER}_run{k}").exists():
            runs[str(k)] = f"{READER}_run{k}"
    verdicts = {r: {} for r in runs}
    for mid, tag in SHEETS:
        for r, suffix in runs.items():
            v = json.loads(out_file(mid, tag, suffix).read_text(encoding="utf-8"))
            for n in v:
                for s in ("X", "Y"):
                    verdicts[r][(mid + tag, n, s)] = v[n][s]
    cells = sorted(verdicts["1"])
    assert all(set(verdicts[r]) == set(cells) for r in runs), "runs cover different items"
    res = {"verdicts_per_run": len(cells)}
    names = sorted(runs)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            same = sum(verdicts[a][c] == verdicts[b][c] for c in cells)
            res[f"run{a}_vs_run{b}"] = {"same": same, "of": len(cells)}
            print(f"  run {a} vs run {b}: same judgment on {same} of {len(cells)} ({100 * same / len(cells):.1f} %)")
    if len(names) > 2:
        all_same = sum(len({verdicts[r][c] for r in names}) == 1 for c in cells)
        res["all_runs_same"] = {"same": all_same, "of": len(cells)}
        print(f"  all {len(names)} runs give the same judgment on {all_same} of {len(cells)}")
    (G / f"blind_read_{READER}_retest_2026-10-28.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


def score_all() -> int:
    """Report the pairwise agreement of the three readers on the 300 judgments of v14ctrl, v14seg and v14think.

    Only the first sheet's v14ctrl judgments are used, so no judgment counts twice."""
    readers = {"first": {}, "glm": {}, "kolibri": {}}
    for mid, tag in SHEETS:
        key = json.loads((G / f"blind_read_key_{mid}_2026-10-01{tag}.json").read_text(encoding="utf-8"))
        files = {"first": G / f"blind_read_verdicts_{mid}_2026-10-01{tag}.json",
                 "glm": out_file(mid, tag, "glm"), "kolibri": out_file(mid, tag, "kolibri")}
        for r, f in files.items():
            v = json.loads(f.read_text(encoding="utf-8"))
            for n in key:
                for side in ("X", "Y"):
                    if tag and key[n][side] == "v14ctrl":
                        continue
                    readers[r][(key[n][side], f"{mid}:{n}")] = v[n][side] == "covered"
    cells = sorted(readers["first"])
    res = {"verdicts": len(cells)}
    for a, b in (("first", "glm"), ("first", "kolibri"), ("glm", "kolibri")):
        po, k = kappa([readers[a][c] for c in cells], [readers[b][c] for c in cells])
        res[f"{a}_vs_{b}"] = {"agree": round(po, 3), "kappa": round(k, 3)}
        print(f"  {a:7s} vs {b:7s}: agreement {po:.2f}  kappa {k:.2f}  ({len(cells)} judgments)")
    all3 = sum(readers["first"][c] == readers["glm"][c] == readers["kolibri"][c] for c in cells)
    res["all_three_agree"] = all3
    print(f"  all three give the same judgment on {all3} of {len(cells)}")
    (G / "blind_read_three_readers_2026-10-28.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    if "--score-all" in sys.argv:
        raise SystemExit(score_all())
    if "--retest" in sys.argv:
        raise SystemExit(retest())
    if "--read" in sys.argv:
        raise SystemExit(read_all())
    raise SystemExit(score())
