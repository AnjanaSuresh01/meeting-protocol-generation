"""Calibrates automatic coverage judges against hand-labeled reference bullets.

The question each judge answers is whether a generated draft records a given bullet of the
reference protocol. Two meetings carry hand labels for the reference bullets that the previous
protocol does not already cover (`goldset/new_bullet_coverage_{april27,june22}_handcheck.json`).
Every arm decides each of these bullets against the v6 draft
(outputs_deliverable/v6/<meeting>/protocol_pro.md), and the script reports its agreement with the
hand labels, how many bullets it calls covered, how many are covered by hand, its false "covered"
calls and its misses, in total and per meeting.

Arms:
    lexical    content-word overlap >= 0.30 (postedit_effort.covered)
    e2e_pro    end-to-end judgment by LISA Pro (Qwen), the model that wrote the drafts
    e2e_code   end-to-end judgment by LISA Code (MiniMax), a different model family
    qa_pro     question from the bullet -> answer from the draft -> answer compared with the
               bullet, all by LISA Pro
    qa_code    the same question-answer method with LISA Code

The end-to-end and question-answer methods follow "Comprehensiveness Metrics for Automatic
Evaluation of Factual Recall in Text Generation" (ACL 2026 Findings), which reports the
question-answer comparison as the more stable one across judge models and finds that small NLI
models lose context on isolated statements, so no NLI model is tried. The cross-family arms follow
the 2026 LLM-judge bias literature (arXiv 2604.22891, 2604.23178, 2609.17857), which reports that
judges favor output of their own model family.

Before any call, 12 candidate draft bullets per reference bullet are retrieved by stem- and
compound-aware overlap, and bullets are sent in batches of 10, so each prompt stays small. A batch
that fails or does not answer within 90 s is skipped, and its bullets count as not covered.

    python judge_calibrate.py                    # all arms
    python judge_calibrate.py e2e_code qa_pro    # selected arms

Writes goldset/judge_calibration.json; arms measured in earlier runs are kept.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import judge_coverage as J                                       # noqa: E402
from postedit_effort import STOP, content, covered               # noqa: E402
from term_source import PREVIOUS                                 # noqa: E402
from thesis_pipeline.text_norm import NormConfig, normalize      # noqa: E402

CFG = NormConfig(casefold=True, strip_punct=True)
SUF = ("ungen", "ung", "heiten", "heit", "keiten", "keit", "ern", "en", "er", "es", "em", "e", "s")
N_CAND = 12
BATCH = 10
PRO, CODE = "lisa-pro-03-2026", "lisa-code-101"


def stem(w: str) -> str:
    for s in SUF:
        if len(w) - len(s) >= 5 and w.endswith(s):
            return w[: -len(s)]
    return w


def stems(text: str) -> set:
    return {stem(p) for w in normalize(text, CFG).split() for p in re.split(r"[-/]", w)
            if len(p) > 3 and p not in STOP}


def candidates(his: str, draft: list[str], draft_stems: list[set]) -> list[int]:
    a = stems(his)
    if not a:
        return list(range(min(N_CAND, len(draft))))
    sc = []
    for i, d in enumerate(draft_stems):
        n = sum(1 for x in a if x in d or (len(x) >= 6 and any(x in y or y in x for y in d)))
        sc.append((n / len(a), i))
    sc.sort(reverse=True)
    return [i for _, i in sc[:N_CAND]]


SYS_E2E = (
    "Du prüfst, ob ein Protokollpunkt inhaltlich in einem Entwurf vorkommt.\n"
    "Für jeden nummerierten PUNKT bekommst du KANDIDATEN aus dem Entwurf.\n"
    "Antworte für jeden Punkt mit genau einer Zeile: <Nummer>|JA|<Kandidatenbuchstabe>  oder  "
    "<Nummer>|NEIN|-\n"
    "JA nur, wenn ein Kandidat DIESELBE Tatsache festhält, auch in anderen Worten. "
    "Ein Kandidat, der nur dasselbe Thema streift, ohne die Tatsache zu nennen, ist NEIN. "
    "Keine Erklärungen, keine weiteren Zeilen."
)
SYS_QA = (
    "Du beantwortest Fragen AUSSCHLIESSLICH aus den gegebenen Entwurfszeilen.\n"
    "Für jede nummerierte FRAGE gib genau eine Zeile: <Nummer>|<kurze Antwort>  oder  "
    "<Nummer>|UNBEANTWORTBAR\n"
    "Erfinde nichts. Wenn die Entwurfszeilen die Frage nicht beantworten, schreibe "
    "UNBEANTWORTBAR. Keine Erklärungen."
)
SYS_Q = (
    "Formuliere zu jedem nummerierten Protokollpunkt EINE kurze Frage, deren Antwort genau die "
    "Tatsache dieses Punktes ist. Gib je Punkt eine Zeile: <Nummer>|<Frage>. Keine Erklärungen."
)
SYS_CMP = (
    "Vergleiche je Zeile eine ANTWORT mit der TATSACHE.\n"
    "Gib genau eine Zeile je Nummer: <Nummer>|JA  wenn die Antwort dieselbe Tatsache ausdrückt "
    "(auch anders formuliert oder allgemeiner, solange die Tatsache erkennbar ist), sonst "
    "<Nummer>|NEIN. Keine Erklärungen."
)


def _lisa(model, sysmsg, user):
    import generate_protocols as G
    # lang_gate=False: the expected answer is a line such as "1|JA|A", not German prose, so the
    # German-output check would re-roll every reply.
    # A short timeout: a normal batch answers in about 1.2 s, and a batch that does not return
    # within 90 s is skipped instead of blocking the whole arm.
    import time as _t
    t0 = _t.time()
    try:
        r = G.lisa(model, sysmsg, user, timeout=90, retries=0, lang_gate=False)
        print(f"      [{model.split('-')[1]} {_t.time()-t0:.1f}s]", end="", flush=True)
        return r
    except Exception as e:
        print(f"      [batch skipped after {_t.time()-t0:.0f}s: {type(e).__name__}]",
              end="", flush=True)
        raise _Skip()


class _Skip(Exception):
    pass


def _parse(raw: str, n: int) -> dict[int, str]:
    out = {}
    for line in raw.splitlines():
        m = re.match(r"\s*(\d+)\s*\|\s*(.+?)\s*$", line)
        if m and 1 <= int(m.group(1)) <= n:
            out[int(m.group(1))] = m.group(2)
    return out


def run_e2e(new, draft, cand, model) -> list[bool]:
    got = [False] * len(new)
    for s in range(0, len(new), BATCH):
        chunk = list(range(s, min(s + BATCH, len(new))))
        parts = []
        for k, i in enumerate(chunk, 1):
            cs = "\n".join(f"   {chr(65+j)}) {draft[c]}" for j, c in enumerate(cand[i]))
            parts.append(f"{k}. PUNKT: {new[i]}\n   KANDIDATEN:\n{cs}")
        try:
            raw = _lisa(model, SYS_E2E, "\n\n".join(parts))
        except _Skip:
            continue          # a skipped batch does not stop the arm
        for k, v in _parse(raw, len(chunk)).items():
            got[chunk[k - 1]] = v.upper().startswith("JA")
    return got


def run_qa(new, draft, cand, model) -> list[bool]:
    qs: dict[int, str] = {}
    for s in range(0, len(new), BATCH):
        chunk = list(range(s, min(s + BATCH, len(new))))
        try:
            raw = _lisa(model, SYS_Q, "\n".join(f"{k}. {new[i]}" for k, i in enumerate(chunk, 1)))
        except _Skip:
            continue          # a skipped batch does not stop the arm
        for k, v in _parse(raw, len(chunk)).items():
            qs[chunk[k - 1]] = v
    ans: dict[int, str] = {}
    for s in range(0, len(new), BATCH):
        chunk = [i for i in range(s, min(s + BATCH, len(new))) if i in qs]
        if not chunk:
            continue
        parts = []
        for k, i in enumerate(chunk, 1):
            cs = "\n".join(f"   - {draft[c]}" for c in cand[i])
            parts.append(f"{k}. FRAGE: {qs[i]}\n   ENTWURFSZEILEN:\n{cs}")
        try:
            raw = _lisa(model, SYS_QA, "\n\n".join(parts))
        except _Skip:
            continue          # a skipped batch does not stop the arm
        for k, v in _parse(raw, len(chunk)).items():
            ans[chunk[k - 1]] = v
    got = [False] * len(new)
    todo = [i for i in ans if "UNBEANTWORTBAR" not in ans[i].upper()]
    for s in range(0, len(todo), BATCH):
        chunk = todo[s:s + BATCH]
        parts = [f"{k}. ANTWORT: {ans[i]}\n   TATSACHE: {new[i]}" for k, i in enumerate(chunk, 1)]
        try:
            raw = _lisa(model, SYS_CMP, "\n\n".join(parts))
        except _Skip:
            continue          # a skipped batch does not stop the arm
        for k, v in _parse(raw, len(chunk)).items():
            got[chunk[k - 1]] = v.upper().startswith("JA")
    return got


def load(mid: str):
    lab = json.loads((REPO / f"goldset/new_bullet_coverage_{mid}_handcheck.json")
                     .read_text(encoding="utf-8"))["verdicts"]
    his_all = J.ref_bullets(REPO / J.REF[mid])
    carried = J.lexical_covered(his_all, J.ref_bullets(REPO / PREVIOUS[mid]))
    new = [t for i, t in enumerate(his_all, 1) if i not in carried]
    assert len(new) == len(lab), (mid, len(new), len(lab))
    draft = J.draft_bullets(REPO / f"outputs_deliverable/v6/{mid}/protocol_pro.md")
    ds = [stems(d) for d in draft]
    cand = [candidates(t, draft, ds) for t in new]
    return new, [v["verdict"] == "covered" for v in lab], draft, cand


ARMS = {
    "lexical":  lambda n, d, c: [bool(content(t)) and covered(content(t), [content(x) for x in d
                                                                          if content(x)])
                                 for t in n],
    "e2e_pro":  lambda n, d, c: run_e2e(n, d, c, PRO),
    "e2e_code": lambda n, d, c: run_e2e(n, d, c, CODE),
    "qa_pro":   lambda n, d, c: run_qa(n, d, c, PRO),
    "qa_code":  lambda n, d, c: run_qa(n, d, c, CODE),
}


def main(which: list[str]) -> int:
    sets = {m: load(m) for m in ("april27", "june22")}
    truth_n = sum(sum(t) for _, t, _, _ in sets.values())
    tot = sum(len(n) for n, _, _, _ in sets.values())
    print(f"Calibration against {tot} hand-labelled bullets ({truth_n} covered by hand = "
          f"{truth_n/tot:.0%})\n")
    print(f"  {'arm':10s}{'agree':>7}{'says cov':>10}{'true cov':>10}{'false-cov':>11}"
          f"{'missed':>8}   per-meeting agreement")
    out = {}
    for arm in which:
        try:
            tp = fp = fn = tn = 0
            per = []
            for mid, (new, truth, draft, cand) in sets.items():
                got = ARMS[arm](new, draft, cand)
                a = sum(g == t for g, t in zip(got, truth))
                per.append(f"{mid} {a/len(new):.0%}")
                for g, t in zip(got, truth):
                    tp += g and t
                    fp += g and not t
                    fn += (not g) and t
                    tn += (not g) and not t
            agree = (tp + tn) / tot
            out[arm] = {"agree": agree, "says_covered": tp + fp, "true": tp + fn,
                        "false_covered": fp, "missed": fn}
            print(f"  {arm:10s}{agree:>6.0%}{tp+fp:>10}{tp+fn:>10}{fp:>11}{fn:>8}   "
                  + ", ".join(per), flush=True)
        except Exception as e:
            print(f"  {arm:10s} FAILED: {type(e).__name__}: {str(e)[:70]}", flush=True)
    f = REPO / "goldset" / "judge_calibration.json"
    prev = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    prev.update(out)                       # keep arms measured in earlier runs
    f.write_text(json.dumps(prev, indent=1, ensure_ascii=False), encoding="utf-8")
    print("\n  -> goldset/judge_calibration.json")
    return 0


if __name__ == "__main__":
    sys.exit(main([a for a in sys.argv[1:] if a in ARMS] or list(ARMS)))
