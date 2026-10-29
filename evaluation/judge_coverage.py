"""LLM-as-judge coverage of reference bullets, reported next to the lexical coverage rule.

A reference protocol is one valid version among several. A lexical test of "did the draft cover
this reference bullet?" scores a draft that records the same fact in different words as a miss,
so the ADD rate measures agreement with the reference's phrasing as much as with its content.
This module asks a LISA model instead: for each reference bullet, is that content present
anywhere in the draft, in any wording?

The judge is made auditable, not trusted:
* It must cite the draft bullet it believes covers each reference bullet. A "covered" answer
  without a valid citation is discarded. A German reader can check each citation quickly, and a
  hallucinated "covered" becomes visible.
* All arms are judged in one run with the same prompt, because LISA is nondeterministic.
* Same-family bias is declared, not hidden. LISA (Qwen) judges protocols that LISA wrote and
  cannot be neutral about its own phrasing, so the lexical count (postedit_effort's rule, see
  lexical_covered) is kept beside the judged count rather than replaced. Where the two
  disagree, the disagreement is the result.

References are the Meeting_*.md files in REF (Backlog bullets and bullets under three words
skipped); drafts are outputs_v10/<model>/<meeting>/protocol_<lisa><arm>.md.

    python judge_coverage.py --model canary_diar --arms "" _fixed_carry [--lisa pro|flash]

Writes goldset/judge_coverage.json.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import generate_protocols as G                                   # noqa: E402

TREE = REPO / "outputs_v10"
OUT = REPO / "goldset" / "judge_coverage.json"
REF = {
    "april27": "Meeting_2026-04-27.md", "may11": "Meeting_2026-05-11.md",
    "may18": "Meeting_2026-05-18.md", "june01": "Meeting_2026-06-01.md",
    "june15": "Meeting_2026-06-15.md", "june22": "Meeting_2026-06-22.md",
}
BACKLOG = re.compile(r"^\s*\**\s*backlog", re.I)
LEX_SIM = 0.80

SYSTEM = (
    "Du prüfst, ob ein automatisch erstelltes Besprechungsprotokoll denselben INHALT enthält wie "
    "ein von Hand geschriebenes Referenzprotokoll.\n\n"
    "Du bekommst:\n"
    "  A) nummerierte Stichpunkte aus dem Referenzprotokoll\n"
    "  B) nummerierte Stichpunkte aus dem generierten Protokoll\n\n"
    "Für JEDEN Referenzpunkt entscheide: Kommt dieser Inhalt im generierten Protokoll vor — "
    "egal mit welchen Worten, egal an welcher Stelle, egal ob kürzer oder länger formuliert?\n\n"
    "WICHTIG:\n"
    "1. Es geht um den INHALT, nicht um die Formulierung. Eine andere Wortwahl, eine andere "
    "Reihenfolge oder eine andere Aufteilung auf mehrere Stichpunkte ist KEIN Fehler.\n"
    "2. Teilweise abgedeckt zählt nur dann als abgedeckt, wenn die Kernaussage da ist. Ein "
    "bloß verwandtes Thema ohne die Aussage zählt NICHT.\n"
    "3. Für jeden als abgedeckt markierten Punkt musst du die Nummer des generierten "
    "Stichpunkts angeben, der ihn abdeckt. Ohne Beleg gilt der Punkt als NICHT abgedeckt.\n\n"
    "Antworte AUSSCHLIESSLICH mit JSON in genau dieser Form, ohne weiteren Text:\n"
    '{"covered": [{"ref": 1, "by": 12}, {"ref": 4, "by": 7}]}\n'
    "Referenzpunkte, die nicht abgedeckt sind, lässt du einfach weg."
)


def ref_bullets(path: Path) -> list[str]:
    out, skip = [], None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"^( *)\*\s+(.*)$", raw)
        if not m:
            continue
        depth = len(m.group(1)) // 4
        body = re.sub(r"\*\*(.*?)\*\*", r"\1", m.group(2)).strip()
        if not body:
            continue
        if BACKLOG.match(body):
            skip = depth
            continue
        if skip is not None:
            if depth > skip:
                continue
            skip = None
        if len(body.split()) >= 3:
            out.append(body)
    return out


def draft_bullets(path: Path) -> list[str]:
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"^ *[*-]\s+(.*)$", line)
        if m and len(m.group(1).split()) >= 2:
            out.append(m.group(1).strip())
    return out


def lexical_covered(ref: list[str], draft: list[str]) -> set[int]:
    """1-based indices of the reference bullets covered under the exact test postedit_effort
    uses, imported rather than re-implemented.

    The ADD rate's own rule is content-word overlap >= 0.30. A comparison of whole normalized
    strings with SequenceMatcher >= 0.80 would be far stricter, so a judge compared against it
    would show a gain that comes from the mismatch between the two rules, not from the judge.
    The comparison is only meaningful against the rule the headline number actually uses.
    """
    from postedit_effort import content, covered
    pool = [content(d) for d in draft]
    pool = [p for p in pool if p]
    out = set()
    for i, r in enumerate(ref, 1):
        c = content(r)
        if c and covered(c, pool):
            out.add(i)
    return out


def judge(ref: list[str], draft: list[str], model_id: str) -> tuple[set[int], int]:
    a = "\n".join(f"{i}. {t}" for i, t in enumerate(ref, 1))
    b = "\n".join(f"{i}. {t}" for i, t in enumerate(draft, 1))
    user = (f"=== A) REFERENZPROTOKOLL ({len(ref)} Punkte) ===\n{a}\n\n"
            f"=== B) GENERIERTES PROTOKOLL ({len(draft)} Punkte) ===\n{b}\n=== ENDE ===")
    raw = G.lisa(model_id, SYSTEM, user, timeout=600, retries=1)
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return set(), 0
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return set(), 0
    ok, bad = set(), 0
    for e in data.get("covered", []):
        r, by = e.get("ref"), e.get("by")
        # the citation must exist — a "covered" answer with no valid evidence is discarded
        if isinstance(r, int) and 1 <= r <= len(ref) and isinstance(by, int) and 1 <= by <= len(draft):
            ok.add(r)
        else:
            bad += 1
    return ok, bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="canary_diar")
    ap.add_argument("--arms", nargs="+", default=["", "_fixed_carry"],
                    help="protocol_pro<ARM>.md suffixes to compare")
    ap.add_argument("--lisa", default="pro")
    a = ap.parse_args()
    model_id = G.LISA_MODELS[a.lisa]

    print("LLM-AS-JUDGE COVERAGE — content, not phrasing")
    print("=" * 92)
    print('The reference protocol is not the only correct version. Other versions')
    print('can be equally good or better.\n')
    print(f"judge: {model_id}   system: {a.model}   arms: {[x or 'raw' for x in a.arms]}\n")
    print(f"{'arm':16}{'meeting':10}{'ref':>6}{'draft':>7}{'lexical':>9}{'judged':>8}{'gain':>7}")
    print("-" * 92)

    results = {}
    for arm in a.arms:
        agg = {"ref": 0, "lex": 0, "judged": 0, "discarded": 0}
        for mid, fn in sorted(REF.items()):
            rp, dp = REPO / fn, TREE / a.model / mid / f"protocol_{a.lisa}{arm}.md"
            if not rp.exists() or not dp.exists() or dp.stat().st_size < 80:
                continue
            ref, draft = ref_bullets(rp), draft_bullets(dp)
            if not ref or not draft:
                continue
            lex = lexical_covered(ref, draft)
            jud, bad = judge(ref, draft, model_id)
            agg["ref"] += len(ref)
            agg["lex"] += len(lex)
            agg["judged"] += len(jud)
            agg["discarded"] += bad
            print(f"{arm or 'raw':16}{mid:10}{len(ref):>6}{len(draft):>7}"
                  f"{len(lex):>9}{len(jud):>8}{len(jud)-len(lex):>+7}", flush=True)
            results.setdefault(arm or "raw", {})[mid] = {
                "ref": len(ref), "draft": len(draft),
                "lexical": sorted(lex), "judged": sorted(jud), "discarded": bad}
        if agg["ref"]:
            print(f"{'':16}{'TOTAL':10}{agg['ref']:>6}{'':>7}"
                  f"{agg['lex']:>9}{agg['judged']:>8}{agg['judged']-agg['lex']:>+7}")
            print(f"{'':16}{'coverage':10}{'':>6}{'':>7}"
                  f"{agg['lex']/agg['ref']:>8.0%}{agg['judged']/agg['ref']:>8.0%}")
            print(f"{'':16}ADD rate  lexical {1-agg['lex']/agg['ref']:.1%}"
                  f"   judged {1-agg['judged']/agg['ref']:.1%}"
                  f"   (judgments discarded for a bad citation: {agg['discarded']})\n")
            results.setdefault(arm or "raw", {})["_total"] = agg

    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print("  The judge shares a model family with the writer (both Qwen), so it cannot be")
    print("  neutral about LISA's own phrasing. The lexical figure is kept beside it rather")
    print("  than replaced; where the two disagree, the disagreement is the finding.")
    print(f"\nwrote -> {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
