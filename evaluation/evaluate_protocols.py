"""Scores every generated protocol (each ASR model in MODELS x LISA flash and pro) against the human-written protocol of its meeting.

Per protocol:
  * coverage      ROUGE-L (LCS) F1 against the human protocol, after casefolding and
                  punctuation removal (a lexical baseline)
  * name_recall   share of the names and terms of the human protocol's roster that appear
                  (thesis_pipeline.metrics.ne_scores)
  * name_prec     name precision, with the known names absent from the human protocol as extra
                  candidates, as a guard against names of people who were not there
  * sections      number of person sections ('##'/'###' headings or bold bullets)
  * words         length; a protocol under 150 or over 2,500 words counts as degenerate
The Backlog sections are removed from the human protocols before scoring (see strip_backlog).
(model, meeting) cells whose transcript transcript_gate.py rated neither "clean" nor "suspect"
are excluded. Systems are ranked by mean coverage with a 95% bootstrap interval, and ranked a
second time on only the meetings on which every system passed the gate.

    PYTHONUTF8=1 python evaluate_protocols.py          # long-form outputs (pipeline_outputs_extracted/outputs)
    PYTHONUTF8=1 python evaluate_protocols.py --v10    # segment-wise outputs (outputs_v10)
    options: --with-backlog keeps the Backlog in the reference; --ungated disables the gate

Writes pipeline_outputs_extracted/evaluation/layer1_protocol_eval.md and .json
(layer1_protocol_eval_v10.* with --v10).
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

from thesis_pipeline.metrics import ne_scores, build_attendance_roster, bootstrap_ci
from thesis_pipeline.text_norm import normalize, NormConfig

REPO = Path(__file__).resolve().parent
# --v10 scores outputs_v10 (segment-wise decoding, diarized protocols); without it, the
# long-form outputs. File names are the same in both, so only the base directory and the
# model folder names change.
USE_V10 = "--v10" in sys.argv
if USE_V10:
    OUT = REPO / "outputs_v10"
    MODELS = ["canary_diar", "cohere", "granite41",
              "whisper_large_v3", "whisper_turbo", "whisper_turbo_de"]
else:
    OUT = REPO / "pipeline_outputs_extracted" / "outputs"
    # six ASR models; granite41 takes the place of the older Granite Speech 3.3 ("granite")
    MODELS = ["whisper", "whisper_turbo", "whisper_turbo_de", "granite41", "cohere", "canary_diar"]
LISA = ["flash", "pro"]

# Human-written reference protocols (.md); a meeting whose file is missing is skipped. The
# april13 minutes are converted from a Wiki PDF export that flattens bullet levels 2-5, so
# april13 is usable for content measures but not for structural depth
# (see evaluate_speaker_attribution.HUMAN).
REF = {
    "march09": "Meeting_2026-03-09.md",
    "april13": "Meeting_2026-04-13.md",
    "april27": "Meeting_2026-04-27.md", "may11": "Meeting_2026-05-11.md",
    "may18": "Meeting_2026-05-18.md", "june01": "Meeting_2026-06-01.md",
    "june15": "Meeting_2026-06-15.md", "june22": "Meeting_2026-06-22.md",
}
KNOWN = json.loads((REPO / "config.json").read_text(encoding="utf-8"))["known_names"]

_NORM = NormConfig(casefold=True, strip_punct=True)
_PERSON = re.compile(r"^\s*(?:#{2,3}\s+(?!#)|\*\s+\*\*)(.+?)(?:\*\*)?\s*$", re.M)


def strip_backlog(ref: str) -> tuple[str, int]:
    """Remove the Backlog sections from a human protocol before scoring against it.

    The generated protocol is a draft that by design does not contain the Backlog, which is
    written by hand into the final protocol, so scoring the draft against the Backlog would
    count a part it is meant to omit as missing. Every bullet headed "Backlog" (bold or plain)
    is dropped together with everything under it, up to the next bullet at the same or a
    shallower indent. Returns the remaining text and the number of dropped lines.

    --with-backlog keeps the full reference, for a comparison with and without the Backlog.
    """
    if "--with-backlog" in sys.argv:
        return ref, 0
    # Matches both heading forms of the human protocols: '* **Backlog**' and a plain
    # per-person '* Backlog' (optionally with a colon).
    head = re.compile(r"^[*-]\s+[*_]*\s*backlog\s*[*_]*\s*:?\s*$", re.I)
    out, dropped, skip_indent = [], 0, None
    for line in ref.splitlines():
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        if skip_indent is not None:
            # leave the backlog block at the next bullet at the same or a shallower level
            if stripped and indent <= skip_indent and stripped[:1] in "*-":
                skip_indent = None
            else:
                dropped += 1
                continue
        if head.match(stripped):
            skip_indent = indent
            dropped += 1
            continue
        out.append(line)
    return "\n".join(out), dropped


def rouge_l_f1(ref: str, hyp: str) -> float:
    r = normalize(ref, _NORM).split()
    h = normalize(hyp, _NORM).split()
    n, m = len(r), len(h)
    if n == 0 or m == 0:
        return 0.0
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        ri = r[i - 1]
        for j in range(1, m + 1):
            tmp = dp[j]
            dp[j] = prev + 1 if ri == h[j - 1] else (dp[j] if dp[j] >= dp[j - 1] else dp[j - 1])
            prev = tmp
    lcs = dp[m]
    prec, rec = lcs / m, lcs / n
    return round(2 * prec * rec / (prec + rec), 4) if (prec + rec) else 0.0


def n_person_sections(md: str) -> int:
    return len(_PERSON.findall(md))


def score_one(protocol: str, reference: str) -> dict:
    roster = build_attendance_roster(reference, KNOWN)          # who/what the human protocol names
    absent = [e for e in KNOWN if e not in roster]              # hallucination guard
    ne = ne_scores(protocol, roster, extra_candidates=absent)
    return {
        "coverage": rouge_l_f1(reference, protocol),
        "name_recall": ne.recall,
        "name_prec": ne.precision,
        "sections": n_person_sections(protocol),
        "words": len(protocol.split()),
    }


def load_gate() -> dict:
    """Return the (model, meeting) cells whose transcript transcript_gate.py rated neither "clean" nor "suspect".

    These cells are excluded from the ranking, because a protocol built from degenerate ASR
    output (for example a looping transcript) does not rank the ASR system. Reads
    transcript_gate_v10.json with --v10, else transcript_gate.json; without the file the
    ranking is ungated. --ungated disables the gate."""
    if "--ungated" in sys.argv:
        return {}
    p = REPO / ("transcript_gate_v10.json" if USE_V10 else "transcript_gate.json")
    if not p.exists():
        print(f"  !! {p.name} not found — run transcript_gate.py --apply first; "
              f"ranking is UNGATED\n")
        return {}
    gate = json.loads(p.read_text(encoding="utf-8"))
    return {(m, mid) for m, ms in gate.items() for mid, c in ms.items()
            if c.get("verdict") not in ("clean", "suspect")}


def main():
    raw = {m: (REPO / f).read_text(encoding="utf-8")
           for m, f in REF.items() if (REPO / f).exists()}
    refs, n_lines = {}, 0
    for m, t in raw.items():
        refs[m], d = strip_backlog(t)
        n_lines += d
    print(f"references: {sorted(refs)}  ({len(refs)} meetings)")
    if n_lines:
        before = sum(len(t.split()) for t in raw.values())
        after = sum(len(t.split()) for t in refs.values())
        print(f"  BACKLOG STRIPPED from the reference: {n_lines} lines, "
              f"{before - after} of {before} words ({(before - after) / before:.0%}).")
        print("  The system must not produce the backlog, so it is not")
        print("  scored against it.  (--with-backlog restores the old reference.)")
    else:
        print("  reference INCLUDES the backlog section (--with-backlog)")
    print()

    excluded = load_gate()
    dropped = []

    rows = {}                                                  # (model,lisa) -> list of per-meeting dicts
    for model in MODELS:
        for lisa in LISA:
            per = []
            for mid, ref in refs.items():
                if (model, mid) in excluded:
                    dropped.append(f"{model}/{mid}")
                    continue
                p = OUT / model / mid / f"protocol_{lisa}.md"
                if p.exists() and p.stat().st_size > 5:
                    per.append(score_one(p.read_text(encoding="utf-8"), ref))
            if per:
                rows[(model, lisa)] = per

    if dropped:
        uniq = sorted(set(dropped))
        print(f"GATED OUT — degenerate ASR input, excluded from the ranking ({len(uniq)} cells):")
        for c in uniq:
            print(f"   {c}")
        print()

    # aggregate + rank by mean coverage
    table = []
    for (model, lisa), per in rows.items():
        cov = [d["coverage"] for d in per]
        _, lo, hi = bootstrap_ci(cov) if len(cov) > 1 else (0.0, min(cov), max(cov))
        table.append({
            "model": model, "lisa": lisa, "n": len(per),
            "coverage": round(statistics.mean(cov), 3),
            "cov_lo": round(lo, 3), "cov_hi": round(hi, 3),
            "name_recall": round(statistics.mean(d["name_recall"] for d in per), 3),
            "name_prec": round(statistics.mean(d["name_prec"] for d in per), 3),
            "sections": round(statistics.mean(d["sections"] for d in per), 1),
            "words": round(statistics.mean(d["words"] for d in per)),
            "degenerate": sum(1 for d in per if d["words"] < 150 or d["words"] > 2500),
        })
    table.sort(key=lambda r: r["coverage"], reverse=True)

    hdr = f"{'rank':>4} {'model':16} {'lisa':5} {'n':>2} {'coverage':>16} {'name_rec':>8} {'name_prec':>9} {'sect':>5} {'words':>6} {'degen':>5}"
    print(hdr); print("-" * len(hdr))
    lines_md = ["| # | ASR model | LISA | n | Coverage (ROUGE-L) | Name recall | Name prec | Sections | Words | Degen |",
                "|--:|---|---|--:|--:|--:|--:|--:|--:|--:|"]
    for i, r in enumerate(table, 1):
        cov = f"{r['coverage']:.3f} [{r['cov_lo']:.2f}-{r['cov_hi']:.2f}]"
        print(f"{i:>4} {r['model']:16} {r['lisa']:5} {r['n']:>2} {cov:>16} "
              f"{r['name_recall']:>8.2f} {r['name_prec']:>9.2f} {r['sections']:>5} {r['words']:>6} {r['degenerate']:>5}")
        lines_md.append(f"| {i} | {r['model']} | {r['lisa']} | {r['n']} | {r['coverage']:.3f} "
                        f"({r['cov_lo']:.2f}–{r['cov_hi']:.2f}) | {r['name_recall']:.2f} | {r['name_prec']:.2f} "
                        f"| {r['sections']:.1f} | {r['words']} | {r['degenerate']} |")

    # Systems scored on different meetings are not comparable, so a second ranking uses only
    # the meetings on which every system passed the gate.
    common = [mid for mid in sorted(refs) if not any((m, mid) in excluded for m in MODELS)]
    print(f"\nCOMMON-SUBSET RANKING — the {len(common)} meeting(s) every system passed the gate on:"
          f" {', '.join(common) if common else '(none)'}")
    if len(common) < 2:
        print("   Too few meetings for a stable cross-system ranking. Report per-system n")
        print("   (table above) and do NOT quote a single winner from it.")
    if common:
        ctab = []
        for model in MODELS:
            for lisa in LISA:
                cov = []
                for mid in common:
                    p = OUT / model / mid / f"protocol_{lisa}.md"
                    if p.exists() and p.stat().st_size > 5:
                        cov.append(score_one(p.read_text(encoding="utf-8"), refs[mid])["coverage"])
                if cov:
                    ctab.append((statistics.mean(cov), model, lisa, len(cov)))
        ctab.sort(reverse=True)
        print(f"\n   {'model':16} {'lisa':5} {'n':>2} {'coverage':>9}")
        for covm, model, lisa, n in ctab:
            print(f"   {model:16} {lisa:5} {n:>2} {covm:>9.3f}")

    evald = REPO / "pipeline_outputs_extracted" / "evaluation"
    evald.mkdir(exist_ok=True)
    suffix = "_v10" if USE_V10 else ""
    (evald / f"layer1_protocol_eval{suffix}.md").write_text(
        "# Layer 1 — Protocol quality vs human reference\n\n"
        f"Meetings scored: {', '.join(sorted(refs))}. Coverage = ROUGE-L F1 (lexical baseline; "
        "AlignScore faithfulness added next). Name recall/precision via metrics.ne_scores against "
        "each meeting's attendance roster. Ranked by mean coverage; bracket = 95% bootstrap CI.\n\n"
        + "\n".join(lines_md) + "\n", encoding="utf-8")
    (evald / f"layer1_protocol_eval{suffix}.json").write_text(
        json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote -> {evald/f'layer1_protocol_eval{suffix}.md'}")


if __name__ == "__main__":
    main()
