"""Replays the naming steps of protokoll.py with two trios of recognizers: A = canary + cohere +
granite41 (the evaluated runs) and B = canary + cohere + whisper_turbo_de (the final trio).
For each meeting, it compares the handover votes, the names-to-check list and the replayed speaker
names of A and B, and checks A against the delivered outputs in outputs_deliverable/v10.
It writes goldset/asr_trio_comparison.json and replaces any existing file of that name.

Usage:
    python compare_asr_trio.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import protokoll as P  # noqa: E402
from handover_naming import name_by_content, name_speakers  # noqa: E402
from term_source import PREVIOUS           # noqa: E402

MEETINGS = ["march09", "april13", "april27", "may11", "may18", "june01", "june15", "june22"]
DATE = {"march09": "2026-03-09", "april13": "2026-04-13", "april27": "2026-04-27", "may11": "2026-05-11",
        "may18": "2026-05-18", "june01": "2026-06-01", "june15": "2026-06-15", "june22": "2026-06-22"}
MAIN = "outputs_v11_final/canary_diar/{mid}/transcript_speakers.txt"
OTHER = "outputs_v10/{model}/{mid}/transcript_speakers.txt"
TRIOS = {"A_evaluated": ["cohere", "granite41"], "B_proposed": ["cohere", "whisper_turbo_de"]}
DELIVERED = REPO / "outputs_deliverable" / "v10"


def read(p: str) -> str:
    return (REPO / p).read_text(encoding="utf-8", errors="replace")


def history(mid: str) -> list[str]:
    """Return the --history list for a meeting: one earlier protocol per date, without the previous
    document's date, with .md preferred over .docx.
    """
    prev_date = re.search(r"(\d{4}-\d{2}-\d{2})", Path(PREVIOUS[mid]).name).group(1)
    by_date = {}
    for p in sorted(REPO.glob("Meeting*"), key=lambda q: (q.suffix != ".md", q.name)):
        m = re.search(r"(\d{4}-\d{2}-\d{2})", p.name)
        if m and p.suffix in (".md", ".docx") and m.group(1) < DATE[mid] and m.group(1) != prev_date:
            by_date.setdefault(m.group(1), p.name)
    return [by_date[d] for d in sorted(by_date)]


def delivered_handover(mid: str) -> dict[str, str]:
    log = (DELIVERED / mid / "run_pro.log").read_text(encoding="utf-8", errors="replace")
    m = re.search(r"named by the (?:chair|meeting lead)'s handover: (.*)", log)
    return dict(x.split("=") for x in m.group(1).split(", ")) if m else {}


def delivered_flags(mid: str) -> list[str]:
    f = DELIVERED / mid / "protocol_pro.namen.md"
    if not f.exists():
        return []
    return [l.strip() for l in f.read_text(encoding="utf-8").splitlines() if l.startswith("* **")]


def flag_lines(rows) -> list[str]:
    import name_check as NC
    md = NC.as_markdown(rows) if rows else ""
    return [l.strip() for l in md.splitlines() if l.startswith("* **")]


def replay_naming(mid: str, handover: dict[str, str]) -> tuple[dict[str, str], dict[str, str]]:
    """Replay the naming chain of protokoll.py (voice profile, handover, topic) with the given handover
    votes and return (replayed mapping, mapping logged by the delivered run). Voice-profile names come
    from the run log, since they do not depend on the other transcripts.
    """
    log = (DELIVERED / mid / "run_pro.log").read_text(encoding="utf-8", errors="replace")

    def grab(pat):
        m = re.search(pat + r": (.*)", log)
        return dict(x.split("=") for x in m.group(1).split(", ")) if m else {}
    mapping = grab(r"named by voice profile")
    used = set(mapping.values())
    for lab, name in handover.items():
        if lab not in mapping and name not in used:
            mapping[lab] = name
            used.add(name)
    mapping = name_by_content(read(MAIN.format(mid=mid)), REPO / PREVIOUS[mid], mapping)
    return mapping, grab(r"speaker names \(voice, handover, topic\)")


def main() -> int:
    report = {}
    for mid in MEETINGS:
        main_tx = read(MAIN.format(mid=mid))
        roster = P.roster_from_document(REPO / PREVIOUS[mid])
        text = (DELIVERED / mid / "protocol_pro.md").read_text(encoding="utf-8")
        hist = [REPO / h for h in history(mid)]
        rec = {"roster": roster, "delivered_handover_applied": delivered_handover(mid),
               "delivered_flags": delivered_flags(mid)}
        for tag, models in TRIOS.items():
            others = [read(OTHER.format(model=m, mid=mid)) for m in models]
            hand = name_speakers([main_tx] + others, roster) if roster else {}
            rows, basis = P.names_to_check(main_tx, others, REPO / PREVIOUS[mid], hist, text)
            rec[tag] = {"handover_votes": hand, "flags": flag_lines(rows), "basis": basis}
        a, b = rec["A_evaluated"], rec["B_proposed"]
        rec["handover_identical"] = a["handover_votes"] == b["handover_votes"]
        names_a, logged = replay_naming(mid, a["handover_votes"])
        names_b, _ = replay_naming(mid, b["handover_votes"])
        rec["names_A_reproduce_log"] = names_a == logged
        rec["names_identical"] = names_a == names_b
        rec["names"] = names_b
        rec["flags_identical"] = a["flags"] == b["flags"]
        rec["A_flags_match_delivered"] = a["flags"] == rec["delivered_flags"]
        rec["flags_only_A"] = [f for f in a["flags"] if f not in b["flags"]]
        rec["flags_only_B"] = [f for f in b["flags"] if f not in a["flags"]]
        report[mid] = rec
        print(f"{mid:8s} handover A={a['handover_votes']} B={b['handover_votes']} identical={rec['handover_identical']}"
              f" | applied in v10: {rec['delivered_handover_applied']}")
        print(f"         flags A={len(a['flags'])} B={len(b['flags'])} identical={rec['flags_identical']}"
              f" | A reproduces delivered list: {rec['A_flags_match_delivered']}")
        print(f"         final names: A reproduces the v10 log {rec['names_A_reproduce_log']},"
              f" B gives the same names {rec['names_identical']}")
    print("ALL 8 MEETINGS: same names with the final trio:",
          all(r["names_identical"] and r["names_A_reproduce_log"] for r in report.values()))
    out = REPO / "goldset" / "asr_trio_comparison.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
