"""Layout repair for the generated team protocols in outputs_v10/; no LLM is called.

It reorders the '## ' person blocks by when each person's longest block of speech starts
(goldset/speaker_identity.json, goldset/segments_v9_all.json), renames '## SPEAKER_n' headings
to UNNAMED_HEADING, and lists '### ' topics filed under more than one person. protokoll.py
imports split_blocks and UNNAMED_HEADING from this module.

    python protocol_repair.py          # report only
    python protocol_repair.py --apply  # rewrite in place, write goldset/protocol_repairs.json
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from evaluate_speaker_attribution import HUMAN, person_key
from entity_correct import squash

REPO = Path(__file__).resolve().parent
V10 = REPO / "outputs_v10"
LOG = REPO / "goldset" / "protocol_repairs.json"
MODELS = ["canary_diar", "cohere", "granite41",
          "whisper_large_v3", "whisper_turbo", "whisper_turbo_de"]
LISAS = ["flash", "pro"]
GAP = 30.0  # seconds; a longer gap between one person's turns starts a new block
UNNAMED_HEADING = "Nicht zugeordnete Beiträge"


def is_team_arm(name: str) -> bool:
    """Return True for a protocol_*_fixed* team file, whose '##' headings are people.
    Senat files are skipped because their '##' headings are agenda items in session order.
    """
    return name.startswith("protocol_") and "_fixed" in name and "_senat" not in name

def speech_blocks(mid: str) -> dict[str, float]:
    """Return person key -> start (s) of the person's longest speech block in meeting `mid`,
    where a block joins turns at most GAP seconds apart and its length is the speaking time.
    The goldset files are read here, not at import, because protokoll.py imports this module
    without them.
    """
    IDENTITY = json.loads((REPO / "goldset" / "speaker_identity.json")
                          .read_text(encoding="utf-8"))["mapping"]
    SEGMENTS = json.loads((REPO / "goldset" / "segments_v9_all.json").read_text(encoding="utf-8"))
    turns: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for s in SEGMENTS:
        if s["meeting_id"] != mid:
            continue
        who = IDENTITY.get(mid, {}).get(s["speaker"])
        if who:
            turns[person_key(who)].append((s["start_s"], s["end_s"]))
    out = {}
    for who, tl in turns.items():
        ts = sorted(tl)
        best_start, best_len = ts[0][0], 0.0
        s, e, spoken = ts[0][0], ts[0][1], ts[0][1] - ts[0][0]
        for a, b in ts[1:]:
            if a - e <= GAP:
                e = max(e, b)
                spoken += b - a
            else:
                if spoken > best_len:
                    best_start, best_len = s, spoken
                s, e, spoken = a, b, b - a
        if spoken > best_len:
            best_start, best_len = s, spoken
        out[who] = best_start
    return out


def split_blocks(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Split a protocol into the text before the first '## ' heading and a list of
    (heading text, block text including the heading line) pairs.
    """
    lines = text.splitlines(keepends=True)
    idx = [i for i, l in enumerate(lines)
           if l.startswith("## ") and not l.startswith("###")]
    if not idx:
        return text, []
    pre = "".join(lines[:idx[0]])
    blocks = []
    for n, i in enumerate(idx):
        j = idx[n + 1] if n + 1 < len(idx) else len(lines)
        blocks.append((lines[i].strip()[3:].strip(), "".join(lines[i:j])))
    return pre, blocks


def repair(text: str, order: dict[str, float]) -> tuple[str, dict]:
    pre, blocks = split_blocks(text)
    if not blocks:
        return text, {}
    changes: dict[str, object] = {}

    # Rename SPEAKER_n headings
    renamed = []
    fixed_blocks = []
    for head, body in blocks:
        if re.fullmatch(r"SPEAKER[_\s]?\d+", head.strip(), re.I):
            renamed.append(head)
            body = body.replace(f"## {head}", f"## {UNNAMED_HEADING}", 1)
            head = UNNAMED_HEADING
        fixed_blocks.append((head, body))
    if renamed:
        changes["unnamed_cluster_renamed"] = renamed

    # Order blocks by speech start
    def key(hb: tuple[str, str]) -> tuple[int, float, int]:
        k = person_key(hb[0])
        if hb[0] == UNNAMED_HEADING:
            return (2, 0.0, 0)  # unassigned content last
        if k in order:
            return (0, order[k], 0)
        return (1, 0.0, 0)  # no speech start: after the timed blocks

    before = [h for h, _ in fixed_blocks]
    ordered = sorted(enumerate(fixed_blocks), key=lambda x: key(x[1]) + (x[0],))
    after = [h for _, (h, _) in ordered]
    if before != after:
        changes["reordered"] = {"was": before, "now": after}

    # Each block must end with a newline; otherwise, after reordering, the next '##' heading
    # is glued to the last line of the block before it.
    parts = []
    for _, (_, body) in ordered:
        if body and not body.endswith("\n"):
            body += "\n"
        parts.append(body)
    out = pre + "".join(parts)
    if not out.endswith("\n"):
        out += "\n"
    return out, changes


def duplicate_topics(text: str) -> list[str]:
    """Return the normalized '### ' topic titles that appear under more than one person."""
    per: dict[str, set[str]] = defaultdict(set)
    person = None
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("## ") and not s.startswith("###"):
            person = person_key(s[3:])
        elif s.startswith("### ") and person:
            per[squash(s[4:])].add(person)
    return sorted(k for k, v in per.items() if len(v) > 1)


def main() -> int:
    apply = "--apply" in sys.argv
    print("PROTOCOL REPAIR — deterministic, post-generation, no API")
    print("=" * 92)
    print("Layout only. Blocks are permuted and one heading is renamed; no text is reworded,")
    print("no bullet changes owner.\n")

    orders = {mid: speech_blocks(mid) for mid in HUMAN}
    log: dict[str, dict] = {}
    n_re = n_un = n_dup = n_files = 0

    for md in sorted(V10.glob("*/*/protocol_*_fixed*.md")):
        model, mid = md.parent.parent.name, md.parent.name
        if mid not in orders or md.stat().st_size < 80:
            continue
        if not is_team_arm(md.name):
            continue
        text = md.read_text(encoding="utf-8")
        new, ch = repair(text, orders[mid])
        dup = duplicate_topics(text)
        if dup:
            ch["duplicate_topics"] = dup
            n_dup += 1
        if not ch:
            continue
        n_files += 1
        n_re += "reordered" in ch
        n_un += "unnamed_cluster_renamed" in ch
        key = f"{model}/{mid}/{md.name}"
        log[key] = ch
        if apply and new != text:
            md.write_text(new, encoding="utf-8")

    print(f"  protocols with a repair                 : {n_files}")
    print(f"    section order restored                : {n_re}")
    print(f"    diarization label renamed to a note   : {n_un}")
    print(f"  protocols with a topic under two people : {n_dup}  (reported, not merged)\n")

    for k, ch in list(log.items())[:6]:
        print(f"  {k}")
        if "reordered" in ch:
            print(f"     was: {' -> '.join(ch['reordered']['was'])}")
            print(f"     now: {' -> '.join(ch['reordered']['now'])}")
        if "unnamed_cluster_renamed" in ch:
            print(f"     renamed: {ch['unnamed_cluster_renamed']}")
        if "duplicate_topics" in ch:
            print(f"     same topic under two people: {ch['duplicate_topics'][:3]}")

    if apply:
        LOG.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nAPPLIED. wrote -> {LOG.relative_to(REPO)}")
    else:
        print("\n(report only — re-run with --apply)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
