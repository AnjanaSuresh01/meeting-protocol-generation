"""Rule-based checks for a generated German protocol: the '## Person' / '### Topic' / '* bullet'
hierarchy, a truncated ending, ASCII spellings such as [UEBERTRAG] and stacked [ÜBERTRAG] markers,
encoding damage (U+FFFD or mojibake) and a text under 40 words. validate_protocol() returns a
ProtocolReport; a single '## ' section and a text under 40 words are listed in problems but do
not affect ok.

Usage, from the folder that holds thesis_pipeline/:
    python -m thesis_pipeline.protocol_validate
main() checks every pipeline_outputs_extracted/outputs/<model>/<meeting>/pro_rulebased_final.md
under the working directory and writes pipeline_outputs_extracted/protocol_validation_report.json.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thesis_pipeline.text_norm import check_encoding, canonicalize_markers  # noqa: E402
from thesis_pipeline.text_norm import CANONICAL_CARRYOVER  # noqa: E402

_H2 = re.compile(r"^##\s+\S", re.MULTILINE)
_H3 = re.compile(r"^###\s+\S", re.MULTILINE)
_BULLET = re.compile(r"^\s*[*-]\s+\S", re.MULTILINE)
_BAD_MARKER = re.compile(r"\[\s*(?:UEBERTRAG|UBERTRAG|Ue?bertrag|UE?BERTRAG)\s*\]")
_DUP_MARKER = re.compile(r"(?:\[ÜBERTRAG\]\s*){2,}")
_SENT_END = re.compile(r"[.!?:)\]\"'»„“0-9]\s*$")
# Complete last lines: a '* ', '- ' or '# ' line, a numbered item, or a horizontal rule such as
# '---'. '## ' and '### ' headings do not match and are checked like prose.
_STRUCTURAL = re.compile(r"^\s*(?:[*#-]|\d+\.)\s|^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")
# Appended by generate_protocols.lisa() when the API still reports finish_reason=length after
# its re-rolls.
TRUNCATION_MARK = "<!-- finish_reason=length: generation truncated after all re-rolls -->"
# A bullet that ends on one of these words counts as truncated. Separable verb particles
# ('fällt aus') and einem, eines, ohne, das, dass are left out: they can end a complete bullet.
_NEVER_FINAL = set("der die den dem des eine einen und oder sowie bzw weil wenn ob "
                   "für von vom zum zur beim im am gegen".split())


@dataclass
class ProtocolReport:
    ok: bool
    n_persons: int
    n_topics: int
    n_bullets: int
    hierarchy_ok: bool
    truncated: bool
    bad_markers: int
    encoding_ok: bool
    problems: list[str] = field(default_factory=list)

    def as_dict(self):
        return asdict(self)


def validate_protocol(text: str) -> ProtocolReport:
    problems: list[str] = []
    n_h2 = len(_H2.findall(text))
    n_h3 = len(_H3.findall(text))
    n_bul = len(_BULLET.findall(text))

    hierarchy_ok = True
    if n_h2 == 0:
        hierarchy_ok = False
        problems.append("no '## Person' headings")
    if n_bul == 0:
        hierarchy_ok = False
        problems.append("no bullet points")
    # A '### ' topic before the first '## ' heading breaks the hierarchy.
    first_h2 = text.find("\n## ") if not text.startswith("## ") else 0
    first_h3 = text.find("\n### ") if not text.startswith("### ") else 0
    if n_h3 and (first_h2 == -1 or (0 <= first_h3 < first_h2)):
        hierarchy_ok = False
        problems.append("a '### Topic' appears before any '## Person'")
    if n_h2 == 1:
        problems.append("only ONE person section (possible degenerate-transcript symptom, cf. C5)")

    # Truncation: a non-structural last line that does not match _SENT_END, or a bullet last line
    # that ends on a word from _NEVER_FINAL.
    last_line = ""
    for ln in reversed(text.splitlines()):
        if ln.strip():
            last_line = ln
            break
    is_structural = bool(_STRUCTURAL.match(last_line))
    truncated = bool(last_line) and not is_structural and not _SENT_END.search(last_line.rstrip())
    if TRUNCATION_MARK in text:
        problems.append("generation truncated: the API reported finish_reason=length after "
                        "every re-roll — the end of this protocol is missing")
    elif not truncated and is_structural and _BULLET.match(last_line):
        tail = re.findall(r"[\wäöüßÄÖÜ]+", last_line.lower())
        truncated = bool(tail) and tail[-1] in _NEVER_FINAL
    if truncated:
        problems.append(f"ends mid-sentence (truncation / finish_reason=length?): "
                        f"...{last_line.rstrip()[-40:]!r}")

    bad = len(_BAD_MARKER.findall(text))
    if bad:
        problems.append(f"{bad} non-canonical carry-over marker(s) (should be {CANONICAL_CARRYOVER})")
    dup = len(_DUP_MARKER.findall(text))
    if dup:
        problems.append(f"{dup} duplicated/stacked [ÜBERTRAG] marker(s) — carry-over accumulation bug")

    enc = check_encoding(text)
    if not enc.healthy:
        problems.append(f"encoding: {enc.n_replacement_chars} U+FFFD, "
                        f"{enc.n_mojibake_hits} mojibake, NFC={enc.is_nfc}")

    if len(text.split()) < 40:
        problems.append(f"near-empty protocol ({len(text.split())} words)")

    ok = hierarchy_ok and not truncated and bad == 0 and dup == 0 and enc.healthy and n_h2 >= 1
    return ProtocolReport(
        ok=ok, n_persons=n_h2, n_topics=n_h3, n_bullets=n_bul,
        hierarchy_ok=hierarchy_ok, truncated=truncated, bad_markers=bad,
        encoding_ok=enc.healthy, problems=problems,
    )


def normalize_protocol(text: str) -> str:
    """Return the text with canonical [ÜBERTRAG] and Rückstand spellings, in Unicode NFC form."""
    from thesis_pipeline.text_norm import to_nfc
    return to_nfc(canonicalize_markers(text))


def main(base: Path = Path("pipeline_outputs_extracted/outputs"),
         which: str = "pro_rulebased_final.md") -> dict:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    report = {}
    print(f"Validating '{which}' across all model/meeting outputs:\n")
    print(f"{'model':16s} {'meeting':9s} {'P':>3s} {'T':>4s} {'B':>4s}  {'ok':3s} problems")
    print("-" * 100)
    n_ok = n_bad = 0
    for md in sorted(base.glob(f"*/*/{which}")):
        model, mid = md.parent.parent.name, md.parent.name
        text = md.read_text(encoding="utf-8")
        r = validate_protocol(text)
        report.setdefault(model, {})[mid] = r.as_dict()
        flag = "OK " if r.ok else "XXX"
        n_ok += r.ok
        n_bad += (not r.ok)
        prob = "; ".join(r.problems)[:56]
        if not r.ok:
            print(f"{model:16s} {mid:9s} {r.n_persons:>3d} {r.n_topics:>4d} "
                  f"{r.n_bullets:>4d}  {flag} {prob}")
    print("-" * 100)
    print(f"{n_ok} passed, {n_bad} flagged (only flagged shown above).")
    out = base.parent / "protocol_validation_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", out)
    return report


if __name__ == "__main__":
    main()
