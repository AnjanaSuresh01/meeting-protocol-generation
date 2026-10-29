"""Tests whether a context question to LISA removes the wrong "klingt wie" spelling suggestions.

name_check.uncertain_single suggests a spelling from earlier protocols whenever a heard word
sounds like a term there (Kölner Phonetik plus letter similarity). This script takes the
suggestions that reach a v10 protocol (near_flag_gate.rows()) and, for each, the sentence of the
Canary transcript (outputs_v11_final) that holds the heard word and the sentence of the earlier
protocol that holds the suggestion. Every suggestion carries a hand label in READ: R a real name
correction (e.g. Gwen -> Qwen), H a spelling harmonized to the earlier protocols (Webseite ->
Website), U a wrong word with a wrong suggestion, W a heard word that was right (e.g. Hund,
Absturz).

The method follows contextual LLM-based named-entity revision (EDM 2026, "Improving Named Entity
Transcription with Contextual LLM-based Revision"): candidates are first filtered by phonetic
similarity, then a language model reads the sentences and decides. Here the model only decides
whether a suggestion is shown; it never changes the protocol. One batched call per meeting goes
to lisa-pro-03-2026 through generate_protocols.lisa (temperature 0.2, reasoning off). The
question uses only the meeting's transcript and the earlier protocols the suggestion came from;
the hand labels are never shown to the model. The summary counts each label among all
suggestions and among those the model answers JA for.

    python name_flag_context_check.py              # ask LISA, score against READ, write JSON
    python name_flag_context_check.py --tag _run2  # the same, written as a separate repeat run
    python name_flag_context_check.py --contexts   # print both sentences of every flag, no LISA call

Writes goldset/name_flag_context_check_2026-10-01<tag>.json.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
_TAG = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else ""
OUT = REPO / "goldset" / f"name_flag_context_check_2026-10-01{_TAG}.json"   # --tag <suffix>: a repeat run
MODEL = "lisa-pro-03-2026"

# Hand labels, assigned by reading each flag in its sentences (as printed by --contexts). R real
# name correction, H harmonizes to the spelling of the earlier protocols, U the heard word is wrong
# but so is the suggestion, W the heard word was right.
READ = {
    ("march09", "Gwen"): "R", ("april27", "Olaf"): "R", ("may18", "Vilo"): "R",
    ("may11", "Webseite"): "H", ("may18", "Webseite"): "H",
    ("may11", "Lloyd"): "U", ("may18", "Lloyd"): "U",
    ("march09", "Absturz"): "W", ("march09", "Anträge"): "W", ("march09", "VLLM"): "W",
    ("april13", "Dienst"): "W", ("april13", "Life"): "W", ("april27", "Mitarbeiterin"): "W",
    ("april27", "Hund"): "W", ("may11", "Splatting"): "W", ("may11", "Hund"): "W",
    ("may18", "Dienst"): "W", ("may18", "Datenschatz"): "W", ("may18", "Frame"): "W",
    ("may18", "Pick"): "W", ("may18", "Kontakt"): "W", ("june01", "Connect"): "W",
    ("june15", "State"): "W", ("june22", "Klasse"): "W", ("june22", "State"): "W",
    ("june22", "Study"): "W",
}

SYSTEM = (
    "Du prüfst Rechtschreibvorschläge für ein deutsches Besprechungsprotokoll. Eine automatische "
    "Spracherkennung hat ein Wort gehört. Ein früheres Protokoll derselben Gruppe enthält einen "
    "ähnlich klingenden Namen oder Fachbegriff. Entscheide für jeden Fall: Ist das gehörte Wort im "
    "Satz A eine Fehlerkennung GENAU DIESES Namens oder Begriffs aus Satz B (dieselbe Sache, nur "
    "falsch geschrieben)? Antworte NEIN, wenn das gehörte Wort im Satz A ein gewöhnliches, richtig "
    "verwendetes Wort ist oder eine andere Sache meint. Antworte je Fall in einer Zeile genau so: "
    "<Nummer>|JA oder <Nummer>|NEIN. Keine Erklärungen.")


def _bound(w: str) -> str:
    return rf"(?<![\wÄÖÜäöüß]){re.escape(w)}(?![\wÄÖÜäöüß])"


def sentence(word: str, text: str, width: int = 110) -> str:
    m = re.search(_bound(word), text)
    if not m:
        return ""
    s = text[max(0, m.start() - width):m.end() + width]
    s = re.sub(r"\[\d\d:\d\d:\d\d\]\s*SPEAKER_\d+:\s*", " ", s)
    return re.sub(r"\s+", " ", s.replace("\n", " ")).strip()


def flags() -> list[dict]:
    import near_flag_gate as G
    from term_source import PREVIOUS
    out = []
    for r in G.rows():
        mid = r["meeting"]
        prev = REPO / PREVIOUS[mid]
        docs = [prev] + G.history(mid, prev)
        tx = (REPO / f"outputs_v11_final/canary_diar/{mid}/transcript_speakers.txt").read_text(
            encoding="utf-8")
        ctx_b = ""
        for d in docs:
            if d.exists():
                t = re.sub(r"<[^>]+>", "", d.read_text(encoding="utf-8", errors="replace"))
                ctx_b = sentence(r["suggest"], t, 80)
                if ctx_b:
                    break
        out.append({"meeting": mid, "heard": r["heard"], "suggest": r["suggest"],
                    "sentence_heard": sentence(r["heard"], tx), "sentence_earlier": ctx_b,
                    "read": READ.get((mid, r["heard"]), "?"), "lexical_gate": r["verdict"]})
    return out


def ask(meeting_rows: list[dict]) -> dict[int, str]:
    import generate_protocols as GP
    lines = []
    for i, f in enumerate(meeting_rows, 1):
        lines.append(f"{i}. gehört: \"{f['heard']}\" — Vorschlag: \"{f['suggest']}\"\n"
                     f"   Satz A (Transkript): {f['sentence_heard']}\n"
                     f"   Satz B (früheres Protokoll): {f['sentence_earlier']}")
    raw = GP.lisa(MODEL, SYSTEM, "\n".join(lines), timeout=300, retries=2, lang_gate=False)
    ans = {}
    for m in re.finditer(r"(\d+)\s*\|\s*(JA|NEIN)", raw, re.I):
        ans[int(m.group(1))] = m.group(2).upper()
    return ans


def main() -> int:
    fl = flags()
    missing = [f for f in fl if f["read"] == "?"]
    assert not missing, f"unread flags: {[(f['meeting'], f['heard']) for f in missing]}"
    if "--contexts" in sys.argv:
        for f in fl:
            print(f"[{f['read']}] {f['meeting']} {f['heard']} -> {f['suggest']}\n   A: {f['sentence_heard']}"
                  f"\n   B: {f['sentence_earlier']}")
        return 0
    by_meeting: dict[str, list[dict]] = {}
    for f in fl:
        by_meeting.setdefault(f["meeting"], []).append(f)
    for mid, rows in by_meeting.items():
        try:
            ans = ask(rows)
        except Exception as e:                 # LISA unreachable: leave these unanswered, continue
            print(f"  {mid}: LISA failed ({type(e).__name__}); these flags stay unanswered", flush=True)
            ans = {}
        for i, f in enumerate(rows, 1):
            f["lisa"] = ans.get(i, "NO ANSWER")
        print(f"  {mid}: {len(rows)} asked, {sum(1 for f in rows if f['lisa'] != 'NO ANSWER')} answered",
              flush=True)

    def tally(rows, key, keep):
        return sum(1 for f in rows if keep(f[key]))
    shown_before = fl
    shown_after = [f for f in fl if f["lisa"] == "JA"]
    unanswered = sum(1 for f in fl if f["lisa"] == "NO ANSWER")
    if unanswered:
        print(f"\n  ** {unanswered} of {len(fl)} flags got no answer — the summary below is "
              "INCOMPLETE")
    print("\n  label  meeting  heard -> suggestion            lexical gate   LISA")
    for f in sorted(fl, key=lambda f: (f["read"], f["meeting"])):
        print(f"  {f['read']:5s}  {f['meeting']:8s} {f['heard'] + ' -> ' + f['suggest']:30s}"
              f" {f['lexical_gate']:12s}   {f['lisa']}")
    res = {}
    for name, rows in (("shown now (no filter)", shown_before), ("shown if LISA says JA", shown_after)):
        c = {k: sum(1 for f in rows if f["read"] == k) for k in "RHUW"}
        res[name] = c
        print(f"\n  {name:24s}: {len(rows):2d} suggestions — real fixes {c['R']}, harmonise {c['H']}, "
              f"wrong suggestion on a wrong word {c['U']}, wrong suggestion on a right word {c['W']}")
    OUT.write_text(json.dumps({"built": "2026-10-01", "model": MODEL,
                               "method": "EDM 2026 contextual LLM revision, used as a show/hide filter only",
                               "labels": "R real fix, H harmonise, U wrong word + wrong suggestion, W right word",
                               "rows": fl, "summary": res}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
