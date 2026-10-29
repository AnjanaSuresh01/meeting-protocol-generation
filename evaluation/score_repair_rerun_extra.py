"""Computes the measures of score_repair_rerun.py that need each arm's own transcript.

compare_v10_rerun.structure_and_names checks invented names against the evaluated transcript for every arm. The
delivered arms were written from another transcript, so a name heard only there would count as invented. Here each
arm is checked with invented_names.apply against the transcript it was given (TX). The script also checks whether
the re-transcribed stretches (the gaps in protocol_pro.repair.json) reach the delivered protocols: a stretch counts
when one of its distinctive words occurs in the protocol of that meeting.

    python score_repair_rerun_extra.py   -> goldset/repair_rerun_extra_2026-10-12.json
"""
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import invented_names as IVN  # noqa: E402
from carry_tag import read_prev  # noqa: E402
from term_source import PREVIOUS  # noqa: E402

MEET = ["march09", "april13", "april27", "may11", "may18", "june01", "june15", "june22"]
TX = {"v10": "outputs_v11_final/canary_diar/{m}/transcript_speakers.txt",
      "v12_evaltx": "outputs_v11_final/canary_diar/{m}/transcript_speakers.txt",
      "v12_delivered": "outputs_deliverable/v12_delivered/{m}/protocol_pro.transcript.txt",
      "v12_delivered_raw": "pipeline_outputs_extracted/outputs/canary_diar/{m}/transcript_speakers.txt"}
COMMON = set("""aber alle also auch dann dass dem den der des die doch eine einen einer eines habe haben hatte
ich ist jetzt mal mit nach noch nicht oder schon sein sind und von was weil wenn werde wird wir zum zur genau
eigentlich irgendwie quasi gerade nochmal immer einfach dafür damit davon""".split())


def main() -> int:
    out = {"invented_own_transcript": {}, "recovered_reaches_protocol": {}}
    for arm, pat in TX.items():
        inv = []
        for m in MEET:
            t = (REPO / f"outputs_deliverable/{arm}/{m}/protocol_pro.md").read_text(encoding="utf-8")
            tx = (REPO / pat.format(m=m)).read_text(encoding="utf-8")
            _, restored, flagged = IVN.apply(t, tx, read_prev(REPO / PREVIOUS[m]))
            inv += [f"{m}: {k}" for k in list(restored) + list(flagged)]
        out["invented_own_transcript"][arm] = inv
        print(f"{arm:18s} invented names against its own transcript: {len(inv)} {inv}")
    # distinctive words of a recovered stretch: at least 7 letters, not in COMMON, absent from the raw transcript
    for arm in ("v12_delivered", "v12_delivered_raw"):
        hits = []
        for m in MEET:
            rep = REPO / f"outputs_deliverable/v12_delivered/{m}/protocol_pro.repair.json"
            gaps = json.loads(rep.read_text(encoding="utf-8"))["gaps"] if rep.exists() else []
            raw = (REPO / TX["v12_delivered_raw"].format(m=m)).read_text(encoding="utf-8").lower()
            prot = (REPO / f"outputs_deliverable/{arm}/{m}/protocol_pro.md").read_text(encoding="utf-8").lower()
            for g in gaps:
                words = {w for w in re.findall(r"[a-zäöüß][a-zäöüß-]{6,}", g["recovered_text"].lower())
                         if w not in COMMON and w not in raw}
                found = sorted(w for w in words if w in prot)
                hits.append({"meeting": m, "span": f"{g['start_s']:.0f}-{g['end_s']:.0f}", "distinctive": len(words),
                             "in_protocol": found})
        out["recovered_reaches_protocol"][arm] = hits
        n = sum(1 for h in hits if h["in_protocol"])
        print(f"\n{arm}: {n} of {len(hits)} recovered stretches have a distinctive word in the protocol")
        for h in hits:
            print(f"   {h['meeting']} {h['span']}: {h['distinctive']} distinctive words, in protocol: {h['in_protocol'][:6]}")
    (REPO / "goldset" / "repair_rerun_extra_2026-10-12.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                                                         encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
