"""Runs protokoll.py on the eight meetings in three arms that differ only in the transcript and its repairs.

All other inputs are the same (the v10 flags --voices --history-vocab --older; Cohere and Whisper
large-v3-turbo-German as --also-transcript; one voice store per arm, meetings in date order):

    v12_evaltx         the evaluated transcript (outputs_v11_final/canary_diar)
    v12_delivered      the transcript written by canary_diarize.py
                       (pipeline_outputs_extracted/outputs/canary_diar), repaired inside protokoll.py by
                       the four transcript repairs of transcript_repair.py
    v12_delivered_raw  the same canary_diarize.py transcript with --no-repair

All three arms use lisa-pro (GLM-5.3-Flash) with LISA_REASONING_EFFORT=low, the setting closest to
"reasoning off" (see generate_protocols.lisa). The evaluated runs used lisa-pro-03-2026
(Qwen3.5-397B-A17B), so the arms are comparable with each other but not with those runs.

    python run_repair_rerun.py [arm ...] [meeting ...]
    -> outputs_deliverable/<arm>/<meeting>/protocol_pro.md and run_pro.log, outputs_deliverable/<arm>/voices.json
"""
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # python -I ignores PYTHONIOENCODING
from term_source import OLDER, PREVIOUS  # noqa: E402

AUDIO = {"march09": "Team-Meeting_2026-03-09.mp3", "april13": "Team-Meeting_2026-04-13.mp3",
         "april27": "Team-Meeting_2026-04-27.flac", "may11": "Team-Meeting_2026-05-11.flac",
         "may18": "Team-Meeting_2026-05-18.flac", "june01": "Team-Meeting_2026-06-01.flac",
         "june15": "Team-Meeting_2026-06-15.flac", "june22": "Team-Meeting_2026-06-22.flac"}
DATE = {m: f"2026-{d}" for m, d in zip(AUDIO, ["03-09", "04-13", "04-27", "05-11", "05-18", "06-01", "06-15", "06-22"])}
CLUSTER = "pipeline_outputs_extracted/outputs/canary_diar/{m}/transcript_speakers.txt"
EVALTX = "outputs_v11_final/canary_diar/{m}/transcript_speakers.txt"
ARMS = {"v12_evaltx": (EVALTX, []), "v12_delivered": (CLUSTER, []), "v12_delivered_raw": (CLUSTER, ["--no-repair"])}
LISA = "lisa-pro"


def history(mid: str) -> list[str]:
    """Every earlier protocol of the series, one per date, the previous document's date excluded (as run_deliverable)."""
    prev_date = re.search(r"(\d{4}-\d{2}-\d{2})", Path(PREVIOUS[mid]).name).group(1)
    by_date = {}
    for p in sorted(REPO.glob("Meeting*"), key=lambda q: (q.suffix != ".md", q.name)):
        m = re.search(r"(\d{4}-\d{2}-\d{2})", p.name)
        if m and p.suffix in (".md", ".docx") and m.group(1) < DATE[mid] and m.group(1) != prev_date:
            by_date.setdefault(m.group(1), p.name)
    return [by_date[d] for d in sorted(by_date)]


def prev_transcript(arm: str, prev: str) -> str:
    """The transcript of the previous meeting as THIS arm produced it, for enrolling its voices."""
    if arm == "v12_delivered":
        return f"outputs_deliverable/{arm}/{prev}/protocol_pro.transcript.txt"
    return ARMS[arm][0].format(m=prev)


def run(arm: str, mid: str) -> None:
    from evaluate_speaker_attribution import HUMAN
    out = REPO / "outputs_deliverable" / arm / mid / "protocol_pro.md"
    if out.exists() and out.stat().st_size > 80:
        print(f"{arm} {mid}: exists, skipped", flush=True)
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    src, extra = ARMS[arm]
    cmd = [sys.executable, "protokoll.py", AUDIO[mid], "--previous", PREVIOUS[mid],
           "--transcript", src.format(m=mid),
           "--also-transcript", f"outputs_v10/cohere/{mid}/transcript_speakers.txt",
           "--also-transcript", f"outputs_v10/whisper_turbo_de/{mid}/transcript_speakers.txt",
           "--lisa", LISA, "--date", DATE[mid], "--out", str(out)] + extra
    if mid in OLDER and (REPO / OLDER[mid]).exists():
        cmd += ["--older", OLDER[mid]]
    hist = history(mid)
    if hist:
        cmd += ["--history"] + hist + ["--history-vocab"]
    order = list(AUDIO)
    cmd += ["--voices", str(REPO / "outputs_deliverable" / arm / "voices.json")]
    i = order.index(mid)
    if i > 0 and order[i - 1] in HUMAN:
        p = order[i - 1]
        cmd += ["--previous-audio", AUDIO[p], "--previous-transcript", prev_transcript(arm, p),
                "--previous-minutes", HUMAN[p]]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "LISA_REASONING_EFFORT": "low"}
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, cwd=REPO)
    (out.parent / "run_pro.log").write_text(" ".join(cmd) + "\n\n" + r.stdout + r.stderr, encoding="utf-8")
    last = [l for l in (r.stdout + r.stderr).splitlines() if l.strip() and "arn" not in l][-2:]
    print(f"{arm} {mid}: exit {r.returncode} | " + " / ".join(x.strip()[:110] for x in last), flush=True)


def main() -> int:
    arms = [a for a in sys.argv[1:] if a in ARMS] or list(ARMS)
    meetings = [m for m in sys.argv[1:] if m in AUDIO] or list(AUDIO)
    for arm in arms:
        for mid in meetings:
            run(arm, mid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
