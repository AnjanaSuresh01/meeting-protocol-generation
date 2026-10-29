"""Runs protokoll.py on the eight meetings with the inputs of the v10 arm and one changed setting.

The inputs and flags are those of the v10 arm (run_deliverable.py --history-vocab), with two differences:

  --cite            the model cites a transcript segment behind every new bullet (off with --no-cite)
  --speakers <json> the speaker names that v10 arrived at, read from v10's own run logs

Supplying v10's speaker names keeps the comparison to one changed setting. v10 named the speakers with voice
profiles that accumulate in date order (voices.json), so a partial rerun cannot reproduce them; with
v10's final names an arm differs from v10 only in the changed setting and in the model's sampling.
Other options: --tag NAME (output folder, default v12cite), --depth and --segments (passed to
protokoll.py), --lisa MODEL (default lisa-pro-03-2026).

    python run_cite_arm.py                 # all 8 meetings, skips meetings already done
    python run_cite_arm.py april27         # one meeting (names come from v10, so order is free)
    python run_cite_arm.py --tag v13depth --no-cite --depth [meetings]

writes outputs_deliverable/<tag>/<meeting>/protocol_pro.md, speakers_from_v10.json and run_pro.log,
and outputs_deliverable/<tag>/MODEL.txt
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from term_source import PREVIOUS, OLDER                      # noqa: E402
# copied from run_deliverable.py, which executes at import and so cannot be imported from
AUDIO = {"march09": "Team-Meeting_2026-03-09.mp3", "april13": "Team-Meeting_2026-04-13.mp3",
         "april27": "Team-Meeting_2026-04-27.flac", "may11": "Team-Meeting_2026-05-11.flac",
         "may18": "Team-Meeting_2026-05-18.flac", "june01": "Team-Meeting_2026-06-01.flac",
         "june15": "Team-Meeting_2026-06-15.flac", "june22": "Team-Meeting_2026-06-22.flac"}
DATE = {m: "2026-" + {"march": "03", "april": "04", "may": "05", "june": "06"}[m[:-2]] + "-" + m[-2:]
        for m in AUDIO}

TAG = "v12cite"
# the same setup (v10's own speaker names) serves any single-change experiment, e.g.
#   python run_cite_arm.py --tag v13depth --no-cite --depth [meetings]
if "--tag" in sys.argv:
    TAG = sys.argv[sys.argv.index("--tag") + 1]
CITE = "--no-cite" not in sys.argv
DEPTH = "--depth" in sys.argv
# segment-level generation (protokoll.py --segments), one LISA call per report
SEGMENTS = "--segments" in sys.argv
# --lisa compares language models: same v10 names, same flags, only the model changes. The output
# keeps the file name protocol_pro.md so every scorer reads it unchanged; the model actually used is
# written to MODEL.txt in the arm folder and to run_pro.log.
LISA = sys.argv[sys.argv.index("--lisa") + 1] if "--lisa" in sys.argv else "lisa-pro-03-2026"


def v10_names(mid: str) -> dict:
    log = (REPO / "outputs_deliverable" / "v10" / mid / "run_pro.log").read_text(
        encoding="utf-8", errors="replace")
    m = re.findall(r"speaker names \(voice, handover, topic\): (.*)", log)
    return dict(x.split("=", 1) for x in m[-1].strip().split(", ")) if m else {}


def history(mid: str) -> list[str]:
    """One document per date before the meeting, .md preferred, the previous document's date
    excluded (the same rule as run_deliverable.py)."""
    prev_date = re.search(r"(\d{4}-\d{2}-\d{2})", Path(PREVIOUS[mid]).name).group(1)
    by_date = {}
    for p in sorted(REPO.glob("Meeting*"), key=lambda q: (q.suffix != ".md", q.name)):
        m = re.search(r"(\d{4}-\d{2}-\d{2})", p.name)
        if m and p.suffix in (".md", ".docx") and m.group(1) < DATE[mid] \
                and m.group(1) != prev_date:
            by_date.setdefault(m.group(1), p.name)
    return [by_date[d] for d in sorted(by_date)]


def main() -> int:
    only = [a for a in sys.argv[1:] if a in AUDIO] or list(AUDIO)
    for mid in only:
        d = REPO / "outputs_deliverable" / TAG / mid
        out = d / "protocol_pro.md"
        if out.exists() and out.stat().st_size > 80:
            print(f"{mid}: exists, skipped")
            continue
        d.mkdir(parents=True, exist_ok=True)
        names = v10_names(mid)
        spk = d / "speakers_from_v10.json"
        spk.write_text(json.dumps(names, ensure_ascii=False, indent=1), encoding="utf-8")
        cmd = [sys.executable, "protokoll.py", AUDIO[mid], "--previous", PREVIOUS[mid],
               "--transcript", f"outputs_v11_final/canary_diar/{mid}/transcript_speakers.txt",
               "--also-transcript", f"outputs_v10/cohere/{mid}/transcript_speakers.txt",
               "--also-transcript", f"outputs_v10/granite41/{mid}/transcript_speakers.txt",
               "--lisa", LISA, "--date", DATE[mid], "--out", str(out),
               "--speakers", str(spk)]
        cmd += ["--cite"] if CITE else []
        cmd += ["--depth"] if DEPTH else []
        cmd += ["--segments"] if SEGMENTS else []
        if mid in OLDER and (REPO / OLDER[mid]).exists():
            cmd += ["--older", OLDER[mid]]
        h = history(mid)
        if h:
            cmd += ["--history"] + h + ["--history-vocab"]
        import time as _time
        (REPO / "outputs_deliverable" / TAG / "MODEL.txt").write_text(LISA + "\n", encoding="utf-8")
        t0 = _time.time()
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        secs = _time.time() - t0
        # the log line remerge_deliverable.py reads, so this arm can be re-merged like any other
        log = r.stdout + r.stderr
        log += "\n   speaker names (voice, handover, topic): " + ", ".join(
            f"{k}={v}" for k, v in names.items()) + "\n"
        log += f"   model: {LISA}   wall time: {secs:.0f} s   exit: {r.returncode}\n"
        (d / "run_pro.log").write_text(log, encoding="utf-8")
        cl = [ln.strip() for ln in log.splitlines() if "citations:" in ln]
        print(f"{mid}: exit {r.returncode} | {cl[-1][:150] if cl else log.strip().splitlines()[-1][:150]}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
