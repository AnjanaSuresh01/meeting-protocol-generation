"""Runs protokoll.py on the eight meetings, one meeting at a time, as a user would run it.

Inputs per meeting: the Canary-1B-v2 transcript (outputs_v11_final/canary_diar), the Cohere and Whisper
large-v3-turbo-German transcripts as --also-transcript (used by the 2-of-3 handover name vote and the
cross-system name check), the previous document (term_source.PREVIOUS) and, where it exists, the document
before it (term_source.OLDER). --evaluated-trio passes Granite Speech 4.1 instead of Whisper
large-v3-turbo-German as the third transcript, as in the evaluated runs (outputs_deliverable/v10).
Other options:

  --tag NAME              output folder outputs_deliverable/NAME (default baseline)
  --lisa MODEL            language model (default lisa-pro-03-2026)
  --voices                one voice store per tag (outputs_deliverable/<tag>/voices.json); by default the
                          meetings run in date order, and before each meeting the voices of the previous
                          recorded meeting are enrolled from its corrected minutes
  --history               the earlier protocols of the series (dated before the meeting, without the
                          previous document) widen the list of names to check
  --history-vocab         the same protocols are also put into the prompt
  --llm-carry, --no-carry passed through to protokoll.py
  meeting ids             run only these meetings (default all eight)

Meetings whose protocol already exists are skipped.

    python run_deliverable.py --tag baseline [--lisa lisa-flash-03-2026] [--voices] [--llm-carry] [may11 ...]
    -> outputs_deliverable/<tag>/<meeting>/protocol_<pro|flash>.md and run_<pro|flash>.log
"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from term_source import PREVIOUS, OLDER

AUDIO = {"march09": "Team-Meeting_2026-03-09.mp3", "april13": "Team-Meeting_2026-04-13.mp3",
         "april27": "Team-Meeting_2026-04-27.flac", "may11": "Team-Meeting_2026-05-11.flac",
         "may18": "Team-Meeting_2026-05-18.flac", "june01": "Team-Meeting_2026-06-01.flac",
         "june15": "Team-Meeting_2026-06-15.flac", "june22": "Team-Meeting_2026-06-22.flac"}
DATE = {"march09": "2026-03-09", "april13": "2026-04-13", "april27": "2026-04-27", "may11": "2026-05-11",
        "may18": "2026-05-18", "june01": "2026-06-01", "june15": "2026-06-15", "june22": "2026-06-22"}

args = sys.argv[1:]
tag = args[args.index("--tag") + 1] if "--tag" in args else "baseline"
lisa = args[args.index("--lisa") + 1] if "--lisa" in args else "lisa-pro-03-2026"
only = [a for a in args if a in AUDIO] or list(AUDIO)
THIRD = "granite41" if "--evaluated-trio" in args else "whisper_turbo_de"   # see docstring
short = "pro" if "pro" in lisa else "flash"
for mid in only:
    out = REPO / "outputs_deliverable" / tag / mid / f"protocol_{short}.md"
    if out.exists() and out.stat().st_size > 80:
        print(f"{mid}: exists, skipped"); continue
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "protokoll.py", AUDIO[mid], "--previous", PREVIOUS[mid],
           "--transcript", f"outputs_v11_final/canary_diar/{mid}/transcript_speakers.txt",
           "--also-transcript", f"outputs_v10/cohere/{mid}/transcript_speakers.txt",
           "--also-transcript", f"outputs_v10/{THIRD}/{mid}/transcript_speakers.txt",
           "--lisa", lisa, "--date", DATE[mid], "--out", str(out)]
    cmd += [a for a in ("--llm-carry", "--no-carry") if a in args]
    if mid in OLDER and (REPO / OLDER[mid]).exists():
        cmd += ["--older", OLDER[mid]]
    # Every earlier protocol of the series, strictly before this meeting's date. --history alone
    # only widens the names-to-check list; --history-vocab also puts them in the prompt (the v10
    # setting). The date filter keeps a meeting's own minutes, which are its scoring reference,
    # out of its own run.
    if "--history" in args or "--history-vocab" in args:
        import re as _re
        # One document per date: some protocols exist in several copies (.md, .docx, "... (1).docx"),
        # and counting each copy would inflate the log's "widened to N documents". The .md is
        # preferred because it is the format every scorer reads. The previous document's date is
        # excluded, not only its file name, so a .docx copy of it is excluded as well.
        prev_date = _re.search(r"(\d{4}-\d{2}-\d{2})", Path(PREVIOUS[mid]).name).group(1)
        by_date = {}
        for p in sorted((REPO).glob("Meeting*"), key=lambda q: (q.suffix != ".md", q.name)):
            m = _re.search(r"(\d{4}-\d{2}-\d{2})", p.name)
            if m and p.suffix in (".md", ".docx") and m.group(1) < DATE[mid] \
                    and m.group(1) != prev_date:
                by_date.setdefault(m.group(1), p.name)
        hist = [by_date[d] for d in sorted(by_date)]
        if hist:
            cmd += ["--history"] + hist
            if "--history-vocab" in args:
                cmd += ["--history-vocab"]
    if "--voices" in args:
        # weekly deployment, in date order: before meeting i, enroll the voices of meeting i-1 from
        # its own corrected minutes, if evaluate_speaker_attribution.HUMAN lists minutes for it
        from evaluate_speaker_attribution import HUMAN
        order = list(AUDIO)
        cmd += ["--voices", str(REPO / "outputs_deliverable" / tag / "voices.json")]
        i = order.index(mid)
        if i > 0 and order[i - 1] in HUMAN:
            p = order[i - 1]
            cmd += ["--previous-audio", AUDIO[p],
                    "--previous-transcript", f"outputs_v11_final/canary_diar/{p}/transcript_speakers.txt",
                    "--previous-minutes", HUMAN[p]]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
    (out.parent / f"run_{short}.log").write_text(r.stdout + r.stderr, encoding="utf-8")
    last = [l for l in (r.stdout + r.stderr).splitlines() if l.strip()][-3:]
    print(f"{mid} {short}: exit {r.returncode} | " + " / ".join(x.strip()[:90] for x in last), flush=True)
