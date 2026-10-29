# -*- coding: utf-8 -*-
"""Computes the per-meeting values behind the pooled post-edit figures, with their totals.

For each of the eight meetings:
  minutes            length of the recording (Team-Meeting_<date>.mp3/.flac, read with soundfile)
  reference_bullets  bullets of the reference protocol after strip_backlog and score_editforward.no_backlog
  add_, delete_,     for the v10 draft and for last week's protocol alone ("prev"): reference bullets
  draft_             the draft does not cover (ADD), draft bullets that cover no reference bullet
                     (DELETE), and the number of draft bullets; coverage is postedit_effort.covered
                     (30% content-word rule), as in score_editforward
  new                reference bullets that last week's protocol does not cover
  new_missed_        new bullets that the v10 and the v12_evaltx drafts do not cover
  leaves*            reference leaves that are not word-for-word copies of last week's protocol, and
                     how many of them the draft and the copy record, taken from
                     goldset/strip_copies_rescore_2026-10-07.json (per_meeting, leaf rule L, S1)

    python per_meeting_2026-10-12.py    -> goldset/per_meeting_2026-10-12.json
"""
import json
import sys
from pathlib import Path

import soundfile as sf

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.stdout.reconfigure(encoding="utf-8")
import postedit_effort as PE  # noqa: E402
import score_editforward as SE  # noqa: E402
from evaluate_protocols import strip_backlog  # noqa: E402

AUDIO = {"march09": "Team-Meeting_2026-03-09.mp3", "april13": "Team-Meeting_2026-04-13.mp3",
         "april27": "Team-Meeting_2026-04-27.flac", "may11": "Team-Meeting_2026-05-11.flac",
         "may18": "Team-Meeting_2026-05-18.flac", "june01": "Team-Meeting_2026-06-01.flac",
         "june15": "Team-Meeting_2026-06-15.flac", "june22": "Team-Meeting_2026-06-22.flac"}
strip = json.loads((REPO / "goldset" / "strip_copies_rescore_2026-10-07.json").read_text(encoding="utf-8"))
leaves = strip["per_meeting"]["L"]["S1"]

out = {}
for mid, wav in AUDIO.items():
    info = sf.info(str(REPO / wav))
    ref = PE.bullets(SE.no_backlog(strip_backlog((REPO / SE.REF[mid]).read_text(encoding="utf-8"))[0]))
    prev = PE.bullets(SE.previous_only(mid))
    row = {"minutes": round(info.frames / info.samplerate / 60, 1), "reference_bullets": len(ref)}
    for arm in ("v10", "prev"):
        draft = prev if arm == "prev" else PE.bullets(SE.arm_text(arm, mid))
        row[f"add_{arm}"] = sum(not PE.covered(r, draft) for r in ref)
        row[f"delete_{arm}"] = sum(not PE.covered(h, ref) for h in draft)
        row[f"draft_{arm}"] = len(draft)
    new = [r for r in ref if not PE.covered(r, prev)]
    row["new"] = len(new)
    for arm in ("v10", "v12_evaltx"):
        hb = PE.bullets(SE.arm_text(arm, mid))
        row[f"new_missed_{arm}"] = sum(not PE.covered(r, hb) for r in new)
    lv = leaves[mid]["recall_new_leaves"]
    row["leaves"] = lv["n"]
    row["leaves_draft"] = lv["draft"]
    row["leaves_copy"] = lv["prev"]
    out[mid] = row
    print(mid, row)
tot = {k: sum(r[k] for r in out.values()) for k in next(iter(out.values())) if k != "minutes"}
tot["minutes"] = round(sum(r["minutes"] for r in out.values()), 1)
out["total"] = tot
print("total", tot)
(REPO / "goldset" / "per_meeting_2026-10-12.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
