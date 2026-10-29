"""Command-line surface of the scripts.

Each script runs with no arguments (or --help) from the repository root. The test
checks the exit code and stable parts of stdout and stderr; line numbers and
traceback paths are not checked.
"""
from __future__ import annotations

import subprocess
import sys

import pytest

from conftest import ROOT

RUNS = [
    ("pipeline/protokoll.py", ["--help"], 0, ["--history-vocab", "--older OLDER"], []),
    ("pipeline/carry_tag.py", ["--help"], 0,
     ["usage: carry_tag.py [-h] [--out OUT] protocol transcript previous"], []),
    ("pipeline/postedit_effort.py", ["--help"], 0, ["--tree TREE", "--suffix SUFFIX"], []),
    ("pipeline/postedit_effort.py", [], 1,
     ["needs evaluate_protocols.py, which is only in evaluation/"], []),
    ("pipeline/name_check.py", ["--help"], 0, ["--max-agreement", "--append"], []),
    ("pipeline/normalize_structure.py", [], 0,
     ["STRUCTURE NORMALISATION — deterministic layout repair", "nothing to repair."], []),
    ("pipeline/protocol_repair.py", [], 1,
     ["PROTOCOL REPAIR — deterministic, post-generation, no API"],
     ["FileNotFoundError", "speaker_identity.json"]),
    ("pipeline/stale_flag.py", [], 1, [],
     ["usage: python stale_flag.py PROTOCOL.md PREVIOUS.md OLDER.md"]),
    ("pipeline/carry_forward.py", [], 1, [],
     ["usage: python carry_forward.py PROTOCOL.md PREVIOUS.md"]),
    ("pipeline/style_exemplars.py", [], 1, [],
     ["usage: python style_exemplars.py PREVIOUS.md TRANSCRIPT.txt"]),
    ("pipeline/style_flag.py", [], 0,
     ["arm v10", "0 bullets flagged for human review across v10"], []),
    ("pipeline/entity_correct.py", [], 1,
     ["AUTOMATIC ENTITY CORRECTION — vocabulary from the previous document alone",
      "skipped: previous document or goldset/clip_asr_v10/ transcripts missing"],
     ["no meeting had data, nothing written"]),
    ("pipeline/term_source.py", [], 0, ["may11", "june22", "0 terms"], []),
    ("pipeline/term_source.py", ["--compare"], 0, ["--compare needs entity_lexicon.py"], []),
    ("pipeline/thesis_pipeline/canary_diarize.py", [], 1, [], ["config.json not found"]),
    ("pipeline/speaker_turns_v11.py", [], 1,
     ["meeting    words  turns  mean s  max s  changes kept"],
     ["segments_v9_all.json"]),
    ("pipeline/transcript_repair.py", [], 1,
     ["usage: python transcript_repair.py <words.json> <diarization.rttm> <audio> [out.txt]"], []),
    ("pipeline/gap_recover.py", [], 1, [], ["segments_v9_all.json"]),
    ("pipeline/verify_short_turns.py", [], 1, [], ["segments_v9_all.json"]),
    ("pipeline/voice_profiles.py", [], 0, [], None),          # None = require empty
    ("recognizers/segment_loop_gate.py", [], 0,
     ["DEGENERATE CLIPS IN FULL (0)", "(report only — re-run with --apply)"], []),
    ("recognizers/transcript_gate.py", [], 1, [], ["LibsndfileError"]),
    ("recognizers/repair_word_speakers.py", [], 1,
     ["WORD -> SPEAKER LABELS — checking all eight meetings"],
     ["transcript_canary_ts.json"]),
    ("recognizers/assemble_v9_transcripts.py", [], 1, [], ["segments_v9_all.json"]),
    ("recognizers/flag_edge_words.py", [], 2, [],
     ["the following arguments are required: --files"]),
    ("recognizers/resegment_v7.py", [], 1, [], ["transcript_canary_ts.json"]),
    ("recognizers/whisper_long_redecode.py", [], 1, [], ["whisper_empty_segments.json"]),
]


@pytest.mark.parametrize("script,args,rc,out_need,err_need", RUNS,
                         ids=[" ".join([r[0], *r[1]]) for r in RUNS])
def test_cli_surface(script, args, rc, out_need, err_need):
    p = subprocess.run([sys.executable, str(ROOT / script), *args],
                       cwd=str(ROOT), capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=300)
    assert p.returncode == rc, f"rc {p.returncode} != {rc}\nSTDERR:\n{p.stderr[-1500:]}"
    for s in out_need:
        assert s in p.stdout, f"missing {s!r} in stdout:\n{p.stdout[-1500:]}"
    if err_need is None:
        assert p.stdout == "" and p.stderr == ""
    else:
        for s in err_need:
            assert s in p.stderr, f"missing {s!r} in stderr:\n{p.stderr[-1500:]}"
