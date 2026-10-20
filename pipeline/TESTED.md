# How this folder was tested

Sections 1 to 3 describe the tests of 15 October 2026. Section 4 describes the checks after the cleanup of 19 October 2026, and section 5 the run of notebook 1c on JupyterHub on the same day.

## 1. Run on JupyterHub (15 October 2026)

Everything ran on the university JupyterHub. Step 1 ran in notebooks 1a and 1b on an NVIDIA A100 (40 GB). Notebook 1c did not exist yet. Step 2 ran in the terminal. LISA did not offer `lisa-pro` that day, so step 2 used `--lisa lisa-flash`. The meetings ran in date order. 27 April enrolled the voices of the 13 April recording.

| | 13 April | 27 April |
|---|---|---|
| Step 1: words (July cluster run) | 6,028 (6,028) | 6,389 (6,481) |
| Step 1: speaker turns (July) | 581 (586) | 719 (549) |
| Step 1: speakers | 4 | 4 |
| Dropped speech recovered | none found | 6 stretches, 258.3 s, 476 words |
| Speakers named | 4 of 4, by topic | 4 of 4, by voice |
| New items inserted | 99 | 53 |
| Sections, words | 4, 2,671 | 4, 2,930 |
| Structure check | passes | passes |

Differences from the delivered runs of 11 and 12 October:
- `lisa-flash` instead of `lisa-pro`
- the series started at 13 April instead of 9 March
- no second recognizer's transcript (`--also-transcript`)

The LISA call failed several times that day. With `lisa-pro`, it failed with HTTP 500, then 404 once the model was gone from LISA's list. One `lisa-flash` call for 13 April also failed with 404. A `lisa-flash` call a minute later, for 27 April, succeeded. 13 April then succeeded on the second attempt, run with `LISA_RETRIES=5`.

## 2. Completeness check (15 October 2026)

This check ran outside JupyterHub, on a copy of the code and of the inputs. The delivered run of 11 and 12 October 2026 (`outputs_deliverable/v12_delivered`) was repeated for three team meetings: 9 March, 13 April and 27 April. The meetings ran in date order with one shared voice store, as in the original run. The test copy held only the code of this folder and the inputs of these three meetings.

Every step ran as delivered except one. The LISA call returned the text the original run had saved (`protocol_pro.neu.md`), so no request went to the service. The transcript repairs, the voice enrollment with WavLM, the speaker naming, the carry-over, the [ÜBERTRAG] and [PRÜFEN] marks and the structure check all ran. The text of the 9 dropped stretches (4 on 9 March, 5 on 27 April) came from `cache_gap_recovery.json`, so Whisper did not decode them again.

| Meeting | Exit code | Output files identical to the delivered run |
|---|---|---|
| 9 March | 0 | 6 of 6 |
| 13 April | 0 | 6 of 6 |
| 27 April | 0 | 6 of 6 |

The six files per meeting are the protocol, `.namen.md`, `.stil.md`, `.neu.md`, `.transcript.txt` and `.repair.json`. They were compared byte for byte.

A Senate run (`--structure senat`, 13 April transcript, a team protocol standing in for the invitation) ended with exit code 0. Its output was identical to the same run with all scripts of the development folder.

The replays loaded 30 files of this folder and no other project file.

## 3. Notebook 1c (15 October 2026)

Notebook 1c needs a GPU and the Cohere model, so it was checked outside JupyterHub with the July cluster files of 27 April (`transcript_canary_ts.json`, `diarization_hyp.rttm`) and two changes made only to the test copy: the GPU check was removed, and Whisper ran on the CPU.

- Cell 1 built 125 speaker turns and found 5 stretches without Canary words, 154 clips of at most 27.9 s. The repaired main transcript of the same files has 124 turns (the voice check moves one short sentence) and 5 recovered stretches.
- Cell 2 decoded the first 3 clips with Whisper large-v3-turbo-German. Each text matched the speech of its turn. The other clips received the Canary text of their turn, to test the rest of the chain.
- Cell 3 (Cohere Transcribe) was not run. Its loading code is identical to cell 5 of `../recognizers/NB_v10_asr.ipynb`, which decoded the 1,939 evaluation clips on JupyterHub.
- `protokoll.py` read both transcripts with `--also-transcript` and reported 3 compared systems. The protocol was byte-identical to the run without the two files. The list of names to check gained 3 entries where Whisper and Canary disagreed.

On 19 October cell 1 was changed: long clips are now split before the 0.5 s of silence is added, not after. With the same files, cell 1 again gives 125 turns, 5 stretches and 154 clips of at most 27.92 s. Before the change, 5 of the 154 clips were pure silence. After it, none is.

## 4. After the cleanup (19 October 2026)

On 19 October the comments were shortened, two files were renamed and the fixes listed in `../CHANGES.md` were applied. For every one of the 114 code files of the repository, a script compared the syntax tree without comments and docstrings with the code before the cleanup plus those fixes. All 114 were identical. The checks of section 2 were then repeated outside JupyterHub with the cleaned code, together with the other folders:

| Check | Result |
|---|---|
| test suite, `python -m pytest` from the repository root | 87 passed |
| delivered runs of 9 March, 13 April and 27 April replayed as in section 2 | 18 of 18 output files identical |
| Senate run as in section 2, cleaned code against the code before the cleanup | 5 of 5 output files identical |
| the ten scripts of `../evaluation/` in a copy with the stored results | 0 result files changed |
| steps 3 and 4 of `../recognizers/README.md` | 116 of 116 files identical: 3 merged Whisper files, `goldset/segment_loop_gate.json` and 112 meeting transcripts |
| cell 1 of notebook 1c with the July files of 27 April (GPU check removed, no decoding) | 154 clips, none of them pure silence |

A Senate run with the Wiki PDF export of 13 April as `--previous` also ended with exit code 0. The cleaned code reads 11,515 characters of text from that PDF, where the code before the fix read 481,313 characters of raw PDF data.

## 5. Notebook 1c and step 2 with three recognizers on JupyterHub (19 October 2026)

The run used the JupyterHub folder of section 1, with the 15 October outputs of notebooks 1a and 1b for 27 April. Notebook 1c and `asr_chunks.py` were the versions of 19 October. Step 2 ran with the `protokoll.py` of 15 October, which the fixes of 19 October change only in messages and in code paths this run does not use. Step 2 used `--lisa lisa-flash` and both `--also-transcript` files.

| Step | Result |
|---|---|
| Notebook 1c, cell 1 | 125 turns, 6 stretches without Canary words, 155 clips of at most 28 s, 50.9 min of audio |
| Notebook 1c, cell 2: Whisper large-v3-turbo-German | 6,884 words in 133 turns, 0 of 131 clips empty |
| Notebook 1c, cell 3: Cohere Transcribe | 6,984 words in 134 turns, 0 of 131 clips empty |
| Step 2: transcript repairs | 6 words moved by the sentence rule, 1 short sentence returned, 124 turns, 6 stretches recovered (258.3 s, 476 words), as on 15 October |
| Step 2: speakers | 4 of 4 named by voice profile |
| Step 2: protocol | 61 new items inserted, 4 sections, 2,868 words, structure check passes |
| Step 2: names to check | 10, with 3 systems compared (15 October, without notebook 1c: 1) |

The 131 clips of cells 2 and 3 are the 125 turns and the 6 stretches. Clips longer than 27 s were cut, which gives 155 pieces. When cell 3 loaded Cohere Transcribe, transformers downloaded newer versions of the model's four code files from Hugging Face (`trust_remote_code`). The model therefore ran with revision `d552dadc2922ba6f6c1f994f8db904d52dcaf942`, not with the revision of the thesis decodes (see `../recognizers/README.md`).

## Not tested

- Step 2 with the cleaned `protokoll.py` on JupyterHub. Sections 2 to 4 ran outside JupyterHub and compare code, not new results.
- A Senate run on JupyterHub, and a Senate run with a real invitation.
- Building the NeMo and pyannote environments from scratch. The runs used existing environments, repaired on 15 October (see `README.md`).
