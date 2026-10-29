# The six speech recognizers

This folder decodes the same 1,939 clips of the eight team meetings with all six recognizers of the thesis, on the university JupyterHub. The 147 clips of the gold set are decoded first. The outputs are the input of `../evaluation/`. They are also the source of the meeting transcripts from which the evaluated protocols were written.

| Code name | Model |
|---|---|
| `canary_diar` | `nvidia/canary-1b-v2` (NeMo) |
| `whisper_large_v3` | `openai/whisper-large-v3` |
| `whisper_turbo` | `openai/whisper-large-v3-turbo` |
| `whisper_turbo_de` | `primeline/whisper-large-v3-turbo-german` |
| `cohere` | `CohereLabs/cohere-transcribe-03-2026` |
| `granite41` | `ibm-granite/granite-speech-4.1-2b` |

This is the code that produced the thesis results. On 19 October 2026 its comments were shortened, and a few messages and defaults were fixed (`../CHANGES.md`). Keep the files in one folder. The scripts import each other by name and find their data relative to their own location.

## Files

| File | Runs on | What it does |
|---|---|---|
| `NB_v10_asr.ipynb` | GPU, kernel "NeMo ASR" | decodes every clip in `goldset/clips_v10/` with the six recognizers, the 147 gold clips first. Writes `goldset/clip_asr_v10/<code name>__v10.json` |
| `NB_whisper_long.ipynb` | GPU, kernel "NeMo ASR" | decodes again, with the three Whisper models, the 219 clips longer than 30 s that `NB_v10_asr.ipynb` left empty for them |
| `NB_biasing_full.ipynb` | GPU, kernel "NeMo ASR" | decodes the 1,939 clips with Canary-1B-v2 twice at beam 4, without and with decoder biasing toward the names of each meeting's previous protocol. Writes `bias_full/sweep_full.json`, which `../evaluation/score_biasing_full.py` scores |
| `asr_chunks.py` | (imported) | splits a clip longer than 28 s at its quietest points. This is the version `NB_whisper_long.ipynb` used |
| `whisper_empty_segments.json` | (data, no speech) | the ids of the 219 clips |
| `whisper_long_redecode.py` | terminal | `--merge` fills the texts of `NB_whisper_long.ipynb` into the three Whisper files of `goldset/clip_asr_v10/` and keeps a backup |
| `segment_loop_gate.py` | terminal | marks clips whose text repeats itself or is empty. `--apply` writes `goldset/segment_loop_gate.json` |
| `assemble_v9_transcripts.py` | terminal | joins the clip texts back into one speaker transcript per meeting and recognizer (`outputs_v10/<code name>/<meeting>/`) |
| `transcript_gate.py` | terminal | checks every meeting transcript for loops, word rate and boilerplate. `--v10 --apply` writes `transcript_gate_v10.json` |
| `resegment_v9.py`, `flag_edge_words.py`, `cut_clips_v9.py` | terminal | regenerate the segments and the clips (optional, see below) |
| `repair_word_speakers.py` | terminal | restores the speaker labels of the 11 May word list, which the segmenter reads (optional, see below) |
| `boundary_lib.py`, `boundary_signals.py`, `build_clip_dataset.py`, `resegment_v7.py`, `resegment_v8.py`, `rater_notes.py`, `thesis_pipeline/` | (imported) | functions used by the scripts above |

## Environment

- The notebooks run in `~/nemoenv` with the kernel "NeMo ASR" (`../environments/README.md`). Run `../environments/patch_nemo_py310.py` once. Without it, Canary fails on the 25 clips longer than 40 s.
- The terminal scripts run in `~/protokollenv`.
- Cohere Transcribe is a gated model on Hugging Face. Accept its conditions once on its model page. Then log in once in a terminal with `~/nemoenv/bin/hf auth login`. The notebooks of this folder do not read `pipeline/.env`.

## Working directory

The notebooks work in their own folder, `recognizers/`, where JupyterLab starts their kernel. The terminal scripts are run from this folder too. All data paths below are relative to it.

## Data

The data are internal recordings and texts of the research group. They are not in the repository and are provided separately.

| Path | Content | Needed by |
|---|---|---|
| `goldset/clips_v10/<segment id>.mp3` | the 1,939 clips (16 kHz mono MP3), unpacked from `clips_v10_gold.zip` and `clips_v10_rest.zip` | the three notebooks |
| `goldset/segments_v9_all.json` | the 1,939 segments (frozen) | `NB_v10_asr.ipynb`, `assemble_v9_transcripts.py` |
| `goldset/goldset_gold_147.json` | the ids of the 147 gold clips, which decide the decode order | `NB_v10_asr.ipynb` |
| `manifest_biasing_full.json` | the 1,939 segments and, for each meeting, the first 120 names and terms of its previous protocol | `NB_biasing_full.ipynb` |
| `Team-Meeting_2026-03-09.mp3`, `Team-Meeting_2026-04-13.mp3`, `Team-Meeting_2026-04-27.flac`, `Team-Meeting_2026-05-11.flac`, `Team-Meeting_2026-05-18.flac`, `Team-Meeting_2026-06-01.flac`, `Team-Meeting_2026-06-15.flac`, `Team-Meeting_2026-06-22.flac` | the eight recordings | `transcript_gate.py`, regeneration |
| `pipeline_outputs_extracted/outputs/canary_diar/<meeting>/transcript_canary_ts.json`, `diarization_hyp.rttm` | Canary word list and speaker turns of each whole meeting | regeneration |
| `pipeline_outputs_extracted/outputs/whisper/<meeting>/transcript_no_prompt_timestamped.txt` | whole-meeting Whisper text, used to correct the clock drift of the word list | regeneration |

To unpack the clips in this folder:

```bash
python -m zipfile -e clips_v10_gold.zip .
python -m zipfile -e clips_v10_rest.zip .
```

## Order of steps

1. GPU: open `NB_v10_asr.ipynb` with the kernel "NeMo ASR" and run the cells in order. After a crash, every model cell can be run again. It skips what is already written. The last cell packs `v10_asr_results.zip`.
2. GPU: open `NB_whisper_long.ipynb` and run it. It reads `whisper_empty_segments.json` from this folder and writes `whisper_long_redecode/<code name>.json`.
3. Terminal: move the outputs to where the merge expects them, then merge:
   ```bash
   source ~/protokollenv/bin/activate
   cd <repository>/recognizers
   mkdir -p goldset && cp whisper_empty_segments.json goldset/
   mv whisper_long_redecode goldset/
   python whisper_long_redecode.py --merge
   ```
4. Terminal: gate the clips, build the meeting transcripts, then gate the transcripts:
   ```bash
   python segment_loop_gate.py --apply
   python assemble_v9_transcripts.py --source all
   python transcript_gate.py --v10 --apply
   ```
5. GPU, only for the decoder-biasing result: open `NB_biasing_full.ipynb` with the kernel "NeMo ASR" and run the cells in order. Copy `bias_full/sweep_full.json` to `../evaluation/bias_full/` and the manifest to `../evaluation/dist/cluster_biasing_full/`.
6. Continue with `../evaluation/`.

## Regenerating segments and clips (optional)

The thesis results were scored on the provided clips. A re-cut is not needed. It is also not guaranteed to be byte-identical, because the MP3 encoder can differ between machines. Use the provided `goldset/segments_v9_all.json` and clips for scoring. To check the segmentation:

```bash
python repair_word_speakers.py           # report; --apply rewrites the 11 May word list
python resegment_v9.py --out goldset/segments_v9_rerun.json
python flag_edge_words.py --files segments_v9_rerun.json
python cut_clips_v9.py --segments goldset/segments_v9_rerun.json --out goldset/clips_rerun
```

Then compare the results with the frozen files.

## Points to know

- Cohere Transcribe loads its model code from Hugging Face (`trust_remote_code=True`), and that code changed after the thesis decodes. On 19 October 2026 the JupyterHub cache held three revisions: `32d9e4ba6271d78168c095c2f90bc173eaad97d2` (24 May), `b1eacc2686a3d08ceaae5f24a88b1d519620bc09` (25 June) and `d552dadc2922ba6f6c1f994f8db904d52dcaf942` (19 October). The decodes of September most likely used `b1eacc26…`, the newest revision in the cache before October. To repeat them with that revision, pass `revision="b1eacc2686a3d08ceaae5f24a88b1d519620bc09"` to the three `from_pretrained` calls in cell 5 ("5. Cohere Transcribe") of `NB_v10_asr.ipynb`.
- `resegment_v9.py` writes `goldset/segments_v9_all.new.json` unless `--out` names another file. The frozen `goldset/segments_v9_all.json` is only replaced when it is named explicitly.
- `flag_edge_words.py` annotates the files named by `--files` in place. It has no default file.
- If a whole-meeting Whisper file of `pipeline_outputs_extracted/outputs/whisper/` is missing, the drift correction falls back to no correction. No error is raised, and the segment boundaries then shift.
- `cut_clips_v9.py` writes to `<out>/clips/<segment id>.mp3`. `NB_v10_asr.ipynb` reads `goldset/clips_v10/<segment id>.mp3`.
- `NB_v10_asr.ipynb` decodes each clip in one pass. The three Whisper models returned no text for the 219 clips longer than 30 s, and step 2 fills them. `whisper_long_redecode.py --merge` expects those 219 entries to be empty and fills only those.
- Granite Speech 4.1 is limited to 200 new tokens per clip. This cuts the text of long clips outside the gold set. Every gold clip is shorter than 30 s.
- When a clip raises an error, a model cell stores an empty text and prints the first 70 characters of the error. Check the printed counts of non-empty results at the end.

## Reproduction check

On 15 October 2026, steps 3 and 4 were run in a copy of this folder. The run started from the Whisper files as `NB_v10_asr.ipynb` had written them and from the three files of `NB_whisper_long.ipynb`. The merged Whisper files, `goldset/segment_loop_gate.json` and all 112 meeting transcripts in `outputs_v10/` were identical to the stored files (line endings ignored). The check was repeated on 19 October with the cleaned code, with the same result. The notebooks need a GPU and were not run again.
