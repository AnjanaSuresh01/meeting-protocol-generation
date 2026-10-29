# Data

This repository holds code only. The data are recordings and texts of internal meetings of a research group at Hochschule Hof. They contain the voices and names of its members, so they are not on GitHub. They are shared separately, only with people the group allows. To get them, ask the author of this repository.

## Do I need the gold set?

| What you want to do | Gold set needed? | Other data needed |
|---|---|---|
| Install the code and run the tests | no | none |
| Write a protocol for a new meeting (`pipeline/protokoll.py`) | no | your recording and the protocol of the meeting before it |
| Transcribe the evaluation clips with the six recognizers (`recognizers/`) | yes: the clips and the segment list | the eight recordings |
| Compute the numbers of Chapter 5 and draw the figures (`evaluation/`) | yes: the reference texts | the recognizer outputs, the stored protocol runs and the human protocols |
| Write the 96 protocols of the thesis again (`pipeline/generate_protocols.py`) | two files of the gold-set folder | the recognizer transcripts and the previous protocols |

`protokoll.py` reads no gold-set file. On 28 October 2026 it ran in a folder with no `goldset/` folder at all, on the 9 March meeting, with LISA replaced by its saved answer. All six output files were identical to those of the delivered run.

## What the gold set is

The gold set is 152 short clips from the eight team meetings. A human corrected the text of each clip. 147 clips are scored and 5 are excluded. Everything about them is in one file, `goldset_v10.json`. For each clip it holds:

- `reference_text`: the corrected text
- `model_hypotheses`: the texts of the six recognizers
- `scoring.eligible`: whether the clip is scored
- `resolution`: who settled the reference text

The audio of the gold clips is in `clips_v10_gold.zip`.

## The three data packages

The data come as three ZIP files, one per folder. Each one holds the folder paths, so it can be unpacked inside the repository folder. Every file then lands where the code reads it. `.gitignore` covers all of these paths, so Git never commits them by mistake.

Unpack a package in the repository folder:

```bash
cd <repository>
python -m zipfile -e data_evaluation.zip .
```

You only need the package of the folder you want to use.

### `data_pipeline.zip`, for `pipeline/`

For the example commands of `README.md` and `pipeline/README.md`:

| Path | Content | Size |
|---|---|---|
| `pipeline/data/Team-Meeting_2026-*.mp3`, `.flac` | the eight recordings | 448 MB |
| `pipeline/data/Meeting_2026-*.md` | the 13 human protocols from 2 March to 22 June 2026 | 0.3 MB |

For the 96 protocols of the thesis ("The 96 protocols of the thesis" in `pipeline/README.md`):

| Path | Content | Size |
|---|---|---|
| `pipeline/outputs_v10/<recognizer>/<meeting>/transcript_speakers.txt` | the 48 meeting transcripts of the six recognizers | under 14 MB |
| `pipeline/goldset/clip_asr_v10/<recognizer>__v10.json` | the clip texts of the six recognizers | 2.7 MB |
| `pipeline/goldset/speaker_identity.json` | which speaker label is which person | under 0.1 MB |
| `pipeline/goldset/segments_v9_all.json` | the 1,939 segments | 2.4 MB |
| `pipeline/Meeting_2026-03-02.md`, `Meeting_2026-04-08.md`, `Meeting_2026-04-20.md`, `Meeting_2026-05-04.md`, `Meeting_2026-05-11.md`, `Meeting_2026-05-18.md`, `Meeting_2026-06-08(1).md`, `Meeting_2026-06-15.md` | the eight previous protocols, named in `PREVIOUS` in `pipeline/term_source.py` | 0.2 MB |
| `pipeline/Meeting-Protocol-FG-MM-AI_2026-03-02.docx` | the previous protocol of 9 March as a Word file, for `--carry` | under 0.1 MB |

A new meeting needs none of this. Put your own recording and last week's protocol into `pipeline/data/`.

### `data_recognizers.zip`, for `recognizers/`

| Path | Content | Size |
|---|---|---|
| `recognizers/clips_v10_gold.zip` | 152 clips: the 147 gold clips and 5 more | 10.0 MB |
| `recognizers/clips_v10_rest.zip` | the other 1,789 clips (2 clips are in both ZIP files, 1,939 different clips in all) | 136.8 MB |
| `recognizers/goldset/segments_v9_all.json` | the 1,939 segments | 2.4 MB |
| `recognizers/goldset/goldset_gold_147.json` | the ids of the 147 gold clips, decoded first | 0.2 MB |
| `recognizers/manifest_biasing_full.json` | input of `NB_biasing_full.ipynb` | 0.3 MB |
| `recognizers/Team-Meeting_2026-*.mp3`, `.flac` | the eight recordings, read by `transcript_gate.py` | 448 MB |

`recognizers/README.md` shows how to unpack the two clip ZIP files into `goldset/clips_v10/`.

### `data_evaluation.zip`, for `evaluation/`

This package holds the recognizer outputs as well. The evaluation therefore runs without a GPU and without running `recognizers/` first.

| Path | Content |
|---|---|
| `evaluation/goldset/goldset_v10.json` | the gold set (see above) |
| `evaluation/goldset/clip_asr_v10/<recognizer>__v10.json` | the clip texts of the six recognizers |
| `evaluation/goldset/goldset_v9_after_second_opinion.json`, `goldset_v9_draw.json`, `rater_noise_floor.json`, `segments_v9_all.json`, `goldset_v10.BEFORE_peinl_merge_20260907.json`, `name_reachability.json` | inputs of `build_goldset_v10.py`, `reference_reliability.py` and `score_biasing_full.py` |
| `evaluation/goldset/reference_reliability.json`, `score_v10_2026-10-01.json`, `wer_block_bootstrap_2026-10-01.json`, `rover_trio_2026-10-09.json`, `vote_subsets_2026-10-10.json`, `asr_trio_comparison.json`, `bias_full_scores.json` | the stored results, to compare a new run with |
| `evaluation/outputs_v10/{cohere,granite41,whisper_turbo_de}/<meeting>/transcript_speakers.txt` | meeting transcripts of three recognizers |
| `evaluation/outputs_v11_final/canary_diar/<meeting>/transcript_speakers.txt` | meeting transcripts of Canary-1B-v2 |
| `evaluation/outputs_deliverable/v10/<meeting>/protocol_pro.md`, `protocol_pro.namen.md`, `run_pro.log` | the evaluated runs of `protokoll.py` |
| `evaluation/Meeting_2026-*.md` | the human protocols |
| `evaluation/bias_full/sweep_full.json` | output of `NB_biasing_full.ipynb` |
| `evaluation/dist/cluster_biasing_full/manifest_biasing_full.json` | input of the biasing run |

These are exactly the files the first ten scripts of `evaluation/README.md` opened on 28 October 2026. With them, all ten scripts ended without error, and the eight result files were identical to the stored ones (line endings ignored).

Part 2 of `evaluation/README.md` (Sections 5.3 to 5.7) and the figure scripts need more files from the same package:

| Path | Content |
|---|---|
| `evaluation/outputs_deliverable/<run>/<meeting>/` | the stored runs of `protokoll.py` (21 run folders, among them `v10`, `v10rerun`, `v10pro_rep`, `v10flash`, `v14ctrl`, `v14seg`, `v14think` and the three GLM runs `v12_*`): `protocol_pro.md`, the saved LISA draft `protocol_pro.neu.md`, `protocol_pro.repair.json`, `protocol_pro.transcript.txt` and `run_pro.log` |
| `evaluation/outputs_v10/<recognizer>/<meeting>/` | the transcripts and the protocols of the six-recognizer grid (`protocol_flash*.md`, `protocol_pro*.md`) |
| `evaluation/outputs_v11_final/canary_diar/<meeting>/` | the evaluated transcripts and their protocols |
| `evaluation/pipeline_outputs_extracted/outputs/<recognizer>/<meeting>/` | the whole-meeting Canary-1B-v2 decode, the diarization (`diarization_hyp.rttm`) and the protocols of earlier runs |
| `evaluation/goldset/speaker_identity.json`, `speaker_embeddings.npz` | the reference naming of the speakers and its voice embeddings |
| `evaluation/goldset/gap_recovered.json`, `verify_short_turns_canary_diar.json`, `ladder_hand_labels.json`, `name_ab_before/` | inputs of the transcript and attribution scripts |
| `evaluation/goldset/blind_read_key_*.json`, `blind_read_verdicts_*.json`, `new_bullet_coverage_*_handcheck.json`, `missed_bullets_categories_2026-10-11.json` | the keys and judgments of the blind readings, and the hand labels |
| `evaluation/goldset/emb_cache_granite_r2.npz`, `rerank_cache_bge_v2_m3.json` | the stored scores of the two learned matchers, so that no model has to be downloaded |
| `evaluation/goldset/*.json` (the other result files) | the stored results, to compare a new run with |
| `evaluation/config.json` | the list of known names, read by `evaluate_protocols.py` when it is loaded |
| `evaluation/der_answers.json`, `evaluation/dist/der_annotation/frames.json` | the 109 labeled frames and the 240 drawn frames of the frame sample |
| `evaluation/transcript_gate.json` | the gate results of the long-form transcripts |
| `evaluation/Team-Meeting_2026-*.mp3`, `.flac` | the eight recordings. Only `per_meeting_2026-10-12.py` reads them, for their length |

On 29 October 2026 these scripts ran on a copy of the full package. The files they opened are the ones listed in this section and the section before it. The results are described in the reproduction check of `evaluation/README.md`.

## Where the packages are kept

The packages are not part of the Git history. The author keeps them in one place and gives the link to people who are allowed to use them. A GitHub release of this private repository, or a folder on the university's servers, both work. GitHub accepts release files up to 2 GiB each, and only collaborators of a private repository can download them. A release of a repository that is made public becomes public too.
