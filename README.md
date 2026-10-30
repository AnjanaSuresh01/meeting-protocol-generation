# Meeting protocol generation

Code of the master's thesis of Anjana Suresh, Hochschule Hof (2026). The system turns the recording of a German team meeting and the protocol of the previous meeting into a draft protocol. Everything runs on the university JupyterHub. The speech recognizers run in GPU notebooks. All other steps run in the JupyterHub terminal. The language model is reached through the institute's LISA service.

This page is the map of the code. It shows what each folder does, the order in which the files run, and which files produced each result of the thesis. Three more pages go into detail:

- [`GETTING_STARTED.md`](GETTING_STARTED.md): accounts, installation, keys, the first test run, and what to do when something fails
- [`DATA.md`](DATA.md): the data each task needs and where the files go. The data are not in this repository.
- [`CODEBASE_MAP.md`](CODEBASE_MAP.md): every file, with what it imports and which files use it

## Folders

| Folder | What it does | Thesis |
|---|---|---|
| [`pipeline/`](pipeline/README.md) | the system: three GPU notebooks write the transcripts, and `protokoll.py` writes the protocol | Chapter 4 |
| [`recognizers/`](recognizers/README.md) | runs the six recognizers on the 1,939 clips of the eight team meetings, and the decoder-biasing run of Canary-1B-v2 | Sections 4.4 and 4.5 |
| [`evaluation/`](evaluation/README.md) | the gold set, the scores of the recognizers, the scoring of the protocols and the runs with GLM 5.3 Flash | Chapter 5 |
| [`environments/`](environments/README.md) | the three Python environments on JupyterHub and their exact package versions | |
| `tests/` | 87 checks that need no data, no keys and no network | |
| `tools/` | `sync_copies.py`, which keeps the copied modules identical | |

Each folder runs on its own. Its scripts import each other by file name and read their data relative to their own folder. Some modules are therefore present in more than one folder, as identical copies. After you edit one of them, run `python tools/sync_copies.py`. `tests/test_copies_in_sync.py` fails if a copy differs.

Do not move files between folders. A moved script no longer finds the modules it imports.

## Run order

### Task A: a protocol for one new meeting

This is the system described in the thesis. It uses three recognizers. Canary-1B-v2 writes the transcript, and Cohere Transcribe and Whisper large-v3-turbo-German check the names. It needs no gold set.

| Order | Runs on | File | Writes into `pipeline/outputs/<MEETING>/` |
|---|---|---|---|
| 1 | GPU, kernel "NeMo ASR" | [`pipeline/cluster/step1a_canary_transcribe.ipynb`](pipeline/cluster/step1a_canary_transcribe.ipynb) | the words with their times |
| 2 | GPU, kernel "PyAnnote Diarization" | [`pipeline/cluster/step1b_pyannote_speakers.ipynb`](pipeline/cluster/step1b_pyannote_speakers.ipynb) | the transcript with speaker labels |
| 3 | GPU, kernel "NeMo ASR" | [`pipeline/cluster/step1c_cohere_whisper_transcribe.ipynb`](pipeline/cluster/step1c_cohere_whisper_transcribe.ipynb) | the Cohere and Whisper transcripts |
| 4 | terminal, `~/protokollenv` | [`pipeline/protokoll.py`](pipeline/protokoll.py) | the protocol, a list of names to check and a list of noun-style bullets |

Put the recording and last week's protocol into `pipeline/data/`. In each notebook, set the first line to the meeting and its recording, for example `MEETING, AUDIO = "april13", "data/Team-Meeting_2026-04-13.mp3"`. Then run its cells in order. Step 4 runs in the terminal:

```bash
source ~/protokollenv/bin/activate
cd <repository>/pipeline
export PYTHONUTF8=1
export LISA_RETRIES=5
python protokoll.py data/Team-Meeting_2026-04-13.mp3 \
  --previous data/Meeting_2026-04-08.md \
  --transcript outputs/april13/transcript_speakers.txt \
  --also-transcript outputs/april13/transcript_speakers_cohere.txt \
  --also-transcript outputs/april13/transcript_speakers_whisper_turbo_de.txt \
  --history data/Meeting_2026-03-02.md data/Meeting_2026-03-09.md --history-vocab \
  --date 2026-04-13 --lisa lisa-flash \
  --out outputs/april13/protocol_flash.md
```

[`pipeline/README.md`](pipeline/README.md) has the commands for later meetings, which reuse the voices of earlier ones. It also covers the Senate format and every option.

### Task B: the results of the thesis

The steps run in this order. Each one reads what the steps before it wrote. The data package of each folder already holds those outputs ([`DATA.md`](DATA.md)). You can therefore start at any step.

| Step | Folder | What runs | Output | Thesis |
|---|---|---|---|---|
| 1 | `recognizers/` | `NB_v10_asr.ipynb` and `NB_whisper_long.ipynb` on the GPU, then `whisper_long_redecode.py --merge`, `segment_loop_gate.py --apply`, `assemble_v9_transcripts.py --source all` and `transcript_gate.py --v10 --apply` in the terminal | the clip texts of the six recognizers and their 48 meeting transcripts | 4.4, 4.5 |
| 2 | `recognizers/` | `NB_biasing_full.ipynb` on the GPU | the decodes with and without decoder biasing | 5.3 |
| 3 | `evaluation/` | the twelve scripts of part 1 of [`evaluation/README.md`](evaluation/README.md), from `build_goldset_v10.py` to `score_biasing_full.py` | the gold set and the scores of the recognizers | 5.1, 5.2, 5.3 |
| 4 | `pipeline/` | `generate_protocols.py --v10 --fixed`, `protocol_repair.py --apply` and `validate_protocols_v10.py` (calls LISA) | the 96 protocols of the six-recognizer grid | 5.4, 5.5 |
| 5 | `evaluation/` | `run_deliverable.py`, `run_cite_arm.py` and `run_repair_rerun.py` (call LISA), then `remerge_deliverable.py` | the evaluated runs of `protokoll.py` and the runs with GLM 5.3 Flash | 4.3, 5.3 to 5.7 |
| 6 | `evaluation/` | the scoring scripts of part 2 of [`evaluation/README.md`](evaluation/README.md) | the results on the protocols | 5.3 to 5.7, Appendix D |

Steps 4 and 5 call LISA. LISA no longer offers the model versions of the thesis ([LISA model](#lisa-model)), so a new run writes different protocols. Step 6 scores the stored runs from the data package and needs no LISA.

The details of each step are in the README of its folder. Step 4 needs inputs copied into `pipeline/`. They are listed in [The 96 protocols of the thesis](pipeline/README.md#the-96-protocols-of-the-thesis).

## Where each result comes from

All scripts below are in `evaluation/` unless another folder is named. [`evaluation/README.md`](evaluation/README.md) lists the result file that each script writes.

| Thesis | Scripts |
|---|---|
| 4.4 segmentation and clips | `recognizers/resegment_v9.py`, `recognizers/boundary_signals.py`, `recognizers/cut_clips_v9.py` |
| 4.5 the six recognizers | `recognizers/NB_v10_asr.ipynb`, `recognizers/NB_whisper_long.ipynb`, `recognizers/whisper_long_redecode.py`, `recognizers/segment_loop_gate.py`, `recognizers/transcript_gate.py` |
| 4.6 the speaker-attributed transcript | the notebooks in `pipeline/cluster/`, `pipeline/transcript_repair.py`, `evaltx_word_retiming.py`, `short_turn_names_2026-10-26.py` |
| 4.7 to 4.9 naming, generation and merge | `pipeline/protokoll.py` and its modules |
| 5.1 the gold set | `build_goldset_v10.py`, `reference_reliability.py` |
| 5.2 recognition accuracy | `score_v10.py`, `wer_block_bootstrap.py`, `number_norm_wer.py`, `check_turbo_seeding.py`, `whisper_lineage_closeness.py`, `rover_trio.py`, `vote_subsets_ci.py`, `vote_vs_singles.py` |
| 5.3 names and domain terms | `name_reachability.py`, `name_cases_after_repair.py`, `name_check_eval.py`, `score_sweep.py`, `score_biasing_full.py`, `name_majority_gate.py`, `entity_recover.py`, `koelner_fix_replay.py`, `near_flag_gate.py`, `name_flag_context_check.py`, `compare_asr_trio.py` |
| 5.4 speaker attribution | `build_der_app.py`, `score_der.py`, `speaker_identity.py`, `evaluate_diarization.py`, `speaking_order_test.py`, `order_rule_compare.py`, `ladder_split.py`, `score_editforward.py`, `strip_copies_rescore.py`, `recount_numbers.py`, `recount_numbers_part2.py` |
| 5.5 where quality is lost | `evaluate_attribution_v10.py`, `compare_lisa_models.py`, `compare_segments.py`, `paired_and_null_2026-10-11.py`, `support_chance_2026-10-26.py`, `prompt_leakage_check.py`, `invented_examples_check_2026-10-12.py`, `check_identical_lines.py` |
| 5.6 editing effort and coverage | `score_editforward.py`, `strip_copies_rescore.py`, `delete_interval_2026-10-12.py`, `unsaid_bullets_check.py`, `thirds_recall.py`, `trend_thirds_2026-10-11.py`, `paired_and_null_2026-10-11.py`, `blind_coverage_read.py`, `blind_read_glm.py`, `coverage_semantic.py`, `coverage_rerank.py`, `judge_calibrate.py`, `depth_compare_ch5.py` |
| 5.7 runs with GLM 5.3 Flash | `run_repair_rerun.py`, `score_repair_rerun.py`, `score_repair_rerun_extra.py`, `model_comparison_2026-10-12.py`, `blind_read_glm.py`, `lisa_pro_probe.py` |
| Appendix D, values per meeting | `per_meeting_2026-10-12.py`, `model_comparison_2026-10-12.py` |

A few counts in the thesis come from reading the outputs, such as the kinds of the 34 missed bullets in Table 5.13. These readings are stored as data files, not computed by a script.

## LISA model

All protocols evaluated in the thesis were written by LISA Pro in the version `lisa-pro-03-2026` (Qwen3.5-397B-A17B). The comparison with LISA Flash used `lisa-flash-03-2026`. LISA no longer offers these versions. LISA Pro now runs on GLM 5.3 Flash. A new run therefore writes different protocols. The thesis reports its runs with GLM 5.3 Flash separately, in Section 5.7.

`protokoll.py` uses `lisa-pro` by default. For every model name that does not end in `-03-2026`, it sets `LISA_REASONING_EFFORT=low`, unless the variable is already set. The scripts that name `lisa-pro-03-2026` in their code (`generate_protocols.py` and the run and judge scripts in `evaluation/`) need the model name changed before they can run.

`LISA_BASE_URL` can end in `/openai`, as in `pipeline/.env.example`, or in `/api`. Both addresses answered requests to `lisa-pro` and `lisa-flash`.

## Data and keys

The recordings, transcripts, reference texts and protocols are internal documents of the research group. They are not in this repository. They come as three data packages, one per folder ([`DATA.md`](DATA.md)). `.gitignore` keeps audio, data folders and the key file out of the repository.

The LISA key, the LISA address and the Hugging Face token go into `pipeline/.env`. Create it from `pipeline/.env.example`. This file is never committed.

## Tests

```bash
source ~/protokollenv/bin/activate
cd <repository>
pip install -r requirements-dev.txt
python -m pytest
```

The suite needs the packages of `~/protokollenv` and pytest. It needs no data, no keys and no network. It checks:

- fixed expected values for the pipeline rules and the recognizer helpers
- the exit code and output of each script run with no arguments or with `--help`
- that every module imports without a `.env` or data
- that the copied modules are byte-identical

Some modules read a data file when they are imported. Without the data they stop with `FileNotFoundError`, and the test expects exactly that. Fifteen scripts in `evaluation/` do their work as soon as they are loaded. They are meant to be run only as scripts, so the import test skips them. Both lists are in `tests/test_structure.py`.

## Getting the code onto JupyterHub

In a JupyterHub terminal:

```bash
mkdir -p ~/Thesis
cd ~/Thesis
git clone https://github.com/AnjanaSuresh01/meeting-protocol-generation.git
```

The repository is private, so `git clone` asks for a GitHub user name and a personal access token. You can also upload a ZIP of the repository through the JupyterHub file browser.
