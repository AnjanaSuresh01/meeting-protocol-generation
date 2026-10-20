# Evaluation of the six recognizers

This folder scores the six recognizers on the gold set of 147 human-corrected clips, tests their differences, measures how far two human readers of the reference disagree, scores the word vote of three recognizers and the decoder biasing of Canary-1B-v2, and replays the speaker naming with the three recognizers of the final system. Everything runs in the JupyterHub terminal on the CPU, in `~/protokollenv` (`../environments/README.md`).

The code is the code that produced the thesis results. On 19 October 2026 its comments were shortened and a few printed messages were fixed (`../CHANGES.md`). The result files did not change. Keep the files in one folder: the scripts import each other by name and find their data relative to their own location. The folder also holds the pipeline modules that `compare_asr_trio.py` imports; they are identical to the files in `../pipeline/`.

## Scripts

| Script | Reads | Writes | Thesis |
|---|---|---|---|
| `build_goldset_v10.py` | `goldset/clip_asr_v10/<code name>__v10.json` (six files), `goldset/goldset_v9_after_second_opinion.json`, `goldset/goldset_v9_draw.json`, `goldset/rater_noise_floor.json`, `goldset/segments_v9_all.json` | `goldset/goldset_v10.json` | the gold set (Section 5.1) |
| `reference_reliability.py` | `goldset/goldset_v10.json`, `goldset/goldset_v10.BEFORE_peinl_merge_20260907.json` | `goldset/reference_reliability.json` | disagreement of two readers, 0.076 WER on 48 clips (Section 5.1) |
| `score_v10.py` | `goldset/goldset_v10.json`, `goldset/reference_reliability.json` | the file named by `--json` | WER and CER of the six recognizers, bootstrap intervals, paired tests with Holm correction (Section 5.2) |
| `wer_block_bootstrap.py` | `goldset/goldset_v10.json`, `goldset/score_v10_2026-10-01.json` | `goldset/wer_block_bootstrap_2026-10-01.json` | the paired tests with clips resampled in blocks (Section 5.2) |
| `check_turbo_seeding.py` | `goldset/goldset_v10.json` | (prints) | WER on the clips that no recognizer reproduces exactly (Section 5.2) |
| `number_norm_wer.py` | `goldset/goldset_v10.json` | (prints) | WER with numbers normalized (Section 5.2) |
| `rover_trio.py` | `goldset/goldset_v10.json` | `goldset/rover_trio_2026-10-09.json` | word vote of three recognizers, Canary-1B-v2 alone 0.099 against 0.082 for the vote of the final three (Section 5.2) |
| `vote_subsets_ci.py` | `goldset/goldset_v10.json` | `goldset/vote_subsets_2026-10-10.json` | the gain of the vote on the subsets of the gold set (Section 5.2) |
| `compare_asr_trio.py` | the meeting transcripts in `outputs_v11_final/canary_diar/`, `outputs_v10/cohere/`, `outputs_v10/granite41/`, `outputs_v10/whisper_turbo_de/`, the run outputs in `outputs_deliverable/v10/`, the protocols `Meeting_2026-*.md` | `goldset/asr_trio_comparison.json` | the naming steps of `protokoll.py` replayed with the evaluated and the final three recognizers: the same 34 names (Sections 4.5 and 5.3) |
| `score_biasing_full.py` | `bias_full/sweep_full.json` (output of `../recognizers/NB_biasing_full.ipynb`), `dist/cluster_biasing_full/manifest_biasing_full.json`, `goldset/goldset_v10.json`, `goldset/name_reachability.json` | `goldset/bias_full_scores.json` | decoder biasing of Canary-1B-v2 toward names from the previous protocol, five pre-registered rules and the verdict (Section 5.3) |

The other `.py` files are modules these scripts import.

## Data

The data are internal recordings and texts of the research group: the gold-set references, the recognizer outputs, the meeting transcripts and the protocols. They are not in the repository and are provided separately. Paths in the table above are relative to this folder. The six files of `goldset/clip_asr_v10/` are the output of `../recognizers/` after its step 3.

## Order of steps

Every script except `check_turbo_seeding.py` and `number_norm_wer.py` writes into `goldset/` and replaces the provided file of the same name. Keep a copy of the provided folder first, so the new results can be compared with it:

```bash
source ~/protokollenv/bin/activate
cd <repository>/evaluation
export PYTHONUTF8=1
cp -r goldset goldset_provided
python build_goldset_v10.py
python reference_reliability.py
python score_v10.py --json goldset/score_v10_2026-10-01.json
python wer_block_bootstrap.py
python check_turbo_seeding.py
python number_norm_wer.py
python rover_trio.py
python vote_subsets_ci.py
python compare_asr_trio.py
python score_biasing_full.py bias_full/sweep_full.json
diff -r goldset_provided goldset
```

`score_biasing_full.py` needs the sweep file as its argument. Without it, the script prints its usage and exits with code 1.

## Points to know

- `score_v10.py` uses the agreement of the two readers from `goldset/reference_reliability.json`. If that file is missing, it prints a warning and shows an older value (0.116 on 5 clips) instead. Run `reference_reliability.py` first.
- `wer_block_bootstrap.py` checks that its plain bootstrap reproduces `goldset/score_v10_2026-10-01.json`, and stops if it does not. In the order above, that file was just written by `score_v10.py`, so the check compares the two scripts with each other. Compare `goldset/score_v10_2026-10-01.json` with the copy in `goldset_provided/` to check the scores themselves.
- `compare_asr_trio.py` imports `protokoll.py` and its modules but does not call LISA.

## Reproduction check

On 15 October 2026 all ten scripts were run in a copy of this folder with the provided data. The outputs of `build_goldset_v10.py` (byte for byte), `reference_reliability.py`, `score_v10.py`, `wer_block_bootstrap.py`, `rover_trio.py`, `vote_subsets_ci.py`, `compare_asr_trio.py` and `score_biasing_full.py` were identical to the stored result files, apart from date and build fields (for `score_v10.py`, its `meta` block). `check_turbo_seeding.py` and `number_norm_wer.py` only print and ended without error. On 19 October the ten scripts were run again with the cleaned code in a copy holding the stored result files. No result file changed.
