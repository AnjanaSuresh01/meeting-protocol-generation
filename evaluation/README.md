# Evaluation

This folder holds the code behind the results of Chapter 5 of the thesis and the scripts that draw its figures. Part 1 scores the six recognizers on the gold set. Part 2 scores the protocols that LISA wrote, including the runs with GLM 5.3 Flash. Part 3 draws the figures.

Everything runs in the JupyterHub terminal, in `~/protokollenv` ([`../environments/README.md`](../environments/README.md)). Only the scripts marked "LISA" call the language model. All others run offline on the CPU.

Keep the files in one folder. The scripts import each other by file name and read their data relative to this folder. The folder also holds copies of pipeline modules, such as `protokoll.py` and `generate_protocols.py`. They are identical to the files in `../pipeline/`.

The data are not in the repository ([`../DATA.md`](../DATA.md)). All paths below are relative to this folder.

## Part 1: the six recognizers (Sections 5.1 to 5.3)

| Script | Reads | Writes | Result |
|---|---|---|---|
| `build_goldset_v10.py` | `goldset/clip_asr_v10/<code name>__v10.json` (six files), `goldset/goldset_v9_after_second_opinion.json`, `goldset/goldset_v9_draw.json`, `goldset/rater_noise_floor.json`, `goldset/segments_v9_all.json` | `goldset/goldset_v10.json` | the gold set (5.1) |
| `reference_reliability.py` | `goldset/goldset_v10.json`, `goldset/goldset_v10.BEFORE_peinl_merge_20260907.json` | `goldset/reference_reliability.json` | the two readers disagree by 0.076 WER on 48 clips (5.1) |
| `score_v10.py` | `goldset/goldset_v10.json`, `goldset/reference_reliability.json` | the file named by `--json` | WER and CER of the six recognizers, bootstrap intervals, paired tests with Holm correction (5.2) |
| `wer_block_bootstrap.py` | `goldset/goldset_v10.json`, `goldset/score_v10_2026-10-01.json` | `goldset/wer_block_bootstrap_2026-10-01.json` | the paired tests with clips resampled in blocks (5.2) |
| `check_turbo_seeding.py` | `goldset/goldset_v10.json` | prints | WER on the clips that no recognizer reproduces exactly (5.2) |
| `number_norm_wer.py` | `goldset/goldset_v10.json` | prints | WER with numbers normalized (5.2) |
| `rover_trio.py` | `goldset/goldset_v10.json` | `goldset/rover_trio_2026-10-09.json` | the word vote of three recognizers. Canary-1B-v2 alone gives 0.099, the vote of the final three 0.082 (5.2) |
| `vote_subsets_ci.py` | `goldset/goldset_v10.json` | `goldset/vote_subsets_2026-10-10.json` | the gain of the vote on parts of the gold set (5.2) |
| `vote_vs_singles.py` | `goldset/goldset_v10.json` | `goldset/vote_vs_singles_2026-10-25.json` | the kept vote against each of the six single recognizers (5.2) |
| `whisper_lineage_closeness.py` | `goldset/goldset_v10.json` | `goldset/whisper_lineage_closeness_2026-10-29.json` | how close each output is to the scored output of Whisper large-v3-turbo, the basis of the "+" marks (4.5, 5.2) |
| `compare_asr_trio.py` | the meeting transcripts, the run outputs in `outputs_deliverable/v10/`, the protocols `Meeting_2026-*.md` | `goldset/asr_trio_comparison.json` | the naming steps of `protokoll.py`, replayed with the evaluated and with the final three recognizers. Both give the same 34 speaker names (4.5, 5.3) |
| `score_biasing_full.py` | `bias_full/sweep_full.json` (output of `../recognizers/NB_biasing_full.ipynb`), `dist/cluster_biasing_full/manifest_biasing_full.json`, `goldset/goldset_v10.json`, `goldset/name_reachability.json` | `goldset/bias_full_scores.json` | decoder biasing toward the names of the previous protocol, five rules fixed before the run (5.3) |

### Order of steps, part 1

Every script except `check_turbo_seeding.py` and `number_norm_wer.py` writes into `goldset/` and replaces the provided file of the same name. Keep a copy of the provided folder first, so that you can compare the new results with it:

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
python vote_vs_singles.py
python whisper_lineage_closeness.py
python compare_asr_trio.py
python score_biasing_full.py bias_full/sweep_full.json
diff -r --strip-trailing-cr goldset_provided goldset
```

The provided files were written on Windows, so their lines end with CR LF. On JupyterHub the new files end their lines with LF. Without `--strip-trailing-cr`, `diff` reports every file as different.

- `score_v10.py` takes the agreement of the two readers from `goldset/reference_reliability.json`. If that file is missing, it prints a warning and shows an older value (0.116 on 5 clips). Run `reference_reliability.py` first.
- `wer_block_bootstrap.py` checks that its plain bootstrap reproduces `goldset/score_v10_2026-10-01.json`, and it stops if it does not.
- `score_biasing_full.py` needs the sweep file as its argument. Without it, the script prints its usage and exits with code 1.

## Part 2: the protocols (Sections 5.3 to 5.7)

These scripts score the protocols of the stored runs in `outputs_deliverable/` and `outputs_v10/` against the human minutes `Meeting_2026-*.md`. They need no LISA, except where the table says so.

### The runs

The runs were written by three scripts that call `protokoll.py`. They need LISA. The model versions of the thesis (`lisa-pro-03-2026`, `lisa-flash-03-2026`) are no longer offered, so the default model names in these scripts fail today. A run with a current model writes new protocols that differ from the stored ones.

| Script | Runs it wrote | Used in |
|---|---|---|
| `run_deliverable.py` (LISA) | `v10`, the evaluated runs, and the repeat `v10rerun` | 4.3, 5.3 to 5.6 |
| `run_cite_arm.py` (LISA) | `v10pro_rep` (second Pro run), `v10flash` (Flash), `v14ctrl`, `v14seg` and `v14think` (repeat, one call per report, reasoning on) | 5.5, 5.6 |
| `run_repair_rerun.py` (LISA) | `v12_evaltx`, `v12_delivered` and `v12_delivered_raw`, the runs with GLM 5.3 Flash | 5.7 |
| `remerge_deliverable.py` | rebuilds the protocols of a run from its saved LISA drafts (`protocol_pro.neu.md`) with the deterministic merge steps, without a model call | 4.9 |

### Section 5.3, names and domain terms

| Script | Writes | What it computes |
|---|---|---|
| `name_reachability.py` | `goldset/name_reachability.json` | sorts each name that the minutes and the recognizers spell differently: in the prompt name list, only in the previous document, or nowhere in it |
| `name_check_eval.py` | prints | the definition of a name case (Kölner Phonetik and a similarity of at least 0.6), and recall and precision of the names-to-check list |
| `name_cases_after_repair.py` | `goldset/name_cases_after_repair_2026-10-29.json` | every run scored on the name cases drawn before and after the Whisper re-decode |
| `name_ab_score.py` | `goldset/name_ab_score.json` | name spellings without and with the prompt rules 5a and 5b |
| `name_majority_gate.py` | prints | whether a majority vote of the recognizers would fix or break the name spellings |
| `entity_recover.py` | prints (`--apply` writes `goldset/entity_recover.json`) | constrained entity correction after the RECOVER method, each edit checked against the minutes of the same meeting |
| `koelner_fix_replay.py` | `goldset/koelner_fix_replay_2026-10-10.json` | the naming steps replayed with the original and with a corrected Kölner Phonetik function |
| `near_flag_gate.py` | prints | each name suggestion from the previous document that reaches a protocol, labeled right, wrong or undecidable |
| `name_flag_context_check.py` (LISA) | `goldset/name_flag_context_check_2026-10-01.json` | LISA asked whether each suggestion is a real misrecognition. `--contexts` prints the sentences without a LISA call |
| `qa_protocols.py` | `outputs_deliverable/qa_report.json` | a layered quality check of a run, which also finds one name in two spellings |

`entity_lexicon.py` and `acronym_swap_check.py` are modules that these scripts import.

### Section 5.4, speaker attribution

| Script | Writes | What it computes |
|---|---|---|
| `build_der_app.py` | `dist/der_annotation/` | draws 240 frames of 2 s and builds the listening page on which each frame was labeled with its speaker. It needs the recordings |
| `score_der.py` | `goldset/der_result.json` | the labeled frames: miss, false alarm and confusion with a Wilson interval |
| `speaker_identity.py` | `goldset/speaker_identity.json` (with `--apply`) | the reference naming of the diarization speakers from the attendance record |
| `evaluate_diarization.py` | `pipeline_outputs_extracted/evaluation/layer3_diarization.json` | the number of diarization clusters per meeting against the attendance record |
| `score_deliverable.py` | prints | each voice name of a run against `goldset/speaker_identity.json` |
| `speaking_order_test.py` | prints | the reporting order recovered from the diarization against the order of the minutes, with a roster and a random baseline |
| `order_rule_compare.py` | prints | the first-utterance rule against the longest-block rule for the order of the report sections |
| `ladder_split.py` | `goldset/ladder_split.json` | topic ownership against "last week's owner", split by whether the owner said the topic in the meeting |
| `attribution_blocklevel.py` | `outputs_deliverable/attribution_blocklevel.json` | topic attribution with block-to-block matching next to the per-person matcher |
| `handover_test.py`, `handover_read.py` | prints | topics whose owner changed since last week, and the evidence for each of them |
| `evaltx_word_retiming.py` | `goldset/evaltx_word_retiming_2026-10-29.json` | how the evaluated transcript timed its words (Section 4.6) |
| `short_turn_names_2026-10-26.py` | `goldset/short_turn_names_2026-10-26.json` | first names inside diarization turns shorter than 1.2 s (Section 4.6) |

`absent_reachability.py` and `thesis_pipeline/diarization_eval.py` are modules that these scripts import.

### Section 5.5, where quality is lost

| Script | Writes | What it computes |
|---|---|---|
| `evaluate_attribution_v10.py` | `goldset/attribution_v10_ab.json` | flat against diarized transcript as LISA input |
| `compare_lisa_models.py` | prints | LISA Flash against a second LISA Pro run, with sign tests per meeting |
| `compare_segments.py` | `goldset/compare_segments_2026-10-01.json` | one LISA call per report against one call per meeting |
| `support_chance_2026-10-26.py` | `goldset/support_chance_2026-10-26.json` | the chance level of the 30% support rule |
| `prompt_leakage_check.py` | prints | where the example fragments of the prompt rules occur |
| `invented_examples_check_2026-10-12.py` | `goldset/invented_examples_check_2026-10-12.json` | that the invented prompt examples occur in no protocol and no transcript |
| `check_identical_lines.py` | prints | draft bullets identical to a bullet of the same week's minutes |

### Section 5.6, editing effort and coverage

| Script | Writes | What it computes |
|---|---|---|
| `score_editforward.py` | prints | pooled ADD and DELETE, split into new and carried bullets, topic ownership and names, against "last week alone" (Table 5.11) |
| `strip_copies_rescore.py` | `goldset/strip_copies_rescore_2026-10-07.json` | the minutes without last week's copied leaf bullets, scored again with two-stage bootstrap intervals (Table 5.12, Figure 5.2) |
| `delete_interval_2026-10-12.py` | `goldset/delete_interval_2026-10-12.json` | the interval of the DELETE difference between the draft and last week's protocol |
| `unsaid_bullets_check.py` | `goldset/unsaid_bullets_2026-10-10.json` | missed new bullets whose content words hardly occur in the transcript |
| `thirds_recall.py`, `trend_thirds_2026-10-11.py` | `goldset/thirds_recall_2026-10-10.json`, prints | recorded new bullets per third of the meeting, and the trend test |
| `paired_and_null_2026-10-11.py` | prints | exact McNemar tests between runs (Table 5.14) and a chance check of the 30% word rule |
| `blind_coverage_read.py` | `goldset/blind_read_sheet_*.md`, `goldset/blind_read_key_*.json` (with `--build`) | the blind reading sheets of the first LLM reader. `--score` scores its judgments |
| `blind_read_glm.py` (LISA for `--read`) | `goldset/blind_read_glm_*_2026-10-28.json` | the second LLM reader, GLM 5.3 Flash, on the same 100 items. `--score` runs offline |
| `coverage_semantic.py`, `coverage_rerank.py` | `goldset/coverage_*_calibration_2026-10-01.json` | the embedding matcher and the cross-encoder matcher, calibrated on april27 and tested on june22 |
| `judge_calibrate.py`, `judge_deliverable.py`, `judge_coverage.py` (LISA) | `goldset/judge_*.json` | LISA as a judge of coverage, against the hand labels |
| `depth_compare_ch5.py`, `number_trace.py` | prints | the share of nested bullets, and the source of every figure in a draft |
| `per_meeting_2026-10-12.py` | `goldset/per_meeting_2026-10-12.json` | the values per meeting of Appendix D. It reads the length of the recordings |

`evaluate_protocols.py` and `postedit_effort.py` are modules that these scripts import. `evaluate_protocols.py` can also run on its own and scores every protocol of the grid.

### Section 5.7, the runs with GLM 5.3 Flash

| Script | Writes | What it computes |
|---|---|---|
| `score_repair_rerun.py` | `goldset/repair_rerun_scores_2026-10-11.json` | structure, naming, post-edit and stripped recall of the three GLM runs, with McNemar tests |
| `score_repair_rerun_extra.py` | `goldset/repair_rerun_extra_2026-10-12.json` | invented names of each GLM run against its own transcript, and whether the re-transcribed stretches reach the protocol |
| `model_comparison_2026-10-12.py` | `goldset/model_comparison_2026-10-12.json` | the Qwen3.5 runs against the GLM run on the same inputs |
| `compare_v10_rerun.py` | prints | two runs compared on self-check, invented names, foreign script and speaker order |
| `lisa_pro_probe.py`, `lisa_models_listing.py` (LISA) | `goldset/lisa_pro_probe_2026-10-12.json`, `goldset/lisa_models_2026-10-12.json` | the reasoning settings of the current LISA Pro, and the model list of the service |

`recount_numbers.py` and `recount_numbers_part2.py` print further counts across Sections 5.2 to 5.6 in one run, among them the naming route per voice (Table 5.8) and the noun-phrase share (Table 5.10). `recount_numbers.py` also runs `score_v10.py` and writes `goldset/score_v10_2026-09-30.json`.

### Order of steps, part 2

The scripts read the stored runs and the stored results. Several of them write into `goldset/` and replace the provided file of the same name, so keep a copy first, as in part 1. Run each script from this folder:

```bash
source ~/protokollenv/bin/activate
cd <repository>/evaluation
export PYTHONUTF8=1
cp -r goldset goldset_provided
# 5.3
python name_reachability.py
python name_cases_after_repair.py
python name_check_eval.py
python name_ab_score.py
python name_majority_gate.py
python entity_recover.py
python koelner_fix_replay.py
python near_flag_gate.py
python name_flag_context_check.py --contexts
python qa_protocols.py v10
# 5.4
python score_der.py der_answers.json
python speaker_identity.py
python evaluate_diarization.py
python score_deliverable.py v10
python speaking_order_test.py
python order_rule_compare.py
python ladder_split.py
python attribution_blocklevel.py v6 v8
python handover_test.py
python evaltx_word_retiming.py
python short_turn_names_2026-10-26.py
# 5.5
python compare_lisa_models.py
python compare_segments.py
python support_chance_2026-10-26.py
python prompt_leakage_check.py
python invented_examples_check_2026-10-12.py
python check_identical_lines.py
# 5.6
python score_editforward.py v10 v10rerun v10flash v10pro_rep
python strip_copies_rescore.py
python delete_interval_2026-10-12.py
python unsaid_bullets_check.py
python thirds_recall.py
python trend_thirds_2026-10-11.py
python paired_and_null_2026-10-11.py
python blind_coverage_read.py --score
python blind_read_glm.py --score
python coverage_semantic.py --calibrate
python coverage_rerank.py --calibrate
python depth_compare_ch5.py
python number_trace.py v10
python per_meeting_2026-10-12.py
# 5.7
python score_repair_rerun.py
python score_repair_rerun_extra.py
python model_comparison_2026-10-12.py
python compare_v10_rerun.py
# counts
python recount_numbers_part2.py
python recount_numbers.py
```

### Points to know, part 2

- Fourteen scripts of part 2 and `whisper_lineage_closeness.py` do their work as soon as Python loads them. They have no `if __name__ == "__main__":` block. Run them only as scripts, as above. `../tests/test_structure.py` lists them and does not import them.
- `evaluate_attribution_v10.py` stops with `KeyError: 'april13'` on the provided data. `ROSTER` in `evaluate_speaker_attribution.py` has no entry for april13, and the reference set contains that meeting. The stored `goldset/attribution_v10_ab.json` was computed before april13 was added to the reference set.
- `name_reachability.py` gives other name cases today than the stored `goldset/name_reachability.json`. The stored file was written before the long Whisper segments were decoded again. `name_cases_after_repair.py` scores the runs on both sets of cases.
- `strip_copies_rescore.py` first checks that `score_editforward.py` reproduces the published values, and it stops if it does not.
- `coverage_semantic.py` and `coverage_rerank.py` use the stored caches `goldset/emb_cache_granite_r2.npz` and `goldset/rerank_cache_bge_v2_m3.json`. Without them they need the models `granite-embedding-311m-multilingual-r2` and `bge-reranker-v2-m3`.
- `trend_thirds_2026-10-11.py` reads `goldset/` relative to the current folder, so start it from this folder.
- `score_editforward.py` rewrites `goldset/ladder_split.json` during its run and then restores it.

## Part 3: the figures

The figure scripts are in `figures/`. Each one writes its PNG file next to itself.

| Script | Figure | Reads |
|---|---|---|
| `figures/fig_system_overview_2026-10-28.py` | 4.1, components and data flow of the system | nothing (drawn from fixed labels) |
| `figures/fig_evaluation_pipeline.py` | 4.2, the evaluation pipeline | nothing (drawn from fixed labels) |
| `figures/fig_segment_lengths_2026-10-28.py` | 4.3, durations of the 1,939 segments | `goldset/segments_v9_all.json` |
| `figures/fig_recognizer_selection_2026-10-29.py` | 4.4, selection of the three recognizers | `goldset/score_v10_2026-10-01.json`, `goldset/rover_trio_2026-10-09.json` |
| `figures/fig5_1_wer_kept_2026-10-29.py` | 5.1, WER per system and of the vote | `goldset/score_v10_2026-10-01.json`, `goldset/rover_trio_2026-10-09.json` |
| `figures/fig_strip_copies.py` | 5.2, origin of the leaf bullets of the minutes | `goldset/strip_copies_rescore_2026-10-07.json` |
| `figures/fig6_1_v3_2026-10-25.py` | 6.1, the system against last week's protocol alone | nothing (the values are written in the script) |
| `figures/fig6_2_chain_2026-10-26.py` | 6.2, where content is lost along the chain | nothing (the values are written in the script) |

```bash
cd <repository>/evaluation
python figures/fig5_1_wer_kept_2026-10-29.py
```

## Reproduction check

Part 1: on 15 October 2026 the first ten scripts of the table were run in a copy of this folder with the provided data. The output of `build_goldset_v10.py` was identical to the stored file, byte for byte. The other result files were identical apart from date and build fields. On 19 October the ten scripts were run again with the cleaned code, and no result file changed. `vote_vs_singles.py` and `whisper_lineage_closeness.py` were added on 29 October. Both wrote result files identical to the stored ones.

Part 2 and part 3: on 29 October 2026 every script of the order above ran once in a copy with the provided data. The data were reset before each script, so that no script saw the output of another. The scripts ran twice, once with the code of the thesis runs and once with the code of this folder. Both versions gave the same result files and the same printed numbers. The only exceptions are the reworded messages and one removed quotation (`../CHANGES.md`). The eight figures were identical to those of the thesis, byte for byte.

Of the result files the run wrote, 39 were identical to the stored ones. These files differed:

- `outputs_deliverable/qa_report.json`: the wording of the messages, with the same findings
- `goldset/strip_copies_rescore_2026-10-07.json`: the removed quotation in the `meta` block
- `goldset/name_cases_after_repair_2026-10-29.json`: the stored file has a block `remaining_29` that the script does not write
- `goldset/repair_rerun_scores_2026-10-11.json`: the stored file has two more pairs (`v10 vs v12_evaltx`, `v10rerun vs v12_evaltx`) that the script does not write
- `goldset/name_reachability.json`: written before the long Whisper segments were decoded again (see above)
- `goldset/name_ab_score.json`, `goldset/entity_variants.json`, `goldset/ladder_split.json`, `outputs_deliverable/attribution_blocklevel.json` and `pipeline_outputs_extracted/evaluation/layer1_protocol_eval.json`: written from an earlier state of the protocol outputs, before they were last regenerated

`per_meeting_2026-10-12.py` needs the recordings. The run had them. `evaluate_attribution_v10.py` stopped with the `KeyError` described above. The scripts that call LISA were not run. Python checked that they compile.
