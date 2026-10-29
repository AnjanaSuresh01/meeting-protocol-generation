# Changes of 19 October 2026

This code produced the thesis results. On 19 October 2026 it was tidied. No change alters a result: the delivered protocol runs, the Senate run, the recognizer transcripts and the evaluation results were reproduced byte for byte afterwards (`pipeline/TESTED.md`, section 4).

## Comments, names and files

- Comments and docstrings were shortened to what explains the code, and the text cells of the notebooks to run instructions. This step did not touch the code: for every file, the syntax tree without docstrings is the same before and after.
- `pipeline/peinl_charge_attribution.py` is now `attribution_baselines.py`, and `evaluation/peinl_notes.py` is now `rater_notes.py`, in every folder that has a copy. The imports were updated.
- Printed messages were reworded in neutral terms.
- The test data use invented names instead of real ones.
- `NOTES.md`, which collected the development history taken out of the comments on 17 October, was removed.

## Fixes

| File | Before | After |
|---|---|---|
| `recognizers/resegment_v7.py` | without `--retreat`, the script crashed when it wrote its output, and it printed fixed file names | writes `segments_v7.json` and prints the files it wrote |
| `recognizers/assemble_v9_transcripts.py` | `--keep-degenerate` was read from the command line, but argparse rejected it | the option is declared |
| `recognizers/resegment_v9.py` | `--out` defaulted to the frozen `goldset/segments_v9_all.json` | defaults to `goldset/segments_v9_all.new.json` |
| `recognizers/flag_edge_words.py` | without arguments, it rewrote two frozen goldset files in place | `--files` is required |
| `recognizers/repair_word_speakers.py` | `--apply` crashed if `transcript_speakers.txt` was missing, and its last hint named a script that is not in the repository | backs the file up only if it exists; the hint is removed |
| `recognizers/NB_v10_asr.ipynb`, `NB_whisper_long.ipynb`, `NB_biasing_full.ipynb` | worked in the fixed folder `/home/jovyan/Thesis/protocol_generator`; two named the kernel `nemo_asr`; one message named `clips_v10.zip` | work in their own folder; kernel `nemo-asr`; the message names the two clip packages |
| `pipeline/cluster/step1c_cohere_whisper_transcribe.ipynb` | did not read the token from `pipeline/.env`, so the gated Cohere model could not be downloaded on an account without a cached copy | cell 1 reads `pipeline/.env` |
| `pipeline/cluster/step1c_cohere_whisper_transcribe.ipynb` | added the 0.5 s silence before it split long clips, so a cut could land in the padding: with the files of 27 April, 5 of 154 pieces were pure silence and went to both recognizers | splits the speech at 27 s first, then pads each piece; no piece is pure silence, and every piece is at most 28 s |
| `pipeline/protokoll.py` | a failed LISA request ended with a traceback | prints the error and exits with code 1 |
| `pipeline/protokoll.py` | `--transcribe-here` failed after its warning when pyannote was not installed | stops at once with a message |
| `pipeline/protokoll.py`, `carry_forward.py` | when no item could be read from `--previous`, nothing was carried, and the program still said it was | prints a warning that names the expected layout |
| `pipeline/protokoll.py` | `--cite` failed late, at the import of `segment_cite.py` | stops at once with a message |
| `pipeline/protokoll.py` | without `--transcript`, the message pointed to `python -m thesis_pipeline.canary_diarize` | points to notebooks 1a and 1b |
| `pipeline/protokoll.py` | a `--transcript` path that does not exist was reported as "No --transcript given" | names the missing file |
| `pipeline/carry_tag.py` | a PDF as previous document was read as raw bytes (style examples, carry-over, [ÜBERTRAG] and [PRÜFEN] marks, the check for invented names) | the PDF text is read with `term_source.read_document` |
| `pipeline/carry_tag.py`, `carry_forward.py`, `attribution_baselines.py`, `protokoll.py` | a file ending in capitals (`.DOCX`, `.MD`) was not recognized | file endings are compared in lower case |
| `pipeline/voice_profiles.py` | minutes that are not `.md` or `.docx` crashed the voice enrollment | enrollment is skipped with a message |
| `pipeline/stale_flag.py`, `carry_forward.py`, `style_exemplars.py`, `transcript_repair.py` | without arguments: `IndexError` or `ValueError` | usage message |
| `pipeline/entity_correct.py` | without data: `KeyError: 'apply'` | names the missing data and exits |
| `pipeline/postedit_effort.py` | needs `evaluate_protocols.py`, which is not in the repository; even `--help` failed | `--help` works, and a run says what is missing |
| `pipeline/term_source.py` | `--compare` compared against an empty list when `entity_lexicon.py` was missing | says that `entity_lexicon.py` is missing |
| `pipeline/thesis_pipeline/canary_diarize.py` | `main()` crashed on the missing `config.json` | says to use notebooks 1a and 1b |
| `pipeline/name_check.py` | help and output said the list is appended to each protocol, and the header said no previous protocol is used | they name the separate file `namen_zum_pruefen.md` that is actually written, and the header names both sources of the list |
| `pipeline/name_check.py` | a PDF previous document was read as raw bytes for its list of known words | the PDF text is read with `term_source.read_document` |
| `pipeline/validate_protocols_v10.py` | with no protocol files, it printed "No violations" and "0 / 0 protocols pass" | says that no files were found and exits with code 2 |
| `pipeline/handover_naming.py` | without transcripts: `ZeroDivisionError` | says what is missing and exits with code 1 |
| `recognizers/thesis_pipeline/goldset_local.py` | without the partner transcripts, it overwrote its four output files with empty lists and then crashed | writes nothing and says why |
| `pipeline/thesis_pipeline/__init__.py` | `__all__` listed modules that do not exist | removed |
| `evaluation/score_v10.py` | without `goldset/reference_reliability.json`, it showed an older floor without a warning | prints a warning |
| `evaluation/compare_asr_trio.py` | read only the old wording of the handover line in run logs | reads the old and the new wording |

The copies of these files in `evaluation/` carry the same changes.

# Additions of 29 October 2026

`evaluation/` now holds the code behind Sections 5.3 to 5.7 of the thesis, the runs with GLM 5.3 Flash, and the scripts that draw the figures. These files were added:

- 63 scripts and `thesis_pipeline/diarization_eval.py` for Sections 5.3 to 5.7
- `vote_vs_singles.py` and `whisper_lineage_closeness.py` for Section 5.2
- 8 figure scripts in `evaluation/figures/`

They are the scripts that produced the stored results, with these changes:

- Docstrings and comments were rewritten in the same way as on 19 October. For every file, the syntax tree without docstrings is the same before and after.
- Printed messages, table labels and the header of the blind reading sheet were reworded in neutral terms, in 76 places. Keys and values that go into result files did not change, with one exception. `strip_copies_rescore.py` no longer stores a quoted request in the `meta` block of its result file.
- Three scripts were renamed after what they compute: `speaking_order_test.py`, `recount_numbers.py` and `recount_numbers_part2.py`. Imports of older module names now point to `attribution_baselines.py` and `rater_notes.py`.
- Seven scripts read a fixed folder on the author's computer. They now read the folder they are in.
- The figure scripts find `goldset/` one folder above their own.
- `postedit_effort.py` and `term_source.py --compare` now say that `evaluate_protocols.py` and `entity_lexicon.py` are only in `evaluation/`, in every copy.
- `tests/test_structure.py` does not import the 15 scripts that do their work when they are loaded. It expects `FileNotFoundError` from 9 more that read a data file when they are loaded.
- `.gitignore` also covers `der_answers.json` and the PNG files that the figure scripts write.

On 29 October each script ran on a copy of the data, once with the code before these changes and once after. Both gave the same result files and the same printed numbers, apart from the reworded messages and the removed quotation. `evaluation/README.md` describes the check.
