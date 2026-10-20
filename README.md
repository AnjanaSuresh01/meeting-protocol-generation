# Meeting protocol generation

Code of the master's thesis of Anjana Suresh, Hochschule Hof (2026). The system turns a German meeting recording and the protocol of the previous meeting into a draft protocol. Everything runs on the university JupyterHub: the speech recognizers in GPU notebooks, all other steps in the JupyterHub terminal, and the language model through the institute's LISA service.

**LISA model.** All protocols evaluated in the thesis were written by LISA Pro in the version `lisa-pro-03-2026` (Qwen3.5-397B-A17B). LISA has since replaced this version with a new model (see [LISA model](#lisa-model) below).

## Start here: generate the protocols

There are two ways to generate protocols. Both run on the university JupyterHub. Before either, get the code onto JupyterHub ([see below](#getting-the-code-onto-jupyterhub)) and build the environments ([`environments/README.md`](environments/README.md)). The recordings, transcripts and protocols are provided separately.

### Route 1: one meeting, from the recording to the protocol

This is the system described in the thesis. It uses three recognizers: Canary-1B-v2 writes the transcript, and Cohere Transcribe and Whisper large-v3-turbo-German check the names. It needs the 35 code files in `pipeline/`.

| Step | Runs on | File | Writes into `pipeline/outputs/<MEETING>/` |
|---|---|---|---|
| 1a | GPU, kernel "NeMo ASR" | [`pipeline/cluster/step1a_canary_transcribe.ipynb`](pipeline/cluster/step1a_canary_transcribe.ipynb) | the words with their times |
| 1b | GPU, kernel "PyAnnote Diarization" | [`pipeline/cluster/step1b_pyannote_speakers.ipynb`](pipeline/cluster/step1b_pyannote_speakers.ipynb) | the transcript with speaker labels |
| 1c | GPU, kernel "NeMo ASR" | [`pipeline/cluster/step1c_cohere_whisper_transcribe.ipynb`](pipeline/cluster/step1c_cohere_whisper_transcribe.ipynb) | the Cohere and Whisper transcripts |
| 2 | terminal, `~/protokollenv` | [`pipeline/protokoll.py`](pipeline/protokoll.py) | the protocol, a list of names to check and a list of noun-style bullets |

In each notebook, set the first line to the meeting and its recording, for example `MEETING, AUDIO = "april13", "data/Team-Meeting_2026-04-13.mp3"`, and run its cells in order. The recording and last week's protocol go into `pipeline/data/`. Then run step 2 in the terminal:

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

The commands for later meetings, which reuse the voices of earlier ones, the Senate format and all options are in [`pipeline/README.md`](pipeline/README.md).

### Route 2: the 96 protocols of the thesis

The thesis compared 8 team meetings × 6 recognizers × 2 LISA versions. Each recognizer transcribes the eight meetings, and LISA Flash and LISA Pro each write a protocol from every one of the 48 transcripts.

1. **Transcripts.** In `recognizers/`, run steps 1 to 4 of [`recognizers/README.md`](recognizers/README.md). They use eight files:
   - `NB_v10_asr.ipynb` decodes the 1,939 clips with the six recognizers (GPU).
   - `NB_whisper_long.ipynb`, with `asr_chunks.py` and `whisper_empty_segments.json`, decodes the 219 long clips again for the three Whisper models (GPU).
   - `whisper_long_redecode.py`, `segment_loop_gate.py`, `assemble_v9_transcripts.py` and `transcript_gate.py` build and check the transcripts (terminal).

   The result is `recognizers/outputs_v10/<recognizer>/<meeting>/transcript_speakers.txt`.
2. **Inputs.** `generate_protocols.py` reads its inputs from its own folder, `pipeline/`. Copy them there:
   ```bash
   cd <repository>
   cp -r recognizers/outputs_v10 pipeline/
   mkdir -p pipeline/goldset
   cp -r recognizers/goldset/clip_asr_v10 pipeline/goldset/
   ```
   Put the provided files `speaker_identity.json` and `segments_v9_all.json` into `pipeline/goldset/`. Put the previous protocols into `pipeline/` itself, not into `pipeline/data/`. These are the eight files named in `PREVIOUS` in `pipeline/term_source.py` and, for `--carry`, `Meeting-Protocol-FG-MM-AI_2026-03-02.docx`.
3. **Protocols.** In the terminal:
   ```bash
   source ~/protokollenv/bin/activate
   cd <repository>/pipeline
   export PYTHONUTF8=1
   python generate_protocols.py --v10 --fixed
   python protocol_repair.py --apply
   python validate_protocols_v10.py
   ```
   `generate_protocols.py` writes `outputs_v10/<recognizer>/<meeting>/protocol_flash_fixed.md` and `protocol_pro_fixed.md`, 96 files. A second run skips files that already exist. `protocol_repair.py` puts the person sections in speaking order and repairs the layout. `validate_protocols_v10.py` checks the structure of every protocol. Besides these three scripts, the step uses `term_source.py`, `entity_correct.py`, `language_gate.py`, `evaluate_speaker_attribution.py` and `thesis_pipeline/` in `pipeline/`.

   With `--carry` added, LISA also receives last week's protocol, and the files are named `protocol_*_fixed_carry.md`. That run skips April 13, because its previous protocol exists only as a PDF, so it gives 84 protocols.

**LISA versions.** `generate_protocols.py` asks LISA for the versions of the thesis, `lisa-flash-03-2026` and `lisa-pro-03-2026` (`LISA_MODELS` near the top of the file). LISA no longer offers them, so every request fails. To run route 2 today, change the two names to `lisa-flash` and `lisa-pro`. The protocols then come from the new models and differ from those of the thesis.

Route 2 was tested on 20 October 2026 on a copy, with the thesis transcripts and with LISA replaced by the saved thesis answers. It wrote 96 protocols, and 84 with `--carry`. The repair and the structure check passed.

### Everything else

The other files are not needed to generate protocols. `evaluation/` computes the numbers of the thesis. The other notebook in `recognizers/` is the decoder-biasing test, and its other scripts cut the segments and clips again, which is optional. `tests/` and `tools/` check the code. [`CODEBASE_MAP.md`](CODEBASE_MAP.md) describes every file.

## Folders

| Folder | Content |
|---|---|
| [`pipeline/`](pipeline/README.md) | the system: three GPU notebooks for the transcripts (Canary-1B-v2 with pyannote, then Cohere Transcribe and Whisper large-v3-turbo-German), and `protokoll.py`, which writes the protocol |
| [`recognizers/`](recognizers/README.md) | the six recognizers of the thesis on the same 1,939 clips of eight team meetings, and the decoder-biasing run of Canary-1B-v2 |
| [`evaluation/`](evaluation/README.md) | word error rates of the six recognizers on the gold set, the statistical tests, the agreement of two human readers, the word vote of three recognizers, the biasing score, and the naming replay with the final three recognizers |
| [`environments/`](environments/README.md) | the three Python environments on JupyterHub, their exact package versions, and a one-line fix for NeMo on Python 3.10 |
| `tests/` | the test suite: fixed expected values for the pipeline rules and the recognizer helpers, the command-line behavior of the scripts, clean imports, and the identity of the duplicated modules |
| `tools/` | `sync_copies.py`, which copies each canonical module over its duplicates as listed in `copy_groups.json` |

Each folder is self-contained. Its scripts import each other by name and find their data relative to their own location. Some modules are therefore present in more than one folder, as identical copies. `tools/sync_copies.py` copies them again after a change, and `tests/test_copies_in_sync.py` fails if a copy differs.

[`CODEBASE_MAP.md`](CODEBASE_MAP.md) shows how the folders connect, the order of the steps, and every file with what it imports and which files use it.

## Getting the code onto JupyterHub

In a JupyterHub terminal:

```bash
mkdir -p ~/Thesis
cd ~/Thesis
git clone https://github.com/AnjanaSuresh01/meeting-protocol-generation.git
```

The repository is private, so `git clone` asks for a GitHub user name and a personal access token. Uploading a ZIP of the repository through the JupyterHub file browser also works.

## Tests

```bash
source ~/protokollenv/bin/activate
cd ~/Thesis/meeting-protocol-generation
pip install -r requirements-dev.txt
python -m pytest
```

The suite needs the packages of `~/protokollenv` and pytest. It needs no data, no keys and no network. It checks the segmentation and gate rules on synthetic input, the exit code and output of each script run with no arguments or `--help`, that every module imports without a `.env` or corpus, and that the duplicated modules are byte-identical. Two scripts read a goldset file when they are imported. Without the corpus they stop with `FileNotFoundError`, and the test expects exactly that (see `tests/test_structure.py`).

After editing a module that has copies, run `python tools/sync_copies.py`.

## Data and keys

The recordings, transcripts, reference texts and protocols are internal documents of the research group. They are not in this repository and are provided separately. `.gitignore` keeps audio, data folders and the key file out of the repository.

The LISA key, the LISA address and the Hugging Face token go into `pipeline/.env`, created from `pipeline/.env.example`. This file is never committed.

## LISA model

All protocols evaluated in the thesis were written by LISA Pro in the version `lisa-pro-03-2026` (Qwen3.5-397B-A17B). The comparison with LISA Flash used `lisa-flash-03-2026`. LISA no longer offers these versions. Since October 2026, LISA Pro runs on a new model, GLM 5.3 Flash. A new run therefore writes different protocols, and they cannot be compared with the results of the thesis.

`protokoll.py` uses `lisa-pro` by default. The commands in `pipeline/README.md` pass `--lisa lisa-flash`, the model of the JupyterHub tests.
