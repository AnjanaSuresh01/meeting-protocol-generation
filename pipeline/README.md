# Meeting protocol pipeline: code

Master's thesis of Anjana Suresh, Hochschule Hof. State of the code: 19 October 2026.

This folder holds the code of the current pipeline and nothing else. The pipeline turns a meeting recording and the previous protocol into a draft protocol. The code is the code of the evaluated runs. On 19 October its comments were shortened, and a few error messages and error paths were fixed. The outputs of the delivered runs did not change (see `TESTED.md`). Earlier versions and experiments are not included. The other folders of the repository hold the environment files (`../environments/`), the code that runs the six speech recognizers (`../recognizers/`) and their evaluation (`../evaluation/`).

Everything runs on the university JupyterHub:

1. **Step 1, GPU, three notebooks:** Canary-1B-v2 transcribes the recording with word timestamps (notebook 1a). pyannote community-1 finds the speakers, and each word goes to the speaker active at its midpoint (notebook 1b). Cohere Transcribe and Whisper large-v3-turbo-German transcribe the same recording on the same speaker turns (notebook 1c).
2. **Step 2, JupyterHub terminal:** `protokoll.py` repairs the transcript, names the speakers, has LISA write this week's items and merges them into last week's protocol.

Notebooks 1a and 1b and step 2 ran this way on 15 October 2026 for the meetings of 13 and 27 April. Notebook 1c and step 2 with all three recognizers ran on JupyterHub on 19 October for 27 April (see `TESTED.md`).

## Files

| File | Role |
|---|---|
| **Step 1** | |
| `cluster/step1a_canary_transcribe.ipynb` | notebook 1a: Canary transcribes (kernel "NeMo ASR") |
| `cluster/step1b_pyannote_speakers.ipynb` | notebook 1b: pyannote finds the speakers and writes the three files step 2 reads (kernel "PyAnnote Diarization") |
| `cluster/step1c_cohere_whisper_transcribe.ipynb` | notebook 1c: Cohere Transcribe and Whisper large-v3-turbo-German transcribe the speaker turns of the main transcript (kernel "NeMo ASR") |
| `asr_chunks.py` | splits clips longer than 28 s at their quietest points (notebook 1c) |
| `thesis_pipeline/canary_diarize.py` | the functions both notebooks call |
| `thesis_pipeline/env.py` | reads the tokens from `.env` |
| **Step 2** | |
| `protokoll.py` | the entry point |
| `transcript_repair.py` | the four transcript repairs: sentence rule, short-turn verification, turn shaping, recovery of dropped speech |
| `speaker_turns_v11.py`, `verify_short_turns.py`, `gap_recover.py` | the rules and constants those repairs use. `gap_recover.py` transcribes dropped speech again with Whisper large-v3-turbo-German |
| `voice_profiles.py` | names voices from earlier meetings (voice store) |
| `handover_naming.py` | names speakers from handover phrases such as "Dann Johannes, bitte" |
| `term_source.py` | names and terms from the previous protocol |
| `generate_protocols.py` | the prompt rules and the LISA client |
| `language_gate.py` | rejects an answer that is not German |
| `style_exemplars.py` | sentences from the previous protocol as style examples |
| `normalize_structure.py`, `protocol_repair.py`, `validate_protocols_v10.py` | repair and check the heading and bullet structure |
| `entity_correct.py`, `invented_names.py`, `known_leaks.py` | spellings from the previous protocol, names found in neither source, one known foreign-script word |
| `carry_forward.py` | copies last week's protocol and inserts this week's items |
| `carry_tag.py`, `stale_flag.py` | the [ÜBERTRAG] and [PRÜFEN] marks |
| `name_check.py`, `name_align.py`, `style_flag.py` | the list of names to check and the list of noun-phrase bullets |
| `evaluate_speaker_attribution.py`, `attribution_baselines.py`, `postedit_effort.py` | helper functions only (reading protocols, word bags, matching). The rest of these files is not used by the pipeline |
| `thesis_pipeline/protocol_validate.py`, `thesis_pipeline/text_norm.py` | structure check and German text normalization |
| **Setup** | |
| `../environments/requirements_cluster_canary.txt`, `..._pyannote.txt`, `..._protokoll.txt` | the exact package versions of the three environments on 15 October (`pip freeze`) |
| `../environments/patch_nemo_py310.py` | one-line fix that lets NeMo 3.0.0 transcribe long recordings on Python 3.10 |
| `../environments/CLUSTER_SETUP.md` | how the NeMo environment is built |
| `requirements.txt`, `.env.example` | packages for step 2, template for the tokens |

Keep the files in one folder. The modules import each other by name and look for files relative to their own location.

## Setup on JupyterHub

Below, `<repository>` stands for the folder that holds the repository on JupyterHub, for example `~/Thesis/meeting-protocol-generation` after the `git clone` of the main `README.md`. This folder is `<repository>/pipeline`.

### Three environments

| Environment | Python | Used for | Main packages on 15 October |
|---|---|---|---|
| `~/nemoenv` (kernel "NeMo ASR") | 3.10 | notebooks 1a and 1c | NeMo 3.0.0, torch 2.13.0+cu130, numpy 1.26.4 |
| `~/envs/pyannote-diar` (kernel "PyAnnote Diarization") | 3.11 | notebook 1b | pyannote.audio 4.0.7, torch 2.14.0, torchcodec 0.16.0 |
| `~/protokollenv` | 3.10 | step 2 | torch 2.6.0 (CPU), transformers 5.14.1, spaCy 3.8.14 |

Step 1 needs two environments. pip refused to add pyannote 4 to the NeMo environment, because pyannote 4 asks for numpy 2.0 or newer and the NeMo environment has numpy 1.26.4. The full package lists are in the three `../environments/requirements_cluster_*.txt` files. They record the environments as they ran and are not meant as install files.

**Build each environment on its own Python.** The NeMo environment was first built on JupyterHub's shared Python. A JupyterHub update later changed that Python from 3.10 to 3.12, and the environment stopped finding its packages. It was repaired by giving it its own Python 3.10 from conda. For a new build, create that Python first:

```bash
conda create -y -p ~/py310 -c conda-forge --override-channels python=3.10
```

- **NeMo environment:** follow `../environments/CLUSTER_SETUP.md`. It builds `~/nemoenv` on `~/py310`, installs PyTorch for CUDA 13.0 first, then `nemo_toolkit[asr]`, then registers the kernel. Afterwards, run the NeMo fix once: `~/nemoenv/bin/python <repository>/environments/patch_nemo_py310.py`. Without it, NeMo 3.0.0 fails on any recording longer than 40 s with `SyntaxError: invalid syntax`, because one line needs Python 3.11.
- **pyannote environment:** a conda environment with Python 3.11 and pyannote.audio 4.0.7, registered as a Jupyter kernel. The one used on 15 October was built in September. For a new one:
  ```bash
  conda create -y -p ~/envs/pyannote-diar -c conda-forge --override-channels python=3.11
  ~/envs/pyannote-diar/bin/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu130
  ~/envs/pyannote-diar/bin/pip install pyannote.audio==4.0.7 ipykernel
  ~/envs/pyannote-diar/bin/python -m ipykernel install --user --name pyannote-diar --display-name "PyAnnote Diarization"
  ```
  pyannote 4 reads audio through torchcodec, which needs FFmpeg. The JupyterHub image has it (`/usr/bin/ffmpeg`, version 6.1.1).
- **Step-2 environment** (built this way on 15 October):
  ```bash
  ~/py310/bin/python -m venv ~/protokollenv
  source ~/protokollenv/bin/activate
  pip install --no-cache-dir -U pip
  pip install --no-cache-dir torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
  pip install --no-cache-dir -r <repository>/pipeline/requirements.txt
  python -m spacy download de_core_news_sm
  ```

The NeMo and step-2 environments were repaired and built on 15 October as described. The two recipes for a *new* NeMo or pyannote environment were not run from scratch that day.

### Folder, data and tokens

The notebooks do not need a fixed folder. Each one changes to `<repository>/pipeline` itself, because JupyterLab starts the kernel in the notebook's folder (`<repository>/pipeline/cluster`). Put the recordings and protocols into `<repository>/pipeline/data/`.

Copy `.env.example` to `.env` in this folder and enter `LISA_API_KEY`, `LISA_BASE_URL` and `HF_TOKEN`. The JupyterHub file browser hides files whose names start with a dot. Upload the file as `env.txt` and rename it in the terminal: `mv env.txt .env`. Two models are gated on Hugging Face: pyannote community-1 (notebook 1b) and Cohere Transcribe (notebook 1c). Accept their conditions once on their model pages, with the account the token belongs to. Notebooks 1b and 1c read the token from `.env`.

Step 2 downloads two models on first use: WavLM (about 0.8 GB) and Whisper large-v3-turbo-German (about 1.6 GB). Models go to `~/.cache/huggingface`. Watch the home quota: on 15 October the home folder was 97% full.

## Step 1: transcripts (three notebooks)

1. Open `cluster/step1a_canary_transcribe.ipynb` with the kernel **NeMo ASR**. Set the first line, for example `MEETING, AUDIO = "april27", "data/Team-Meeting_2026-04-27.flac"`, and run the cell. It prints the number of words.
2. Open `cluster/step1b_pyannote_speakers.ipynb` with the kernel **PyAnnote Diarization**. Set the same first line and run the cell. It prints words, turns and speakers.
3. Open `cluster/step1c_cohere_whisper_transcribe.ipynb` with the kernel **NeMo ASR**. Set the same first line and run the four cells in order. Cell 1 builds the speaker turns of the main transcript from the files of notebook 1b, with the same functions as `transcript_repair.py`, and adds the stretches without Canary words. Cell 2 runs Whisper large-v3-turbo-German, cell 3 Cohere Transcribe (with the three transformers-5 fixes of `../recognizers/NB_v10_asr.ipynb`). Cell 4 prints the two options for step 2.

`outputs/<MEETING>/` then holds `transcript_speakers.txt`, `transcript_canary_ts.json` and `diarization_hyp.rttm` (notebooks 1a and 1b), and `transcript_speakers_cohere.txt`, `transcript_speakers_whisper_turbo_de.txt` and `step1c_segments.json` (notebook 1c). The two transcripts of notebook 1c carry the same `SPEAKER_NN` labels as the main transcript, which the handover vote needs. Together, notebooks 1a and 1b call the same functions in the same order as `thesis_pipeline.canary_diarize.process_meeting()`. The `main()` of `canary_diarize.py` reads a `config.json` that is not part of the repository and needs NeMo and pyannote in one environment, so it is not used.

## Step 2: protocol (JupyterHub terminal)

```bash
source ~/protokollenv/bin/activate
cd <repository>/pipeline
export PYTHONUTF8=1
export LISA_RETRIES=5
```

**Team meetings in date order.** The first meeting of a series has no voice store yet:

```bash
python protokoll.py data/Team-Meeting_2026-04-13.mp3 \
  --previous data/Meeting_2026-04-08.md \
  --transcript outputs/april13/transcript_speakers.txt \
  --also-transcript outputs/april13/transcript_speakers_cohere.txt \
  --also-transcript outputs/april13/transcript_speakers_whisper_turbo_de.txt \
  --history data/Meeting_2026-03-02.md data/Meeting_2026-03-09.md --history-vocab \
  --date 2026-04-13 --lisa lisa-flash \
  --out outputs/april13/protocol_flash.md
```

Every later meeting enrolls the voices of the meeting before it:

```bash
python protokoll.py data/Team-Meeting_2026-04-27.flac \
  --previous data/Meeting_2026-04-20.md \
  --older data/Meeting_2026-04-13.md \
  --transcript outputs/april27/transcript_speakers.txt \
  --also-transcript outputs/april27/transcript_speakers_cohere.txt \
  --also-transcript outputs/april27/transcript_speakers_whisper_turbo_de.txt \
  --history data/Meeting_2026-03-02.md data/Meeting_2026-03-09.md data/Meeting_2026-04-08.md data/Meeting_2026-04-13.md --history-vocab \
  --voices outputs/voices.json \
  --previous-audio data/Team-Meeting_2026-04-13.mp3 \
  --previous-transcript outputs/april13/protocol_flash.transcript.txt \
  --previous-minutes data/Meeting_2026-04-13.md \
  --date 2026-04-27 --lisa lisa-flash \
  --out outputs/april27/protocol_flash_voices.md
```

**Senate meeting:**

```bash
python protokoll.py data/senat.wav --structure senat \
  --previous data/einladung.md \
  --transcript outputs/senat/transcript_speakers.txt \
  --also-transcript outputs/senat/transcript_speakers_cohere.txt \
  --also-transcript outputs/senat/transcript_speakers_whisper_turbo_de.txt \
  --lisa lisa-flash --out outputs/senat/protokoll.md
```

The invitation can also be a `.docx` or `.pdf` file.

**The previous protocol of a team meeting** must be written like the human minutes: the person as a bold top-level bullet (`*   **Name**`), the topics and details as nested `*` bullets, in Markdown or Word. Carry-over reads no other layout. A protocol in this program's own output layout (`## Person`, `### Topic`) or a Wiki PDF export gives no items, and `protokoll.py` then prints a warning and carries nothing forward.

| Option | Effect |
|---|---|
| `--structure senat` | Senate structure (TOP n), no carry-over |
| `--older` | the protocol before `--previous`. Items that are probably finished get the mark [PRÜFEN] |
| `--voices` with `--previous-audio`, `--previous-transcript`, `--previous-minutes` | names voices from the previous meeting. Use one voice store per series, in date order. The minutes must be `.md` or `.docx` |
| `--history`, `--history-vocab` | earlier protocols of the series for the name list and the prompt vocabulary |
| `--speakers map.json` | a known mapping `{"SPEAKER_00": "Name"}`, used as given |
| `--also-transcript` | another recognizer's transcript of the same recording (notebook 1c), repeatable. With two of them, two of the three recognizers must agree on each handover name, and words on which they disagree go to the list of names to check |
| `--no-repair` | skips the four transcript repairs |
| `--no-mark-carried` | no [ÜBERTRAG] and no [PRÜFEN] marks |
| `--lisa` | LISA model |

Next to the protocol, the program writes `.namen.md` (names to check), `.stil.md` (noun-phrase bullets), `.neu.md` (this week's items before the merge), `.transcript.txt` (the repaired transcript) and `.repair.json` (what the repairs changed). The exit code is 2 when the structure check fails. The protocol is written in that case too. The exit code is 1 when LISA does not answer; the program then prints the error and writes no protocol.

## Notes

- **LISA model.** All protocols evaluated in the thesis were written by LISA Pro in the version `lisa-pro-03-2026` (Qwen3.5-397B-A17B). The comparison with LISA Flash used `lisa-flash-03-2026`. LISA no longer offers these versions. Since October 2026, LISA Pro runs on a new model, GLM 5.3 Flash. A new run therefore writes different protocols, and they cannot be compared with the results of the thesis. `protokoll.py` uses `lisa-pro` by default. The commands above pass `--lisa lisa-flash`, the model of the JupyterHub tests.
- **LISA errors.** On 15 October some requests failed with HTTP 500 or 404 while the next one succeeded. `LISA_RETRIES=5` makes the client try up to six times.
- **Recognizer versions.** A new transcript is close to, but not always identical with, the transcripts used in the thesis. On 15 October, 13 April gave the same 6,028 words as in July, and 27 April gave 6,389 words against 6,481.
- **Three recognizers.** In the evaluated runs, the Cohere Transcribe and Whisper large-v3-turbo-German transcripts came from the decode of the 1,939 evaluation clips (`../recognizers/`). Notebook 1c makes them for any new recording. Without `--also-transcript`, `protokoll.py` runs with Canary alone, and the name check compares against earlier protocols only.
- **Experiment options.** `protokoll.py` still contains options from earlier experiments: `--cite`, `--segments`, `--depth`, `--llm-carry` and `--transcribe-here`. They are not part of the current pipeline. `--cite` stops with a message, because it needs `segment_cite.py`, which is not included. `--transcribe-here` diarizes inside step 2, which needs pyannote in the step-2 environment.
- **Recovered speech** is cached in `cache_gap_recovery.json`, so a second run of the same meeting skips the Whisper decoding.

See `TESTED.md` for the test results.
