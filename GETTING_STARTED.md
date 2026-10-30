# Getting started

This page is for someone who has never seen this project. It says what to get, what to install and which file to run first.

## 1. What this repository does

The code turns the recording of a German meeting into a draft protocol (the minutes). It also holds the code that compared six speech recognizers, and the code that computed the numbers of the master's thesis it belongs to.

| Folder | Use it to | Needs the gold set? |
|---|---|---|
| `pipeline/` | write a protocol for one meeting | no |
| `recognizers/` | transcribe the 1,939 evaluation clips with six speech recognizers | yes |
| `evaluation/` | compute the numbers of Chapter 5 of the thesis | yes |
| `environments/` | install the three Python environments | no |
| `tests/`, `tools/` | check the code | no |

The gold set and the other data are not in this repository. [`DATA.md`](DATA.md) explains why, which files each task needs, and where they go.

## 2. Read the files in this order

1. [`README.md`](README.md): the map of the code, the run order, and which file produced each result
2. this page: what to do, step by step
3. [`DATA.md`](DATA.md): the data
4. [`environments/README.md`](environments/README.md): installation
5. the README of the folder you want to use: [`pipeline/`](pipeline/README.md), [`recognizers/`](recognizers/README.md) or [`evaluation/`](evaluation/README.md)
6. [`CODEBASE_MAP.md`](CODEBASE_MAP.md): every file, what it imports and which files use it, when you need the details

## 3. What you need before you start

| You need | For | How to get it |
|---|---|---|
| access to this repository | the repository is private | the owner adds your GitHub account (Settings, Collaborators) |
| an account on the university JupyterHub with a GPU | everything runs there | the university's computing service |
| a LISA key and the LISA address | LISA is the language model that writes the protocol | the institute that runs LISA |
| a Hugging Face account and an access token | downloading the speech models | huggingface.co, Settings, Access Tokens |
| the data packages | recordings, protocols, gold set | the author of this repository, see [`DATA.md`](DATA.md) |

You can install the code and run the tests without the data.

## 4. Get the code

On JupyterHub, open a terminal (File, New, Terminal) and type:

```bash
mkdir -p ~/Thesis
cd ~/Thesis
git clone https://github.com/AnjanaSuresh01/meeting-protocol-generation.git
```

The repository is private. Git therefore asks for your GitHub user name and, instead of the password, a personal access token (github.com, Settings, Developer settings, Personal access tokens).

Without Git: on the repository page, click the green **Code** button, then **Download ZIP**. Upload the ZIP file with the upload button of the JupyterHub file browser, and unpack it in the terminal with `python -m zipfile -e meeting-protocol-generation-main.zip .`

In all instructions, `<repository>` means the folder you now have: `~/Thesis/meeting-protocol-generation` after `git clone`, or `meeting-protocol-generation-main` after the ZIP.

Do not move files between folders or out of them. The scripts of each folder import each other by name and find their data relative to their own location.

## 5. Install, once

Follow [`environments/README.md`](environments/README.md). It builds three environments. Build them in this order:

| Order | Environment | Used for |
|---|---|---|
| 1 | `~/py310` | a Python 3.10 of its own, the base of `~/protokollenv` and `~/nemoenv` |
| 2 | `~/protokollenv` | `protokoll.py`, all terminal scripts, the tests |
| 3 | `~/nemoenv` (kernel "NeMo ASR") | the GPU notebooks of Canary, Whisper, Cohere and Granite. Run `environments/patch_nemo_py310.py` once afterwards |
| 4 | `~/envs/pyannote-diar` (kernel "PyAnnote Diarization") | notebook 1b, which finds the speakers |

## 6. First check: run the tests

```bash
source ~/protokollenv/bin/activate
cd <repository>
pip install -r requirements-dev.txt
python -m pytest
```

The last line should say `87 passed`. The tests need no data, no keys and no internet. If they pass, every module imports, the copied modules match, and `~/protokollenv` works.

## 7. Enter the keys

1. Copy `pipeline/.env.example` to `pipeline/.env` and enter `LISA_API_KEY`, `LISA_BASE_URL` and `HF_TOKEN`. The JupyterHub file browser hides names that start with a dot. Upload the file as `env.txt` and rename it in the terminal: `mv env.txt .env`.
2. Two models need your consent on Hugging Face. Open their model pages while logged in and accept the conditions: `pyannote/speaker-diarization-community-1` and `CohereLabs/cohere-transcribe-03-2026`.
3. For the notebooks in `recognizers/`, log in once in a terminal: `~/nemoenv/bin/hf auth login`. They do not read `pipeline/.env`.

## 8. Get the data

Ask for the package of the task you want ([`DATA.md`](DATA.md)) and unpack it in `<repository>`:

```bash
cd <repository>
python -m zipfile -e data_pipeline.zip .
```

For your own meeting, no package is needed. Put the recording and last week's protocol into `pipeline/data/`.

## 9. Choose a task and run the files in this order

### Task A: a protocol for one meeting (no gold set)

This is the system of the thesis. Steps 1 to 3 are notebooks on the GPU. Step 4 runs in the terminal.

| Order | Open | Kernel or environment | Writes into `pipeline/outputs/<meeting>/` |
|---|---|---|---|
| 1 | `pipeline/cluster/step1a_canary_transcribe.ipynb` | NeMo ASR | the words with their times |
| 2 | `pipeline/cluster/step1b_pyannote_speakers.ipynb` | PyAnnote Diarization | `transcript_speakers.txt`, `transcript_canary_ts.json`, `diarization_hyp.rttm` |
| 3 | `pipeline/cluster/step1c_cohere_whisper_transcribe.ipynb` | NeMo ASR | `transcript_speakers_cohere.txt`, `transcript_speakers_whisper_turbo_de.txt` |
| 4 | `pipeline/protokoll.py` | terminal, `~/protokollenv` | the protocol, a list of names to check, a list of noun-style bullets |

In each notebook, set the first line to your meeting and its recording, for example `MEETING, AUDIO = "april13", "data/Team-Meeting_2026-04-13.mp3"`. Then run all cells from top to bottom (Run, Run All Cells). The command for step 4 is in [`README.md`](README.md#task-a-a-protocol-for-one-new-meeting). The commands for later meetings and for the Senate format are in [`pipeline/README.md`](pipeline/README.md).

Last week's protocol must be written like the human minutes: each person as a bold top-level bullet, the topics as nested bullets, in Markdown or Word. Any other layout gives a warning, and nothing is carried over from last week.

### Task B: the six speech recognizers (needs `data_recognizers.zip`)

All in `recognizers/`. The details are in [`recognizers/README.md`](recognizers/README.md).

| Order | Run | Where |
|---|---|---|
| 1 | unpack the clips: `python -m zipfile -e clips_v10_gold.zip .` and `python -m zipfile -e clips_v10_rest.zip .` | terminal |
| 2 | `NB_v10_asr.ipynb` | GPU, NeMo ASR |
| 3 | `NB_whisper_long.ipynb` | GPU, NeMo ASR |
| 4 | move the outputs, then `python whisper_long_redecode.py --merge` | terminal |
| 5 | `python segment_loop_gate.py --apply` | terminal |
| 6 | `python assemble_v9_transcripts.py --source all` | terminal |
| 7 | `python transcript_gate.py --v10 --apply` | terminal |
| 8 | `NB_biasing_full.ipynb`, only for the decoder-biasing result | GPU, NeMo ASR |

### Task C: the numbers of the thesis (needs `data_evaluation.zip`)

All in `evaluation/`, in the terminal, on the CPU. Task B is not needed first, because the package holds the recognizer outputs. Run the scripts of parts 1 and 2 of [`evaluation/README.md`](evaluation/README.md) in the order given there, then compare the new results with the stored ones.

### Task D: the 96 protocols of the thesis (needs `data_pipeline.zip`)

In `pipeline/`, in the terminal: `generate_protocols.py --v10 --fixed`, then `protocol_repair.py --apply`, then `validate_protocols_v10.py`. Read "The 96 protocols of the thesis" in [`pipeline/README.md`](pipeline/README.md#the-96-protocols-of-the-thesis) first. The LISA versions of the thesis no longer exist, so the new protocols differ from those of the thesis.

## 10. When something goes wrong

| Message | Cause | What to do |
|---|---|---|
| `ModuleNotFoundError` for a file of this repository | the script was started from another folder, or a file was moved | `cd` into the script's folder and keep the files together |
| `FileNotFoundError` with `goldset/`, `outputs_v10/` or `data/` | the data package is missing or was unpacked in the wrong place | unpack it in `<repository>` ([`DATA.md`](DATA.md)) |
| `FileNotFoundError` when the protocol is written | the folder given with `--out` does not exist | create it with `mkdir -p` |
| `SyntaxError: invalid syntax` from NeMo | the NeMo fix was not run | `~/nemoenv/bin/python <repository>/environments/patch_nemo_py310.py` |
| 401 or 403 from Hugging Face | the token is missing or the model conditions were not accepted | step 7 |
| LISA answers 404 or 500 | the model name no longer exists, or LISA is busy | use `--lisa lisa-flash` or `--lisa lisa-pro` and set `LISA_RETRIES=5` |
| `Disk quota exceeded` | the models fill the home folder (43 GB during the tests) | delete unused models in `~/.cache/huggingface` |
