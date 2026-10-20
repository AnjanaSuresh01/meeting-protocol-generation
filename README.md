# Meeting protocol generation

Code of the master's thesis of Anjana Suresh, Hochschule Hof (2026). The system turns a German meeting recording and the protocol of the previous meeting into a draft protocol. Everything runs on the university JupyterHub: the speech recognizers in GPU notebooks, all other steps in the JupyterHub terminal, and the language model through the institute's LISA service.

**LISA model.** All protocols evaluated in the thesis were written by LISA Pro in the version `lisa-pro-03-2026` (Qwen3.5-397B-A17B). LISA has since replaced this version with a new model (see [LISA model](#lisa-model) below).

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

## Where to start

- **A protocol for a new recording:** `environments/`, then `pipeline/`.
- **The recognizer comparison of the thesis:** `environments/`, then `recognizers/`, then `evaluation/`.

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
