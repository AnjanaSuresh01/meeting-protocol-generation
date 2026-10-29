# Environments on JupyterHub

All code of this repository runs on the university JupyterHub, in three Python environments.

| Environment | Python | Kernel | Runs |
|---|---|---|---|
| `~/nemoenv` | 3.10 | "NeMo ASR" (registered as `nemo-asr`) | `pipeline/cluster/step1a_canary_transcribe.ipynb`, `pipeline/cluster/step1c_cohere_whisper_transcribe.ipynb`, `recognizers/NB_v10_asr.ipynb`, `recognizers/NB_whisper_long.ipynb`, `recognizers/NB_biasing_full.ipynb` |
| `~/envs/pyannote-diar` | 3.11 | "PyAnnote Diarization" (`pyannote-diar`) | `pipeline/cluster/step1b_pyannote_speakers.ipynb` |
| `~/protokollenv` | 3.10 | none, used in the terminal | `pipeline/protokoll.py`, every `.py` script in `recognizers/` and `evaluation/`, and the tests |

The notebooks need a GPU. An NVIDIA A100 (40 GB) was used. The terminal scripts run on the CPU.

## Files

| File | Content |
|---|---|
| `requirements_cluster_canary.txt` | `pip freeze` of `~/nemoenv` on 15 October 2026: NeMo 3.0.0, torch 2.13.0+cu130, transformers 5.15.0, numpy 1.26.4 |
| `requirements_cluster_pyannote.txt` | `pip freeze` of `~/envs/pyannote-diar`: pyannote.audio 4.0.7, torch 2.14.0, torchcodec 0.16.0 |
| `requirements_cluster_protokoll.txt` | `pip freeze` of `~/protokollenv`: torch 2.6.0 (CPU), transformers 5.14.1, spaCy 3.8.14 with `de_core_news_sm` 3.8.0, numpy 2.2.3 |
| `CLUSTER_SETUP.md` | Part A: how `~/nemoenv` is built. PyTorch for CUDA 13.0 comes first, then `nemo_toolkit[asr]`, then the kernel and the NeMo fix |
| `patch_nemo_py310.py` | one-line fix for NeMo 3.0.0 on Python 3.10 |

The three freeze files record the environments as they ran. They are not install files. They list CUDA builds such as `torch==2.13.0+cu130`, which are not on the default package index. For that reason, `CLUSTER_SETUP.md` installs PyTorch first, from the CUDA index.

## Building the environments

1. Give each environment its own Python, so that a JupyterHub update of the shared Python cannot break it. A venv built on the shared Python stopped finding its packages when that Python changed from 3.10 to 3.12.
   ```bash
   conda create -y -p ~/py310 -c conda-forge --override-channels python=3.10
   ```
2. For `~/nemoenv`, follow `CLUSTER_SETUP.md`. The notebook messages call it "CLUSTER_SETUP.md Part A". Its step 6 runs the NeMo fix:
   ```bash
   ~/nemoenv/bin/python <repository>/environments/patch_nemo_py310.py
   ```
   NeMo 3.0.0 has one line that needs Python 3.11 or newer. Without the fix, Canary fails on every clip longer than 40 s with `SyntaxError: invalid syntax`. The script keeps the original file as `streaming_utils.py.orig_py311`.
3. `~/envs/pyannote-diar` is a conda environment with Python 3.11 and pyannote.audio 4.0.7. pyannote 4 needs numpy 2.0 or newer, so it cannot share `~/nemoenv`, which has numpy 1.26.4. The requirement comes from pyannote-core 6.0.1. The September environment has numpy 1.26.4 all the same (see its freeze file), and notebook 1b ran in it on 15 October. A new build gets numpy 2. pyannote 4 reads audio through torchcodec, which needs FFmpeg. The JupyterHub image has `/usr/bin/ffmpeg` 6.1.1.
   ```bash
   conda create -y -p ~/envs/pyannote-diar -c conda-forge --override-channels python=3.11
   ~/envs/pyannote-diar/bin/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu130
   ~/envs/pyannote-diar/bin/pip install pyannote.audio==4.0.7 ipykernel
   ~/envs/pyannote-diar/bin/python -m ipykernel install --user --name pyannote-diar --display-name "PyAnnote Diarization"
   ```
4. `~/protokollenv`:
   ```bash
   ~/py310/bin/python -m venv ~/protokollenv
   source ~/protokollenv/bin/activate
   pip install --no-cache-dir -U pip
   pip install --no-cache-dir torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
   pip install --no-cache-dir -r <repository>/pipeline/requirements.txt
   python -m spacy download de_core_news_sm
   ```
   For the tests, also run `pip install -r <repository>/requirements-dev.txt`.

On 15 October 2026, step 1 and the fix of step 2 repaired `~/nemoenv`, and step 4 built `~/protokollenv`. `~/envs/pyannote-diar` was an existing environment from September. The full builds of steps 2 and 3 were not repeated from scratch that day.

## Models

The models are downloaded from Hugging Face on first use, into `~/.cache/huggingface`:

- `nvidia/canary-1b-v2`
- `openai/whisper-large-v3`, `openai/whisper-large-v3-turbo`, `primeline/whisper-large-v3-turbo-german`
- `CohereLabs/cohere-transcribe-03-2026`, loaded with `trust_remote_code=True` (gated: Hugging Face token and accepted conditions needed)
- `ibm-granite/granite-speech-4.1-2b`
- `pyannote/speaker-diarization-community-1` (gated: Hugging Face token and accepted conditions needed)
- `microsoft/wavlm-base-plus-sv`

Watch the home quota. On 15 October 2026 the home folder was 97% full, with 43 GB in `~/.cache/huggingface`.
