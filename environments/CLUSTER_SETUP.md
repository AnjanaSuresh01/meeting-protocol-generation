# Part A: the NeMo environment (`~/nemoenv`)

This environment runs the notebooks with the kernel "NeMo ASR": notebooks 1a and 1c in `pipeline/cluster/` and the three notebooks in `recognizers/`. Building it takes 15 to 30 minutes, mostly downloads. Run the steps in a JupyterHub terminal, not in a notebook.

The order matters: PyTorch first, from the CUDA index, then NeMo. An unpinned NeMo install into a shared environment once replaced torch with a different CUDA build without updating torchaudio, and the environment stopped working.

## 1. Create the environment on its own Python

Skip the first line if `~/py310` already exists.

```bash
conda create -y -p ~/py310 -c conda-forge --override-channels python=3.10
~/py310/bin/python -m venv ~/nemoenv
source ~/nemoenv/bin/activate
pip install -U pip wheel
```

## 2. PyTorch for CUDA 13.0

```bash
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu130
python -c "import torch, torchaudio; print(torch.__version__, torchaudio.__version__, torch.cuda.is_available())"
```

The last value must be `True`. If it is `False`, stop: either no GPU is visible or the CUDA build is wrong.

## 3. NeMo

```bash
pip install -U "nemo_toolkit[asr]"
pip install soundfile librosa
```

## 4. Jupyter kernel

```bash
pip install ipykernel
python -m ipykernel install --user --name nemo-asr --display-name "NeMo ASR"
```

## 5. Check

```bash
python -c "
import torch, torchaudio, nemo
from nemo.collections.asr.models import ASRModel
print('torch     ', torch.__version__)
print('torchaudio', torchaudio.__version__)
print('nemo      ', nemo.__version__)
print('CUDA      ', torch.cuda.is_available(), torch.version.cuda)
print('ASRModel imported OK')
"
```

If this prints without a traceback, the environment is complete.

## 6. NeMo fix for Python 3.10

```bash
~/nemoenv/bin/python <repository>/environments/patch_nemo_py310.py
```

NeMo 3.0.0 has one line that needs Python 3.11. Without the fix, Canary fails on every recording or clip longer than 40 s with `SyntaxError: invalid syntax`.

## 7. Record the versions

```bash
pip freeze > ~/requirements_nemoenv_now.txt
diff ~/requirements_nemoenv_now.txt <repository>/environments/requirements_cluster_canary.txt
```

`requirements_cluster_canary.txt` is the environment of 15 October 2026. A new build can differ in minor versions, because steps 2 and 3 install the newest releases.

## Using the kernel

Open the notebook and choose **Kernel → Change Kernel → "NeMo ASR"** (top right of the notebook). If "NeMo ASR" is not listed, reload the browser tab. To confirm, run `import sys; print(sys.executable)` in a cell. The path must contain `nemoenv`.

## If `accelerate` fails to import

A fresh install can bring `accelerate` 1.14.0, which has a circular import. Test it:

```bash
python -c "import accelerate; print(accelerate.__version__, 'OK')"
```

If it raises `ImportError: cannot import name 'dispatch_model'`, install an older version:

```bash
pip install --no-deps "accelerate<1.14"
```

`--no-deps` keeps pip from changing torch. Then repeat step 5.

## Rules

1. Install packages in the terminal, never from a notebook cell.
2. Never install into JupyterHub's shared base environment.
3. PyTorch first, from the CUDA index, then everything else.
4. After any change, repeat steps 5 and 7.

## Disk space

`~/nemoenv` takes about 6 to 9 GB. If step 2 or 3 fails with a quota or "No space left" error:

```bash
du -sh ~/nemoenv ~/.cache/pip ~/.cache/huggingface
pip cache purge
```

The Hugging Face cache holds the model weights. A model deleted there is downloaded again on its next use.
