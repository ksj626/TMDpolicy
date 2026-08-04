# Fixed environment

The supported runtime is native Linux x86_64, Python 3.12, an NVIDIA GPU with
BF16, PyTorch 2.11.0/torchvision 0.26.0 from the explicit CUDA 12.8 wheel index,
and PyPI `lerobot==0.6.1`. CUDA 12.8 must be compatible with the host driver; if
the driver requires another official PyTorch CUDA wheel, set
`TMD_CUDA_INDEX_URL` and update the two torch pins together before creation.

Create and verify without starting a dataset download or training job:

```bash
cd /home/dmsdmswns/TMDpolicy
bash scripts/setup/create_environment.sh
```

The script performs, in order: Conda Python 3.12 creation; conda-forge ffmpeg;
explicit CUDA torch/torchvision wheels; PyPI
`lerobot[training,pi,smolvla,libero,evaluation]==0.6.1`; editable TMDpolicy; then
imports and CUDA/BF16/API verification. `conda-history.yml` describes the same
history and `constraints.txt` records validated runtime versions.

Authenticate before model work:

```bash
conda activate tmdpolicy
huggingface-cli login
```

The account must have accepted the gated `google/paligemma-3b-pt-224` license;
PI0.5 tokenization/model loading otherwise fails. For headless LIBERO and
repo-local caches:

```bash
export MUJOCO_GL=egl
export HF_HOME=/home/dmsdmswns/TMDpolicy/.cache/huggingface
export HF_LEROBOT_HOME=/home/dmsdmswns/TMDpolicy/.cache/lerobot
```

`MUJOCO_GL=egl` selects NVIDIA EGL rather than a desktop display. `HF_HOME`
holds Hub models/tokenizers; `HF_LEROBOT_HOME` holds LeRobot datasets. Model
weights remain external immutable assets; no site-packages file is modified.
