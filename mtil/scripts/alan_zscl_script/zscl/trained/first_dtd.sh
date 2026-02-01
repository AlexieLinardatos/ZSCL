#!/bin/bash
#SBATCH --job-name=zscl_Dtd
#SBATCH --time=02:30:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/scratch/alexie/logs/%x-%j.out
#SBATCH --signal=USR1@60

set -euo pipefail
mkdir -p /scratch/alexie/logs

echo "[`date`] Host: $(hostname)"
nvidia-smi

module load cuda/12.6
module load python/3.11.5

# ----------------------------
# Per-job environment (fresh)
# ----------------------------
ENV_DIR="$SLURM_TMPDIR/env"

echo "[`date`] Checking whether module Python has tkinter..."
if python -c "import tkinter" >/dev/null 2>&1; then
  echo "[`date`] tkinter OK on module Python. Creating venv..."
  python -m venv "$ENV_DIR"
  source "$ENV_DIR/bin/activate"
else
  echo "[`date`] tkinter missing on module Python."
  echo "[`date`] Falling back to conda env with tk (reliable on HPC)."

  # Try to load a conda module (names vary)
  if module avail 2>&1 | egrep -qi "miniconda|anaconda"; then
    # Prefer miniconda if present, else anaconda
    if module avail 2>&1 | egrep -qi "miniconda"; then
      module load miniconda3 || true
    else
      module load anaconda3 || true
    fi
  fi

  if ! command -v conda >/dev/null 2>&1; then
    echo "[`date`] ERROR: conda not available, and module Python lacks tkinter."
    echo "Ask your cluster admins for a Python module built with Tk, or use a conda module."
    exit 3
  fi

  # Create a per-job conda env inside SLURM_TMPDIR
  # Using -p avoids name collisions and keeps it job-local.
  CONDA_ENV_DIR="$SLURM_TMPDIR/conda-env"
  conda create -y -p "$CONDA_ENV_DIR" python=3.11 tk pip

  # Activate (conda needs shell hook)
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV_DIR"

  python -c "import tkinter; print('tkinter ok via conda')"
fi

which python
python -V
which pip

# ----------------------------
# Python deps (og style)
# ----------------------------
# If compute nodes have restricted internet, these may fail.
# In that case, build once on a login node, or use wheels/cache.
pip install --upgrade pip

# PyTorch cu126
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Other deps
pip install tqdm ftfy regex wilds pandas
pip install git+https://github.com/modestyachts/ImageNetV2_pytorch

# ----------------------------
# Go to repo (adjust if needed)
# ----------------------------
REPO_ROOT="$HOME/projects/def-fqureshi/alexie/ZSCL"
cd "$REPO_ROOT/mtil"

mkdir -p logs

TARGET_DATASET="DTD"
SAVE_PATH="ckpt/clean/5000_iter/zscl/trained"
MODEL_PATH="${SAVE_PATH}/DTD_trained"
mkdir -p "${SAVE_PATH}" "${MODEL_PATH}"

MODEL_NAME="${TARGET_DATASET}.pth"
CKPT_PATH="${SAVE_PATH}/${MODEL_NAME}"

DATASETS="DTD,MNIST,EuroSAT,Flowers,ObjectNet"

# ----------------------------
# OG-ish load logic
# ----------------------------
LOAD_ARGS=""
START_ITERATION_ARGS=""

if [ -f "${CKPT_PATH}" ]; then
  echo "[`date`] Found existing checkpoint: ${CKPT_PATH}"
  LOAD_ARGS="--load ${CKPT_PATH}"
  # START_ITERATION_ARGS left empty on purpose for "resume-ish" behavior
else
  echo "[`date`] No checkpoint found at ${CKPT_PATH}. Starting fresh."
  # No --load when starting from scratch (safer than undefined PREV_LOAD_PATH)
  START_ITERATION_ARGS="--start-iteration 0"
fi

# ----------------------------
# Stage 1 (og: iterations 0)
# ----------------------------
echo "[`date`] Stage 1: init/eval"
srun python -m src.main \
  --train-mode=whole \
  --train-dataset="${TARGET_DATASET}" \
  --lr=1e-5 \
  --ls 0.2 \
  --iterations 0 \
  --method ZSCL \
  --image_loss \
  --text_loss \
  --we \
  --avg_freq 100 \
  --l2 1 \
  --ref-dataset ImageNet \
  --ref-sentences conceptual_captions \
  --save "${SAVE_PATH}" \
  --eval-datasets "${DATASETS}" \
  --eval-interval 250 \
  --custom-finetune \
  --max-evaluation-size 500

# ----------------------------
# Stage 2 (train 5000)
# ----------------------------
echo "[`date`] Stage 2: train"
srun python -m src.main \
  --train-mode=whole \
  --train-dataset="${TARGET_DATASET}" \
  --lr=1e-5 \
  --ls 0.2 \
  --iterations 5000 \
  --method ZSCL \
  --image_loss \
  --text_loss \
  --we \
  --avg_freq 100 \
  --l2 1 \
  --ref-dataset ImageNet \
  --ref-sentences conceptual_captions \
  --save "${MODEL_PATH}" \
  --eval-datasets "${DATASETS}" \
  --eval-interval 250 \
  --custom-finetune \
  --max-evaluation-size 500 \
  --load "${CKPT_PATH}" \
  --start-iteration 0

echo "[`date`] Done."
