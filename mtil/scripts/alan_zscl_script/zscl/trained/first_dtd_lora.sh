#!/bin/bash
#SBATCH --job-name=zscl_Dtd_lora
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

  if module avail 2>&1 | egrep -qi "miniconda|anaconda"; then
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

  CONDA_ENV_DIR="$SLURM_TMPDIR/conda-env"
  conda create -y -p "$CONDA_ENV_DIR" python=3.11 tk pip
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
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install tqdm ftfy regex wilds pandas
pip install git+https://github.com/modestyachts/ImageNetV2_pytorch

# ----------------------------
# Go to repo (adjust if needed)
# ----------------------------
REPO_ROOT="$HOME/projects/def-fqureshi/alexie/ZSCL"
cd "$REPO_ROOT/mtil"

mkdir -p logs

TARGET_DATASET="DTD"
SAVE_PATH="ckpt/clean/5000_iter/zscl/lora/trained"
MODEL_PATH="${SAVE_PATH}/DTD_trained"
mkdir -p "${SAVE_PATH}" "${MODEL_PATH}"

MODEL_NAME="${TARGET_DATASET}.pth"
CKPT_PATH="${SAVE_PATH}/${MODEL_NAME}"

DATASETS="DTD,MNIST,EuroSAT,Flowers"

# ----------------------------
# OGD options (LoRA-only OGD)
# ----------------------------
# For first task, keep OGD_MEMORY_IN empty so projection is a no-op
# (there is no previous-task subspace yet). Memory will be saved at task end.
USE_OGD=1
OGD_MEMORY_IN=""
OGD_MEMORY_OUT="${MODEL_PATH}/ogd_memory.pth"
OGD_ARGS=""
if [ "${USE_OGD}" -eq 1 ]; then
  OGD_ARGS="\
  --ogd-enable \
  --ogd-params-scope lora \
  --ogd-memory-budget-per-task 32 \
  --ogd-sample-batches 8 \
  --ogd-basis-method qr \
  --ogd-projection-mode basis \
  --ogd-log-interval 200 \
  --ogd-save-path ${OGD_MEMORY_OUT}"
  if [ -n "${OGD_MEMORY_IN}" ]; then
    OGD_ARGS="${OGD_ARGS} --ogd-memory-path ${OGD_MEMORY_IN}"
  fi
fi

# ----------------------------
# Stage 1 (init/eval)
# ----------------------------
echo "[`date`] Stage 1: init/eval (LoRA)"
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
  --max-evaluation-size 500 \
  --use_lora \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.1

# ----------------------------
# Stage 2 (train 5000)
# ----------------------------
echo "[`date`] Stage 2: train (LoRA)"
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
  --use_lora \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.1 \
  ${OGD_ARGS} \
  --load "${CKPT_PATH}" \
  --start-iteration 0

echo "[`date`] Done."
