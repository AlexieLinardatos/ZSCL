#!/bin/bash
#SBATCH --job-name=p2_base_dtd
#SBATCH --time=01:30:00
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
  echo "[`date`] tkinter OK. Creating venv..."
  python -m venv "$ENV_DIR"
  source "$ENV_DIR/bin/activate"
else
  echo "[`date`] tkinter missing. Falling back to conda..."
  if module avail 2>&1 | egrep -qi "miniconda|anaconda"; then
    if module avail 2>&1 | egrep -qi "miniconda"; then
      module load miniconda3 || true
    else
      module load anaconda3 || true
    fi
  fi
  if ! command -v conda >/dev/null 2>&1; then
    echo "[`date`] ERROR: conda not available and tkinter missing."
    exit 3
  fi
  CONDA_ENV_DIR="$SLURM_TMPDIR/conda-env"
  conda create -y -p "$CONDA_ENV_DIR" python=3.11 tk pip
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV_DIR"
  python -c "import tkinter; print('tkinter ok via conda')"
fi

which python; python -V; which pip

# ----------------------------
# Python deps
# ----------------------------
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install tqdm ftfy regex wilds pandas
pip install git+https://github.com/modestyachts/ImageNetV2_pytorch

# ----------------------------
# Paths
# ----------------------------
REPO_ROOT="$HOME/projects/def-fqureshi/alexie/ZSCL"
cd "$REPO_ROOT/mtil"
mkdir -p logs

TARGET_DATASET="DTD"
BASE_SAVE="ckpt/phase2.1/baseline"            # Stage 1 zero-shot checkpoint goes here
MODEL_PATH="${BASE_SAVE}/DTD_trained"         # Stage 2 trained checkpoint goes here
mkdir -p "${BASE_SAVE}" "${MODEL_PATH}"

CKPT_PATH="${BASE_SAVE}/${TARGET_DATASET}.pth"   # written by Stage 1, read by Stage 2
DATASETS="DTD,MNIST,EuroSAT,Flowers,ImageNet"

LORA_ARGS="--use_lora --lora_r 8 --lora_alpha 16 --lora_dropout 0.1"

# ----------------------------
# Stage 1: zero-shot eval
# (runs 0 iterations, just evaluates pretrained CLIP and saves the checkpoint)
# ----------------------------
echo "[`date`] Stage 1: zero-shot eval"
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
  --avg_freq 50 \
  --l2 1 \
  --ref-dataset ImageNet \
  --ref-sentences conceptual_captions \
  --save "${BASE_SAVE}" \
  --eval-datasets "${DATASETS}" \
  --eval-interval 250 \
  --custom-finetune \
  --max-evaluation-size 500 \
  $LORA_ARGS

# ----------------------------
# Stage 2: train DTD (task 1 of 4)
# ----------------------------
echo "[`date`] Stage 2: train DTD"
srun python -m src.main \
  --train-mode=whole \
  --train-dataset="${TARGET_DATASET}" \
  --lr=1e-5 \
  --ls 0.2 \
  --iterations 2000 \
  --method ZSCL \
  --image_loss \
  --text_loss \
  --we \
  --avg_freq 50 \
  --l2 1 \
  --ref-dataset ImageNet \
  --ref-sentences conceptual_captions \
  --save "${MODEL_PATH}" \
  --eval-datasets "${DATASETS}" \
  --eval-interval 250 \
  --custom-finetune \
  --max-evaluation-size 500 \
  $LORA_ARGS \
  --load "${CKPT_PATH}" \
  --start-iteration 0

echo "[`date`] Done. Checkpoint: ${MODEL_PATH}/DTD.pth"
