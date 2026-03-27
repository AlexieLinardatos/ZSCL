#!/bin/bash
#SBATCH --job-name=p2_base_flowers
#SBATCH --time=01:15:00
#SBATCH --mem=48GB
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

TARGET_DATASET="Flowers"
SAVE_PATH="ckpt/phase2.1/baseline/DTD_trained/MNIST_trained/EuroSAT_trained/Flowers_trained"
PREV_LOAD_PATH="ckpt/phase2.1/baseline/DTD_trained/MNIST_trained/EuroSAT_trained/EuroSAT.pth"
mkdir -p "${SAVE_PATH}"

MODEL_NAME="${TARGET_DATASET}.pth"
DATASETS="DTD,MNIST,EuroSAT,Flowers,ImageNet"
LORA_ARGS="--use_lora --lora_r 8 --lora_alpha 16 --lora_dropout 0.1"

# Resume if checkpoint exists, otherwise load from previous task
LOAD=""
START_ITERATION=""
if [ -f "${SAVE_PATH}/${MODEL_NAME}" ]; then
  echo "[`date`] Resuming from ${SAVE_PATH}/${MODEL_NAME}"
  LOAD="--load ${SAVE_PATH}/${MODEL_NAME}"
else
  echo "[`date`] Loading from previous task: ${PREV_LOAD_PATH}"
  LOAD="--load ${PREV_LOAD_PATH}"
  START_ITERATION="--start-iteration 0"
fi

echo "[`date`] Training Flowers (task 4 of 4)"
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
  --save "${SAVE_PATH}" \
  --eval-datasets "${DATASETS}" \
  --eval-interval 250 \
  --custom-finetune \
  --max-evaluation-size 500 \
  $LORA_ARGS \
  ${LOAD} \
  ${START_ITERATION}

echo "[`date`] Done. Checkpoint: ${SAVE_PATH}/Flowers.pth"
