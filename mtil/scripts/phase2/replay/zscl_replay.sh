#!/bin/bash
#SBATCH --job-name=p2_replay
#SBATCH --time=10:00:00
#SBATCH --mem=48GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/scratch/alexie/logs/%x-%j.out
#SBATCH --signal=USR1@60

# NOTE: This is ONE job that trains all 4 tasks back-to-back.
# The replay buffer lives in Python memory across tasks, so it cannot
# be split into separate jobs without extra serialization.
# Time budget: ~2h per task x 4 tasks + overhead = 10h requested.

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

SAVE_PATH="ckpt/phase2/replay"
mkdir -p "${SAVE_PATH}"

DATASETS="DTD,MNIST,EuroSAT,Flowers,ImageNet"
LORA_ARGS="--use_lora --lora_r 8 --lora_alpha 16 --lora_dropout 0.1"

# ----------------------------
# Run all 4 tasks with replay
#
# Key differences from baseline scripts:
#   --use_replay              activates the replay outer loop
#   --dataset_order           defines the task sequence (handled by Python)
#   --replay_budget 500       total exemplar budget across all tasks
#   --replay_batch_size 32    replay samples mixed in each training step
#   --replay_loss_weight 1.0  weight of replay CE loss
#
#   NO --custom-finetune      (routing is automatic when --use_replay is set)
#   NO --train-dataset        (outer loop sets this per task from --dataset_order)
#   NO --load                 (outer loop chains checkpoints automatically)
# ----------------------------
echo "[`date`] Starting ZSCL + Replay: DTD -> MNIST -> EuroSAT -> Flowers"

srun python -m src.main \
  --train-mode=whole \
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
  --save "${SAVE_PATH}" \
  --eval-datasets "${DATASETS}" \
  --eval-interval 250 \
  --max-evaluation-size 500 \
  $LORA_ARGS \
  --use_replay \
  --replay_budget 500 \
  --replay_batch_size 32 \
  --replay_loss_weight 1.0 \
  --dataset_order DTD,MNIST,EuroSAT,Flowers

echo "[`date`] Done."
echo "[`date`] Checkpoints: ${SAVE_PATH}/{DTD,MNIST,EuroSAT,Flowers}.pth"
echo "[`date`] Eval CSVs:   ${SAVE_PATH}/metrics_{DTD,MNIST,EuroSAT,Flowers}.csv"
