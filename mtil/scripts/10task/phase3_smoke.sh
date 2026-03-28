#!/bin/bash
#SBATCH --job-name=p3_smoke
#SBATCH --time=00:30:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --account=def-fqureshi
#SBATCH --output=/scratch/alexie/logs/%x-%j.out

# Smoke test: 2 tasks, 10 iterations each, to verify inter-task transitions work

set -euo pipefail
mkdir -p /scratch/alexie/logs
echo "[`date`] Host: $(hostname)"

module load cuda/12.2
module load python/3.11.5

ENV_DIR="$SLURM_TMPDIR/env"
if python -c "import tkinter" >/dev/null 2>&1; then
  python -m venv "$ENV_DIR"
  source "$ENV_DIR/bin/activate"
else
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
fi

which python; python -V; which pip
pip install --upgrade pip
pip install --no-index torch torchvision
pip install --no-index tqdm ftfy regex pandas scipy
pip install --no-index wandb
export WANDB_MODE=offline

export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_ROOT="$HOME/projects/def-fqureshi/alexie/ZSCL"
cd "$REPO_ROOT/mtil"

SAVE_PATH="ckpt/10task/phase3_smoke"
mkdir -p "${SAVE_PATH}"

echo "[`date`] Starting Phase 3 smoke test (2 tasks, 10 iters each)"

srun python -m phase3.train_phase3 \
  --train-mode=whole \
  --lr=1e-5 \
  --ls 0.2 \
  --iterations 10 \
  --method ZSCL \
  --image_loss \
  --text_loss \
  --we \
  --avg_freq 5 \
  --l2 1 \
  --ref-dataset ImageNet \
  --ref-sentences conceptual_captions \
  --save "${SAVE_PATH}" \
  --eval-datasets "DTD,EuroSAT" \
  --eval-interval 10 \
  --use_replay \
  --replay_budget 100 \
  --replay_batch_size 4 \
  --replay_loss_weight 0.75 \
  --dataset_order DTD,EuroSAT \
  --lambda_replay_teacher_distill 0.1

echo "[`date`] Smoke test passed. Both tasks completed."
