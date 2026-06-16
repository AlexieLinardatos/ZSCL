#!/bin/bash
#SBATCH --job-name=v4dyn_smoke
#SBATCH --time=00:30:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --account=def-fqureshi
#SBATCH --output=/scratch/alexie/logs/%x-%j.out

# Smoke test for the Adaptive ZSCL extension (NS4 / dynamic hyperparameter
# scheduling) on Canada Compute.  3 tasks x 10 iters so the per-task schedules
# actually vary across the sequence:
#   - warmup_cooldown lambda_RTD peaks on the MIDDLE task (Caltech101)
#   - ramp_up zscl_scale goes 1.0 -> 1.25 -> 1.5
#   - lr_scale_by_class differs across class counts (100 / 102 / 100)
# Watch the per-task log line:
#   [Phase3 schedule] Task k/3 '<name>': lr=...  zscl_scale=...  lambda_RTD=...
# and confirm losses_<task>.csv is written for each task.
# This does NOT validate accuracy — only that scheduling + 3 task transitions run.

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

SAVE_PATH="ckpt/11task/phase3_v4dyn_smoke"
rm -rf "${SAVE_PATH}"          # fresh start every smoke run (no resume confusion)
mkdir -p "${SAVE_PATH}"

echo "[`date`] Starting Adaptive ZSCL smoke (3 tasks, 10 iters each, schedules ON)"

srun python -m phase3.train_phase3 \
  --train-mode=whole \
  --lr=5e-6 \
  --ls 0.1 \
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
  --eval-datasets "Aircraft,Caltech101,CIFAR100" \
  --eval-interval 10 \
  --use_replay \
  --replay_budget 150 \
  --replay_batch_size 4 \
  --replay_loss_weight 1.0 \
  --dataset_order Aircraft,Caltech101,CIFAR100 \
  --lambda_replay_teacher_distill 0.3 \
  --zscl_loss_schedule ramp_up \
  --zscl_loss_min 1.0 \
  --zscl_loss_max 1.5 \
  --lr_scale_by_class \
  --lambda_rtd_schedule warmup_cooldown \
  --lambda_rtd_min 0.5 \
  --lambda_rtd_max 1.5

echo "[`date`] Smoke passed. Check the [Phase3 schedule] lines above show lr/zscl_scale/lambda_RTD varying across the 3 tasks."
