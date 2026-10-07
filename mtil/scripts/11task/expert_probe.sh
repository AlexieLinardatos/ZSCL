#!/bin/bash
#SBATCH --job-name=expert_probe
#SBATCH --time=8:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --account=def-fqureshi_gpu
#SBATCH --output=/scratch/alexie/logs/%x-%A_%a.out
#SBATCH --array=0-8

# Plasticity probe: go/no-go for expert distillation.
#
# Every 11-task run sits at Diag ~87 (accuracy right after training a task),
# whether ZSCL / RD are on or off, and forgetting is under 1 point. So +1 Last
# has to come from learning each task better. Expert distillation only helps if
# an unconstrained per-task expert beats the student's Diag by a clear margin.
#
# Each array task trains ONE dataset alone from zero-shot CLIP with every anchor
# removed (no ZSCL, no L2, no weight averaging, no replay distillation) and
# everything else identical to remind_11t.sh: same trainer, iterations, batch
# size, label smoothing, eval.
#
#   index -> TASK x LR      5e-6 is the student's lr: isolates the anchors.
#                           1e-5 / 2e-5 add plasticity on top.
#
# Compare the diagonal against the student (REMIND R1):
#   Aircraft 52.90 (go if >= ~56)   DTD 78.40 (>= ~81)   StanfordCars 86.83 (>= ~89)
#
#   sbatch scripts/11task/expert_probe.sh
#   python scripts/compute_metrics.py ckpt/11task/expert_probe/<TASK>_lr<LR>/task_summary.csv
set -euo pipefail
mkdir -p /scratch/alexie/logs
echo "[`date`] Host: $(hostname)"
nvidia-smi

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
export PYTHONPATH="$REPO_ROOT/mtil/scripts/4task/phase3:${PYTHONPATH:-}"
mkdir -p logs

TASKS=(Aircraft DTD StanfordCars)
LRS=(5e-6 1e-5 2e-5)
declare -A ITERS=([Aircraft]=3000 [DTD]=1500 [StanfordCars]=3000)

TASK=${TASKS[$((SLURM_ARRAY_TASK_ID / 3))]}
LR=${LRS[$((SLURM_ARRAY_TASK_ID % 3))]}
SAVE_PATH="ckpt/11task/expert_probe/${TASK}_lr${LR}"
mkdir -p "${SAVE_PATH}"

if [ -s "${SAVE_PATH}/task_summary.csv" ]; then
  echo "ERROR: ${SAVE_PATH}/task_summary.csv exists and would be appended to."
  exit 1
fi

echo "[`date`] Expert probe: ${TASK}  lr=${LR}  iters=${ITERS[$TASK]}  -> ${SAVE_PATH}"

# Phase 3 insists on --use_replay; with one task the buffer is filled once
# after training and never read. --method finetune drops ZSCL; omitting --we
# and --l2 drops weight averaging and the L2 anchor.
srun python -m phase3.train_phase3 \
  --train-mode=whole \
  --method finetune \
  --lr="${LR}" \
  --ls 0.1 \
  --iterations "${ITERS[$TASK]}" \
  --save "${SAVE_PATH}" \
  --eval-datasets "${TASK}" \
  --eval-interval 500 \
  --use_replay \
  --replay_budget 100 \
  --replay_batch_size 8 \
  --no_replay_teacher_distill \
  --batch-size-eval 16 \
  --dataset_order "${TASK}" \
  --task_iterations "${TASK}:${ITERS[$TASK]}"

echo "[`date`] Done: ${SAVE_PATH}"
