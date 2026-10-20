#!/bin/bash
#SBATCH --job-name=11t_p3_v4
#SBATCH --time=72:00:00
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --account=def-fqureshi_gpu
#SBATCH --output=/scratch/alexie/logs/%x-%j.out
#SBATCH --signal=USR1@60

# Phase 3 v4: two changes over v3:
#   1. lambda_replay_teacher_distill 0.5 -> 0.3
#      At 0.5 RTD takes 33% of gradient budget, CE only 3%.
#      At 0.3 RTD drops to ~25%, CE gets ~5-6% — RTD still steers but less suppressive.
#      At 0.1 RTD was "not steering" (too weak), so 0.3 is the middle ground.
#   2. Aircraft 2000 -> 3000 iters
#      Aircraft is fine-grained (100 classes), first task trained, hardest to learn.
#      53.59% final acc in v3 — more CE exposure needed.
#
# All other settings identical to v3.

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

TAG=""
# Previous-task teacher (multi-teacher distillation). PREV is the weight of an
# extra distil term on replay exemplars whose teacher is the previous task's
# checkpoint, scored against the exemplars' own class names. PREV_MIX < 1
# averages that teacher with frozen CLIP after the softmax (0.5 = plain
# average). Unset, the run and its save path are unchanged.
#   PREV=0.3 sbatch phase3_no_lora_11t_v4.sh                 # previous model alone
#   PREV=0.3 PREV_MIX=0.5 sbatch phase3_no_lora_11t_v4.sh    # average with frozen CLIP
PREV_FLAGS=""
if [ -n "${PREV:-}" ]; then
  PREV_FLAGS="--lambda_prev_distill ${PREV} --prev_distill_mix ${PREV_MIX:-1.0}"
  TAG="${TAG}_prev${PREV}_mix${PREV_MIX:-1.0}"
fi

SAVE_PATH="ckpt/11task/phase3_no_lora_v4${TAG}"
mkdir -p "${SAVE_PATH}"

if [ -n "${TAG}" ] && [ -s "${SAVE_PATH}/task_summary.csv" ]; then
  echo "ERROR: ${SAVE_PATH}/task_summary.csv exists and would be appended to."
  exit 1
fi

EVAL_DATASETS="Aircraft,Caltech101,CIFAR100,DTD,EuroSAT,Flowers,Food,MNIST,OxfordPet,StanfordCars,SUN397,ImageNet"

echo "[`date`] Starting Phase 3 v4 — lambda_RTD=0.3, Aircraft 3000 iters"

# ---------------------------------------------------------------------------
# FLAG LEGEND — ExRD HEADLINE BASELINE (all extensions compare against this).
# (Inline comments can't go inside the \-continued command below.)
#   --train-mode=whole                fine-tune the whole CLIP model
#   --lr / --ls / --iterations        LR, label smoothing, fallback iter count
#   --method ZSCL --image_loss --text_loss   ZSCL distillation, both branches
#   --we --avg_freq 50                WiSE weight averaging every 50 steps
#   --l2 1                            L2-to-reference weight penalty
#   --ref-dataset ImageNet --ref-sentences conceptual_captions   ZSCL ref data
#   --use_replay --replay_budget 11000 --replay_batch_size 8   episodic buffer
#                                     (proportional-by-class allocation = default)
#   --lambda_replay_teacher_distill 0.3   replay teacher-distill (RTD) weight
#   --dataset_order ...               11-task Order I (standard)
#   --task_iterations ...             per-task step budget
# ---------------------------------------------------------------------------
srun python -m phase3.train_phase3 \
  --train-mode=whole \
  --lr=5e-6 \
  --ls 0.1 \
  --iterations 1500 \
  --method ZSCL \
  --image_loss \
  --text_loss \
  --we \
  --avg_freq 50 \
  --l2 1 \
  --ref-dataset ImageNet \
  --ref-sentences conceptual_captions \
  --save "${SAVE_PATH}" \
  --eval-datasets "${EVAL_DATASETS}" \
  --eval-interval 500 \
  --use_replay \
  --replay_budget 11000 \
  --replay_batch_size 8 \
  ${PREV_FLAGS} \
  --replay_loss_weight 1.0 \
  --batch-size-eval 16 \
  --dataset_order Aircraft,Caltech101,CIFAR100,DTD,EuroSAT,Flowers,Food,MNIST,OxfordPet,StanfordCars,SUN397 \
  --lambda_replay_teacher_distill 0.3 \
  --task_iterations "Aircraft:3000,Caltech101:1000,CIFAR100:1500,DTD:1500,EuroSAT:1000,Flowers:1500,Food:1500,MNIST:800,OxfordPet:1500,StanfordCars:3000,SUN397:5000"

echo "[`date`] Done. Checkpoints in ${SAVE_PATH}/"
