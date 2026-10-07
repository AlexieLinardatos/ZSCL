#!/bin/bash
#SBATCH --job-name=11t_featrep_v0
#SBATCH --time=72:00:00
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --account=def-fqureshi_gpu
#SBATCH --output=/scratch/alexie/logs/%x-%j.out
#SBATCH --signal=USR1@60

# Feature replay V0 (pure): v4 with the buffer storing 512-d embeddings instead
# of images. Every other flag is byte-identical to phase3_no_lora_11t_v4.sh, so
# the A/B against v4 isolates the storage change.
#
# Two things change as a consequence, both forced rather than chosen:
#   1. Replay CE scores stored embeddings against live text embeddings, so its
#      gradient reaches the text tower only. L_zscl and L_RD already supply the
#      image-side gradient (both compute teacher text embeddings under no_grad).
#   2. L_RD has no pixels to distil on, so --rd_source defaults to 'current'
#      (the current task's batch). Rebuttal control C1b measured this swap at
#      66.66 vs 66.72 Transfer, i.e. free. To attribute cleanly, run v4 with
#      --rd_source current as a third arm.
#
# Storage: 11,000 exemplars at 224x224x3 fp32 = 6.6 GB -> 512-d fp16 = 11 MB.
#
# Note on replay_batch_size: kept at 8 to match v4 exactly. Feature replay makes
# a replay sample nearly free (no vision forward), so a much larger batch is
# affordable — but that is a separate experiment, not this one.

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
# RD_SOURCE as in remind_11t.sh: unset keeps the default ('current').
RD_FLAG=""
if [ -n "${RD_SOURCE:-}" ]; then
  RD_FLAG="--rd_source ${RD_SOURCE}"
  TAG="${TAG}_rd${RD_SOURCE}"
fi

# Saturation sweep. BUDGET is the total exemplar count (default 11000);
# setting it also encodes only each task's share, since encoding the whole
# budget at 100k+ exhausts host memory. RBS is the replay batch size (default
# 8): every run sees the same ~170k replay samples, so at large budgets each
# exemplar is replayed about once, and RBS=32 tests whether a plateau is a data
# limit or a replay-compute limit. A huge BUDGET (e.g. 2000000) keeps every
# training image. Unset, both reproduce the original run and save path.
#   LAYER=2 PQ_M=128 RD_SOURCE=none BUDGET=44000 sbatch remind_11t.sh
#   RD_SOURCE=none BUDGET=2000000 RBS=32 sbatch featrep_v0_11t.sh
BUDGET_FLAGS="--replay_budget ${BUDGET:-11000} --replay_batch_size ${RBS:-8}"
if [ -n "${BUDGET:-}" ]; then
  BUDGET_FLAGS="${BUDGET_FLAGS} --replay_encode_share"
  TAG="${TAG}_b${BUDGET}"
fi
[ -n "${RBS:-}" ] && TAG="${TAG}_rbs${RBS}"

SAVE_PATH="ckpt/11task/featrep_v0${TAG}"
mkdir -p "${SAVE_PATH}"

if [ -s "${SAVE_PATH}/task_summary.csv" ]; then
  echo "ERROR: ${SAVE_PATH}/task_summary.csv exists and would be appended to."
  exit 1
fi

EVAL_DATASETS="Aircraft,Caltech101,CIFAR100,DTD,EuroSAT,Flowers,Food,MNIST,OxfordPet,StanfordCars,SUN397,ImageNet"

echo "[`date`] Starting feature replay V0 — storage=feature, lambda_RTD=0.3"

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
  --replay_storage feature \
  ${BUDGET_FLAGS} \
  ${RD_FLAG} \
  --drift_probe_size 64 \
  --replay_loss_weight 1.0 \
  --batch-size-eval 16 \
  --dataset_order Aircraft,Caltech101,CIFAR100,DTD,EuroSAT,Flowers,Food,MNIST,OxfordPet,StanfordCars,SUN397 \
  --lambda_replay_teacher_distill 0.3 \
  --task_iterations "Aircraft:3000,Caltech101:1000,CIFAR100:1500,DTD:1500,EuroSAT:1000,Flowers:1500,Food:1500,MNIST:800,OxfordPet:1500,StanfordCars:3000,SUN397:5000"

echo "[`date`] Done. Checkpoints in ${SAVE_PATH}/"
