#!/bin/bash
#SBATCH --job-name=11t_remind
#SBATCH --time=72:00:00
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --account=def-fqureshi_gpu
#SBATCH --output=/scratch/alexie/logs/%x-%j.out
#SBATCH --signal=USR1@60

# REMIND (Hayes et al., ECCV 2020) adapted to CLIP, as a baseline arm against
# feature replay. Identical to featrep_v0_11t.sh except for what the buffer
# stores and which half of the network is allowed to train.
#
# The mechanism, and why it is a different bet from feature replay:
#   - Stores PQ-compressed activations from the MIDDLE of the vision tower
#     (block LAYER), not the final embedding. Replayed samples still flow
#     through the upper blocks, so the image tower keeps a real gradient path.
#   - Freezes every block BELOW the split after task 0. Stored activations
#     therefore cannot go stale: drift is eliminated by construction rather
#     than corrected afterwards, which is the opposite of --feature_adapt.
#   - Costs plasticity. Half the vision tower stops learning after task 0,
#     which on MTIL (MNIST through SUN397) may hurt more than on a single
#     ImageNet stream. That trade is what this arm measures.
#
# Storage per exemplar at layer 6, m=32:  197 tokens x 32 B = 6.3 KB
#   vs. pixel replay  602 KB   (96x smaller)
#   vs. feature replay  1 KB   (6x larger, but with gradient and no staleness)
#
# Parameterised by environment variable:
#   LAYER=6  PQ_M=32  sbatch remind_11t.sh    # the default configuration
#   LAYER=3  PQ_M=32  sbatch remind_11t.sh    # freeze less, store earlier
#   LAYER=9  PQ_M=32  sbatch remind_11t.sh    # freeze more, compress better
#   LAYER=6  PQ_M=16  sbatch remind_11t.sh    # half the storage, coarser codes
#
# Run src/test_remind.py first: it verifies the split forward pass reproduces
# the unsplit one exactly, which is the failure mode that would otherwise show
# up only as quietly bad accuracy hours in.

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

LAYER="${LAYER:-6}"
PQ_M="${PQ_M:-32}"

SAVE_PATH="ckpt/11task/remind_l${LAYER}_m${PQ_M}"
mkdir -p "${SAVE_PATH}"

EVAL_DATASETS="Aircraft,Caltech101,CIFAR100,DTD,EuroSAT,Flowers,Food,MNIST,OxfordPet,StanfordCars,SUN397,ImageNet"

echo "[`date`] Starting REMIND — split at block ${LAYER}, PQ m=${PQ_M}"

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
  --replay_storage remind \
  --remind_layer "${LAYER}" \
  --remind_pq_m "${PQ_M}" \
  --replay_budget 11000 \
  --replay_batch_size 8 \
  --replay_loss_weight 1.0 \
  --batch-size-eval 16 \
  --dataset_order Aircraft,Caltech101,CIFAR100,DTD,EuroSAT,Flowers,Food,MNIST,OxfordPet,StanfordCars,SUN397 \
  --lambda_replay_teacher_distill 0.3 \
  --drift_probe_size 64 \
  --task_iterations "Aircraft:3000,Caltech101:1000,CIFAR100:1500,DTD:1500,EuroSAT:1000,Flowers:1500,Food:1500,MNIST:800,OxfordPet:1500,StanfordCars:3000,SUN397:5000"

echo "[`date`] Done. Checkpoints in ${SAVE_PATH}/"
