#!/bin/bash
#SBATCH --job-name=11t_p3_v5
#SBATCH --time=72:00:00
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --account=def-fqureshi_gpu
#SBATCH --output=/scratch/alexie/logs/%x-%j.out
#SBATCH --signal=USR1@60

# Phase 3 v5: three changes over v4, aimed at Last -> 86% with ImageNet flat.
#   1. replay_loss_weight 1.0 -> 1.5
#      v4 forgetting: CIFAR100 -3.93, EuroSAT -3.15, DTD -2.13.
#      Stronger replay CE directly attacks these (no ImageNet impact —
#      replay buffer never contains ImageNet).
#   2. replay_batch_size 8 -> 16
#      Less noisy replay gradient -> more consistent anti-forgetting pressure.
#      40GB MIG slice + gradient checkpointing on both transformers
#      should fit; if OOM, fall back to 12.
#   3. task_iterations bumps on under-trained / plasticity-limited tasks:
#        CIFAR100  1500 -> 2000   (raises peak, buffers against forgetting)
#        DTD       1500 -> 2000   (78.19 peak -> want ~80+ peak)
#        Aircraft  3000 -> 3500   (54.52 final, still plasticity-bound)
#
# Unchanged from v4: lambda_RTD=0.3, L2=1, lr=5e-6, replay_budget=11000.

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

SAVE_PATH="ckpt/11task/phase3_no_lora_v5"
mkdir -p "${SAVE_PATH}"

EVAL_DATASETS="Aircraft,Caltech101,CIFAR100,DTD,EuroSAT,Flowers,Food,MNIST,OxfordPet,StanfordCars,SUN397,ImageNet"

echo "[`date`] Starting Phase 3 v5 — replay_w=1.5, replay_bs=16, longer CIFAR100/DTD/Aircraft"

# FLAG LEGEND — see phase3_no_lora_11t_v4.sh for the shared ExRD block.
# v5 = v4 tuned for Last: --replay_loss_weight 1.5 (stronger replay CE),
# --replay_batch_size 16 (less noisy replay grad), and longer --task_iterations
# on CIFAR100/DTD/Aircraft. RTD=0.3, L2=1, lr=5e-6, budget=11000 unchanged.
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
  --replay_batch_size 16 \
  --replay_loss_weight 1.5 \
  --batch-size-eval 16 \
  --dataset_order Aircraft,Caltech101,CIFAR100,DTD,EuroSAT,Flowers,Food,MNIST,OxfordPet,StanfordCars,SUN397 \
  --lambda_replay_teacher_distill 0.3 \
  --task_iterations "Aircraft:3500,Caltech101:1000,CIFAR100:2000,DTD:2000,EuroSAT:1000,Flowers:1500,Food:1500,MNIST:800,OxfordPet:1500,StanfordCars:3000,SUN397:5000"

echo "[`date`] Done. Checkpoints in ${SAVE_PATH}/"
