#!/bin/bash
#SBATCH --job-name=reb_rd_prev
#SBATCH --time=72:00:00
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --account=def-fqureshi_gpu
#SBATCH --output=/scratch/alexie/logs/%x-%j.out
#SBATCH --signal=USR1@60

# Rebuttal control C2 (Reviewer zvRY, Q3): must the RD teacher be the frozen
# pre-trained CLIP, or does any teacher give the same gain?
#
# Identical to phase3_no_lora_11t_v4.sh (the headline ExRD run) in every
# respect — same replay images, same 10,599 caption anchors, same lambda=0.3,
# same per-task iteration schedule, same 11k buffer with supervised replay CE —
# except that the RD term's teacher is the checkpoint produced by task t-1
# (the student's own initialisation for the current task, i.e. the classic
# LwF/iCaRL teacher choice) instead of the pre-trained CLIP checkpoint.  Its
# caption embeddings come from that same checkpoint's text encoder, so the
# distillation target is that teacher's own alignment distribution.
#
# The ZSCL branch keeps the frozen pre-trained teacher, so this is a
# single-factor swap of the RD teacher rather than a different method.
#
# Reading: if Transfer holds up, the anchor identity does not matter and any
# self-distillation signal suffices.  If Transfer collapses toward the no-anchor
# ablation (64.38), the *frozen pre-trained* anchor is the load-bearing part.

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
# Both package roots: phase3 (unmodified) and the rebuttal control wrapper.
export PYTHONPATH="$REPO_ROOT/mtil/scripts/4task/phase3:$REPO_ROOT/mtil/scripts/rebuttal:${PYTHONPATH:-}"
mkdir -p logs

SAVE_PATH="ckpt/11task/rebuttal_rd_prev_teacher"
mkdir -p "${SAVE_PATH}"

EVAL_DATASETS="Aircraft,Caltech101,CIFAR100,DTD,EuroSAT,Flowers,Food,MNIST,OxfordPet,StanfordCars,SUN397,ImageNet"

echo "[`date`] Starting control C2 — RD with previous-task checkpoint as teacher"

# Flags are v4's verbatim; the only addition is --rd_teacher prev_task.
srun python -m rd_controls.run_rd_control \
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
  --replay_loss_weight 1.0 \
  --batch-size-eval 16 \
  --dataset_order Aircraft,Caltech101,CIFAR100,DTD,EuroSAT,Flowers,Food,MNIST,OxfordPet,StanfordCars,SUN397 \
  --lambda_replay_teacher_distill 0.3 \
  --task_iterations "Aircraft:3000,Caltech101:1000,CIFAR100:1500,DTD:1500,EuroSAT:1000,Flowers:1500,Food:1500,MNIST:800,OxfordPet:1500,StanfordCars:3000,SUN397:5000" \
  --rd_teacher prev_task

echo "[`date`] Done. Checkpoints in ${SAVE_PATH}/"
