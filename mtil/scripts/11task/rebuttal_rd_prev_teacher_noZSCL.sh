#!/bin/bash
#SBATCH --job-name=reb_prev_nz
#SBATCH --time=72:00:00
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --account=def-fqureshi_gpu
#SBATCH --output=/scratch/alexie/logs/%x-%j.out
#SBATCH --signal=USR1@60

# Rebuttal control C2b (Reviewer zvRY, Q3), HIGH-CONTRAST variant: must the RD
# teacher be the frozen pre-trained CLIP, measured where RD is the only anchor?
#
# C2 (rebuttal_rd_prev_teacher.sh) keeps the frozen teacher on the ZSCL branch,
# which props Transfer up regardless of what the RD teacher does, so it can only
# ever move ~0.4 points.  This variant removes the ZSCL reference stream
# (--no_existing_distill) so the previous-task checkpoint is the ONLY teacher in
# the run.  Iteration-matched reference points:
#
#   ablation_no_zscl_no_rd    (no anchor at all)              Transfer 64.38
#   ablation_no_zscl_with_rd  (RD, frozen pre-trained teacher) Transfer 66.72
#
# This is the decisive form of the reviewer's question.  Land near 64.38 and a
# rolling self-teacher supplies no zero-shot preservation, i.e. the frozen
# pre-trained anchor is the load-bearing component, exactly as the paper claims.
# Land near 66.72 and any teacher suffices, which would refute that claim -- in
# which case report it; it is a stronger paper for having tested it.

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

SAVE_PATH="/scratch/alexie/ckpt/11task/rebuttal_rd_prev_teacher_noZSCL"
mkdir -p "${SAVE_PATH}"

EVAL_DATASETS="Aircraft,Caltech101,CIFAR100,DTD,EuroSAT,Flowers,Food,MNIST,OxfordPet,StanfordCars,SUN397,ImageNet"

echo "[`date`] Starting control C2b — RD with previous-task teacher, ZSCL branch OFF"

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
  --no_existing_distill \
  --rd_teacher prev_task

echo "[`date`] Done. Checkpoints in ${SAVE_PATH}/"
