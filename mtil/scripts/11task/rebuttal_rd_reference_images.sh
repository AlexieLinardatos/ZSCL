#!/bin/bash
#SBATCH --job-name=reb_rd_ref
#SBATCH --time=72:00:00
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --account=def-fqureshi_gpu
#SBATCH --output=/scratch/alexie/logs/%x-%j.out
#SBATCH --signal=USR1@60

# Rebuttal control C3 (Reviewer zvRY, Q2): the reference-image arm of the same
# control.
#
# Identical to phase3_no_lora_11t_v4.sh (the headline ExRD run) except that the
# RD term is evaluated on the ZSCL reference batch (ImageNet images) instead of
# on buffer exemplars.  Same teacher, same caption anchors, same lambda=0.3,
# same number of distilled images.
#
# Note this run is by construction a re-weighting of the ZSCL term: RD on
# reference images with the frozen teacher IS L_ZSCL, so the run measures
# ZSCL at effective weight 1.3 plus replay.  It is included because it is the
# exact single-factor counterpart of C1, but the equivalence should be stated
# in the response rather than hidden — and it is the lowest-priority of the
# three runs if GPU time is tight.
#
# Reading: comparing C3 with ExRD isolates image identity with everything else
# fixed; comparing C3 with the "+ proportional replay" row of Table 3
# (86.44 / 77.57 / 68.00) checks that the extra 0.3 ZSCL weight alone does not
# reproduce RD's Transfer gain.

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

SAVE_PATH="ckpt/11task/rebuttal_rd_reference_images"
mkdir -p "${SAVE_PATH}"

EVAL_DATASETS="Aircraft,Caltech101,CIFAR100,DTD,EuroSAT,Flowers,Food,MNIST,OxfordPet,StanfordCars,SUN397,ImageNet"

echo "[`date`] Starting control C3 — RD on ZSCL reference images, frozen teacher"

# Flags are v4's verbatim; the only addition is --rd_image_source reference.
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
  --rd_image_source reference

echo "[`date`] Done. Checkpoints in ${SAVE_PATH}/"
