#!/bin/bash
#SBATCH --job-name=reb_r0
#SBATCH --time=72:00:00
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --account=def-fqureshi_gpu
#SBATCH --output=/scratch/alexie/logs/%x-%j.out
#SBATCH --signal=USR1@60

# Rebuttal baseline C0 (Reviewer zvRY, Q1): iteration-matched replay-only run.
#
# The existing lambda=0 ablation (ckpt/11task/ablation_prop_replay, 86.44 Last /
# 68.00 Transfer) was trained with a UNIFORM 1500 iters/task, whereas the
# headline ExRD run (phase3_no_lora_11t_v4.sh) uses the per-task schedule
# (Aircraft 3000, StanfordCars 3000, SUN397 5000, MNIST 800, ...).  A per-task
# breakdown between those two runs therefore mixes the RD effect with a
# per-task iteration-budget effect — visible as the two largest apparent RD
# gains landing exactly on StanfordCars (3000 vs 1500 iters) and SUN397
# (5000 vs 1500).
#
# This run is v4 with lambda=0 and nothing else changed, so ExRD minus this run
# is a clean single-factor per-task measurement of what RD contributes.  It is
# the reference the per-task analysis script expects by default, and it also
# supplies an iteration-matched lambda=0 row for the sensitivity table.
#
# It uses no rd_controls machinery — plain phase3 with --no_replay_teacher_distill.

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

SAVE_PATH="ckpt/11task/rebuttal_replay_only_matched"
mkdir -p "${SAVE_PATH}"

EVAL_DATASETS="Aircraft,Caltech101,CIFAR100,DTD,EuroSAT,Flowers,Food,MNIST,OxfordPet,StanfordCars,SUN397,ImageNet"

echo "[`date`] Starting matched replay-only baseline (lambda=0, v4 iteration schedule)"

# Flags are v4's verbatim; the only change is --no_replay_teacher_distill (lambda=0).
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
  --replay_loss_weight 1.0 \
  --batch-size-eval 16 \
  --dataset_order Aircraft,Caltech101,CIFAR100,DTD,EuroSAT,Flowers,Food,MNIST,OxfordPet,StanfordCars,SUN397 \
  --lambda_replay_teacher_distill 0.3 \
  --task_iterations "Aircraft:3000,Caltech101:1000,CIFAR100:1500,DTD:1500,EuroSAT:1000,Flowers:1500,Food:1500,MNIST:800,OxfordPet:1500,StanfordCars:3000,SUN397:5000" \
  --no_replay_teacher_distill

echo "[`date`] Done. Checkpoints in ${SAVE_PATH}/"
