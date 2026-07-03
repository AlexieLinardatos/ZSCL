#!/bin/bash
#SBATCH --job-name=11t_llm_anchor_lam03
#SBATCH --time=72:00:00
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --account=def-fqureshi_gpu
#SBATCH --output=/scratch/alexie/logs/%x-%j.out
#SBATCH --signal=USR1@60

# NS1: LLM-anchored text encoder.
# Same v4 (ExRD) config + a frozen sentence-embedding LLM (intfloat/e5-large-v2,
# see --llm_anchor_model below) as a semantic anchor for CLIP's text encoder:
# CLIP text feats are projected into the LLM space and pulled toward the LLM's
# embeddings via a cosine-drift loss. lambda = 0.3 (same magnitude as lambda_RTD).
# This is the "anchor ADDED on top of ZSCL" config (ZSCL text branch still on);
# the substitution variant lives on branch llm-anchor-no-zscl-text.

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
# transformers must be available offline (compute nodes have no internet).
# If the Alliance wheelhouse lacks it, run prestage_hf_anchor.sh on a login
# node first OR `pip download transformers` to ~/projects/.../wheels.
pip install --no-index transformers
export WANDB_MODE=offline

# HF cache lives in project space (persistent, pre-staged on a login node via
# mtil/scripts/prestage_hf_anchor.sh). Offline mode forbids any network call.
export HF_HOME="$HOME/projects/def-fqureshi/alexie/hf_cache"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# Pre-flight: fail fast with a clear message if the anchor model isn't cached.
ANCHOR_MODEL="intfloat/e5-large-v2"
ANCHOR_DIR="$HF_HOME/hub/models--${ANCHOR_MODEL//\//--}"
if [ ! -d "$ANCHOR_DIR" ]; then
  echo "[FATAL] anchor model not pre-staged at $ANCHOR_DIR"
  echo "        run: bash mtil/scripts/prestage_hf_anchor.sh   (on a LOGIN node)"
  exit 4
fi
echo "[`date`] anchor model cache OK: $ANCHOR_DIR"

export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_ROOT="$HOME/projects/def-fqureshi/alexie/ZSCL"
cd "$REPO_ROOT/mtil"
export PYTHONPATH="$REPO_ROOT/mtil/scripts/4task/phase3:${PYTHONPATH:-}"
mkdir -p logs

SAVE_PATH="ckpt/11task/phase3_llm_anchor_lam03"
mkdir -p "${SAVE_PATH}"

EVAL_DATASETS="Aircraft,Caltech101,CIFAR100,DTD,EuroSAT,Flowers,Food,MNIST,OxfordPet,StanfordCars,SUN397,ImageNet"

echo "[`date`] Starting Phase 3 — LLM anchor (e5-large-v2) lambda=0.3"

# ---------------------------------------------------------------------------
# FLAG LEGEND (inline comments can't go inside the \-continued command below)
#   ExRD baseline block: see multi_teacher_merge_v3.sh for the shared legend
#   (train-mode/lr/ls/method/image+text_loss/we/l2/ref-*/use_replay/RTD/...).
#
#   LLM anchor (NS1, this experiment):
#     --lambda_llm_anchor 0.3        weight on the cosine-drift anchor loss
#                                    (0 disables => pure ExRD)
#     --llm_anchor_model e5-large-v2 frozen HF sentence model = the anchor space
#     --llm_anchor_hidden 1024       hidden width of the CLIP->LLM projection MLP
#                                    (512 -> 1024 -> e5 dim 1024)
#     --llm_anchor_batch_size 64     #reference captions sampled per step for the
#                                    anchor loss (embeddings precomputed+cached)
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
  --replay_loss_weight 1.0 \
  --batch-size-eval 16 \
  --dataset_order Aircraft,Caltech101,CIFAR100,DTD,EuroSAT,Flowers,Food,MNIST,OxfordPet,StanfordCars,SUN397 \
  --lambda_replay_teacher_distill 0.3 \
  --lambda_llm_anchor 0.3 \
  --llm_anchor_model intfloat/e5-large-v2 \
  --llm_anchor_hidden 1024 \
  --llm_anchor_batch_size 64 \
  --task_iterations "Aircraft:3000,Caltech101:1000,CIFAR100:1500,DTD:1500,EuroSAT:1000,Flowers:1500,Food:1500,MNIST:800,OxfordPet:1500,StanfordCars:3000,SUN397:5000"

echo "[`date`] Done. Checkpoints in ${SAVE_PATH}/"
