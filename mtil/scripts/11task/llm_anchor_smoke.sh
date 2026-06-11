#!/bin/bash
#SBATCH --job-name=11t_llm_anchor_smoke
#SBATCH --time=01:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --account=def-fqureshi_gpu
#SBATCH --output=/scratch/alexie/logs/%x-%j.out
#SBATCH --signal=USR1@60

# NS1 smoke test: 100 iters on the first task (Aircraft) to validate the full
# pipeline before committing to a 72h production run.
#
# What this verifies:
#   1. Offline HF cache loads (TRANSFORMERS_OFFLINE=1).
#   2. CC captions resolve from $HOME/scratch/data.
#   3. LLM embeddings get computed and cached to $SAVE_PATH.
#   4. Projection head initializes and joins the optimizer.
#   5. Forward + backward through anchor loss runs without OOM.
#   6. anchor loss visibly decreases in the loss CSV.
#
# Expected wall time: ~10-25 min (50% LLM precompute, 50% training).

set -euo pipefail
mkdir -p /scratch/alexie/logs
echo "[`date`] Host: $(hostname)"
nvidia-smi

module load cuda/12.2
module load python/3.11.5

ENV_DIR="$SLURM_TMPDIR/env"
python -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"

pip install --upgrade pip
pip install --no-index torch torchvision
pip install --no-index tqdm ftfy regex pandas scipy
pip install --no-index wandb
pip install --no-index transformers
export WANDB_MODE=offline

export HF_HOME="$HOME/projects/def-fqureshi/alexie/hf_cache"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

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

SAVE_PATH="ckpt/11task/llm_anchor_smoke"
mkdir -p "${SAVE_PATH}"

# Single task, no chained eval, low iteration count.
EVAL_DATASETS="Aircraft"

echo "[`date`] Smoke: 100 iters on Aircraft only, lambda=0.3"

srun python -m phase3.train_phase3 \
  --train-mode=whole \
  --lr=5e-6 \
  --ls 0.1 \
  --iterations 100 \
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
  --eval-interval 100 \
  --use_replay \
  --replay_budget 1000 \
  --replay_batch_size 8 \
  --replay_loss_weight 1.0 \
  --batch-size-eval 16 \
  --dataset_order Aircraft \
  --lambda_replay_teacher_distill 0.3 \
  --lambda_llm_anchor 0.3 \
  --llm_anchor_model intfloat/e5-large-v2 \
  --llm_anchor_hidden 1024 \
  --llm_anchor_batch_size 64 \
  --task_iterations "Aircraft:100"

echo "[`date`] Smoke done. Inspect:"
echo "  ${SAVE_PATH}/losses_Aircraft.csv  (llm_anchor column should trend down)"
echo "  ${SAVE_PATH}/llm_anchor_cache_*.pt  (LLM embeddings cache)"
echo "  ${SAVE_PATH}/llm_anchor_proj_Aircraft.pth  (saved projection)"
