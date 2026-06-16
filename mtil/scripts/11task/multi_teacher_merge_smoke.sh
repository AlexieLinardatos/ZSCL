#!/bin/bash
#SBATCH --job-name=11t_mtm_smoke
#SBATCH --time=01:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --account=def-fqureshi_gpu
#SBATCH --output=/scratch/alexie/logs/%x-%j.out

# Smoke test for trajectory-aware multi-teacher distillation (NS3).
#
# Goals:
#   1. Verify δ_t and s_t files are written after each task.
#   2. Verify that at task 3 (the 4th task), the trainer logs:
#        [MultiTeacherMerge] strategy=data_driven α=0.1 τ=1.0
#        num_deltas=3 weights=[w0, w1, w2] norm_ratio=...
#      (i.e. merged teacher built from 3 prior tasks).
#   3. Verify no OOM on the 3g.40gb MIG slice.
#   4. Verify identical baseline behavior when --use_multi_teacher_merge is
#      removed (run this script twice — once with the flag, once without —
#      and confirm both finish without errors).
#
# Picks 4 tasks with semantically distinct domains so the cosine-similarity
# routing in data-driven weights has something to discriminate on:
#   DTD (textures), EuroSAT (satellite), MNIST (digits), Flowers (botany).
#
# 250 iters per task to safely clear the LR scheduler warmup (a pre-existing
# bug in src/utils.py:28 trips ZeroDivisionError when steps == warmup_length).

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

SAVE_PATH="ckpt/11task/phase3_multi_teacher_merge_smoke"
mkdir -p "${SAVE_PATH}"

EVAL_DATASETS="DTD,EuroSAT,MNIST,Flowers"

echo "[`date`] Starting NS3 smoke — 4 tasks × 250 iters, data-driven merge"

srun python -m phase3.train_phase3 \
  --train-mode=whole \
  --lr=5e-6 \
  --ls 0.1 \
  --iterations 250 \
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
  --eval-interval 250 \
  --use_replay \
  --replay_budget 400 \
  --replay_batch_size 4 \
  --replay_loss_weight 1.0 \
  --batch-size-eval 16 \
  --dataset_order DTD,EuroSAT,MNIST,Flowers \
  --lambda_replay_teacher_distill 0.3 \
  --use_multi_teacher_merge \
  --merge_strategy data_driven \
  --merge_alpha 0.1 \
  --merge_softmax_temp 1.0 \
  --merge_signature_batches 5 \
  --merge_dtype fp16

echo "[`date`] Smoke complete."
echo "[`date`] Inspect ${SAVE_PATH}/deltas/ and ${SAVE_PATH}/signatures/:"
ls -lh "${SAVE_PATH}/deltas/" "${SAVE_PATH}/signatures/" || true
