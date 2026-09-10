#!/bin/bash
#SBATCH --job-name=featrep_smoke
#SBATCH --time=00:40:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --account=def-fqureshi
#SBATCH --output=/scratch/alexie/logs/%x-%j.out

# Smoke test for feature replay (--replay_storage feature).
#
# Two passes of 3 tasks x 10 iterations: pure feature replay (V0), then the same
# thing with label-propagation drift adaptation (V2). Between them they exercise
# everything the extension touches:
#   task 0 -> boundary encode -> task 1 replay CE on stored features
#          -> adapt stored features -> boundary encode + rebalance
#          -> task 2 replay CE over two tasks
#
# What to check in the log:
#   [Phase3] Replay storage mode: feature
#   [Phase3 config] ... replay_storage=feature  rd_source=current
#   FeatureReplayBuffer(... dim=512, size=0.1 MB)   <- not gigabytes
#   replay_sup=<nonzero>                            <- CE is actually firing
#   [Phase3] Drift: observed_drift_cos=...          <- pass 2 only
#   drift_adaptation.csv written with 2 rows        <- pass 2 only

set -euo pipefail
mkdir -p /scratch/alexie/logs
echo "[`date`] Host: $(hostname)"

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

run_pass () {
  local name="$1"; shift
  local save_path="ckpt/11task/featrep_smoke_${name}"
  rm -rf "${save_path}"
  mkdir -p "${save_path}"

  echo ""
  echo "[`date`] === pass '${name}' (3 tasks, 10 iters each) ==="

  srun python -m phase3.train_phase3 \
    --train-mode=whole \
    --lr=5e-6 \
    --ls 0.1 \
    --iterations 10 \
    --method ZSCL \
    --image_loss \
    --text_loss \
    --we \
    --avg_freq 5 \
    --l2 1 \
    --ref-dataset ImageNet \
    --ref-sentences conceptual_captions \
    --save "${save_path}" \
    --eval-datasets "DTD,EuroSAT" \
    --eval-interval 10 \
    --use_replay \
    --replay_storage feature \
    --replay_budget 200 \
    --replay_batch_size 8 \
    --replay_loss_weight 1.0 \
    --batch-size-eval 16 \
    --dataset_order DTD,EuroSAT,MNIST \
    --lambda_replay_teacher_distill 0.3 \
    "$@"

  echo "[`date`] pass '${name}' completed."
}

# V0: pure feature replay, stored features never touched again.
run_pass v0

# V2: same, plus label propagation with augmented anchors at each boundary.
run_pass lp \
  --feature_adapt lp \
  --feature_adapt_anchors both \
  --feature_adapt_samples 200

echo ""
echo "[`date`] Smoke test passed. Drift log:"
cat ckpt/11task/featrep_smoke_lp/drift_adaptation.csv
