#!/bin/bash
#SBATCH --job-name=anchor_probe
#SBATCH --time=6:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --account=def-fqureshi_gpu
#SBATCH --output=/scratch/alexie/logs/%x-%A_%a.out
#SBATCH --array=0-3

# Which anchor costs plasticity? Follow-up to expert_probe.sh.
#
# Aircraft is task 0, so the student and the expert both start from zero-shot
# CLIP with no replay. Student Diag 52.9 vs expert 57.9 at the same lr (5e-6),
# and ablation_no_zscl_no_rd (ZSCL off, L2 + WE on) is still 52.5, so ZSCL is
# not the cost. WE cannot be either: the per-task eval runs on the LIVE model,
# and on task 0 weight averaging only builds a separate averaged copy that the
# live model never sees. That leaves the L2 anchor (sum of squared weight
# differences to the task-start weights, x1). This sweeps its strength with
# everything else exactly as in the 11-task runs.
#
#   index -> arm        expected Aircraft if L2 is the cost
#   0  full  (l2 1)     ~52.5  (control; reproduces the 11-task task-0 row)
#   1  l2 0.1           in between
#   2  l2 0.01          close to the expert
#   3  l2 0             ~57-58 (expert minus whatever ZSCL costs)
#
# ImageNet is evaluated too, as the zero-shot price of each setting.
#
#   sbatch scripts/11task/anchor_probe.sh
#   for d in ckpt/11task/anchor_probe/*/; do echo "== $d"; cat $d/task_summary.csv; done
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

ARMS=(l2_1 l2_0.1 l2_0.01 l2_0)
L2S=(1 0.1 0.01 0)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}
L2=${L2S[$SLURM_ARRAY_TASK_ID]}
SAVE_PATH="ckpt/11task/anchor_probe/Aircraft_${ARM}"
mkdir -p "${SAVE_PATH}"

if [ -s "${SAVE_PATH}/task_summary.csv" ]; then
  echo "ERROR: ${SAVE_PATH}/task_summary.csv exists and would be appended to."
  exit 1
fi

echo "[`date`] Anchor probe: Aircraft  l2=${L2}  -> ${SAVE_PATH}"

# Flags identical to the 11-task baseline except --l2, the eval set and the
# single-task order. No replay is read on task 0, so RD is moot.
srun python -m phase3.train_phase3 \
  --train-mode=whole \
  --lr=5e-6 \
  --ls 0.1 \
  --iterations 3000 \
  --method ZSCL \
  --image_loss \
  --text_loss \
  --we \
  --avg_freq 50 \
  --l2 "${L2}" \
  --ref-dataset ImageNet \
  --ref-sentences conceptual_captions \
  --save "${SAVE_PATH}" \
  --eval-datasets "Aircraft,ImageNet" \
  --eval-interval 1000 \
  --use_replay \
  --replay_budget 100 \
  --replay_batch_size 8 \
  --batch-size-eval 16 \
  --dataset_order Aircraft \
  --task_iterations "Aircraft:3000"

echo "[`date`] Done: ${SAVE_PATH}"
