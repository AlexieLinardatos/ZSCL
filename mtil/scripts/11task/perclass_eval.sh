#!/bin/bash
#SBATCH --job-name=perclass_eval
#SBATCH --time=02:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --account=def-fqureshi_gpu
#SBATCH --output=/scratch/alexie/logs/%x-%j.out

# Runs per-class ImageNet zero-shot eval on two checkpoints:
#   1. Our full method (phase3_no_lora_v4/SUN397.pth)
#   2. ZSCL baseline (ablation_zscl_only/SUN397.pth)
# Outputs two JSON files to ckpt/11task/perclass_results/.
# Copy those JSONs locally and run scripts/analyze_imagenet_groups.py to plot.

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

which python; python -V
pip install --upgrade pip
pip install --no-index torch torchvision
pip install --no-index tqdm ftfy regex pandas scipy
export WANDB_MODE=offline

export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_ROOT="$HOME/projects/def-fqureshi/alexie/ZSCL"
cd "$REPO_ROOT/mtil"
export PYTHONPATH="$REPO_ROOT/mtil:${PYTHONPATH:-}"

DATA_LOCATION="$HOME/projects/def-fqureshi/alexie/datasets"
OUT_DIR="ckpt/11task/perclass_results"
mkdir -p "$OUT_DIR"

echo "[`date`] Evaluating full method (phase3_no_lora_v4)..."
python scripts/perclass_imagenet_eval.py \
  --checkpoint ckpt/11task/phase3_no_lora_v4/SUN397.pth \
  --data-location "$DATA_LOCATION" \
  --output "$OUT_DIR/perclass_ours.json"

echo "[`date`] Evaluating ZSCL baseline (ablation_zscl_only)..."
python scripts/perclass_imagenet_eval.py \
  --checkpoint ckpt/11task/ablation_zscl_only/SUN397.pth \
  --data-location "$DATA_LOCATION" \
  --output "$OUT_DIR/perclass_zscl.json"

echo "[`date`] Done. Results in $OUT_DIR/"
echo "Copy to local machine with:"
echo "  scp alexie@nibi.alliancecan.ca:$REPO_ROOT/mtil/$OUT_DIR/*.json ."
