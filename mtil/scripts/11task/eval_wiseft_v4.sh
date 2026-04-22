#!/bin/bash
#SBATCH --job-name=wiseft_v4
#SBATCH --time=03:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --account=def-fqureshi_gpu
#SBATCH --output=/scratch/alexie/logs/%x-%j.out

# Test-time WiSE-FT sweep on the final v4 checkpoint.
# Interpolates between pre-trained CLIP (alpha=0.0) and v4 fine-tuned (alpha=1.0)
# and evaluates on all 11 MTIL tasks + ImageNet at each alpha.

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

CKPT="ckpt/11task/phase3_no_lora_v4/SUN397.pth"
EVAL_DATASETS="Aircraft,Caltech101,CIFAR100,DTD,EuroSAT,Flowers,Food,MNIST,OxfordPet,StanfordCars,SUN397,ImageNet"

if [ ! -f "${CKPT}" ]; then
  echo "[`date`] ERROR: checkpoint not found at ${CKPT}"
  echo "Listing ckpt/11task/phase3_no_lora_v4/:"
  ls -la ckpt/11task/phase3_no_lora_v4/ || true
  exit 2
fi

for ALPHA in 0.70 0.80 0.85 0.90 0.95 1.00; do
  echo ""
  echo "=============================================="
  echo "[`date`] WiSE-FT eval with alpha=${ALPHA}"
  echo "=============================================="
  OUT_DIR="ckpt/11task/wiseft_v4/alpha_${ALPHA}"
  mkdir -p "${OUT_DIR}"

  srun python -m src.main \
    --eval-only \
    --wise-ft \
    --alpha ${ALPHA} \
    --load "${CKPT}" \
    --eval-datasets "${EVAL_DATASETS}" \
    --batch-size-eval 16 \
    --save "${OUT_DIR}" 2>&1 | tee "${OUT_DIR}/eval.log"
done

echo ""
echo "[`date`] All alphas done."
echo ""
echo "=== Summary: per-dataset accuracy by alpha ==="
for ALPHA in 0.70 0.80 0.85 0.90 0.95 1.00; do
  echo ""
  echo "--- alpha=${ALPHA} ---"
  grep -E "Evaluate on|Top-1" "ckpt/11task/wiseft_v4/alpha_${ALPHA}/eval.log" | tail -40
done
