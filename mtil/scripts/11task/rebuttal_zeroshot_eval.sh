#!/bin/bash
#SBATCH --job-name=reb_zs_eval
#SBATCH --time=08:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --account=def-fqureshi_gpu
#SBATCH --output=/scratch/alexie/logs/%x-%j.out

# Measures pre-trained CLIP ViT-B/16 zero-shot accuracy on all 12 eval sets.
#
# Needed by scripts/rebuttal/per_task_rd_analysis.py to answer the second half
# of Reviewer zvRY's Q1 — whether RD helps everywhere or only on tasks close to
# CLIP pre-training — without hardcoding zero-shot numbers copied from another
# paper.  Writes ckpt/rebuttal/zeroshot/evaluate_all_results.csv.
#
# Evaluation only: no training, ~1 GPU-hour.

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

export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_ROOT="$HOME/projects/def-fqureshi/alexie/ZSCL"
cd "$REPO_ROOT/mtil"

ZS_DIR="/scratch/alexie/ckpt/rebuttal/zeroshot"
mkdir -p "${ZS_DIR}"

EVAL_DATASETS="Aircraft,Caltech101,CIFAR100,DTD,EuroSAT,Flowers,Food,MNIST,OxfordPet,StanfordCars,SUN397,ImageNet"

# evaluate() appends, so start from a clean file to avoid duplicated rows.
rm -f "${ZS_DIR}/evaluate_all_results.csv"

python scripts/rebuttal/save_zeroshot_ckpt.py --out "${ZS_DIR}/zeroshot.pth"

echo "[`date`] Evaluating pre-trained CLIP zero-shot on ${EVAL_DATASETS}"

srun python -m src.main \
  --eval-only \
  --train-mode=whole \
  --load "${ZS_DIR}/zeroshot.pth" \
  --eval-datasets "${EVAL_DATASETS}" \
  --batch-size-eval 16

echo "[`date`] Done. Results in ${ZS_DIR}/evaluate_all_results.csv"
