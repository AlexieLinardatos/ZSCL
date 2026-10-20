#!/bin/bash
#SBATCH --job-name=wiseft_all
#SBATCH --time=12:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --account=def-fqureshi_gpu
#SBATCH --output=/scratch/alexie/logs/%x-%A_%a.out
#SBATCH --array=0-2

# Test-time WiSE-FT on EVERY task checkpoint of a finished 11-task run, so the
# full accuracy matrix (and with it the canonical upper-triangle Transfer) can be
# computed. eval_wiseft_v4.sh only merged the final checkpoint, which gives Last
# and an ImageNet number but not Transfer.
#
# Each task checkpoint is merged with zero-shot CLIP,
#   theta = (1 - alpha) * theta_zeroshot + alpha * theta_task,
# evaluated on all 12 datasets, and written as one row of a task_summary.csv in
# the same format the trainer writes, so compute_metrics.py reads it unchanged.
#
#   index -> alpha
#   0  1.00   no merge. NOT the same as the run's own task_summary.csv: that one
#             evaluated the live model, while the saved checkpoint is the
#             weight-averaged (--we) model. This row is the averaged model.
#   1  0.95
#   2  0.90
#
#   sbatch scripts/11task/eval_wiseft_alltasks.sh                     # REMIND R1
#   RUN=phase3_no_lora_v4 sbatch scripts/11task/eval_wiseft_alltasks.sh
#   python scripts/compute_metrics.py ckpt/11task/wiseft_all/<RUN>/alpha_<A>/task_summary.csv
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

RUN="${RUN:-remind_l2_m128_rdnone}"
ALPHAS=(1.00 0.95 0.90)
ALPHA=${ALPHAS[$SLURM_ARRAY_TASK_ID]}
TASKS=(Aircraft Caltech101 CIFAR100 DTD EuroSAT Flowers Food MNIST OxfordPet StanfordCars SUN397)
EVAL_DATASETS="Aircraft,Caltech101,CIFAR100,DTD,EuroSAT,Flowers,Food,MNIST,OxfordPet,StanfordCars,SUN397,ImageNet"

OUT="ckpt/11task/wiseft_all/${RUN}/alpha_${ALPHA}"
mkdir -p "${OUT}"
SUMMARY="${OUT}/task_summary.csv"
if [ -s "${SUMMARY}" ]; then
  echo "ERROR: ${SUMMARY} exists. Move it before re-running."
  exit 1
fi

for T in "${TASKS[@]}"; do
  if [ ! -f "ckpt/11task/${RUN}/${T}.pth" ]; then
    echo "ERROR: missing ckpt/11task/${RUN}/${T}.pth"
    ls -la "ckpt/11task/${RUN}/" || true
    exit 2
  fi
done

echo "task_idx,task_name,${EVAL_DATASETS},avg" > "${SUMMARY}.tmp"

for i in "${!TASKS[@]}"; do
  T=${TASKS[$i]}
  LOG="${OUT}/eval_${T}.log"
  echo "[`date`] ${RUN}  alpha=${ALPHA}  checkpoint ${i}: ${T}"

  srun python -m src.main \
    --eval-only \
    --wise-ft \
    --alpha "${ALPHA}" \
    --load "ckpt/11task/${RUN}/${T}.pth" \
    --eval-datasets "${EVAL_DATASETS}" \
    --batch-size-eval 16 \
    --save "${OUT}/tmp_${T}" 2>&1 | tee "${LOG}"

  # One "Top-1 accuracy: X" line per eval dataset, in --eval-datasets order.
  ACCS=$(grep -oP "Top-1 accuracy: \K[0-9.]+" "${LOG}" || true)
  N=$(echo "${ACCS}" | grep -c . || true)
  if [ "${N}" -ne 12 ]; then
    echo "ERROR: expected 12 accuracies in ${LOG}, found ${N}."
    exit 3
  fi
  ROW=$(echo "${ACCS}" | paste -sd, -)
  AVG=$(echo "${ACCS}" | awk '{s+=$1} END {printf "%.4f", s/NR}')
  echo "${i},${T},${ROW},${AVG}" >> "${SUMMARY}.tmp"
done

mv "${SUMMARY}.tmp" "${SUMMARY}"
echo "[`date`] Done: ${SUMMARY}"
python scripts/compute_metrics.py "${SUMMARY}"
