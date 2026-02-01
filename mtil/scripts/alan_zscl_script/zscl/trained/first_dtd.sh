#!/bin/bash
#SBATCH --job-name=zscl_Dtd
#SBATCH --time=02:30:00                 # max time
#SBATCH --mem=32GB                      # memory
#SBATCH --cpus-per-task=4               # number of CPU cores
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/scratch/alexie/logs/%x-%j.out
#SBATCH --signal=USR1@60
#SBATCH --requeue

set -euo pipefail
mkdir -p /scratch/alexie/logs

# IMPORTANT:
# Make sure this exists BEFORE you submit (Slurm won't create parent dirs for --output):
#   mkdir -p /scratch/alexie/ZSCL/mtil/logs

# Handle Slurm's warning signal (USR1) 60s before walltime: requeue instead of dying silently
handle_usr1 () {
  echo "[`date`] Caught USR1 (about to hit walltime). Requeuing job ${SLURM_JOB_ID}..."
  scontrol requeue "$SLURM_JOB_ID"
}
trap 'handle_usr1' USR1

echo "[`date`] Job starting on host: $(hostname)"
nvidia-smi

module load python/3.11.5
module load cuda/12.6

source /scratch/alexie/venvs/zscl/bin/activate

# Optional sanity checks
which python
python -V
which pip

# Cache pip downloads per-job (reduces repeated downloads)
export PIP_CACHE_DIR="$SLURM_TMPDIR/pip-cache"

# If your code uses a DATA_LOCATION, set it explicitly for the cluster
# (Change this path to wherever you stage datasets on Sharknet)
export DATA_LOCATION="/scratch/alexie/data"
mkdir -p "$DATA_LOCATION"

# Make expected dataset aliases
ln -sfn "$DATA_LOCATION/imagenet_10classes" "$DATA_LOCATION/ImageNet"

# Quick sanity checks (fail early with a clear message)
test -d "$DATA_LOCATION/ImageNet/Data/CLS-LOC/train" || { echo "Missing ImageNet train at $DATA_LOCATION/ImageNet/Data/CLS-LOC/train"; exit 2; }
test -d "$DATA_LOCATION/conceptual_captions/cc_data/val" || { echo "Missing CC at $DATA_LOCATION/conceptual_captions/cc_data/val"; exit 2; }
test -f "$DATA_LOCATION/conceptual_captions/Validation_GCC-1.1.0-Validation_output.csv" || { echo "Missing CC CSV in $DATA_LOCATION/conceptual_captions"; exit 2; }
test -f "$DATA_LOCATION/MNIST/raw/train-images-idx3-ubyte.gz" || { echo "Missing MNIST raw gz in $DATA_LOCATION/MNIST/raw"; exit 2; }


REPO_ROOT="$HOME/projects/def-fqureshi/alexie/ZSCL"
cd "$REPO_ROOT/mtil"
mkdir -p logs ckpt/clean/5000_iter/zscl/trained

# ---- Python deps ----
# NOTE: If compute nodes have no internet, these installs will fail.
# In that case, you should build an env once on a login node and reuse it.

# ---- Experiment config ----
TARGET_DATASET="DTD"

SAVE_PATH="ckpt/clean/5000_iter/zscl/trained"
MODEL_PATH="${SAVE_PATH}/DTD_trained"
mkdir -p "${SAVE_PATH}" "${MODEL_PATH}"

MODEL_NAME="${TARGET_DATASET}.pth"
BASE_CKPT="${SAVE_PATH}/${MODEL_NAME}"

DATASETS="DTD,MNIST,EuroSAT,Flowers"

# ---- Stage 1: create an initial checkpoint + quick eval ----
# (We use iterations=1 instead of 0 to ensure a checkpoint file is actually written.)
echo "[`date`] Stage 1: warm start + eval (writing ${BASE_CKPT})"

srun python -m src.main \
  --train-mode=whole \
  --train-dataset="${TARGET_DATASET}" \
  --lr=1e-5 \
  --ls 0.2 \
  --iterations 1 \
  --method ZSCL \
  --image_loss \
  --text_loss \
  --we \
  --avg_freq 100 \
  --l2 1 \
  --ref-dataset ImageNet \
  --ref-sentences conceptual_captions \
  --save "${SAVE_PATH}" \
  --eval-datasets "${DATASETS}" \
  --eval-interval 250 \
  --custom-finetune \
  --max-evaluation-size 500

# Confirm the checkpoint exists before stage 2
if [ ! -f "${BASE_CKPT}" ]; then
  echo "[`date`] ERROR: Expected checkpoint not found: ${BASE_CKPT}"
  echo "Your code may not be writing checkpoints at iterations=1. Check src.main saving logic."
  exit 1
fi

# ---- Stage 2: full training run ----
echo "[`date`] Stage 2: training 5000 iterations (loading ${BASE_CKPT})"

srun python -m src.main \
  --train-mode=whole \
  --train-dataset="${TARGET_DATASET}" \
  --lr=1e-5 \
  --ls 0.2 \
  --iterations 5000 \
  --method ZSCL \
  --image_loss \
  --text_loss \
  --we \
  --avg_freq 100 \
  --l2 1 \
  --ref-dataset ImageNet \
  --ref-sentences conceptual_captions \
  --save "${MODEL_PATH}" \
  --eval-datasets "${DATASETS}" \
  --eval-interval 250 \
  --custom-finetune \
  --max-evaluation-size 500 \
  --load "${BASE_CKPT}" \
  --start-iteration 0

echo "[`date`] Job finished."
