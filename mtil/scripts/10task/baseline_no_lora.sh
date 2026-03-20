#!/bin/bash
#SBATCH --job-name=10t_base_nolora
#SBATCH --time=20:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/scratch/alexie/logs/%x-%j.out
#SBATCH --signal=USR1@60

# 10-task Baseline: ZSCL, no replay, no LoRA (full fine-tuning)
# Our hyperparams: 2000 iter/task, avg_freq=50, lr=1e-5

set -euo pipefail
mkdir -p /scratch/alexie/logs
echo "[`date`] Host: $(hostname)"
nvidia-smi

module load cuda/12.6
module load python/3.11.5

ENV_DIR="$SLURM_TMPDIR/env"
echo "[`date`] Checking whether module Python has tkinter..."
if python -c "import tkinter" >/dev/null 2>&1; then
  echo "[`date`] tkinter OK. Creating venv..."
  python -m venv "$ENV_DIR"
  source "$ENV_DIR/bin/activate"
else
  echo "[`date`] tkinter missing. Falling back to conda..."
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
  python -c "import tkinter; print('tkinter ok via conda')"
fi

which python; python -V; which pip
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install tqdm ftfy regex wilds pandas
pip install git+https://github.com/modestyachts/ImageNetV2_pytorch

REPO_ROOT="$HOME/projects/def-fqureshi/alexie/ZSCL"
cd "$REPO_ROOT/mtil"
mkdir -p logs

SAVE_PATH="ckpt/10task/baseline_no_lora"
mkdir -p "${SAVE_PATH}"

EVAL_DATASETS="Aircraft,Caltech101,CIFAR100,DTD,EuroSAT,Flowers,Food,MNIST,OxfordPet,StanfordCars,ImageNet"
TASKS=(Aircraft Caltech101 CIFAR100 DTD EuroSAT Flowers Food MNIST OxfordPet StanfordCars)

# Zero-shot eval
echo "[`date`] Zero-shot evaluation..."
srun python -m src.main \
  --train-mode=whole \
  --train-dataset=Aircraft \
  --lr=1e-5 \
  --ls 0.2 \
  --iterations 0 \
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
  --eval-interval 500

PREV_CKPT="${SAVE_PATH}/Aircraft.pth"

for i in "${!TASKS[@]}"; do
  TASK="${TASKS[$i]}"

  if [ -f "${SAVE_PATH}/${TASK}.pth" ]; then
    echo "[`date`] Skipping ${TASK} (checkpoint exists)"
    PREV_CKPT="${SAVE_PATH}/${TASK}.pth"
    continue
  fi

  echo "[`date`] Training task $((i+1))/${#TASKS[@]}: ${TASK}"
  srun python -m src.main \
    --train-mode=whole \
    --train-dataset="${TASK}" \
    --lr=1e-5 \
    --ls 0.2 \
    --iterations 2000 \
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
    --custom-finetune \
    --load "${PREV_CKPT}" \
    --ref-model "${PREV_CKPT}" \
    --start-iteration 0

  PREV_CKPT="${SAVE_PATH}/${TASK}.pth"
  echo "[`date`] Done: ${TASK}"
done

echo "[`date`] All ${#TASKS[@]} tasks complete. Checkpoints in ${SAVE_PATH}/"
