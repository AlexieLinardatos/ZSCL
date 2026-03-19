#!/bin/bash
#SBATCH --job-name=10t_replay_lora
#SBATCH --time=20:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/scratch/alexie/logs/%x-%j.out
#SBATCH --signal=USR1@60

# 10-task: ZSCL + Replay + LoRA
# Best ImageNet preservation variant from Phase 2.1

set -euo pipefail
mkdir -p /scratch/alexie/logs
echo "[`date`] Host: $(hostname)"
nvidia-smi

module load cuda/12.6
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
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install tqdm ftfy regex wilds pandas
pip install git+https://github.com/modestyachts/ImageNetV2_pytorch

REPO_ROOT="$HOME/projects/def-fqureshi/alexie/ZSCL"
cd "$REPO_ROOT/mtil"
mkdir -p logs

SAVE_PATH="ckpt/10task/replay_lora"
mkdir -p "${SAVE_PATH}"

EVAL_DATASETS="Aircraft,Caltech101,CIFAR10,CIFAR100,DTD,EuroSAT,Flowers,Food,MNIST,OxfordPet,ImageNet"
LORA_ARGS="--use_lora --lora_r 8 --lora_alpha 16 --lora_dropout 0.1"

echo "[`date`] Starting 10-task ZSCL + Replay + LoRA"

srun python -m src.main \
  --train-mode=whole \
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
  --max-evaluation-size 500 \
  $LORA_ARGS \
  --use_replay \
  --replay_budget 2000 \
  --replay_batch_size 32 \
  --replay_loss_weight 0.75 \
  --dataset_order Aircraft,Caltech101,CIFAR10,CIFAR100,DTD,EuroSAT,Flowers,Food,MNIST,OxfordPet

echo "[`date`] Done. Checkpoints in ${SAVE_PATH}/"
