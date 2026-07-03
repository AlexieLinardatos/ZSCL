#!/bin/bash
#SBATCH --job-name=11t_mtm_v3_a01
#SBATCH --time=72:00:00
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --account=def-fqureshi_gpu
#SBATCH --output=/scratch/alexie/logs/%x-%j.out
#SBATCH --signal=USR1@60

# Trajectory-Aware Multi-Teacher Distillation — v3 (data-driven weighting).
# Headline proof-of-concept for NS3.
#
# Diff vs phase3_no_lora_11t_v4.sh (ExRD baseline):
#   + --use_multi_teacher_merge
#   + --merge_strategy data_driven
#   + --merge_alpha 0.1
#   + --merge_softmax_temp 1.0
#   + --merge_signature_batches 10
#   + --merge_dtype fp16
#
# Everything else (lr, iterations, λ_RTD, replay budget, per-task schedule)
# identical to ExRD v4. No LLM-anchor args.
#
# Acceptance bar for proof-of-concept:
#   Transfer ≥ 69.5   AND   Last not regressing more than 0.3 pp from 86.3.

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
# --no-index: install only from the cluster's local wheelhouse (no internet on
# compute nodes). torch/torchvision + the CLIP/eval deps.
pip install --no-index torch torchvision
pip install --no-index tqdm ftfy regex pandas scipy
pip install --no-index wandb
export WANDB_MODE=offline   # log locally; no calls to wandb.ai from the node

# Allow the CUDA allocator to grow segments — avoids OOM fragmentation on the
# 40GB MIG slice when the merged teacher + student are both resident.
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_ROOT="$HOME/projects/def-fqureshi/alexie/ZSCL"
# cd into the deeper phase3 package so `python -m phase3.train_phase3`
# resolves to scripts/4task/phase3/phase3/ (which has the multi-teacher
# merge args/logic) rather than the stale top-level mtil/phase3/.
cd "$REPO_ROOT/mtil/scripts/4task/phase3"
export PYTHONPATH="$REPO_ROOT/mtil:${PYTHONPATH:-}"
mkdir -p "$REPO_ROOT/mtil/logs"

SAVE_PATH="$REPO_ROOT/mtil/ckpt/11task/phase3_multi_teacher_merge_v3_a01"
mkdir -p "${SAVE_PATH}"

EVAL_DATASETS="Aircraft,Caltech101,CIFAR100,DTD,EuroSAT,Flowers,Food,MNIST,OxfordPet,StanfordCars,SUN397,ImageNet"

echo "[`date`] Starting NS3 v3 — data-driven multi-teacher merge (α=0.1, τ=1.0)"

# ---------------------------------------------------------------------------
# FLAG LEGEND (inline comments can't go inside the \-continued command below)
#
#   ExRD baseline block (identical to phase3_no_lora_11t_v4.sh):
#     --train-mode=whole        fine-tune the whole CLIP model
#     --lr / --ls / --iterations base LR, label smoothing, fallback iter count
#     --method ZSCL             enable ZSCL reference distillation
#     --image_loss --text_loss  both branches of the ZSCL distillation
#     --we --avg_freq 50        weight averaging (WiSE) every 50 steps
#     --l2 1                    L2-to-reference weight penalty
#     --ref-dataset / --ref-sentences  ImageNet imgs + Conceptual Captions text
#     --use_replay --replay_*   episodic replay buffer (budget 11k, bs 8)
#     --lambda_replay_teacher_distill 0.3   replay teacher-distill (RTD) weight
#     --dataset_order / --task_iterations   11-task order + per-task step budget
#
#   Multi-teacher merge (NS3, this experiment):
#     --use_multi_teacher_merge      turn on the merged teacher (off => ExRD)
#     --merge_strategy data_driven   w_i = softmax(cos(s_t,s_i)/τ)·α  (v3)
#     --merge_alpha 0.1              total mass α spread over prior task-vectors
#     --merge_softmax_temp 1.0       τ in the signature-similarity softmax
#     --merge_signature_batches 10   #batches used to estimate each task sig s_t
#     --merge_dtype fp16             store task-vector deltas on disk as fp16
# ---------------------------------------------------------------------------
srun python -m phase3.train_phase3 \
  --train-mode=whole \
  --lr=5e-6 \
  --ls 0.1 \
  --iterations 1500 \
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
  --use_replay \
  --replay_budget 11000 \
  --replay_batch_size 8 \
  --replay_loss_weight 1.0 \
  --batch-size-eval 16 \
  --dataset_order Aircraft,Caltech101,CIFAR100,DTD,EuroSAT,Flowers,Food,MNIST,OxfordPet,StanfordCars,SUN397 \
  --lambda_replay_teacher_distill 0.3 \
  --task_iterations "Aircraft:3000,Caltech101:1000,CIFAR100:1500,DTD:1500,EuroSAT:1000,Flowers:1500,Food:1500,MNIST:800,OxfordPet:1500,StanfordCars:3000,SUN397:5000" \
  --use_multi_teacher_merge \
  --merge_strategy data_driven \
  --merge_alpha 0.1 \
  --merge_softmax_temp 1.0 \
  --merge_signature_batches 10 \
  --merge_dtype fp16

echo "[`date`] Done. Checkpoints in ${SAVE_PATH}/"
