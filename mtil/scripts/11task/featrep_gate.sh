#!/bin/bash
#SBATCH --job-name=featrep_gate
#SBATCH --time=02:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --account=def-fqureshi_gpu
#SBATCH --output=/scratch/alexie/logs/%x-%j.out

# START HERE for the feature-replay extension. Self-contained: installs its own
# env, runs both pre-flight checks, prints a verdict. No training.
#
# Two steps, cheapest first:
#
#   1. Estimator self-test. Synthetic drift with a known answer, no CLIP and no
#      data. Confirms the four drift estimators are wired correctly before any
#      GPU time goes into them. Hard-fails the job if one makes features worse.
#
#   2. Drift gate. Re-encodes a saved pixel replay buffer with every per-task
#      checkpoint of a finished run and reports, per task, the cosine between a
#      feature as stored and the same feature under the final encoder. This is
#      the go/no-go for the whole direction:
#
#        > 0.95        stored features stay valid -> pure feature replay is
#                      licensed, and the reason is that ZSCL+RD hold the image
#                      encoder nearly still. Strong figure in its own right.
#        0.85 - 0.95   drift is real -> the adaptation ladder (--feature_adapt)
#                      is doing the work, and becomes the contribution.
#        < 0.85        drift dominates -> pure feature replay will not hold.
#
#      Run against the ZSCL+RD run and, when available, a no-ZSCL control using
#      the *same* buffer images, so the contrast isolates the trajectory rather
#      than the exemplars.
#
# Optional environment overrides:
#   RD_RUN=phase3_no_lora_v4        run whose checkpoints + buffer to measure
#   CTRL_RUN=ablation_no_zscl_no_rd control run for contrast ("" to skip)
#   QUICK=1                         subsample to 200 exemplars/task (~2 min)
#
# Next step after this passes: sbatch scripts/11task/featrep_smoke.sh

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

RESULTS_DIR="$REPO_ROOT/mtil/thesis_results/featrep_gate"
mkdir -p "${RESULTS_DIR}"

DATASET_ORDER="Aircraft,Caltech101,CIFAR100,DTD,EuroSAT,Flowers,Food,MNIST,OxfordPet,StanfordCars,SUN397"

RD_RUN="${RD_RUN:-phase3_no_lora_v4}"
CTRL_RUN="${CTRL_RUN:-ablation_no_zscl_no_rd}"

QUICK_FLAG=""
if [[ "${QUICK:-0}" == "1" ]]; then
  QUICK_FLAG="--max-per-task 200"
  echo "[`date`] QUICK mode: subsampling to 200 exemplars/task"
fi

# ---------------------------------------------------------------------------
# Step 1: estimator self-test (CPU, seconds)
# ---------------------------------------------------------------------------
echo ""
echo "=========================================================="
echo "[`date`] STEP 1/2 — drift estimator self-test"
echo "=========================================================="

python -m src.test_feature_adaptation

echo "[`date`] Estimators OK."

# ---------------------------------------------------------------------------
# Step 2: drift gate
# ---------------------------------------------------------------------------
# Checkpoint dirs have moved between /scratch and the repo across runs, so
# search both rather than hardcoding one and failing on the other.
find_run_dir () {
  local run_name="$1"
  local root
  for root in \
      "$REPO_ROOT/mtil/ckpt/11task" \
      "/scratch/alexie/ckpt/11task" \
      "$REPO_ROOT/mtil/ckpt" \
      "/scratch/alexie/ckpt"; do
    if [[ -f "${root}/${run_name}/replay_buffer_memory.pt" ]]; then
      echo "${root}/${run_name}"
      return 0
    fi
  done
  return 1
}

find_ckpt_dir () {
  local run_name="$1"
  local root
  for root in \
      "$REPO_ROOT/mtil/ckpt/11task" \
      "/scratch/alexie/ckpt/11task" \
      "$REPO_ROOT/mtil/ckpt" \
      "/scratch/alexie/ckpt"; do
    if compgen -G "${root}/${run_name}/*.pth" > /dev/null; then
      echo "${root}/${run_name}"
      return 0
    fi
  done
  return 1
}

echo ""
echo "=========================================================="
echo "[`date`] STEP 2/2 — representation drift gate"
echo "=========================================================="

if ! BUFFER_DIR="$(find_run_dir "${RD_RUN}")"; then
  echo ""
  echo "[`date`] ERROR: no replay_buffer_memory.pt found for run '${RD_RUN}'."
  echo "The gate needs a saved *pixel* buffer to re-encode. Locate one with:"
  echo "    find \$HOME/projects/def-fqureshi/alexie/ZSCL /scratch/alexie \\"
  echo "         -name replay_buffer_memory.pt 2>/dev/null"
  echo "then re-submit with RD_RUN=<that run's directory name>."
  exit 4
fi

BUFFER="${BUFFER_DIR}/replay_buffer_memory.pt"
echo "[`date`] Buffer:      ${BUFFER}"
echo "[`date`] Checkpoints: ${BUFFER_DIR}"

srun python -m src.measure_drift \
  --buffer "${BUFFER}" \
  --ckpt-dir "${BUFFER_DIR}" \
  --dataset_order "${DATASET_ORDER}" \
  --label "${RD_RUN}" \
  --out "${RESULTS_DIR}/drift_${RD_RUN}.csv" \
  ${QUICK_FLAG}

# Control arm: same buffer images, a different training trajectory. Skipped
# without failing the job, since the headline number is the ZSCL+RD one.
if [[ -n "${CTRL_RUN}" ]]; then
  if CTRL_DIR="$(find_ckpt_dir "${CTRL_RUN}")"; then
    echo ""
    echo "[`date`] Control arm: re-encoding the SAME buffer with '${CTRL_RUN}'"
    echo "[`date`] Checkpoints: ${CTRL_DIR}"
    srun python -m src.measure_drift \
      --buffer "${BUFFER}" \
      --ckpt-dir "${CTRL_DIR}" \
      --dataset_order "${DATASET_ORDER}" \
      --label "${CTRL_RUN}" \
      --out "${RESULTS_DIR}/drift_${CTRL_RUN}.csv" \
      ${QUICK_FLAG}
  else
    echo ""
    echo "[`date`] No checkpoints found for control run '${CTRL_RUN}' — skipping."
    echo "[`date`] The ZSCL+RD number above still stands on its own; the control"
    echo "[`date`] only strengthens the claim that distillation is what holds the"
    echo "[`date`] encoder still. Re-submit with CTRL_RUN=<run name> to add it."
  fi
fi

echo ""
echo "=========================================================="
echo "[`date`] Gate complete. CSVs in ${RESULTS_DIR}/"
ls -la "${RESULTS_DIR}/"
echo ""
echo "Read the verdict line printed above for each run, then:"
echo "  cosine > 0.95   -> sbatch scripts/11task/featrep_v0_11t.sh"
echo "  cosine < 0.95   -> ADAPT=lp ANCHORS=both sbatch scripts/11task/featrep_adapt_11t.sh"
echo "Either way, run scripts/11task/featrep_smoke.sh first to check the pipeline."
echo "=========================================================="
