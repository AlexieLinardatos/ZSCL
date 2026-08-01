#!/bin/bash
#SBATCH --job-name=reb_smoke
#SBATCH --time=02:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --account=def-fqureshi_gpu
#SBATCH --output=/scratch/alexie/logs/%x-%j.out

# End-to-end mock run of all four RD control variants, at 1/1000th the cost.
#
# Run this BEFORE submitting the real 13.5 h controls.  The failed 2026-07-28
# attempt proved the controls parse, patch and build their models, but it died
# at wandb.init on task 1 — before a replay buffer exists, so the RD term had
# never actually executed.  This job reaches task 2 in minutes and therefore
# exercises the parts that only run once a buffer is populated:
#
#   * the patched RD loss being called and returning a finite, non-zero value
#   * --rd_image_source dispatching to current-task / reference images
#   * --rd_teacher prev_task loading the task-1 checkpoint as a teacher and
#     re-deriving its caption embeddings
#   * peak GPU memory with the extra teacher resident on a 40 GB MIG slice
#
# Two tasks (MNIST -> EuroSAT), 20 iterations each, 200-exemplar buffer.
# Accuracy from this job is meaningless; only the assertions at the end matter.

set -uo pipefail
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
pip install --no-index wandb
export WANDB_MODE=offline

export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_ROOT="$HOME/projects/def-fqureshi/alexie/ZSCL"
cd "$REPO_ROOT/mtil"
export PYTHONPATH="$REPO_ROOT/mtil/scripts/4task/phase3:$REPO_ROOT/mtil/scripts/rebuttal:${PYTHONPATH:-}"

SMOKE_ROOT="/scratch/alexie/ckpt/rebuttal/smoke"
rm -rf "${SMOKE_ROOT}"
mkdir -p "${SMOKE_ROOT}"

# The stub-level unit test first: catches wiring errors in 2 s, no GPU needed.
echo "=========== unit test (rd_controls wiring) ==========="
python scripts/rebuttal/test_rd_controls.py || echo "UNIT TEST FAILED"

run_variant () {
  local name="$1"; shift
  local save="${SMOKE_ROOT}/${name}"
  mkdir -p "${save}"
  echo
  echo "=========== smoke variant: ${name}  ($*) ==========="
  # Same flags as the real controls, scaled down; keeps --we/--l2/ZSCL on so
  # the loss composition and memory profile match the full runs.
  python -m rd_controls.run_rd_control \
    --train-mode=whole \
    --lr=5e-6 \
    --ls 0.1 \
    --iterations 20 \
    --loss-interval 5 \
    --method ZSCL \
    --image_loss \
    --text_loss \
    --we \
    --avg_freq 50 \
    --l2 1 \
    --ref-dataset ImageNet \
    --ref-sentences conceptual_captions \
    --save "${save}" \
    --eval-datasets "MNIST,EuroSAT" \
    --eval-interval 1000 \
    --use_replay \
    --replay_budget 200 \
    --replay_batch_size 4 \
    --replay_loss_weight 1.0 \
    --batch-size-eval 16 \
    --dataset_order MNIST,EuroSAT \
    --lambda_replay_teacher_distill 0.3 \
    "$@" > "${save}/run.log" 2>&1
  echo "exit=$? (full output: ${save}/run.log)"
  tail -n 3 "${save}/run.log"
}

run_variant replay      --rd_image_source replay
run_variant current     --rd_image_source current
run_variant reference   --rd_image_source reference
run_variant prev_task   --rd_teacher prev_task

echo
echo "================== ASSERTIONS =================="
python - "${SMOKE_ROOT}" <<'PY'
import csv, os, sys

root = sys.argv[1]
variants = ["replay", "current", "reference", "prev_task"]
# Task 2 of the smoke sequence; RD can only fire once a buffer exists.
LOSS_CSV = "losses_EuroSAT.csv"
failures = []

print(f"{'variant':<12}{'reached t2':>11}{'RD rows':>9}{'RD first':>10}"
      f"{'RD last':>10}{'zscl':>9}  note")
for v in variants:
    d = os.path.join(root, v)
    log = os.path.join(d, "run.log")
    csv_path = os.path.join(d, LOSS_CSV)
    note = ""

    reached = os.path.exists(csv_path)
    rd_vals, zscl_last = [], None
    if reached:
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        rd_vals = [float(r["replay_teacher"]) for r in rows if r.get("replay_teacher")]
        if rows:
            zscl_last = float(rows[-1]["zscl"])

    ok = reached and rd_vals and any(x > 0 for x in rd_vals)
    if not reached:
        failures.append(f"{v}: never reached task 2 (no {LOSS_CSV}) — see {log}")
    elif not rd_vals:
        failures.append(f"{v}: task 2 ran but the RD term was never logged")
    elif not any(x > 0 for x in rd_vals):
        failures.append(f"{v}: RD stayed exactly 0 — the loss is not being applied")
    if any(x != x or x in (float('inf'), float('-inf')) for x in rd_vals):
        failures.append(f"{v}: RD produced NaN/inf")

    if v == "prev_task":
        built = os.path.exists(log) and any(
            "Building previous-task RD teacher" in line for line in open(log, errors="ignore"))
        note = "teacher built from task-1 ckpt" if built else "TEACHER NOT BUILT"
        if not built:
            failures.append("prev_task: never loaded the previous-task checkpoint "
                            "(fell back to the frozen teacher — the control is a no-op)")

    print(f"{v:<12}{str(reached):>11}{len(rd_vals):>9}"
          f"{(f'{rd_vals[0]:.4f}' if rd_vals else '-'):>10}"
          f"{(f'{rd_vals[-1]:.4f}' if rd_vals else '-'):>10}"
          f"{(f'{zscl_last:.2f}' if zscl_last is not None else '-'):>9}  {note}")

print()
print("Expected pattern: all four reach task 2 with non-zero RD; the prev_task "
      "run starts near 0 (student == teacher at task start) while the frozen-teacher "
      "runs start well above 0; 'replay', 'current' and 'reference' differ from each "
      "other because they distil different images.")
print()
if failures:
    print("FAILED:")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("ALL SMOKE ASSERTIONS PASSED — safe to submit the 13.5 h controls.")
PY

echo "[`date`] Smoke test finished. Artifacts under ${SMOKE_ROOT}"
