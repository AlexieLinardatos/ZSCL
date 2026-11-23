#!/bin/bash
# Local (lab A4000) version of exemplar_sel.sh: same flags, no SLURM.
# Runs the selection strategies one after another, most important first.
# Usage (from mtil/):
#   DATA=/path/to/data bash scripts/11task/cvpr/exemplar_sel_local.sh
#   SELS="herding" DATA=... bash scripts/11task/cvpr/exemplar_sel_local.sh   # one method
#   SMOKE=1 DATA=... bash ...   # ~quick VRAM/data check, writes to *_smoke
# Rerunning the same command resumes from the last completed task.

set -uo pipefail
DATA="${DATA:?set DATA to the dataset root (holds all 11 datasets, ImageNet, conceptual_captions)}"
SELS="${SELS:-coverage herding el2n gcr}"
BUDGET="${BUDGET:-11000}"
cd "$(dirname "$0")/../../.."   # -> mtil/
export PYTHONPATH="$PWD/scripts/4task/phase3:${PYTHONPATH:-}"
mkdir -p logs

ORDER=Aircraft,Caltech101,CIFAR100,DTD,EuroSAT,Flowers,Food,MNIST,OxfordPet,StanfordCars,SUN397
EVAL_DATASETS="$ORDER,ImageNet"
TASK_ITERS="Aircraft:3000,Caltech101:1000,CIFAR100:1500,DTD:1500,EuroSAT:1000,Flowers:1500,Food:1500,MNIST:800,OxfordPet:1500,StanfordCars:3000,SUN397:5000"
SUFFIX=""
if [ "${SMOKE:-0}" = 1 ]; then
  ORDER=Aircraft,Caltech101; EVAL_DATASETS=DTD; TASK_ITERS="Aircraft:50,Caltech101:50"; SUFFIX=_smoke
fi

for SEL in $SELS; do
  SAVE_PATH="ckpt/11task/cvpr/imgsel_${SEL}_b${BUDGET}${SUFFIX}"
  mkdir -p "$SAVE_PATH"
  LOG="logs/imgsel_${SEL}_b${BUDGET}${SUFFIX}.out"
  echo "[`date`] strategy=$SEL budget=$BUDGET -> $SAVE_PATH (log: $LOG)"
  python -m phase3.train_phase3 \
    --data-location "$DATA" \
    --train-mode=whole --lr=5e-6 --ls 0.1 --iterations 1500 \
    --method ZSCL --image_loss --text_loss \
    --we --avg_freq 50 --l2 1 \
    --ref-dataset ImageNet --ref-sentences conceptual_captions \
    --save "$SAVE_PATH" \
    --eval-datasets "$EVAL_DATASETS" --eval-interval 500 \
    --use_replay --replay_budget "$BUDGET" --replay_encode_share \
    --replay_batch_size 8 --replay_loss_weight 1.0 --batch-size-eval 16 \
    --dataset_order "$ORDER" \
    --lambda_replay_teacher_distill 0.3 \
    --exemplar_strategy "$SEL" \
    --task_iterations "$TASK_ITERS" 2>&1 | tee -a "$LOG"
  [ "${PIPESTATUS[0]}" -ne 0 ] && { echo "[`date`] $SEL FAILED, stopping"; exit 1; }
done
echo "[`date`] All done."
