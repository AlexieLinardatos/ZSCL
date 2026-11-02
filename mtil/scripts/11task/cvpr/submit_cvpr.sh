#!/bin/bash
# CVPR 2027 replay-buffer study: every run still missing from cvpr2027_main.tex.
# Run from mtil/ on Nibi:  bash scripts/11task/cvpr/submit_cvpr.sh [P1|P2|P3|P4|all]
# Each line is independent; comment out anything already finished. Every run
# writes to its own ckpt/11task/... dir and refuses to append to an existing one.
set -euo pipefail
D=scripts/11task
WHICH="${1:-all}"
want() { [ "$WHICH" = all ] || [ "$WHICH" = "$1" ]; }

# P1: RQ1 + RQ2 on the v4 schedule (Tab. 1, Tab. 2, 11k row)
if want P1; then
  sbatch $D/robustness/buf_11k_nord.sh                 # Replay + ZSCL, no RD
  sbatch $D/cvpr/zscl_only_v4.sh                       # distillation only
  sbatch $D/cvpr/no_anchor_no_replay_v4.sh             # fine-tune row
  sbatch $D/rebuttal_zeroshot_eval.sh                  # zero-shot row
fi

# P2: RQ5 same-MB + RQ7 precision (Tab. 4, Tab. 7)
if want P2; then
  BUDGET=2000000 sbatch $D/featrep_v0_11t.sh            # every image as fp16 embedding
  LAYER=2 PQ_M=128 BUDGET=65000 sbatch $D/remind_11t.sh            # tokens filling ~1.66 GB
  PREC=fp32 sbatch $D/featrep_v0_11t.sh
  PREC=int8 sbatch $D/featrep_v0_11t.sh
  PREC=int4 sbatch $D/featrep_v0_11t.sh
  PQ_M=128 REMIND_EXTRA="--remind_no_pq" sbatch $D/remind_11t.sh   # unquantized tokens, tap 6
fi

# P3: RQ3 image selection (Tab. 3)
if want P3; then
  for s in herding el2n gcr coverage; do SEL=$s BUDGET=1000 sbatch $D/cvpr/exemplar_sel.sh; done
  for s in herding el2n gcr; do SEL=$s BUDGET=11000 sbatch $D/cvpr/exemplar_sel.sh; done  # old runs used 13k
  sbatch $D/cvpr/alloc_equal_task_v4.sh
fi

# P4: RQ6 feature selection (Tab. 6)
if want P4; then
  BUDGET=1000 sbatch $D/featrep_v0_11t.sh                          # random baseline at 1k
  for s in herding coverage el2n; do
    SEL=$s BUDGET=1000 sbatch $D/featrep_v0_11t.sh
    SEL=$s sbatch $D/featrep_v0_11t.sh
  done
fi
