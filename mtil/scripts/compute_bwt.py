"""
Compute Backward Transfer (BWT) and per-task forgetting from task_summary.csv.

BWT_i = R[T, i] - R[i, i]   where R[i, j] = acc on task j after training task i.
BWT < 0 means forgetting; mean Forgetting = -mean(BWT).

Reference: Lopez-Paz & Ranzato, "Gradient Episodic Memory for Continual
Learning", NeurIPS 2017.

Usage (from mtil/):
    python scripts/compute_bwt.py [path_to_task_summary.csv ...]

Defaults to ckpt/11task/phase3_no_lora_v4/task_summary.csv if no args given.
Pass multiple paths to compare runs side by side.
"""
import csv
import os
import sys

DEFAULT = "ckpt/11task/phase3_no_lora_v4/task_summary.csv"


def bwt_from_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    meta = {"task_idx", "task_name", "avg"}
    task_cols = [c for c in rows[0].keys() if c not in meta and c != "ImageNet"]
    final_row = rows[-1]
    out = []
    for r in rows:
        task = r["task_name"]
        if task in task_cols and r.get(task, "") and final_row.get(task, ""):
            peak = float(r[task])
            final = float(final_row[task])
            out.append((task, peak, final, final - peak))
    return out


def main():
    paths = sys.argv[1:] if len(sys.argv) > 1 else [DEFAULT]
    for path in paths:
        if not os.path.exists(path):
            print(f"MISSING: {path}")
            continue
        data = bwt_from_csv(path)
        if not data:
            print(f"EMPTY: {path}")
            continue
        print(f"\n{'='*60}\nBWT report: {path}\n{'='*60}")
        print(f"{'Task':<15} {'Peak':>8} {'Final':>8} {'BWT_i':>8}")
        print("-" * 60)
        for task, peak, final, bwt in data:
            print(f"{task:<15} {peak:>8.2f} {final:>8.2f} {bwt:>+8.2f}")
        bwts = [b for _, _, _, b in data]
        excl_last = bwts[:-1] if len(bwts) > 1 else bwts
        print("-" * 60)
        print(f"{'Mean BWT (all)':<24} {sum(bwts)/len(bwts):>+.2f}")
        print(f"{'Mean BWT (excl. last)':<24} {sum(excl_last)/len(excl_last):>+.2f}")
        print(f"{'Mean Forgetting':<24} {-sum(excl_last)/len(excl_last):>.2f}")


if __name__ == "__main__":
    main()
