"""
Compute Last, Avg, and Transfer metrics from task_summary.csv.

Usage:
    python scripts/compute_metrics.py <path_to_task_summary.csv>

Metrics:
    Last     = mean accuracy of all tasks in the FINAL row (after last task trained)
    Avg      = mean of diagonal (accuracy of task i right after training task i)
    Transfer = mean ImageNet accuracy across all rows (zero-shot preservation)
"""

import csv
import sys
import os

def compute_metrics(csv_path):
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("Empty CSV.")
        return

    # Infer task columns (everything except task_idx, task_name, avg, ImageNet)
    all_cols = list(rows[0].keys())
    meta_cols = {"task_idx", "task_name", "avg"}
    task_cols = [c for c in all_cols if c not in meta_cols and c != "ImageNet"]

    print(f"Tasks found: {task_cols}\n")
    print(f"{'Task':<15} {'Trained After':>14}   {'Accuracy':>10}")
    print("-" * 45)

    # ---- Avg: diagonal ----
    diag_vals = []
    for row in rows:
        task_name = row["task_name"]
        if task_name in task_cols:
            val = row.get(task_name, "")
            if val:
                acc = float(val)
                diag_vals.append(acc)
                print(f"  {task_name:<13} {'after '+task_name:>14}   {acc:>9.2f}%")

    print()

    # ---- Last: final row ----
    last_row = rows[-1]
    last_task = last_row["task_name"]
    last_vals = []
    print(f"Last row = after '{last_task}':")
    for tc in task_cols:
        val = last_row.get(tc, "")
        if val:
            acc = float(val)
            last_vals.append(acc)
            print(f"  {tc:<20} {acc:.2f}%")

    print()

    # ---- Transfer: ImageNet column ----
    transfer_vals = []
    print("ImageNet (Transfer) per row:")
    for row in rows:
        val = row.get("ImageNet", "")
        if val:
            acc = float(val)
            transfer_vals.append(acc)
            print(f"  After {row['task_name']:<15} ImageNet = {acc:.2f}%")

    print()
    print("=" * 45)
    print(f"  Avg      = {sum(diag_vals)/len(diag_vals):.2f}%  (over {len(diag_vals)} tasks)")
    print(f"  Last     = {sum(last_vals)/len(last_vals):.2f}%  (over {len(last_vals)} tasks in final row)")
    print(f"  Transfer = {sum(transfer_vals)/len(transfer_vals):.2f}%  (mean ImageNet, {len(transfer_vals)} rows)")
    print("=" * 45)

    if len(diag_vals) < len(task_cols):
        missing = set(task_cols) - set(r["task_name"] for r in rows)
        print(f"\n  NOTE: Missing rows for: {missing}")
        print("  Metrics are partial — rerun after final task completes.")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "ckpt/11task/phase3_no_lora_v2/task_summary.csv"
    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)
    compute_metrics(path)
