"""
Compute Last, Avg, and Transfer metrics from task_summary.csv.

Usage:
    python scripts/compute_metrics.py <path_to_task_summary.csv>

Metrics:
    Last       = mean accuracy over the trained tasks in the FINAL row.
    Avg        = mean of the WHOLE trained-task matrix (ImageNet excluded).
                 This is the ZSCL definition and the one the results table uses.
    Diag       = mean of the diagonal (task i right after training task i).
                 Reported for reference only -- older versions of this script
                 mislabelled this as "Avg", which is ~9 points higher.
    Transfer   = reported two ways, because the project and the literature
                 disagree on the definition:
      - ImageNet column: mean ImageNet accuracy across all rows. ImageNet is
        never trained in this order, so it is a held-out zero-shot probe.
        This is the project/ExRD convention.
      - Upper triangle: mean accuracy on tasks not yet trained. This is the
        canonical ZSCL convention that ZSCL/BCL/DIKI/AFA report, so it is the
        number to use when placing a run in their tables. The first task has
        no "before" rows and so drops out of this average by construction.

Rows are assumed to be in training order, one row per task boundary.
"""

import csv
import sys
import os


def mean(vals):
    """Mean, or None for an empty list -- so a missing column prints n/a."""
    return sum(vals) / len(vals) if vals else None


def fmt(val):
    return "  n/a" if val is None else f"{val:.2f}%"


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

    # Training order, as given by the row sequence. Used to place the diagonal
    # for the upper-triangle transfer; may be shorter than task_cols on a
    # partial run.
    order = [r["task_name"] for r in rows]

    print(f"Tasks found: {task_cols}\n")
    print(f"{'Task':<15} {'Trained After':>14}   {'Accuracy':>10}")
    print("-" * 45)

    # ---- Diagonal (reference only) ----
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

    # ---- Avg: mean of the whole trained-task matrix ----
    matrix_vals = []
    for row in rows:
        for tc in task_cols:
            val = row.get(tc, "")
            if val:
                matrix_vals.append(float(val))

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

    # ---- Transfer (a): ImageNet held-out column ----
    transfer_vals = []
    print("ImageNet (Transfer, project convention) per row:")
    for row in rows:
        val = row.get("ImageNet", "")
        if val:
            acc = float(val)
            transfer_vals.append(acc)
            print(f"  After {row['task_name']:<15} ImageNet = {acc:.2f}%")
    if not transfer_vals:
        print("  (no ImageNet column in this run)")

    print()

    # ---- Transfer (b): upper triangle, canonical ZSCL ----
    # Row i is the state after training task i, so columns order[i+1:] are
    # tasks this model has not seen yet.
    upper_vals = []
    per_task_upper = {}
    for i, row in enumerate(rows):
        for j in range(i + 1, len(order)):
            val = row.get(order[j], "")
            if val:
                acc = float(val)
                upper_vals.append(acc)
                per_task_upper.setdefault(order[j], []).append(acc)

    if per_task_upper:
        print("Upper-triangle transfer (canonical ZSCL) per task:")
        for tc in order:
            if tc in per_task_upper:
                col = per_task_upper[tc]
                print(f"  {tc:<20} {mean(col):.2f}%   (over {len(col)} rows)")
        print(f"  {order[0]:<20}   n/a   (first task: no untrained rows)")
        print()

    print("=" * 62)
    print(f"  Avg                 = {fmt(mean(matrix_vals))}"
          f"  (whole matrix, {len(matrix_vals)} cells, ImageNet excl.)")
    print(f"  Last                = {fmt(mean(last_vals))}"
          f"  (final row, {len(last_vals)} tasks)")
    print(f"  Transfer [ImageNet] = {fmt(mean(transfer_vals))}"
          f"  (held-out column, {len(transfer_vals)} rows)")
    print(f"  Transfer [upper-tri]= {fmt(mean(upper_vals))}"
          f"  (canonical ZSCL, {len(upper_vals)} cells)")
    print("-" * 62)
    print(f"  Diag (reference)    = {fmt(mean(diag_vals))}"
          f"  (NOT the table's Avg)")
    print("=" * 62)

    if len(diag_vals) < len(task_cols):
        missing = set(task_cols) - set(order)
        print(f"\n  NOTE: Missing rows for: {missing}")
        print("  Metrics are partial — rerun after final task completes.")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "ckpt/11task/phase3_no_lora_v2/task_summary.csv"
    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)
    compute_metrics(path)
